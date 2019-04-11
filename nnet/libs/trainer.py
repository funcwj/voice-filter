# wujian@2018

import os
import sys
import time
import logging

from itertools import permutations
from collections import defaultdict

import torch as th
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch import autograd


def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


def get_logger(
        name,
        format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
        date_format="%Y-%m-%d %H:%M:%S",
        file=False):
    """
    Get logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(
        name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class SimpleTimer(object):
    """
    A simple timer
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60


class ProgressReporter(object):
    """
    A simple progress reporter
    """

    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.reset()

    def reset(self):
        self.loss = []
        self.timer = SimpleTimer()

    def add(self, loss):
        self.loss.append(loss)
        N = len(self.loss)
        if not N % self.period:
            avg = sum(self.loss[-self.period:]) / self.period
            self.logger.info("Processed {:d} batches "
                             "(loss = {:+.2f})...".format(N, avg))

    def report(self, details=False):
        N = len(self.loss)
        if details:
            sstr = ",".join(map(lambda f: "{:.2f}".format(f), self.loss))
            self.logger.info("Loss on {:d} batches: {}".format(N, sstr))
        return {
            "loss": sum(self.loss) / N,
            "batches": N,
            "cost": self.timer.elapsed()
        }


class Trainer(object):
    def __init__(self,
                 nnet,
                 checkpoint="checkpoint",
                 optimizer="adam",
                 gpuid=0,
                 optimizer_kwargs=None,
                 gradient_clip=None,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 logging_period=100,
                 resume=None,
                 no_impr=6):
        if not th.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )
        self.device = th.device("cuda:{}".format(gpuid[0]))
        self.gpuid = gpuid
        if checkpoint:
            os.makedirs(checkpoint, exist_ok=True)
        self.checkpoint = checkpoint
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)

        self.gradient_clip = gradient_clip
        self.logging_period = logging_period
        self.cur_epoch = 0  # zero based
        self.no_impr = no_impr

        if resume:
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume, self.cur_epoch))
            # load nnet
            nnet.load_state_dict(cpt["model_state_dict"])
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(
                optimizer, optimizer_kwargs, state=cpt["optim_state_dict"])
        else:
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True)
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6

        # logging
        self.logger.info("Model summary:\n{}".format(nnet))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.num_params))
        if gradient_clip:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(gradient_clip))

    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        th.save(
            cpt,
            os.path.join(self.checkpoint,
                         "{0}.pt.tar".format("best" if best else "last")))

    def create_optimizer(self, optimizer, kwargs, state=None):
        supported_optimizer = {
            "sgd": th.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": th.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": th.optim.Adam,  # weight_decay, lr
            "adadelta": th.optim.Adadelta,  # weight_decay, lr
            "adagrad": th.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": th.optim.Adamax  # lr, weight_decay
            # ...
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.nnet.parameters(), **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def compute_loss(self, egs):
        raise NotImplementedError

    def train(self, data_loader):
        self.logger.info("Set train mode...")
        self.nnet.train()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        # with autograd.detect_anomaly():
        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.device)

            self.optimizer.zero_grad()
            loss = self.compute_loss(egs)
            loss.backward()
            if self.gradient_clip:
                clip_grad_norm_(self.nnet.parameters(), self.gradient_clip)
            self.optimizer.step()

            reporter.add(loss.item())
        return reporter.report()

    def eval(self, data_loader):
        self.logger.info("Set eval mode...")
        self.nnet.eval()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.device)
                loss = self.compute_loss(egs)
                reporter.add(loss.item())
        return reporter.report(details=True)

    def run(self,
            train_loader,
            dev_loader,
            num_epoches=50,
            num_batch_per_epoch=4000):
        # make dilated conv faster
        th.backends.cudnn.benchmark = True
        # avoid alloc memory from gpu0
        th.cuda.set_device(self.gpuid[0])
        stats = dict()
        # check if save is OK
        self.save_checkpoint(best=False)
        cv = self.eval(dev_loader)
        best_loss = cv["loss"]
        self.logger.info("START FROM EPOCH {:d}, LOSS = {:.4f}".format(
            self.cur_epoch, best_loss))
        no_impr = 0
        stop = False
        trained_batches = 0
        train_reporter = ProgressReporter(
            self.logger, period=self.logging_period)
        # make sure not inf
        self.scheduler.best = best_loss
        # set train mode
        self.nnet.train()
        while True:
            # trained on several batches
            for egs in train_loader:
                trained_batches = (trained_batches + 1) % num_batch_per_epoch
                # update per-batch
                egs = load_obj(egs, self.device)
                self.optimizer.zero_grad()
                loss = self.compute_loss(egs)
                loss.backward()
                if self.gradient_clip:
                    clip_grad_norm_(self.nnet.parameters(), self.gradient_clip)
                self.optimizer.step()
                # record loss
                train_reporter.add(loss.item())
                # if trained on batches done, start evaluation
                if trained_batches == 0:
                    self.cur_epoch += 1
                    cur_lr = self.optimizer.param_groups[0]["lr"]
                    stats[
                        "title"] = "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(
                            cur_lr, self.cur_epoch)
                    tr = train_reporter.report()
                    stats["tr"] = "train = {:+.4f}({:.2f}m/{:d})".format(
                        tr["loss"], tr["cost"], tr["batches"])
                    cv = self.eval(dev_loader)
                    stats["cv"] = "dev = {:+.4f}({:.2f}m/{:d})".format(
                        cv["loss"], cv["cost"], cv["batches"])
                    stats["scheduler"] = ""
                    if cv["loss"] > best_loss:
                        no_impr += 1
                        stats["scheduler"] = "| no impr, best = {:.4f}".format(
                            self.scheduler.best)
                    else:
                        best_loss = cv["loss"]
                        no_impr = 0
                        self.save_checkpoint(best=True)
                    self.logger.info(
                        "{title} {tr} | {cv} {scheduler}".format(**stats))
                    # schedule here
                    self.scheduler.step(cv["loss"])
                    # flush scheduler info
                    sys.stdout.flush()
                    # save last checkpoint
                    self.save_checkpoint(best=False)
                    # reset reporter
                    train_reporter.reset()
                    # early stop or not
                    if no_impr == self.no_impr:
                        self.logger.info(
                            "Stop training cause no impr for {:d} epochs".
                            format(no_impr))
                        stop = True
                        break
                    if self.cur_epoch == num_epoches:
                        stop = True
                        break
                    # enable train mode
                    self.nnet.train()
            if stop:
                break
        self.logger.info("Training for {:d}/{:d} epoches done!".format(
            self.cur_epoch, num_epoches))


class SiSnrTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SiSnrTrainer, self).__init__(*args, **kwargs)

    def sisnr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
        Return:
        sisnr: N tensor
        """

        def l2norm(mat, keepdim=False):
            return th.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - th.mean(x, dim=-1, keepdim=True)
        s_zm = s - th.mean(s, dim=-1, keepdim=True)
        t = th.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        return 20 * th.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def compute_loss(self, egs):
        # flatten for parallel module
        self.nnet.flatten_parameters()
        # N x S
        est = th.nn.parallel.data_parallel(
            self.nnet, (egs["mix"], egs["emb"]), device_ids=self.gpuid)
        # N
        snr = self.sisnr(est, egs["ref"])
        return -th.sum(snr) / est.size(0)