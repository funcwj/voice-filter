#!/usr/bin/env python

# wujian@2018

import os
import json
import pprint
import argparse
import random

from libs.trainer import SiSnrTrainer, get_logger
from libs.dataset import make_dataloader
from nnet import VoiceFilter
from conf import trainer_conf, nnet_conf, train_data, dev_data


def run(args):
    gpuids = tuple(map(int, args.gpus.split(",")))

    if args.checkpoint:
        os.makedirs(args.checkpoint, exist_ok=True)

    logger = get_logger(os.path.join(args.checkpoint, "trainer.log"),
                        file=True)
    logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))

    nnet = VoiceFilter(**nnet_conf)
    trainer = SiSnrTrainer(nnet,
                           gpuid=gpuids,
                           checkpoint=args.checkpoint,
                           resume=args.resume,
                           logger=logger,
                           **trainer_conf)

    data_conf = {
        "train": train_data,
        "dev": dev_data,
    }
    # dump configs
    for conf, fname in zip([nnet_conf, trainer_conf, data_conf],
                           ["mdl.json", "trainer.json", "data.json"]):
        with open(os.path.join(args.checkpoint, fname), "w") as f:
            json.dump(conf, f, indent=4, sort_keys=False)

    train_loader = make_dataloader(train=True,
                                   data_kwargs=train_data,
                                   batch_size=args.batch_size,
                                   cache_size=args.cache_size,
                                   chunk_size=args.chunk_size)
    dev_loader = make_dataloader(train=False,
                                 data_kwargs=dev_data,
                                 batch_size=args.batch_size,
                                 cache_size=args.cache_size,
                                 chunk_size=args.chunk_size)

    trainer.run(train_loader,
                dev_loader,
                eval_interval=args.eval_interval,
                num_epoches=args.epoches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to start train voice-filter, configured from conf.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epoches",
                        type=int,
                        default=50,
                        help="Number of training epoches")
    parser.add_argument("--gpus",
                        type=str,
                        default="0,1",
                        help="Training on which GPUs (one or more, egs "
                        "0, 0,1)")
    parser.add_argument("--eval-interval",
                        type=int,
                        default=3000,
                        help="Number of batches trained per epoch (for larger "
                        "training dataset)")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Directory to dump models")
    parser.add_argument("--resume",
                        type=str,
                        default="",
                        help="Exist model to resume training from")
    parser.add_argument("--chunk-size",
                        type=int,
                        default=64256,
                        help="Chunk size to feed networks")
    parser.add_argument("--batch-size",
                        type=int,
                        default=16,
                        help="Number of utterances in each batch")
    parser.add_argument("--cache-size",
                        type=int,
                        default=16,
                        help="Number of chunks cached in dataloader")
    args = parser.parse_args()
    run(args)