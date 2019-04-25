#!/usr/bin/env python

# wujian@2018

import os
import argparse

import torch as th
import numpy as np

from nnet import VoiceFilter

from libs.audio import WaveReader, write_wav
from libs.dataset import NumpyReader
from libs.evaluator import Evaluator
from libs.trainer import get_logger

from kaldi_python_io import ScriptReader

logger = get_logger(__name__)


class NnetComputer(Evaluator):
    """
    Evaluator implementation
    """

    def __init__(self, *args, **kwargs):
        super(NnetComputer, self).__init__(*args, **kwargs)

    def compute(self, mix, emb):
        with th.no_grad():
            mix = th.from_numpy(mix).to(self.device)
            emb = th.from_numpy(emb).to(self.device)
            spk = self.nnet(mix, emb)
            return spk.detach().squeeze().cpu().numpy()


def run(args):
    mix_reader = WaveReader(args.mix_scp, sr=args.fs)
    spk_embed = NumpyReader(
        args.emb_scp) if args.format == "numpy" else ScriptReader(args.emb_scp,
                                                                  matrix=False)
    os.makedirs(args.dump_dir, exist_ok=True)
    computer = NnetComputer(VoiceFilter, args.checkpoint, gpu_id=args.gpu)
    for key, mix in mix_reader:
        logger.info("Compute on utterance {}...".format(key))
        emb = spk_embed[key]
        emb = emb / (np.linalg.norm(emb, 2) + 1e-8)
        spk = computer.compute(mix, emb)
        norm = np.linalg.norm(mix, np.inf)
        # norm
        spk = spk * norm / np.max(np.abs(spk))
        write_wav(os.path.join(args.dump_dir, "{}.wav".format(key)),
                  spk,
                  sr=args.fs)
    logger.info("Compute over {:d} utterances".format(len(mix_reader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to do speaker aware separation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint", type=str, help="Directory of checkpoint")
    parser.add_argument("--mix-scp",
                        type=str,
                        required=True,
                        help="Rspecifier for input waveform")
    parser.add_argument("--emb-scp",
                        type=str,
                        required=True,
                        help="Rspecifier for speaker embeddings")
    parser.add_argument("--emb-format",
                        type=str,
                        dest="format",
                        choices=["kaldi", "numpy"],
                        default="kaldi",
                        help="Storage type for speaker embeddings")
    parser.add_argument("--gpu",
                        type=int,
                        default=-1,
                        help="GPU-id to offload model to, -1 means "
                        "running on CPU")
    parser.add_argument("--fs",
                        type=int,
                        default=16000,
                        help="Sample rate for mixture input")
    parser.add_argument("--dump-dir",
                        type=str,
                        default="spk",
                        help="Directory to dump separated speakers out")
    args = parser.parse_args()
    run(args)