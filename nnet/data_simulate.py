#!/usr/bin/env python

# wujian@2019

import os
import csv
import random
import argparse

import tqdm
import numpy as np
from libs.audio import WaveReader, write_wav


def mix_audio(src, itf):
    """
    According to the paper, seems they do not scale speakers via SNRs
    """
    min_len = min(src.size, itf.size)
    src_beg = random.randint(0, src.size - min_len)
    itf_beg = random.randint(0, itf.size - min_len)
    src_seg = src[src_beg:src_beg + min_len]
    itf_seg = itf[itf_beg:itf_beg + min_len]
    mix_seg = src_seg + itf_seg
    scale = random.uniform(0.5, 0.9) / np.max(np.abs(mix_seg))
    return src_seg * scale, mix_seg * scale


def run(args):
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
    wave_reader = WaveReader(args.wav_scp)
    with open(os.path.join(args.dump_dir, "emb.key"), "w") as emb:
        with open(args.csv, "r") as f:
            reader = csv.reader(f)
            for ids in tqdm.tqdm(reader):
                src_id, ref_id, itf_id = ids
                emb.write("{}\t{}\n".format("_".join(ids), ref_id))
                src = wave_reader[src_id]
                itf = wave_reader[itf_id]
                src, mix = mix_audio(src, itf)
                write_wav(
                    os.path.join(args.dump_dir,
                                 "src/{}.wav".format("_".join(ids))), src)
                write_wav(
                    os.path.join(args.dump_dir,
                                 "mix/{}.wav".format("_".join(ids))), mix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to simulate data for VoiceFilter training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "wav_scp",
        type=str,
        help="Rspecifier of wave files for Librispeech dataset")
    parser.add_argument(
        "csv",
        type=str,
        help="CSV files obtained from https://github.com/google/speaker-id")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="voice_data",
        help="Directory of output data triplet")
    args = parser.parse_args()
    run(args)
