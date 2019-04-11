# wujian@2018

import random
import torch as th
import numpy as np

from torch.utils.data.dataloader import default_collate
from kaldi_python_io import ScriptReader

from .audio import Reader, WaveReader


def make_dataloader(train=True,
                    data_kwargs=None,
                    chunk_size=32000,
                    batch_size=16,
                    cache_size=32):
    perutt_loader = PeruttLoader(shuffle=train, **data_kwargs)
    return DataLoader(
        perutt_loader,
        train=train,
        chunk_size=chunk_size,
        batch_size=batch_size,
        cache_size=cache_size)


class NumpyReader(Reader):
    """
    Sequential/Random Reader for numpy's ndarray(*.npy) file
    """

    def __init__(self, npy_scp):
        super(NumpyReader, self).__init__(npy_scp)

    def _load(self, key):
        return np.load(self.index_dict[key])


class PeruttLoader(object):
    """
    Per Utterance Loader
    """

    def __init__(self,
                 shuffle=True,
                 mix_scp="",
                 ref_scp="",
                 emb_scp="",
                 embed_format="kaldi",
                 sr=16000):
        if embed_format not in ["kaldi", "numpy"]:
            raise RuntimeError(
                "Unknown embedding format {}".format(embed_format))
        self.mix = WaveReader(mix_scp, sr=sr)
        self.ref = WaveReader(ref_scp, sr=sr)
        self.emb = NumpyReader(
            emb_scp) if embed_format == "numpy" else ScriptReader(
                emb_scp, matrix=False)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.mix.index_keys)
        for key, mix in self.mix:
            eg = dict()
            eg["mix"] = mix
            eg["ref"] = self.ref[key]
            emb = self.emb[key]
            eg["emb"] = emb / (np.linalg.norm(emb, 2) + 1e-8)
            yield eg


class ChunkSplitter(object):
    """
    Split utterance into small chunks
    """

    def __init__(self, chunk_size, train=True, hop=16000):
        self.chunk_size = chunk_size
        self.hop = hop
        self.train = train

    def _make_chunk(self, eg, s):
        """
        Make a chunk instance, which contains:
            "emb": ndarray,
            "mix": ndarray,
            "ref": ndarray
        """
        chunk = dict()
        # support for multi-channel
        chunk["emb"] = eg["emb"]
        chunk["mix"] = eg["mix"][..., s:s + self.chunk_size]
        chunk["ref"] = eg["ref"][..., s:s + self.chunk_size]
        return chunk

    def split(self, eg):
        N = eg["mix"].shape[-1]
        # too short, throw away
        if N < self.hop:
            return []
        chunks = []
        # padding zeros
        if N < self.chunk_size:
            chunk = dict()
            P = self.chunk_size - N
            pad_width = ((0, 0), (0, P)) if eg["mix"].ndim == 2 else (0, P)
            chunk["mix"] = np.pad(eg["mix"], pad_width, "constant")
            chunk["emb"] = eg["emb"]
            chunk["ref"] = np.pad(eg["ref"], (0, P), "constant")
            chunks.append(chunk)
        else:
            # random select start point for training
            s = random.randint(0, N % self.hop) if self.train else 0
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.hop
        return chunks


class DataLoader(object):
    """
    Online dataloader for chunk-level loss
    """

    def __init__(self,
                 perutt_loader,
                 chunk_size=32000,
                 batch_size=16,
                 cache_size=16,
                 train=True):
        self.loader = perutt_loader
        self.cache_size = cache_size * batch_size
        self.batch_size = batch_size
        self.train = train
        self.splitter = ChunkSplitter(
            chunk_size, train=train, hop=chunk_size // 2)

    def _fetch_batch(self):
        while True:
            if len(self.load_list) >= self.cache_size:
                break
            try:
                eg = next(self.load_iter)
                cs = self.splitter.split(eg)
                self.load_list.extend(cs)
            except StopIteration:
                self.stop_iter = True
                break
        if self.train:
            random.shuffle(self.load_list)
        N = len(self.load_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            batch = default_collate(self.load_list[s:s + self.batch_size])
            blist.append(batch)
        # update load_list
        rn = N % self.batch_size
        if rn:
            self.load_list = self.load_list[-rn:]
        else:
            self.load_list = []
        return blist

    def __iter__(self):
        # reset flags
        self.load_iter = iter(self.loader)
        self.stop_iter = False
        self.load_list = []

        while not self.stop_iter:
            bs = self._fetch_batch()
            for obj in bs:
                yield obj