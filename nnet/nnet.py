#!/usr/bin/env python

# wujian@2019

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from stft import STFT, iSTFT

EPSILON = th.finfo(th.float32).eps


class Conv2dBlock(nn.Module):
    """
    2D convolutional blocks used in VoiceFilter
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(5, 5),
                 dilation=(1, 1)):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=1,
                              dilation=dilation,
                              padding=tuple(
                                  d * (k - 1) // 2
                                  for k, d in zip(kernel_size, dilation)))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        x: N x F x T
        """
        x = self.bn(self.conv(x))
        return F.relu(x)


class VoiceFilter(nn.Module):
    """
    Reference from
        VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking
    """

    def __init__(self,
                 frame_len,
                 frame_hop,
                 round_pow_of_two=True,
                 embedding_dim=512,
                 log_mag=False,
                 mvn_mag=False,
                 lstm_dim=400,
                 linear_dim=600,
                 l2_norm=True,
                 bidirectional=False,
                 non_linear="relu"):
        super(VoiceFilter, self).__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": th.sigmoid,
            "tanh": th.tanh
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError(
                "Unsupported non-linear function: {}".format(non_linear))
        N = 2**math.ceil(
            math.log2(frame_len)) if round_pow_of_two else frame_len
        num_bins = N // 2 + 1

        self.stft = STFT(frame_len,
                         frame_hop,
                         round_pow_of_two=round_pow_of_two)
        self.istft = iSTFT(frame_len,
                           frame_hop,
                           round_pow_of_two=round_pow_of_two)
        self.cnn_f = Conv2dBlock(1, 64, kernel_size=(7, 1))
        self.cnn_t = Conv2dBlock(64, 64, kernel_size=(1, 7))
        blocks = []
        for d in range(5):
            blocks.append(
                Conv2dBlock(64, 64, kernel_size=(5, 5), dilation=(1, 2**d)))
        self.cnn_tf = nn.Sequential(*blocks)
        self.proj = Conv2dBlock(64, 8, kernel_size=(1, 1))
        self.lstm = nn.LSTM(8 * num_bins + embedding_dim,
                            lstm_dim,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.mask = nn.Sequential(
            nn.Linear(lstm_dim * 2 if bidirectional else lstm_dim, linear_dim),
            nn.ReLU(), nn.Linear(linear_dim, num_bins))
        self.non_linear = supported_nonlinear[non_linear]
        self.embedding_dim = embedding_dim
        self.l2_norm = l2_norm
        self.log_mag = log_mag
        self.bn = nn.BatchNorm1d(num_bins) if mvn_mag else None

    def flatten_parameters(self):
        self.lstm.flatten_parameters()

    def check_args(self, x, e):
        if x.dim() != e.dim():
            raise RuntimeError(
                "{} got invalid input dim: x/e = {:d}/{:d}".format(
                    self.__name__, x.dim(), e.dim()))
        if e.size(-1) != self.embedding_dim:
            raise RuntimeError("input embedding dim do not match with "
                               "network's, {:d} vs {:d}".format(
                                   e.size(-1), self.embedding_dim))

    def forward(self, x, e, return_mask=False):
        """
        x: N x S
        e: N x D
        """
        if x.dim() == 1:
            x = th.unsqueeze(x, 0)
            e = th.unsqueeze(e, 0)
        if self.l2_norm:
            e = e / th.norm(e, 2, dim=1, keepdim=True)

        # N x S => N x F x T
        mag, ang = self.stft(x)

        # clip
        y = th.clamp(mag, min=EPSILON)
        # apply log
        if self.log_mag:
            y = th.log(y)
        # apply bn
        if self.bn:
            y = self.bn(y)

        N, _, T = mag.shape
        # N x 1 x F x T
        y = th.unsqueeze(y, 1)
        # N x D => N x D x T
        e = th.unsqueeze(e, 2).repeat(1, 1, T)

        y = self.cnn_f(y)
        y = self.cnn_t(y)
        y = self.cnn_tf(y)
        # N x C x F x T
        y = self.proj(y)
        # N x CF x T
        y = y.view(N, -1, T)
        # N x (CF+D) x T
        f = th.cat([y, e], 1)
        # N x T x (CF+D)
        f = th.transpose(f, 1, 2)
        f, _ = self.lstm(f)
        # N x T x F
        m = self.non_linear(self.mask(f))
        if return_mask:
            return m
        # N x F x T
        m = th.transpose(m, 1, 2)
        # N x S
        s = self.istft(mag * m, ang, squeeze=True)
        return s


def run():
    x = th.rand(1, 2000)
    e = th.rand(1, 512)

    nnet = VoiceFilter(256, 128)
    print(nnet)
    s = nnet(x, e, return_mask=True)
    print(s.squeeze().shape)


if __name__ == "__main__":
    run()
