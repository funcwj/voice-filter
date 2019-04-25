#!/usr/bin/env python

# wujian@2019

import math
import torch as th

import torch.nn.functional as F
import torch.nn as nn

EPSILON = th.finfo(th.float32).eps


def init_kernel(frame_len,
                frame_hop,
                round_pow_of_two=True,
                window="sqrt_hann"):
    if window != "sqrt_hann":
        raise RuntimeError("Now only support sqrt hanning window in order "
                           "to make signal perfectly reconstructed")
    # FFT points
    N = 2**math.ceil(math.log2(frame_len)) if round_pow_of_two else frame_len
    # window
    W = th.hann_window(frame_len)**0.5
    S = 0.5 * (N * N / frame_hop)**0.5
    # F x N/2+1 x 2
    K = th.rfft(th.eye(N) / S, 1)[:frame_len]
    # 2 x N/2+1 x F
    K = th.transpose(K, 0, 2) * W
    # N+2 x 1 x F
    K = th.reshape(K, (N + 2, 1, frame_len))
    return K


class STFTBase(nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it 
           could achieve perfect reconstruction after overlap-add
        2) Now haven't consider padding problems yet
    """

    def __init__(self,
                 frame_len,
                 frame_hop,
                 window="sqrt_hann",
                 round_pow_of_two=True):
        super(STFTBase, self).__init__()
        K = init_kernel(frame_len,
                        frame_hop,
                        round_pow_of_two=round_pow_of_two,
                        window=window)
        self.K = nn.Parameter(K, requires_grad=False)
        self.stride = frame_hop
        self.window = window

    def extra_repr(self):
        return "window={0}, stride={1}, kernel_size={2[0]}x{2[2]}".format(
            self.window, self.stride, self.K.shape)


class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Accept raw waveform and output magnitude and phase
        x: input signal, N x 1 x S or N x S
        m: magnitude, N x F x T
        p: phase, N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                x.dim()))
        # if N x S, reshape N x 1 x S
        if x.dim() == 2:
            x = th.unsqueeze(x, 1)
        # N x 2F x T
        c = F.conv1d(x, self.K, stride=self.stride, padding=0)
        # N x F x T
        r, i = th.chunk(c, 2, dim=1)
        m = (r**2 + i**2)**0.5
        p = th.atan2(i, r)
        return m, p


class iSTFT(STFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p, squeeze=False):
        """
        Accept phase & magnitude and output raw waveform
        m, p: N x F x T
        s: N x C x S
        """
        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                p.dim()))
        # if F x T, reshape 1 x F x T
        if p.dim() == 2:
            p = th.unsqueeze(p, 0)
            m = th.unsqueeze(m, 0)
        r = m * th.cos(p)
        i = m * th.sin(p)
        # N x 2F x T
        c = th.cat([r, i], dim=1)
        # N x 2F x T
        s = F.conv_transpose1d(c, self.K, stride=self.stride, padding=0)
        if squeeze:
            s = th.squeeze(s)
        return s