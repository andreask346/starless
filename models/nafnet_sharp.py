# EasySharp model: PSF-conditioned NAFNet U-Net for deconvolution/sharpening.
#
# The degradation PSF (as a small conditioning vector) is injected via FiLM
# (feature-wise affine modulation) at every scale, so ONE model adapts to any
# optics — feed it a tight PSF for gentle sharpening or a wide one for heavy
# deconvolution. Output is a residual toward sharp:  sharp = input + delta.
# No GAN, no stochasticity: deterministic and faithful by design.

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nafnet_star import LayerNorm2d, SimpleGate, NAFBlock, FFCBlock


class FiLM(nn.Module):
    """Feature-wise linear modulation from the conditioning vector."""

    def __init__(self, cond_dim, ch):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim, ch), nn.GELU(),
                                 nn.Linear(ch, ch * 2))

    def forward(self, x, cond):
        gb = self.net(cond)                       # (B, 2*ch)
        g, b = gb.chunk(2, dim=1)
        return x * (1 + g[:, :, None, None]) + b[:, :, None, None]


class CondNAFBlock(nn.Module):
    def __init__(self, ch, cond_dim):
        super().__init__()
        self.block = NAFBlock(ch)
        self.film = FiLM(cond_dim, ch)

    def forward(self, x, cond):
        return self.film(self.block(x), cond)


class SharpNAFNet(nn.Module):
    def __init__(self, in_ch=3, cond_dim=4, width=32, enc_blocks=(1, 1, 1, 6),
                 mid_blocks=4, dec_blocks=(1, 1, 1, 1), use_ffc=True):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for n in enc_blocks:
            self.encoders.append(nn.ModuleList(
                [CondNAFBlock(ch, cond_dim) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
            ch *= 2
        mids = []
        for i in range(mid_blocks):
            mids.append(FFCBlock(ch) if (use_ffc and i % 2 == 1)
                        else CondNAFBlock(ch, cond_dim))
        self.middle = nn.ModuleList(mids)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1, bias=False), nn.PixelShuffle(2)))
            ch //= 2
            self.decoders.append(nn.ModuleList(
                [CondNAFBlock(ch, cond_dim) for _ in range(n)]))
        self.head = nn.Conv2d(width, in_ch, 3, padding=1)

    def forward(self, x, cond):
        feats = []
        y = self.intro(x)
        for enc, down in zip(self.encoders, self.downs):
            for blk in enc:
                y = blk(y, cond)
            feats.append(y)
            y = down(y)
        for m in self.middle:
            y = m(y, cond) if isinstance(m, CondNAFBlock) else m(y)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(feats)):
            y = up(y) + skip
            for blk in dec:
                y = blk(y, cond)
        delta = self.head(y)
        return (x + delta).clamp(0.0, 1.0)        # residual toward sharp


def build(width=32, cond_dim=4, use_ffc=True):
    return SharpNAFNet(width=width, cond_dim=cond_dim,
                       enc_blocks=(1, 1, 1, 6), mid_blocks=4,
                       dec_blocks=(1, 1, 1, 1), use_ffc=use_ffc)
