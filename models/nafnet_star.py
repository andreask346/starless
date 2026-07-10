# Starless model: NAFNet-style norm-free U-Net + optional Fourier (FFC)
# bottleneck + non-negative star-residual head.
#
#   S = relu(head(features));  starless = input - min(S, input)
#
# so starless + stars == input EXACTLY, and the net can only remove light
# where light exists (no dark holes below zero, no invented nebulosity).
# Norm-free (LayerNorm2d over channels only) -> none of StarNet's
# BatchNorm-at-inference tile pathology.

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, ch, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(ch))
        self.b = nn.Parameter(torch.zeros(ch))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.w[None, :, None, None] + self.b[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, ch, expand=2):
        super().__init__()
        dw = ch * expand
        self.norm1 = LayerNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.gate = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv3 = nn.Conv2d(dw // 2, ch, 1)
        self.norm2 = LayerNorm2d(ch)
        self.conv4 = nn.Conv2d(ch, dw, 1)
        self.conv5 = nn.Conv2d(dw // 2, ch, 1)
        self.beta = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x):
        y = self.conv2(self.conv1(self.norm1(x)))
        y = self.gate(y)
        y = y * self.sca(y)
        x = x + self.conv3(y) * self.beta
        y = self.gate(self.conv4(self.norm2(x)))
        return x + self.conv5(y) * self.gamma


class FFCBlock(nn.Module):
    """Fourier bottleneck block (LaMa-style): image-wide receptive field so
    huge halos/spikes are seen whole. rfft2 -> 1x1 convs on stacked
    real/imag -> irfft2, residual. ONNX: needs opset >= 17 (DFT); the
    model builder can disable it (use_ffc=False) -> extra NAFBlocks."""

    def __init__(self, ch):
        super().__init__()
        self.norm = LayerNorm2d(ch)
        self.fconv = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 2, 1), nn.GELU(),
            nn.Conv2d(ch * 2, ch * 2, 1))
        self.local = NAFBlock(ch)
        self.beta = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x):
        y = self.norm(x)
        # FFT ops need fp32 (bf16 autocast unsupported; ONNX DFT is fp32 too)
        with torch.autocast(device_type=x.device.type if x.is_cuda else "cpu",
                            enabled=False):
            yf = y.float()
            # full fft2/ifft2 (not rfft): the ONNX DFT op forbids
            # inverse+onesided together, which is what irfft would emit
            f = torch.fft.fft2(yf, norm="ortho")
            f = torch.cat([f.real, f.imag], dim=1)
            f = self.fconv(f)
            re, im = f.chunk(2, dim=1)
            y2 = torch.fft.ifft2(torch.complex(re, im), norm="ortho").real
        x = x + y2.to(x.dtype) * self.beta
        return self.local(x)


class StarNAFNet(nn.Module):
    """width=32 experiment / width=64 ship. enc_blocks like [1,1,1,6]."""

    def __init__(self, in_ch=3, width=32, enc_blocks=(1, 1, 1, 6),
                 mid_blocks=4, dec_blocks=(1, 1, 1, 1), use_ffc=True):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
            ch *= 2
        mids = []
        for i in range(mid_blocks):
            if use_ffc and i % 2 == 1:
                mids.append(FFCBlock(ch))
            else:
                mids.append(NAFBlock(ch))
        self.middle = nn.Sequential(*mids)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1, bias=False), nn.PixelShuffle(2)))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
        self.head = nn.Conv2d(width, in_ch, 3, padding=1)

    def forward(self, x):
        feats = []
        y = self.intro(x)
        for enc, down in zip(self.encoders, self.downs):
            y = enc(y)
            feats.append(y)
            y = down(y)
        y = self.middle(y)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(feats)):
            y = up(y) + skip
            y = dec(y)
        s = F.relu(self.head(y))            # non-negative star flux
        s = torch.minimum(s, x.clamp_min(0.0))   # cannot remove more than exists
        return s                            # stars layer; starless = x - s


def build(width=32, use_ffc=True):
    return StarNAFNet(width=width,
                      enc_blocks=(1, 1, 1, 6), mid_blocks=4,
                      dec_blocks=(1, 1, 1, 1), use_ffc=use_ffc)
