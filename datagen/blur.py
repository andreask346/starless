# EasySharp — synthetic training data for AI deconvolution / sharpening.
#
# Ground-truth strategy (the honest one): render a scene SHARP at a known
# tight PSF (target), then convolve with a KNOWN wider degradation PSF +
# aberrations + seeing to make the input. The model learns input -> target
# CONDITIONED on the degradation PSF, so one model handles any optics and we
# always have exact ground truth. This sidesteps the "amateur frames aren't
# truly sharp" trap by never using a real frame as the sharp target.
#
#   target  = sharp scene (nebula + stars at PSF_tight)
#   psf     = PSF_wide (Moffat core + Zernike aberrations + spikes)
#   input   = target (*) psf  + noise
#   cond    = [fwhm_in/ref, beta, elong, ...]  fed to the network
#
# The network predicts a residual toward sharp; a re-blur consistency loss
# (||psf (*) output - input||) keeps it faithful and prevents hallucination.

import math

import torch
import torch.nn.functional as F

try:
    from synth import (moffat_stamp, pupil_spike_psf, procedural_background,
                       sample_star_field, add_noise, presentation, _rand,
                       _loguniform)
except ImportError:
    from datagen.synth import (moffat_stamp, pupil_spike_psf,
                               procedural_background, sample_star_field,
                               add_noise, presentation, _rand, _loguniform)


PSF_REF = 2.0            # reference FWHM the conditioning is normalized to
COND_DIM = 4             # [fwhm_ratio, beta, elong, spike_frac]


def zernike_kernel(size, coeffs, device):
    """Small aberration kernel from low-order Zernike phase over a pupil,
    via |FFT(exp(i*phase))|^2. coeffs: dict of {name: amplitude(radians)}."""
    n = size
    ax = torch.linspace(-1, 1, n, device=device)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    r2 = xx ** 2 + yy ** 2
    pupil = (r2 <= 1.0).float()
    rho = torch.sqrt(r2.clamp(max=1.0))
    theta = torch.atan2(yy, xx)
    phase = torch.zeros_like(xx)
    # defocus, astigmatism (0/45), coma (x/y), trefoil
    phase += coeffs.get("defocus", 0.0) * (2 * r2 - 1)
    phase += coeffs.get("astig0", 0.0) * (rho ** 2) * torch.cos(2 * theta)
    phase += coeffs.get("astig45", 0.0) * (rho ** 2) * torch.sin(2 * theta)
    phase += coeffs.get("comax", 0.0) * (3 * rho ** 3 - 2 * rho) * torch.cos(theta)
    phase += coeffs.get("comay", 0.0) * (3 * rho ** 3 - 2 * rho) * torch.sin(theta)
    wf = pupil * torch.exp(1j * phase)
    otf = torch.fft.fftshift(torch.fft.fft2(wf))
    psf = (otf.real ** 2 + otf.imag ** 2)
    c = n // 2
    h = size // 2
    psf = psf[c - h:c - h + size, c - h:c - h + size]
    return psf / psf.sum()


def sample_degradation(gen, device):
    """One degradation PSF per training image. Returns (kernel, cond vector)."""
    ksize = 33
    fwhm = _loguniform(2.2, 9.0, 1, gen=gen).item()      # input FWHM
    beta = _rand(1.8, 4.5, 1, gen=gen).item()
    elong = _rand(0.0, 0.4, 1, gen=gen).item()
    theta = _rand(0, math.pi, 1, gen=gen).item()
    core = moffat_stamp(ksize, fwhm * (1 + elong), fwhm, theta, beta, device)
    kernel = core
    spike_frac = 0.0
    if torch.rand(1, generator=gen).item() < 0.3:        # spikes on some
        spike_frac = _rand(0.05, 0.3, 1, gen=gen).item()
        nv = int(torch.randint(4, 10, (1,), generator=gen).item())
        spikes = pupil_spike_psf(ksize, nv,
                                 _rand(0.006, 0.02, 1, gen=gen).item(),
                                 _rand(0, math.pi, 1, gen=gen).item(),
                                 _rand(0.0, 0.3, 1, gen=gen).item(), device)
        kernel = (1 - spike_frac) * core + spike_frac * spikes
    if torch.rand(1, generator=gen).item() < 0.5:        # aberrations
        amp = _rand(0.3, 2.0, 1, gen=gen).item()
        z = zernike_kernel(ksize, dict(
            comax=_rand(-amp, amp, 1, gen=gen).item(),
            comay=_rand(-amp, amp, 1, gen=gen).item(),
            astig0=_rand(-amp, amp, 1, gen=gen).item(),
            astig45=_rand(-amp, amp, 1, gen=gen).item()), device)
        kernel = 0.6 * kernel + 0.4 * z
    kernel = kernel / kernel.sum()
    cond = torch.tensor([fwhm / PSF_REF, beta / 3.0, elong, spike_frac],
                        device=device)
    return kernel, cond


def conv_psf(img, kernel):
    """FFT convolution of a (3,H,W) image with a single kernel."""
    c, h, w = img.shape
    ks = kernel.shape[0]
    pad = ks // 2
    padded = F.pad(img[None], (pad, pad, pad, pad), mode="reflect")[0]
    gh, gw = padded.shape[1:]
    K = torch.zeros(gh, gw, device=img.device)
    K[:ks, :ks] = kernel
    K = torch.roll(K, shifts=(-pad, -pad), dims=(0, 1))
    Kf = torch.fft.rfft2(K)
    out = torch.stack([
        torch.fft.irfft2(torch.fft.rfft2(padded[ch]) * Kf, s=(gh, gw))
        for ch in range(c)])
    return out[:, pad:pad + h, pad:pad + w]


def render_sharp_scene(h, w, gen, device):
    """A SHARP ground-truth scene: procedural nebula + stars at a tight PSF
    (FWHM ~PSF_REF). This is the target the model should reconstruct."""
    try:
        from synth import make_psf, _splat
    except ImportError:
        from datagen.synth import make_psf, _splat
    bg = procedural_background(h, w, gen, device)
    pos, flux, color = sample_star_field(h, w, gen, device)
    stars = torch.zeros(3, h, w, device=device)
    # sharp stars: tight Moffat, no spikes/halos
    p = dict(fwhm_x=PSF_REF, fwhm_y=PSF_REF, theta=0.0, beta=3.0,
             spike_frac=0.0, n_vanes=0, vane_width=0.0, rotation=0.0,
             obstruction=0.0, halo_frac=0.0, halo_fwhm=PSF_REF * 6)
    psf = make_psf(31, p, device)
    for ch in range(3):
        _splat(stars[ch], psf, pos, flux * color[:, ch] * 1e-4)
    return (bg + stars).clamp(0, 2)


def make_pair(h, w, sharp_scene, seed, device="cuda"):
    """One deconvolution triplet: (input, target_sharp, cond)."""
    gen = torch.Generator().manual_seed(seed)
    if sharp_scene is None:
        sharp = render_sharp_scene(h, w, gen, device)
    else:
        sharp = sharp_scene.to(device)
    kernel, cond = sample_degradation(gen, device)
    blurred = conv_psf(sharp, kernel)
    noisy = add_noise(blurred, gen, device)
    T = presentation(gen, device)
    inp = T(noisy).clamp(0, 1)
    tgt = T(sharp).clamp(0, 1)
    reblur_ref = T(blurred).clamp(0, 1)          # noise-free blurred, for the
    return (inp.float(), tgt.float(), cond.float(),  # re-blur consistency loss
            kernel.float(), reblur_ref.float())


class SharpDataset(torch.utils.data.Dataset):
    """On-the-fly deconvolution pairs. Optional sharp-ish reference tiles
    (their sharpest data) can seed scenes, but default is fully synthetic
    for clean ground truth."""

    def __init__(self, ref_dir=None, crop=256, length=100000, device="cpu"):
        import glob
        import os
        self.crop = crop
        self.length = length
        self.device = device
        self.refs = sorted(glob.glob(os.path.join(ref_dir, "*.npy"))) \
            if ref_dir else []

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        inp, tgt, cond, kernel, reblur = make_pair(
            self.crop, self.crop, None, seed=i * 6271 + 3, device=self.device)
        return inp, tgt, cond, kernel, reblur
