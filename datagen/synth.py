# Starless — synthetic training data engine.
#
# Generates (input, starless, stars) triplets with EXACT ground truth:
#   linear:       sky = background + stars_linear
#   + noise:      noisy = sky + noise(sky)          (noise stays in the target;
#                 star removal must not denoise)
#   presentation: input    = T(noisy)
#                 starless = T(noisy - stars_linear + eps_consistency)
#                 stars    = input - starless        (>= 0, exact recomposition)
# where T is a random monotonic presentation transform (identity / MTF
# autostretch / arcsinh / GHS-like), identical for input and target.
#
# PSFs are physically motivated: elliptical Moffat core+wings, pupil-FFT
# diffraction spikes (spider vanes or lens iris), reflection halos, chromatic
# scale differences, and field-varying aberrations (corner coma/elongation) —
# the wide-lens regime the commercial tools are weakest at is a first-class
# citizen here.
#
# Everything runs in torch (CPU or CUDA) for on-the-fly batch generation.

import math

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ utils
def _rand(lo, hi, *shape, device="cpu", gen=None):
    return torch.rand(*shape, device=device, generator=gen) * (hi - lo) + lo


def _loguniform(lo, hi, *shape, device="cpu", gen=None):
    return torch.exp(_rand(math.log(lo), math.log(hi), *shape,
                           device=device, gen=gen))


# ------------------------------------------------------------------ PSF bank
def moffat_stamp(size, fwhm_x, fwhm_y, theta, beta, device):
    """Elliptical Moffat, unit flux, subpixel-centerable via later shift."""
    ax = torch.arange(size, device=device) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    ct, st = math.cos(theta), math.sin(theta)
    xr = xx * ct + yy * st
    yr = -xx * st + yy * ct
    alpha_x = fwhm_x / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))
    alpha_y = fwhm_y / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))
    r2 = (xr / alpha_x) ** 2 + (yr / alpha_y) ** 2
    psf = (1.0 + r2) ** (-beta)
    return psf / psf.sum()


def pupil_spike_psf(size, n_vanes, vane_width, rotation, obstruction, device):
    """|FFT|^2 of an aperture: circular pupil + spider vanes (or lens-iris
    polygon edges) -> physically correct diffraction spikes."""
    n = size * 2
    ax = torch.linspace(-1, 1, n, device=device)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    r = torch.sqrt(xx ** 2 + yy ** 2)
    pupil = (r <= 1.0).float()
    if obstruction > 0:
        pupil *= (r >= obstruction).float()
    for k in range(n_vanes):
        th = rotation + k * math.pi / max(n_vanes / 2.0, 1.0)
        d = torch.abs(xx * math.sin(th) - yy * math.cos(th))
        along = xx * math.cos(th) + yy * math.sin(th)
        pupil *= 1.0 - ((d < vane_width) & (along > 0)).float()
    otf = torch.fft.fftshift(torch.fft.fft2(pupil))
    psf = (otf.real ** 2 + otf.imag ** 2)
    c = n // 2
    h = size // 2
    psf = psf[c - h:c - h + size, c - h:c - h + size]
    return psf / psf.sum()


def make_psf(size, params, device):
    """Compose core + spikes + halo into one unit-flux PSF stamp."""
    core = moffat_stamp(size, params["fwhm_x"], params["fwhm_y"],
                        params["theta"], params["beta"], device)
    psf = core
    if params["spike_frac"] > 0:
        spikes = pupil_spike_psf(size, params["n_vanes"],
                                 params["vane_width"], params["rotation"],
                                 params["obstruction"], device)
        # spikes carry the high-frequency structure; keep the Moffat core
        psf = (1 - params["spike_frac"]) * core + params["spike_frac"] * spikes
    if params["halo_frac"] > 0:
        halo = moffat_stamp(size, params["halo_fwhm"], params["halo_fwhm"],
                            0.0, 1.8, device)
        psf = (1 - params["halo_frac"]) * psf + params["halo_frac"] * halo
    return psf / psf.sum()


_PRESETS = None


def _load_presets():
    global _PRESETS
    if _PRESETS is None:
        import json
        import os
        p = os.path.join(os.path.dirname(__file__), "psf_presets.json")
        try:
            _PRESETS = json.load(open(p))
        except Exception:
            _PRESETS = []
    return _PRESETS


def sample_psf_family(gen, device):
    """One optical system per training image (shared by all its stars).
    50% of the time anchor on a REAL calibrated preset (Andreas's lenses:
    undersampled FWHM ~1.5-1.9px, elongation 0.24-0.43) with jitter."""
    presets = _load_presets()
    kind = torch.randint(0, 4, (1,), generator=gen).item()
    if presets and torch.rand(1, generator=gen).item() < 0.5:
        pr = presets[torch.randint(0, len(presets), (1,), generator=gen).item()]
        fwhm = pr["fwhm_med"] * _rand(0.85, 1.35, 1, gen=gen).item()
        elong = max(pr["elong_corner"], pr["elong_center"]) \
            * _rand(0.7, 1.3, 1, gen=gen).item()
    else:
        fwhm = _loguniform(1.4, 9.0, 1, gen=gen).item()
        elong = _rand(0.0, 0.45, 1, gen=gen).item()
    fam = dict(
        fwhm=fwhm,
        elong=elong,                       # field-varying elongation strength
        beta=_rand(1.8, 4.5, 1, gen=gen).item(),
        theta0=_rand(0, math.pi, 1, gen=gen).item(),
        spike_frac=0.0, n_vanes=0, vane_width=0.0, rotation=0.0,
        obstruction=0.0, halo_frac=0.0, halo_fwhm=fwhm * 6,
        chroma=_rand(0.95, 1.12, 3, gen=gen).tolist(),  # per-channel FWHM scale
    )
    if kind == 1:                          # Newtonian: 4-vane spider
        fam.update(spike_frac=_rand(0.1, 0.45, 1, gen=gen).item(), n_vanes=4,
                   vane_width=_rand(0.006, 0.02, 1, gen=gen).item(),
                   rotation=_rand(0, math.pi, 1, gen=gen).item(),
                   obstruction=_rand(0.15, 0.35, 1, gen=gen).item())
    elif kind == 2:                        # camera lens: iris cross-screen
        fam.update(spike_frac=_rand(0.05, 0.3, 1, gen=gen).item(),
                   n_vanes=int(torch.randint(6, 10, (1,), generator=gen).item()),
                   vane_width=_rand(0.004, 0.012, 1, gen=gen).item(),
                   rotation=_rand(0, math.pi, 1, gen=gen).item())
    if torch.rand(1, generator=gen).item() < 0.4:      # reflection halo
        fam.update(halo_frac=_rand(0.02, 0.15, 1, gen=gen).item(),
                   halo_fwhm=fwhm * _rand(4, 14, 1, gen=gen).item())
    return fam


# ------------------------------------------------------------------ star field
def sample_star_field(h, w, gen, device):
    """Positions, fluxes, colors. Density spans sparse field -> Milky Way."""
    density = _loguniform(30, 4000, 1, gen=gen).item()   # stars per 512^2
    n = max(int(density * (h * w) / (512 * 512)), 1)
    n = min(n, 20000)
    pos = torch.rand(n, 2, generator=gen).to(device) * torch.tensor(
        [float(w), float(h)], device=device)
    if torch.rand(1, generator=gen).item() < 0.15:       # occasional cluster
        cn = max(int(n * 0.3), 10)
        c = torch.rand(2, generator=gen).to(device) * torch.tensor(
            [float(w), float(h)], device=device)
        r = _rand(20, 120, 1, gen=gen).item()
        cl = c + torch.randn(cn, 2, generator=gen).to(device) * r
        pos = torch.cat([pos, cl])
        n += cn
    # power-law flux (bright tail), scaled so a handful saturate
    flux = _loguniform(1.0, 3e4, n, gen=gen).to(device)
    # star colors: blackbody-ish RGB ratios
    t = _rand(0.0, 1.0, n, gen=gen).to(device)           # 0=red .. 1=blue
    color = torch.stack([1.1 - 0.5 * t, torch.ones_like(t), 0.5 + 0.7 * t],
                        dim=1)
    color /= color.mean(dim=1, keepdim=True)
    return pos, flux, color


def render_stars(h, w, fam, pos, flux, color, gen, device,
                 flux_scale=1e-4):
    """Render the linear star image (3,H,W). Field-varying PSF: elongation
    grows toward corners, oriented radially (coma/astigmatism-like)."""
    out = torch.zeros(3, h, w, device=device)
    n = len(pos)
    if n == 0:
        return out
    # bucket stars by brightness -> stamp size (big stamps only when needed)
    stamp_for = torch.where(flux > 3000, 255,
                torch.where(flux > 300, 127,
                torch.where(flux > 30, 63, 31))).to(torch.long)
    cx, cy = w / 2.0, h / 2.0
    rmax = math.hypot(cx, cy)
    for size in (31, 63, 127, 255):
        idx = torch.nonzero(stamp_for == size).ravel()
        if len(idx) == 0:
            continue
        # group by coarse field radius so PSF varies across the frame
        for ring in range(3):
            r_lo, r_hi = ring / 3.0, (ring + 1) / 3.0
            rr = torch.hypot(pos[idx, 0] - cx, pos[idx, 1] - cy) / rmax
            sel = idx[(rr >= r_lo) & (rr < r_hi)]
            if len(sel) == 0:
                continue
            rmid = (r_lo + r_hi) / 2.0
            el = fam["elong"] * rmid                    # radial elongation
            # radial orientation: mean angle of the ring group
            mean_theta = math.atan2(
                float((pos[sel, 1] - cy).mean()), float((pos[sel, 0] - cx).mean()))
            for ch in range(3):
                f = fam["fwhm"] * fam["chroma"][ch]
                p = dict(fwhm_x=f * (1 + el), fwhm_y=f, theta=mean_theta,
                         beta=fam["beta"], spike_frac=fam["spike_frac"],
                         n_vanes=fam["n_vanes"], vane_width=fam["vane_width"],
                         rotation=fam["rotation"],
                         obstruction=fam["obstruction"],
                         halo_frac=fam["halo_frac"] if size >= 127 else 0.0,
                         halo_fwhm=fam["halo_fwhm"])
                psf = make_psf(size, p, device)
                _splat(out[ch], psf, pos[sel], flux[sel] * color[sel, ch]
                       * flux_scale)
    return out


def _splat(img, psf, pos, amp):
    """Add psf stamps at subpixel positions — vectorized impulse-grid method:
    scatter bilinear-weighted deltas with index_put_, then ONE convolution
    with the (flipped) PSF kernel. ~30x faster than per-star stamping."""
    h, w = img.shape
    s = psf.shape[0]
    half = s // 2
    device = img.device
    x = pos[:, 0].clamp(-half + 1, w + half - 2)
    y = pos[:, 1].clamp(-half + 1, h + half - 2)
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    fx, fy = x - x0, y - y0
    pad = half + 1
    grid = torch.zeros(h + 2 * pad, w + 2 * pad, device=device)
    xi = x0.long() + pad
    yi = y0.long() + pad
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            grid.index_put_((yi + dy, xi + dx), amp * wx * wy,
                            accumulate=True)
    # FFT convolution: cost independent of PSF size (255px halo kernels
    # make spatial conv2d minutes-slow on CPU)
    gh, gw = grid.shape
    K = torch.zeros(gh, gw, device=device)
    K[:s, :s] = psf
    K = torch.roll(K, shifts=(-half, -half), dims=(0, 1))
    out = torch.fft.irfft2(torch.fft.rfft2(grid) * torch.fft.rfft2(K),
                           s=(gh, gw))
    img += out[pad:pad + h, pad:pad + w]


# ------------------------------------------------------------------ backgrounds
def procedural_background(h, w, gen, device):
    """Multi-octave smooth 'nebula' + gradient — placeholder/augmenter used
    alongside real starless plates."""
    img = torch.zeros(3, h, w, device=device)
    base_hue = torch.rand(3, 1, 1, generator=gen).to(device) * 0.5 + 0.5
    for octave in range(4):
        cell = 2 ** (7 - octave)
        gh, gw = max(h // cell, 2), max(w // cell, 2)
        noise = torch.randn(1, 3, gh, gw, generator=gen).to(device)
        up = F.interpolate(noise, size=(h, w), mode="bicubic",
                           align_corners=False)[0]
        img += up * (0.5 ** octave)
    img = torch.tanh(img * 0.4) * 0.5 + 0.5
    amp = _loguniform(0.005, 0.25, 1, gen=gen).to(device)
    img = img * base_hue * amp
    gx = torch.linspace(0, _rand(-0.02, 0.02, 1, gen=gen).item(), w,
                        device=device)
    gy = torch.linspace(0, _rand(-0.02, 0.02, 1, gen=gen).item(), h,
                        device=device)
    img = img + gx[None, None, :] + gy[None, :, None] \
        + _rand(0.002, 0.08, 1, gen=gen).to(device)
    return img.clamp_min(0)


# ------------------------------------------------------------------ noise + presentation
def add_noise(img, gen, device):
    """Post-stack noise: shot (signal-dependent) + read (flat), channel-correlated."""
    a = _loguniform(1e-5, 3e-3, 1, gen=gen).to(device)     # shot scale
    b = _loguniform(1e-4, 4e-3, 1, gen=gen).to(device)     # read floor
    sigma = torch.sqrt(a * img.clamp_min(0) + b ** 2)
    return img + torch.randn(img.shape, generator=gen).to(device) * sigma


def _mtf(x, m):
    return ((m - 1) * x) / ((2 * m - 1) * x - m)


def presentation(gen, device):
    """Random monotonic display transform T applied to BOTH input and target."""
    kind = torch.randint(0, 4, (1,), generator=gen).item()
    if kind == 0:
        return lambda x: x                                        # linear
    if kind == 1:
        m = _rand(0.01, 0.25, 1, gen=gen).item()                  # MTF stretch
        return lambda x: _mtf(x.clamp(0, 1), m)
    if kind == 2:
        a = _loguniform(20, 800, 1, gen=gen).item()               # arcsinh
        return lambda x: torch.asinh(x.clamp_min(0) * a) / math.asinh(a)
    b = _rand(2.0, 8.0, 1, gen=gen).item()                        # GHS-like
    return lambda x: (1 - (1 - x.clamp(0, 1)) ** b) ** (1.0 / b)


# ------------------------------------------------------------------ main entry
def make_pair(h, w, background, seed, device="cuda"):
    """One training triplet. background: (3,H,W) linear starless tensor
    (real plate crop or None -> procedural)."""
    gen = torch.Generator().manual_seed(seed)
    if background is None:
        bg = procedural_background(h, w, gen, device)
    else:
        bg = background.to(device)
    fam = sample_psf_family(gen, device)
    pos, flux, color = sample_star_field(h, w, gen, device)
    stars_lin = render_stars(h, w, fam, pos, flux, color, gen, device)

    sky = bg + stars_lin
    # saturation: clip highlights like a real stacked frame
    sat = _rand(0.7, 1.0, 1, gen=gen).to(device)
    sky = torch.minimum(sky, sat)
    starless_lin = torch.minimum(bg, sat)

    noisy = add_noise(sky, gen, device)
    noise = noisy - sky
    starless_noisy = starless_lin + noise      # same noise: removal != denoise

    T = presentation(gen, device)
    inp = T(noisy).clamp(0, 1)
    tgt_starless = T(starless_noisy).clamp(0, 1)
    tgt_starless = torch.minimum(tgt_starless, inp)   # stars layer >= 0 exactly
    return inp.float(), tgt_starless.float(), (inp - tgt_starless).float()


class PairDataset(torch.utils.data.Dataset):
    """On-the-fly pairs over a folder of linear starless background tiles
    (float32 .npy, (3,H,W), values 0..1). Empty folder -> fully procedural."""

    def __init__(self, bg_dir=None, crop=256, length=100000, device="cpu",
                 base_index=0):
        import glob
        import os
        self.crop = crop
        self.length = length
        self.device = device
        self.base = base_index          # absolute-step offset for resume
        self.bgs = sorted(glob.glob(os.path.join(bg_dir, "*.npy"))) \
            if bg_dir else []

    def __len__(self):
        return self.length

    def _bg_crop(self, j, gen):
        import numpy as np
        if not self.bgs or torch.rand(1, generator=gen).item() < 0.25:
            return None
        arr = np.load(self.bgs[j % len(self.bgs)], mmap_mode="r")
        c, h, w = arr.shape
        if h < self.crop or w < self.crop:
            return None
        y = torch.randint(0, h - self.crop + 1, (1,), generator=gen).item()
        x = torch.randint(0, w - self.crop + 1, (1,), generator=gen).item()
        t = torch.from_numpy(np.ascontiguousarray(
            arr[:, y:y + self.crop, x:x + self.crop])).float()
        if torch.rand(1, generator=gen).item() < 0.5:
            t = torch.flip(t, dims=[2])
        k = torch.randint(0, 4, (1,), generator=gen).item()
        t = torch.rot90(t, k, dims=[1, 2])
        t = t * float(_loguniform(0.3, 2.0, 1, gen=gen))
        return t.clamp(0, 1)

    def __getitem__(self, i):
        j = self.base + i                        # absolute sample index
        gen = torch.Generator().manual_seed(j * 9973 + 12345)
        bg = self._bg_crop(j, gen)
        inp, starless, stars = make_pair(self.crop, self.crop, bg,
                                         seed=j * 7919 + 7, device=self.device)
        return inp, starless, stars
