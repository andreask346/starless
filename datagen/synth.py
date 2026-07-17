# Starless — synthetic training data engine (v1 + v2 recipes).
#
# Generates (input, starless, stars) triplets with EXACT ground truth:
#   linear:       sky = background + stars_linear          (stars occluded by
#                 any foreground silhouette BEFORE compositing — physics-true)
#   + noise:      noisy = sky + noise(sky)          (noise stays in the target;
#                 star removal must not denoise)
#   presentation: input    = T(noisy)
#                 starless = T(noisy - stars_linear + eps_consistency)
#                 stars    = input - starless        (>= 0, exact recomposition)
# where T is a random monotonic presentation transform, identical for input
# and target.
#
# v2 recipe (default_config(2), the default) — driven by the forensic census
# of a real 7008x4672 stretched frame (median 0.38, star peaks 0.19-0.68
# above bg, noise sigma ~0.0025, saturated flat discs up to ~30 px radius):
#   1. bright/saturated star extension: 10-20% bright subpopulation with
#      x10-1000 flux boost, saturation clip widened to 0.5-1.0 (per-channel
#      jitter -> color-fringed rims), explicit "monster" stars whose flux is
#      solved so the clipped plateau reaches a target 6-45 px radius, halos
#      always on for them.
#   2. target-driven presentation: composed chains that land the background
#      median at U(0.10, 0.45) via a solved MTF (Siril-autostretch style),
#      plus per-channel gain jitter, mild curves, saturation boost and
#      explicit per-channel background NEUTRALIZATION. 25-30% stay linear.
#   3. realistic noise: spatially correlated (kernel / resample cycle) and
#      channel-correlated via a sampled mixing coefficient; noise added
#      BEFORE presentation so grain stretches realistically.
#   4. unresolved-starfield glow: dense sub-noise star population rendered
#      into the BACKGROUND (keep-content) + granular bg octave -> diffuse
#      Milky Way texture is explicitly not-stars.
#   5. foreground silhouettes (p~0.3): ridgeline+tree polygons that OCCLUDE
#      the star layer; optional twilight edge glow; terrestrial point lights
#      on the silhouette (keep-content).
#   6. field-geometry decoupling: star elongation computed from a virtual
#      2-8k full-frame center, not the crop center.
#   7. envelope widening: FWHM 1.4-12, defocus discs, guiding-error
#      elongation, RGB misregistration, dark-halo oversharpened profiles.
#   8. do-not-remove distractors: galaxy blobs / nebula knots / comet-like
#      blobs in the background (keep-content).
#
# default_config(1) reproduces the shipped v1 engine bit-exactly (same RNG
# consumption order).
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


# ------------------------------------------------------------------ config
def default_config(version=2):
    """All data-recipe knobs. version=1 == the shipped v1 engine (bit-exact
    RNG order); version=2 == the post-forensics recipe (default)."""
    cfg = dict(
        version=version,
        # --- star flux / saturation
        flux_lo=1.0, flux_hi=3e4, flux_scale=1e-4,
        bright_frac=0.15,            # bright subpopulation fraction
        bright_boost_lo=10.0, bright_boost_hi=1000.0,
        monster_p=0.35,              # p(image has saturated "monster" stars)
        monster_n_max=3,
        monster_r_lo=6.0, monster_r_hi=45.0,   # target clipped-disc radius px
        sat_lo=0.5, sat_hi=1.0,      # saturation clip level range
        sat_chan_jitter=0.05,        # per-channel clip jitter -> fringed rims
        halo_min_stamp=63,           # halo render gate (v1 was 127)
        # --- PSF envelope
        fwhm_lo=1.4, fwhm_hi=12.0,
        defocus_p=0.12,              # defocused/soft profiles
        guide_p=0.15,                # guiding-error elongation
        chan_shift_p=0.30,           # RGB misregistration
        dark_halo_p=0.08,            # oversharpened dark-halo profiles
        # --- field geometry
        virtual_field=True,          # elongation from virtual frame center
        vframe_lo=2000, vframe_hi=8000,
        # --- presentation
        presentation="v2",
        pure_linear_frac=0.27,
        tgt_median_lo=0.10, tgt_median_hi=0.45,
        chan_gain_jitter=0.05,
        curve_p=0.5, satboost_p=0.4, neutralize_p=0.8,
        # --- noise
        noise="v2",
        noise_a_lo=1e-5, noise_a_hi=3e-3,
        noise_b_lo=1e-4, noise_b_hi=4e-3,
        noise_spatial_p=0.7,         # spatially-correlated grain probability
        noise_chan_rho_hi=0.9,       # max channel-correlation coefficient
        # --- background keep-content
        glow_p=0.6,                  # unresolved starfield glow
        glow_density_lo=2000.0, glow_density_hi=30000.0,
        glow_peak_sigma=2.0,         # glow star peaks < this x noise sigma
        bg_granular_p=0.5,           # fine granular octave in procedural bg
        silhouette_p=0.30,           # foreground occlusion probability
        distractor_p=0.35,           # galaxies / knots / comets (keep)
    )
    if version == 1:
        cfg.update(
            bright_frac=0.0, monster_p=0.0,
            sat_lo=0.7, sat_hi=1.0, sat_chan_jitter=0.0,
            halo_min_stamp=127,
            fwhm_lo=1.4, fwhm_hi=9.0,
            defocus_p=0.0, guide_p=0.0, chan_shift_p=0.0, dark_halo_p=0.0,
            virtual_field=False,
            presentation="v1", pure_linear_frac=0.25,
            chan_gain_jitter=0.0, curve_p=0.0, satboost_p=0.0,
            neutralize_p=0.0,
            noise="v1", noise_spatial_p=0.0, noise_chan_rho_hi=0.0,
            glow_p=0.0, bg_granular_p=0.0, silhouette_p=0.0, distractor_p=0.0,
        )
    return cfg


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


def _gauss_blur_stamp(psf, sigma):
    """Gaussian-blur a PSF stamp (separable conv, sum-preserving)."""
    k = int(2 * math.ceil(2.5 * sigma) + 1)
    k = min(k, psf.shape[0] // 2 * 2 - 1)
    if k < 3:
        return psf
    ax = torch.arange(k, device=psf.device) - (k - 1) / 2.0
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    x = psf[None, None]
    x = F.conv2d(x, g.view(1, 1, 1, k), padding=(0, k // 2))
    x = F.conv2d(x, g.view(1, 1, k, 1), padding=(k // 2, 0))
    return x[0, 0]


def _disc_blur_stamp(psf, radius):
    """Convolve a PSF stamp with a flat disc (defocus)."""
    r = max(radius, 0.6)
    k = int(2 * math.ceil(r) + 1)
    k = min(k, psf.shape[0] // 2 * 2 - 1)
    if k < 3:
        return psf
    ax = torch.arange(k, device=psf.device) - (k - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    disc = ((xx ** 2 + yy ** 2) <= r * r).float()
    disc = disc / disc.sum()
    return F.conv2d(psf[None, None], disc[None, None], padding=k // 2)[0, 0]


def make_psf(size, params, device):
    """Compose core + spikes + halo (+ optional defocus disc and dark-halo
    oversharpening ring) into one unit-flux PSF stamp."""
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
    dr = params.get("defocus_r", 0.0)
    if dr > 0:
        psf = _disc_blur_stamp(psf, dr)
    dk = params.get("dark_halo_k", 0.0)
    if dk > 0:                    # oversharpened profile: negative ring
        s = params.get("dark_halo_s", params["fwhm_x"] * 2.0)
        psf = (1.0 + dk) * psf - dk * _gauss_blur_stamp(psf, s)
    return psf / psf.sum().clamp_min(1e-8)


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


def sample_psf_family(gen, device, cfg=None):
    """One optical system per training image (shared by all its stars).
    50% of the time anchor on a REAL calibrated preset (Andreas's lenses:
    undersampled FWHM ~1.5-1.9px, elongation 0.24-0.43) with jitter."""
    v2 = cfg is not None and cfg.get("version", 1) != 1
    presets = _load_presets()
    kind = torch.randint(0, 4, (1,), generator=gen).item()
    if presets and torch.rand(1, generator=gen).item() < 0.5:
        pr = presets[torch.randint(0, len(presets), (1,), generator=gen).item()]
        fwhm = pr["fwhm_med"] * _rand(0.85, 1.35, 1, gen=gen).item()
        elong = max(pr["elong_corner"], pr["elong_center"]) \
            * _rand(0.7, 1.3, 1, gen=gen).item()
    else:
        f_lo = cfg["fwhm_lo"] if v2 else 1.4
        f_hi = cfg["fwhm_hi"] if v2 else 9.0
        fwhm = _loguniform(f_lo, f_hi, 1, gen=gen).item()
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
    if v2:
        # per-channel halo scale jitter -> chromatic fringe on bright rims
        fam["halo_chroma"] = _rand(0.92, 1.18, 3, gen=gen).tolist()
        if torch.rand(1, generator=gen).item() < cfg["defocus_p"]:
            fam["defocus_r"] = _rand(1.0, 6.0, 1, gen=gen).item()
        if torch.rand(1, generator=gen).item() < cfg["guide_p"]:
            fam["guide_elong"] = _rand(0.06, 0.45, 1, gen=gen).item()
            fam["guide_theta"] = _rand(0, math.pi, 1, gen=gen).item()
        if torch.rand(1, generator=gen).item() < cfg["chan_shift_p"]:
            s = _rand(0.1, 0.5, 1, gen=gen).item()
            fam["chan_shift"] = (torch.randn(3, 2, generator=gen) * s) \
                .clamp(-0.8, 0.8).tolist()
        if torch.rand(1, generator=gen).item() < cfg["dark_halo_p"]:
            fam["dark_halo_k"] = _rand(0.10, 0.35, 1, gen=gen).item()
            fam["dark_halo_s"] = fwhm * _rand(1.5, 3.0, 1, gen=gen).item()
    return fam


# ------------------------------------------------------------------ star field
def sample_star_field(h, w, gen, device, cfg=None):
    """Positions, fluxes, colors. Density spans sparse field -> Milky Way.
    v2: a bright subpopulation gets a x10-1000 flux boost so peak/saturation
    overshoot spans ~1x-1000x (the census showed bright stars were absent)."""
    v2 = cfg is not None and cfg.get("version", 1) != 1
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
    f_lo = cfg["flux_lo"] if v2 else 1.0
    f_hi = cfg["flux_hi"] if v2 else 3e4
    flux = _loguniform(f_lo, f_hi, n, gen=gen).to(device)
    if v2 and cfg["bright_frac"] > 0:
        is_b = (torch.rand(n, generator=gen) < cfg["bright_frac"]).to(device)
        boost = _loguniform(cfg["bright_boost_lo"], cfg["bright_boost_hi"],
                            n, gen=gen).to(device)
        flux = torch.where(is_b, flux * boost, flux)
    # star colors: blackbody-ish RGB ratios
    t = _rand(0.0, 1.0, n, gen=gen).to(device)           # 0=red .. 1=blue
    color = torch.stack([1.1 - 0.5 * t, torch.ones_like(t), 0.5 + 0.7 * t],
                        dim=1)
    color /= color.mean(dim=1, keepdim=True)
    return pos, flux, color


def render_stars(h, w, fam, pos, flux, color, gen, device,
                 flux_scale=1e-4, geom=None, halo_min_stamp=127):
    """Render the linear star image (3,H,W). Field-varying PSF: elongation
    grows toward corners, oriented radially (coma/astigmatism-like).
    geom=dict(cx, cy, rmax) decouples the optical center from the crop center
    (virtual full-frame geometry); None -> crop center (v1)."""
    out = torch.zeros(3, h, w, device=device)
    n = len(pos)
    if n == 0:
        return out
    # bucket stars by brightness -> stamp size (big stamps only when needed)
    stamp_for = torch.where(flux > 3000, 255,
                torch.where(flux > 300, 127,
                torch.where(flux > 30, 63, 31))).to(torch.long)
    if geom is None:
        cx, cy = w / 2.0, h / 2.0
        rmax = math.hypot(cx, cy)
    else:
        cx, cy, rmax = geom["cx"], geom["cy"], geom["rmax"]
    chan_shift = fam.get("chan_shift")
    for size in (31, 63, 127, 255):
        idx = torch.nonzero(stamp_for == size).ravel()
        if len(idx) == 0:
            continue
        # group by coarse field radius so PSF varies across the frame
        for ring in range(3):
            r_lo, r_hi = ring / 3.0, (ring + 1) / 3.0
            rr = torch.hypot(pos[idx, 0] - cx, pos[idx, 1] - cy) / rmax
            if geom is not None:
                rr = rr.clamp(max=0.999)
            sel = idx[(rr >= r_lo) & (rr < r_hi)]
            if len(sel) == 0:
                continue
            rmid = (r_lo + r_hi) / 2.0
            el_r = fam["elong"] * rmid                  # radial elongation
            # radial orientation: mean angle of the ring group
            mean_theta = math.atan2(
                float((pos[sel, 1] - cy).mean()), float((pos[sel, 0] - cx).mean()))
            ge = fam.get("guide_elong", 0.0)            # guiding error adds a
            el = el_r + ge                              # frame-wide component
            theta_use = fam.get("guide_theta", mean_theta) if ge > el_r \
                else mean_theta
            for ch in range(3):
                f = fam["fwhm"] * fam["chroma"][ch]
                halo_ch = fam.get("halo_chroma", (1.0, 1.0, 1.0))[ch]
                p = dict(fwhm_x=f * (1 + el), fwhm_y=f, theta=theta_use,
                         beta=fam["beta"], spike_frac=fam["spike_frac"],
                         n_vanes=fam["n_vanes"], vane_width=fam["vane_width"],
                         rotation=fam["rotation"],
                         obstruction=fam["obstruction"],
                         halo_frac=fam["halo_frac"]
                         if size >= halo_min_stamp else 0.0,
                         halo_fwhm=fam["halo_fwhm"] * halo_ch,
                         defocus_r=fam.get("defocus_r", 0.0),
                         dark_halo_k=fam.get("dark_halo_k", 0.0),
                         dark_halo_s=fam.get("dark_halo_s", 0.0))
                psf = make_psf(size, p, device)
                p_ch = pos[sel]
                if chan_shift is not None:              # RGB misregistration
                    p_ch = p_ch + torch.tensor(chan_shift[ch], device=device)
                _splat(out[ch], psf, p_ch, flux[sel] * color[sel, ch]
                       * flux_scale)
    return out


def render_monsters(h, w, fam, gen, device, cfg, sat):
    """Saturated 'monster' stars: flux is SOLVED so the clipped plateau
    reaches a target disc radius (6-45 px -> covers the real frame's ~30 px
    flat-white discs). Halos always on; per-channel flux + halo jitter make
    the color-fringed rim."""
    out = torch.zeros(3, h, w, device=device)
    n = int(torch.randint(1, cfg["monster_n_max"] + 1, (1,),
                          generator=gen).item())
    size = 255
    c = (size - 1) // 2
    sat_lvl = float(sat.mean())
    for _ in range(n):
        px = _rand(0.05, 0.95, 1, gen=gen).item() * w
        py = _rand(0.05, 0.95, 1, gen=gen).item() * h
        r_d = _rand(cfg["monster_r_lo"], cfg["monster_r_hi"], 1,
                    gen=gen).item()
        fringe = _rand(0.9, 1.1, 3, gen=gen).tolist()
        halo_j = _rand(0.92, 1.18, 3, gen=gen).tolist()
        pos1 = torch.tensor([[px, py]], device=device)
        for ch in range(3):
            f = fam["fwhm"] * fam["chroma"][ch]
            p = dict(fwhm_x=f, fwhm_y=f, theta=0.0, beta=fam["beta"],
                     spike_frac=fam["spike_frac"], n_vanes=fam["n_vanes"],
                     vane_width=fam["vane_width"], rotation=fam["rotation"],
                     obstruction=fam["obstruction"],
                     halo_frac=max(fam["halo_frac"], 0.04),  # halo ALWAYS on
                     halo_fwhm=fam["halo_fwhm"] * halo_j[ch],
                     defocus_r=fam.get("defocus_r", 0.0))
            psf = make_psf(size, p, device)
            rr = min(int(round(r_d)), c - 1)
            prof = float(psf[c, c + rr].clamp_min(1e-12))
            amp = min(sat_lvl / prof * fringe[ch], 1e7)
            _splat(out[ch], psf, pos1, torch.tensor([amp], device=device))
    return out


def render_unresolved_glow(h, w, fam, gen, device, cfg, sigma0):
    """Dense SUB-THRESHOLD star population (peaks < ~2x noise sigma), part of
    the BACKGROUND (keep-content): teaches the model that smooth Milky Way
    glow / unresolved starfield texture is explicitly NOT stars."""
    density = _loguniform(cfg["glow_density_lo"], cfg["glow_density_hi"], 1,
                          gen=gen).item()
    n = min(int(density * (h * w) / (512 * 512)), 60000)
    out = torch.zeros(3, h, w, device=device)
    if n < 1:
        return out
    pos = torch.rand(n, 2, generator=gen).to(device) * torch.tensor(
        [float(w), float(h)], device=device)
    if torch.rand(1, generator=gen).item() < 0.6:   # Milky-Way-like band
        nb = n // 2
        th = _rand(0, math.pi, 1, gen=gen).item()
        t = (torch.rand(nb, generator=gen).to(device) - 0.5) * 2.5 \
            * math.hypot(h, w)
        off = torch.randn(nb, generator=gen).to(device) \
            * _rand(0.05, 0.25, 1, gen=gen).item() * min(h, w)
        cx = _rand(0.2, 0.8, 1, gen=gen).item() * w
        cy = _rand(0.2, 0.8, 1, gen=gen).item() * h
        bx = cx + t * math.cos(th) - off * math.sin(th)
        by = cy + t * math.sin(th) + off * math.cos(th)
        pos[:nb, 0] = bx.clamp(0, w - 1)
        pos[:nb, 1] = by.clamp(0, h - 1)
    t = _rand(0.0, 1.0, n, gen=gen).to(device)
    color = torch.stack([1.1 - 0.5 * t, torch.ones_like(t), 0.5 + 0.7 * t],
                        dim=1)
    color /= color.mean(dim=1, keepdim=True)
    f = max(fam["fwhm"], 1.6)
    size = 31
    psf = moffat_stamp(size, f, f, 0.0, fam["beta"], device)
    peak = float(psf.max())
    amp_max = cfg["glow_peak_sigma"] * float(sigma0) / max(peak, 1e-8)
    flux = _loguniform(0.05, 1.0, n, gen=gen).to(device) * amp_max
    for ch in range(3):
        _splat(out[ch], psf, pos, flux * color[:, ch])
    return out


def render_distractors(h, w, gen, device, cfg):
    """DO-NOT-REMOVE content rendered into the background: small galaxy
    blobs, compact nebula knots, comet-like blobs (keep-content)."""
    img = torch.zeros(3, h, w, device=device)
    ax_y = torch.arange(h, device=device, dtype=torch.float32)
    ax_x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ax_y, ax_x, indexing="ij")
    n = int(torch.randint(1, 5, (1,), generator=gen).item())
    for _ in range(n):
        kind = torch.randint(0, 3, (1,), generator=gen).item()
        cx0 = _rand(0.1, 0.9, 1, gen=gen).item() * w
        cy0 = _rand(0.1, 0.9, 1, gen=gen).item() * h
        amp = _loguniform(0.01, 0.15, 1, gen=gen).to(device)
        col = (1.0 + _rand(-0.25, 0.25, 3, gen=gen).to(device))
        if kind == 0:            # small galaxy: elliptical exp disc + bulge
            smaj = _rand(3.0, 14.0, 1, gen=gen).item()
            ratio = _rand(0.25, 1.0, 1, gen=gen).item()
            th = _rand(0, math.pi, 1, gen=gen).item()
            xr = (xx - cx0) * math.cos(th) + (yy - cy0) * math.sin(th)
            yr = -(xx - cx0) * math.sin(th) + (yy - cy0) * math.cos(th)
            r = torch.sqrt((xr / smaj) ** 2 + (yr / (smaj * ratio)) ** 2)
            blob = torch.exp(-1.7 * r) + 0.8 * torch.exp(-(3.5 * r) ** 2)
        elif kind == 1:          # compact nebula knot
            s = _rand(2.0, 6.0, 1, gen=gen).item()
            r2 = ((xx - cx0) ** 2 + (yy - cy0) ** 2) / (2 * s * s)
            blob = torch.exp(-r2)
        else:                    # comet-like: head + exponential tail
            s = _rand(1.5, 4.0, 1, gen=gen).item()
            th = _rand(0, 2 * math.pi, 1, gen=gen).item()
            L = _rand(8.0, 40.0, 1, gen=gen).item()
            along = (xx - cx0) * math.cos(th) + (yy - cy0) * math.sin(th)
            perp = -(xx - cx0) * math.sin(th) + (yy - cy0) * math.cos(th)
            head = torch.exp(-((xx - cx0) ** 2 + (yy - cy0) ** 2)
                             / (2 * s * s))
            tail = torch.exp(-along.clamp_min(0) / L) \
                * torch.exp(-(perp ** 2) / (2 * (1.6 * s) ** 2)) \
                * (along > 0).float()
            blob = head + 0.6 * tail
        img += amp * col[:, None, None] * blob[None]
    return img


def render_silhouette(h, w, gen, device, cfg):
    """Hard-edged dark foreground: procedural ridgeline + simple tree
    'fractals' (stacked triangles). Returns (mask (1,H,W), fg (3,H,W) with
    terrestrial point lights [keep-content], sky_glow (3,H,W) twilight edge
    glow to add ABOVE the ridge)."""
    xs = torch.arange(w, device=device, dtype=torch.float32)
    base = _rand(0.55, 0.90, 1, gen=gen).item() * h
    walk = torch.randn(w, generator=gen).to(device).cumsum(0)
    k = min(31, (w // 2) * 2 - 1)
    walk = F.avg_pool1d(walk[None, None], k, stride=1, padding=k // 2)[0, 0]
    walk = walk - walk.mean()
    amp = _rand(0.02, 0.18, 1, gen=gen).item() * h
    ridge = base + walk / walk.abs().max().clamp_min(1e-6) * amp
    n_tree = int(torch.randint(0, 10, (1,), generator=gen).item())
    for _ in range(n_tree):
        xt = _rand(0, w - 1, 1, gen=gen).item()
        ht = _rand(4.0, 28.0, 1, gen=gen).item()
        wd = max(ht * _rand(0.25, 0.6, 1, gen=gen).item(), 1.0)
        base_y = float(ridge[int(min(max(xt, 0), w - 1))])
        tri = (ht * (1 - (xs - xt).abs() / wd)).clamp_min(0)          # tier 1
        tri2 = (0.65 * ht * (1 - (xs - xt).abs() / (wd * 0.5))
                ).clamp_min(0)                                        # tier 2
        cand = base_y - torch.maximum(tri, tri2)
        keep = torch.maximum(tri, tri2) > 0
        ridge = torch.where(keep, torch.minimum(ridge, cand), ridge)
    yy = torch.arange(h, device=device, dtype=torch.float32)[:, None]
    mask = (yy - ridge[None, :] + 1.0).clamp(0, 1)[None]     # 1px feather
    # foreground plate: near-black, slightly tinted + textured
    lvl = float(_loguniform(0.002, 0.04, 1, gen=gen))
    tint = (1.0 + _rand(-0.15, 0.15, 3, gen=gen).to(device))
    tex = torch.randn(1, 1, max(h // 8, 2), max(w // 8, 2),
                      generator=gen).to(device)
    tex = F.interpolate(tex, size=(h, w), mode="bilinear",
                        align_corners=False)[0]
    fg = (lvl * tint[:, None, None] * (1 + 0.3 * tex)).clamp_min(0)
    # terrestrial point lights ON the silhouette (keep-content)
    n_l = int(torch.randint(0, 5, (1,), generator=gen).item())
    for _ in range(n_l):
        lx = _rand(0.05, 0.95, 1, gen=gen).item() * w
        ry = float(ridge[int(min(max(lx, 0), w - 1))])
        if ry + 4 >= h - 1:
            continue
        ly = _rand(min(ry + 4, h - 2), h - 1, 1, gen=gen).item()
        s = _rand(0.7, 2.0, 1, gen=gen).item()
        a = _rand(0.05, 0.5, 1, gen=gen).item()
        warm = torch.tensor([1.0, _rand(0.45, 0.75, 1, gen=gen).item(),
                             _rand(0.15, 0.45, 1, gen=gen).item()],
                            device=device)
        yy2, xx2 = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij")
        spot = torch.exp(-((xx2 - lx) ** 2 + (yy2 - ly) ** 2) / (2 * s * s))
        fg = fg + a * warm[:, None, None] * spot[None]
    # twilight edge glow above the ridge (sky side)
    glow = torch.zeros(3, h, w, device=device)
    if torch.rand(1, generator=gen).item() < 0.5:
        ga = float(_loguniform(0.01, 0.08, 1, gen=gen))
        scale = _rand(6.0, 40.0, 1, gen=gen).item()
        d_above = (ridge[None, :] - yy).clamp_min(0)
        prof = torch.exp(-d_above / scale)
        if torch.rand(1, generator=gen).item() < 0.6:   # warm twilight
            gcol = torch.tensor([1.0, 0.65, 0.4], device=device)
        else:                                           # cold/moon glow
            gcol = torch.tensor([0.65, 0.8, 1.0], device=device)
        glow = ga * gcol[:, None, None] * prof[None]
    return mask, fg, glow


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
def procedural_background(h, w, gen, device, cfg=None):
    """Multi-octave smooth 'nebula' + gradient — placeholder/augmenter used
    alongside real starless plates. v2 optionally adds a fine GRANULAR octave
    (unresolved-texture look, keep-content)."""
    v2 = cfg is not None and cfg.get("version", 1) != 1
    img = torch.zeros(3, h, w, device=device)
    base_hue = torch.rand(3, 1, 1, generator=gen).to(device) * 0.5 + 0.5
    for octave in range(4):
        cell = 2 ** (7 - octave)
        gh, gw = max(h // cell, 2), max(w // cell, 2)
        noise = torch.randn(1, 3, gh, gw, generator=gen).to(device)
        up = F.interpolate(noise, size=(h, w), mode="bicubic",
                           align_corners=False)[0]
        img += up * (0.5 ** octave)
    if v2 and cfg["bg_granular_p"] > 0 \
            and torch.rand(1, generator=gen).item() < cfg["bg_granular_p"]:
        for cell in (4, 2):                     # granular glow octaves
            gh, gw = max(h // cell, 2), max(w // cell, 2)
            fine = torch.randn(1, 3, gh, gw, generator=gen).to(device)
            up = F.interpolate(fine, size=(h, w), mode="bicubic",
                               align_corners=False)[0]
            img += up * _rand(0.02, 0.08, 1, gen=gen).item()
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
    """v1 noise (kept for blur.py / EasySharp): shot (signal-dependent) +
    read (flat). NOTE: per-pixel white AND per-channel independent — the v2
    path (sample_noise_params/apply_noise) adds the spatial and channel
    correlation real stacks have."""
    a = _loguniform(1e-5, 3e-3, 1, gen=gen).to(device)     # shot scale
    b = _loguniform(1e-4, 4e-3, 1, gen=gen).to(device)     # read floor
    sigma = torch.sqrt(a * img.clamp_min(0) + b ** 2)
    return img + torch.randn(img.shape, generator=gen).to(device) * sigma


def sample_noise_params(gen, cfg):
    """Sample the per-image noise model (v2). Drawn BEFORE the image is
    assembled so sigma is known to the glow renderer and can be returned to
    the training loop (the loss needs local sigma)."""
    p = dict(
        a=float(_loguniform(cfg["noise_a_lo"], cfg["noise_a_hi"], 1, gen=gen)),
        b=float(_loguniform(cfg["noise_b_lo"], cfg["noise_b_hi"], 1, gen=gen)),
        rho=0.0, spatial=None, ksigma=0.0, factor=1.0,
    )
    if cfg["noise_chan_rho_hi"] > 0:
        p["rho"] = float(_rand(0.0, cfg["noise_chan_rho_hi"], 1, gen=gen))
    if torch.rand(1, generator=gen).item() < cfg["noise_spatial_p"]:
        if torch.rand(1, generator=gen).item() < 0.5:
            p["spatial"] = "kernel"                # 1-4 px correlation kernel
            p["ksigma"] = float(_rand(0.4, 1.6, 1, gen=gen))
        else:
            p["spatial"] = "resample"              # up/down resample cycle
            p["factor"] = float(_rand(1.3, 2.2, 1, gen=gen))
    return p


def apply_noise(img, params, gen, device):
    """v2 noise: shot+read sigma map, TRUE channel correlation (mixing
    coefficient rho) and spatial correlation (kernel or resample cycle),
    variance-preserving so the returned sigma map stays exact.
    Returns (noisy, sigma) — sigma in linear units."""
    c, h, w = img.shape
    sigma = torch.sqrt(params["a"] * img.clamp_min(0) + params["b"] ** 2)
    n_ind = torch.randn(c, h, w, generator=gen).to(device)
    if params["rho"] > 0:
        n_sh = torch.randn(1, h, w, generator=gen).to(device)
        r = params["rho"]
        n = math.sqrt(1.0 - r) * n_ind + math.sqrt(r) * n_sh
    else:
        n = n_ind
    if params["spatial"] == "kernel":
        s = params["ksigma"]
        k = int(2 * math.ceil(2.5 * s) + 1)
        ax = torch.arange(k, device=device) - (k - 1) / 2.0
        g = torch.exp(-0.5 * (ax / s) ** 2)
        g = g / g.sum()
        x = n[None]
        x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"),
                     g.view(1, 1, 1, k).expand(c, 1, 1, k), groups=c)
        x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"),
                     g.view(1, 1, k, 1).expand(c, 1, k, 1), groups=c)
        n = x[0]
        g2 = torch.outer(g, g)
        n = n / torch.sqrt((g2 ** 2).sum())     # restore unit marginal std
    elif params["spatial"] == "resample":
        f = params["factor"]
        sh, sw = max(int(h / f), 2), max(int(w / f), 2)
        small = torch.randn(1, c, sh, sw, generator=gen).to(device)
        n = F.interpolate(small, size=(h, w), mode="bicubic",
                          align_corners=False)[0]
        n = n / n.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return img + n * sigma, sigma


def _mtf(x, m):
    return ((m - 1) * x) / ((2 * m - 1) * x - m)


def _solve_mtf_m(b, t):
    """Midtone m such that _mtf(b, m) == t (Siril-autostretch style)."""
    b = min(max(b, 1e-4), 0.98)
    t = min(max(t, 0.02), 0.98)
    if abs(b - t) < 1e-4:
        return 0.5                                # m=0.5 -> identity
    m = b * (t - 1.0) / (2.0 * b * t - t - b)
    return min(max(m, 1e-3), 1.0 - 1e-3)


def _solve_asinh_a(b, t):
    """a such that asinh(a*b)/asinh(a) == t; None if unreachable (t<=b)."""
    b = min(max(b, 1e-4), 0.98)
    if t <= b + 1e-4:
        return None
    lo, hi = 1.0, 1e5
    for _ in range(48):
        mid = math.sqrt(lo * hi)
        v = math.asinh(mid * b) / math.asinh(mid)
        if v < t:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def presentation(gen, device):
    """v1: random monotonic display transform T applied to BOTH input and
    target (kept verbatim for blur.py / the v1 recipe)."""
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


def presentation_v2(gen, device, cfg, ref):
    """TARGET-DRIVEN composed presentation chain. A background-median target
    is sampled U(tgt_median_lo, tgt_median_hi); the base stretch (MTF or
    arcsinh) is SOLVED to land the reference background there, then optional
    per-channel gain jitter, mild curves, saturation boost and explicit
    per-channel background NEUTRALIZATION are composed on top.
    `ref` = the noisy starless image (statistics only). Returns a pure,
    deterministic T applied identically to input AND targets."""
    if torch.rand(1, generator=gen).item() < cfg["pure_linear_frac"]:
        return lambda x: x
    b_med = float(ref.median().clamp(0, 1))
    t = float(_rand(cfg["tgt_median_lo"], cfg["tgt_median_hi"], 1, gen=gen))
    use_asinh = torch.rand(1, generator=gen).item() < 0.3
    a_par = _solve_asinh_a(b_med, t) if use_asinh else None
    if a_par is not None:
        la = math.asinh(a_par)

        def base(x):
            return torch.asinh(x.clamp_min(0) * a_par) / la
    else:
        m = _solve_mtf_m(b_med, t)

        def base(x):
            return _mtf(x.clamp(0, 1), m)
    # per-channel gain jitter (pre-stretch, linear domain)
    j = cfg["chan_gain_jitter"]
    gains = (1.0 + _rand(-j, j, 3, gen=gen)).to(device)[:, None, None] \
        if j > 0 else None
    gamma_v = float(_rand(0.85, 1.18, 1, gen=gen)) \
        if torch.rand(1, generator=gen).item() < cfg["curve_p"] else 1.0
    sat_k = float(_rand(1.05, 1.45, 1, gen=gen)) \
        if torch.rand(1, generator=gen).item() < cfg["satboost_p"] else 1.0

    def chain(x):
        y = x * gains if gains is not None else x
        y = base(y)
        if gamma_v != 1.0:
            y = y.clamp_min(0) ** gamma_v
        if sat_k != 1.0:
            mu = y.mean(0, keepdim=True)
            y = mu + sat_k * (y - mu)
        return y

    off = None
    if torch.rand(1, generator=gen).item() < cfg["neutralize_p"]:
        ry = chain(ref)
        med_ch = ry.flatten(1).median(dim=1).values[:, None, None]
        off = med_ch.mean() - med_ch          # per-channel offsets to grey

    if off is None:
        return chain
    return lambda x: chain(x) + off


# ------------------------------------------------------------------ main entry
def _make_pair_v1(h, w, background, gen, device):
    """The shipped v1 recipe, RNG-order identical to the original code."""
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
    clip_mask = (bg + stars_lin >= sat).any(0)
    sky = torch.minimum(sky, sat)
    starless_lin = torch.minimum(bg, sat)

    # inlined add_noise (identical draws) so sigma can be returned
    a = _loguniform(1e-5, 3e-3, 1, gen=gen).to(device)
    b = _loguniform(1e-4, 4e-3, 1, gen=gen).to(device)
    sigma_lin = torch.sqrt(a * sky.clamp_min(0) + b ** 2)
    noisy = sky + torch.randn(sky.shape, generator=gen).to(device) * sigma_lin
    noise = noisy - sky
    starless_noisy = starless_lin + noise      # same noise: removal != denoise

    T = presentation(gen, device)
    mask = torch.zeros(1, h, w, device=device)
    return dict(bg=bg, fam=fam, sky=sky, sat=sat, clip_mask=clip_mask,
                starless_lin=starless_lin, stars_lin=stars_lin,
                sigma_lin=sigma_lin, noisy=noisy, noise=noise,
                starless_noisy=starless_noisy, T=T, mask=mask,
                flags=dict(monster=False, silhouette=False))


def _make_pair_v2(h, w, background, gen, device, cfg):
    """The v2 recipe (see module docstring)."""
    if background is None:
        bg = procedural_background(h, w, gen, device, cfg)
    else:
        bg = background.to(device)
    fam = sample_psf_family(gen, device, cfg)
    nz = sample_noise_params(gen, cfg)          # noise model known up front
    # saturation level (per-channel jitter -> color-fringed clipped rims)
    sat = _rand(cfg["sat_lo"], cfg["sat_hi"], 1, gen=gen).to(device)
    if cfg["sat_chan_jitter"] > 0:
        jj = cfg["sat_chan_jitter"]
        sat = (sat * (1.0 + _rand(-jj, jj, 3, gen=gen).to(device))) \
            .clamp(0.3, 1.0)[:, None, None]
    else:
        sat = sat.expand(3)[:, None, None].contiguous()

    # ---- keep-content additions to the BACKGROUND
    if cfg["distractor_p"] > 0 \
            and torch.rand(1, generator=gen).item() < cfg["distractor_p"]:
        bg = bg + render_distractors(h, w, gen, device, cfg)
    if cfg["glow_p"] > 0 \
            and torch.rand(1, generator=gen).item() < cfg["glow_p"]:
        med = float(bg.median())
        sigma0 = math.sqrt(nz["a"] * max(med, 0.0) + nz["b"] ** 2)
        bg = bg + render_unresolved_glow(h, w, fam, gen, device, cfg, sigma0)

    # ---- field geometry: virtual full frame, crop somewhere inside it
    geom = None
    if cfg["virtual_field"]:
        wv = _rand(cfg["vframe_lo"], cfg["vframe_hi"], 1, gen=gen).item()
        hv = wv * _rand(0.55, 1.0, 1, gen=gen).item()
        wv, hv = max(wv, float(w)), max(hv, float(h))
        x0 = _rand(0, wv - w, 1, gen=gen).item()
        y0 = _rand(0, hv - h, 1, gen=gen).item()
        geom = dict(cx=wv / 2.0 - x0, cy=hv / 2.0 - y0,
                    rmax=math.hypot(wv / 2.0, hv / 2.0))

    # ---- stars
    pos, flux, color = sample_star_field(h, w, gen, device, cfg)
    stars_lin = render_stars(h, w, fam, pos, flux, color, gen, device,
                             flux_scale=cfg["flux_scale"], geom=geom,
                             halo_min_stamp=cfg["halo_min_stamp"])
    has_monster = False
    if cfg["monster_p"] > 0 \
            and torch.rand(1, generator=gen).item() < cfg["monster_p"]:
        stars_lin = stars_lin + render_monsters(h, w, fam, gen, device, cfg,
                                                sat)
        has_monster = True

    # ---- foreground silhouette OCCLUDES the star layer (physics fix)
    mask = torch.zeros(1, h, w, device=device)
    has_sil = False
    if cfg["silhouette_p"] > 0 \
            and torch.rand(1, generator=gen).item() < cfg["silhouette_p"]:
        mask, fg, sglow = render_silhouette(h, w, gen, device, cfg)
        bg = (bg + sglow) * (1.0 - mask) + fg * mask
        stars_lin = stars_lin * (1.0 - mask)
        has_sil = True

    sky = bg + stars_lin
    clip_mask = (sky >= sat).any(0)
    sky = torch.minimum(sky, sat)
    starless_lin = torch.minimum(bg, sat)

    noisy, sigma_lin = apply_noise(sky, nz, gen, device)
    noise = noisy - sky
    starless_noisy = starless_lin + noise      # same noise: removal != denoise

    if cfg["presentation"] == "v1":
        T = presentation(gen, device)
    else:
        T = presentation_v2(gen, device, cfg, starless_noisy)
    return dict(bg=bg, fam=fam, sky=sky, sat=sat, clip_mask=clip_mask,
                starless_lin=starless_lin, stars_lin=stars_lin,
                sigma_lin=sigma_lin, noisy=noisy, noise=noise,
                starless_noisy=starless_noisy, T=T, mask=mask,
                flags=dict(monster=has_monster, silhouette=has_sil))


def make_pair(h, w, background, seed, device="cuda", cfg=None,
              return_extras=False):
    """One training triplet. background: (3,H,W) linear starless tensor
    (real plate crop or None -> procedural). cfg: see default_config()
    (None -> v2 defaults). return_extras=True additionally returns a dict
    with the presented-units local noise sigma, silhouette mask, and all
    intermediate layers (for the loss, tests and coverage reports)."""
    if cfg is None:
        cfg = default_config(2)
    gen = torch.Generator().manual_seed(seed)
    if cfg.get("version", 1) == 1:
        s = _make_pair_v1(h, w, background, gen, device)
    else:
        s = _make_pair_v2(h, w, background, gen, device, cfg)

    T = s["T"]
    inp = T(s["noisy"]).clamp(0, 1)
    tgt_starless = T(s["starless_noisy"]).clamp(0, 1)
    tgt_starless = torch.minimum(tgt_starless, inp)   # stars layer >= 0 exactly
    out = (inp.float(), tgt_starless.float(), (inp - tgt_starless).float())
    if not return_extras:
        return out
    # local noise scale in PRESENTED units: push sigma through T at the
    # local background level (offset terms in T cancel in the difference)
    lo = T(s["starless_lin"]).clamp(0, 1)
    hi = T(s["starless_lin"] + s["sigma_lin"]).clamp(0, 1)
    sigma_pres = (hi - lo).abs().clamp_min(1e-5).float()
    extras = dict(sigma=sigma_pres, mask=s["mask"], noise=s["noise"],
                  sky=s["sky"], starless_lin=s["starless_lin"],
                  stars_lin=s["stars_lin"], noisy=s["noisy"],
                  starless_noisy=s["starless_noisy"], sat=s["sat"],
                  clip_mask=s["clip_mask"], sigma_lin=s["sigma_lin"],
                  T=T, fam=s["fam"], flags=s["flags"])
    return out + (extras,)


class PairDataset(torch.utils.data.Dataset):
    """On-the-fly pairs over a folder of linear starless background tiles
    (float32 .npy, (3,H,W), values 0..1). Empty folder -> fully procedural.
    Returns (input, starless, stars, sigma) — sigma is the per-pixel local
    noise scale in presented units (the v2 loss normalizes by it; the v1
    loss ignores it)."""

    def __init__(self, bg_dir=None, crop=256, length=100000, device="cpu",
                 base_index=0, cfg=None):
        import glob
        import os
        self.crop = crop
        self.length = length
        self.device = device
        self.base = base_index          # absolute-step offset for resume
        self.cfg = cfg if cfg is not None else default_config(2)
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
        inp, starless, stars, extras = make_pair(
            self.crop, self.crop, bg, seed=j * 7919 + 7, device=self.device,
            cfg=self.cfg, return_extras=True)
        return inp, starless, stars, extras["sigma"]
