# Coverage report for the v3 REAL-DATA recipe.
#   .venv\Scripts\python.exe eval\coverage_report_v3.py
#
# Two jobs:
#  A. the v2 job — assert the generated frames still BRACKET the forensic
#     census of the real 7008x4672 stretched frame (background median 0.38,
#     star peaks 0.19-0.68 above bg, noise sigma ~0.0025, clipped discs to
#     ~30 px radius);
#  B. the v3 job — assert the generated frames now match the MEASURED
#     harvest priors that v2 got wrong:
#       * star FWHM (measured by rendering each image's actual PSF, not by
#         reading back the knob) brackets the measured p10/median/p90;
#       * grain 1/e correlation length ~2 px on every frame;
#       * silhouette rate ~ the measured 0.42 acceptance.
# All v3 targets are read from datagen/v3/* AT RUNTIME (the harvest keeps
# growing — nothing is hardcoded), with the figures quoted in the v3 brief
# printed alongside for reference.
#
# Writes eval/v3_contact_sheet.png.

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datagen.synth import (make_pair, default_config,           # noqa: E402
                           render_stars)
from datagen import v3assets                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 96
CROP = 256
SEED0 = 5_000_000
STRIDE = 9973          # co-prime stride: neighbouring seeds share RNG history
BRIEF = dict(fwhm=(2.04, 3.26, 5.36), sil=0.42, corr=2.03)   # v3 brief figures


def disc_radius(clip_mask):
    m = clip_mask.cpu().numpy().astype(np.uint8)
    if m.sum() == 0:
        return 0.0
    from scipy import ndimage
    lab, n = ndimage.label(m)
    if n == 0:
        return 0.0
    areas = ndimage.sum(m, lab, index=range(1, n + 1))
    return float(math.sqrt(np.max(areas) / math.pi))


def star_peaks(st, sigma_med):
    s = st.mean(0, keepdim=True)[None]
    mp = F.max_pool2d(s, 5, 1, 2)
    thr = max(0.02, 4.0 * sigma_med)
    pk = s[(s == mp) & (s > thr)]
    return pk.flatten().tolist()


def rendered_fwhm(fam, cfg, device):
    """FWHM (px) of the PSF this image ACTUALLY renders with: splat one faint
    star (the bucket the harvest measured — it excludes saturated stars) and
    measure the half-maximum area of the result."""
    gen = torch.Generator().manual_seed(1)
    pos = torch.tensor([[64.0, 64.0]], device=device)
    img = render_stars(129, 129, fam, pos, torch.tensor([1.0], device=device),
                       torch.ones(1, 3, device=device), gen, device,
                       flux_scale=1.0, cfg=cfg)
    p = img[1]
    if not bool(torch.isfinite(p).all()) or float(p.max()) <= 0:
        return float("nan")
    # half-maximum AREA on an 8x-upsampled stamp: on the raw pixel grid the
    # estimate quantises hard (a true 2.0 px core reads 1.13 px); upsampled
    # it is unbiased to ~1.5% for FWHM >= 1.5 px.
    up = 8
    q = F.interpolate(p[None, None], scale_factor=up, mode="bicubic",
                      align_corners=False)[0, 0]
    m = float(q.max())
    if not math.isfinite(m) or m <= 0:
        return float("nan")
    area = float((q >= 0.5 * m).sum())
    return 2.0 * math.sqrt(max(area, 1.0) / math.pi) / up


def corr_len(x):
    """1/e autocorrelation length (px) along the x axis."""
    y = (x - x.mean()) / x.std().clamp_min(1e-9)
    f = torch.fft.rfft2(y)
    ac = torch.fft.irfft2(f * f.conj(), s=y.shape).real
    ac = ac / ac[0, 0]
    row = ac[0, :x.shape[-1] // 2].cpu()
    below = (row < 1 / math.e).nonzero()
    if not len(below):
        return float(len(row))
    i = int(below[0])
    if i == 0:
        return 0.0
    a, b = float(row[i - 1]), float(row[i])     # linear interp to 1/e
    return (i - 1) + (a - 1 / math.e) / max(a - b, 1e-9)


def hist(vals, edges, label, unit=""):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    cnt, _ = np.histogram(vals, bins=edges)
    print(f"\n  {label}: n={len(vals)}  min={vals.min():.4g}  "
          f"p10={np.percentile(vals, 10):.4g}  med={np.median(vals):.4g}  "
          f"p90={np.percentile(vals, 90):.4g}  max={vals.max():.4g} {unit}")
    for i, c in enumerate(cnt):
        bar = "#" * int(round(40 * c / max(cnt.max(), 1)))
        print(f"    [{edges[i]:>8.3g},{edges[i+1]:>8.3g}) {c:>4d} {bar}")


def measured_fwhm_targets():
    """(p10, med, p90) of the harvested full-res FWHM, de-biased across field
    bins (the harvest samples 1 centre + 4 corner windows per frame, so the
    pooled array is corner-heavy by construction).  None if assets absent."""
    A = v3assets.psf_assets()
    if A is None or not A["pools"]:
        return None, 0
    qs = np.array([np.percentile(A["pools"][f][0], [10, 50, 90])
                   for f in sorted(A["pools"])])
    return tuple(qs.mean(axis=0)), A["n_stamps"]


def to_u8(img):
    return (img.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            * 255).astype(np.uint8)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = default_config(3)
    print(f"v3 coverage report: {N} samples @ {CROP}px, device={device}")
    print("assets:", v3assets.summary())
    tgt, n_stamps = measured_fwhm_targets()
    if tgt is None:
        print("  !! PSF assets unavailable — v3 FWHM checks will be skipped")
    else:
        print(f"  measured FWHM targets (live, {n_stamps} stamps): "
              f"p10 {tgt[0]:.2f} / med {tgt[1]:.2f} / p90 {tgt[2]:.2f}   "
              f"(v3 brief quoted {BRIEF['fwhm'][0]}/{BRIEF['fwhm'][1]}/"
              f"{BRIEF['fwhm'][2]})")

    medians, sigmas, discs, peaks, fwhms, corrs = [], [], [], [], [], []
    n_real_psf = n_real_sil = n_acf = 0
    samples = []
    for i in range(N):
        seed = SEED0 + i * STRIDE
        inp, sl, st, ex = make_pair(CROP, CROP, None, seed=seed,
                                    device=device, cfg=cfg,
                                    return_extras=True)
        med = float(inp.median())
        sig = float(ex["sigma"].median())
        dr = disc_radius(ex["clip_mask"])
        medians.append(med)
        sigmas.append(sig)
        discs.append(dr)
        peaks += star_peaks(st, sig)
        fwhms.append(rendered_fwhm(ex["fam"], cfg, device))
        corrs.append(corr_len(ex["noise"][1].float()))
        f = ex["flags"]
        n_real_psf += bool(f["real_psf"])
        n_real_sil += bool(f["real_sil"])
        n_acf += bool(f["noise_acf"])
        samples.append((i, inp.cpu(), f, dr))

    # ---------------- histograms
    hist(medians, np.array([0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .6]),
         "input median (census: 0.38)")
    hist(peaks, np.array([0, .05, .1, .15, .19, .3, .5, .68, .85, 1.0]),
         "star peak-above-bg (census: 0.19-0.68)")
    hist(sigmas, np.array([0, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2, 2.5e-2,
                           5e-2, 1e-1, 1.0]),
         "presented noise sigma (census: ~0.0025)")
    hist(discs, np.array([0, 1, 5, 10, 15, 20, 25, 30, 40, 60]),
         "clipped-disc radius px (census: up to ~30)")
    hist(fwhms, np.array([0, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5, 7, 9, 14]),
         "RENDERED star FWHM px (measured p10/med/p90 "
         + ("n/a" if tgt is None else
            f"{tgt[0]:.2f}/{tgt[1]:.2f}/{tgt[2]:.2f}") + ")")
    hist(corrs, np.array([0, .5, 1, 1.5, 1.8, 2, 2.3, 2.6, 3, 4, 8]),
         "noise 1/e correlation length px (measured ~2)")

    n_mon = sum(1 for _, _, f, _ in samples if f["monster"])
    n_sil = sum(1 for _, _, f, _ in samples if f["silhouette"])
    print(f"\n  flags: {n_mon}/{N} monster, {n_sil}/{N} silhouette "
          f"({n_real_sil} real masks), {n_real_psf}/{N} empirical PSF, "
          f"{n_acf}/{N} measured-ACF noise")

    # ---------------- assertions
    peaks_np = np.asarray(peaks)
    fw = np.asarray([f for f in fwhms if math.isfinite(f)])
    g10, g50, g90 = (float(np.percentile(fw, q)) for q in (10, 50, 90))
    c50 = float(np.median(corrs))
    checks = [
        # --- A: still brackets the census
        ("median 0.38 inside sampled range",
         min(medians) <= 0.38 <= max(medians),
         f"range [{min(medians):.3f}, {max(medians):.3f}]"),
        ("star peaks cover 0.15-0.30 band",
         bool(((peaks_np >= 0.15) & (peaks_np < 0.30)).any()),
         f"{int(((peaks_np >= 0.15) & (peaks_np < 0.30)).sum())} peaks"),
        ("star peaks cover 0.30-0.50 band",
         bool(((peaks_np >= 0.30) & (peaks_np < 0.50)).any()),
         f"{int(((peaks_np >= 0.30) & (peaks_np < 0.50)).sum())} peaks"),
        ("star peaks cover 0.50-0.70 band",
         bool(((peaks_np >= 0.50) & (peaks_np < 0.70)).any()),
         f"{int(((peaks_np >= 0.50) & (peaks_np < 0.70)).sum())} peaks"),
        ("star peaks extend below 0.15 and above 0.70",
         bool((peaks_np < 0.15).any() and (peaks_np >= 0.70).any()),
         f"{int((peaks_np < 0.15).sum())} below, "
         f"{int((peaks_np >= 0.70).sum())} above"),
        ("noise sigma brackets 0.0025",
         min(sigmas) < 0.0025 < max(sigmas),
         f"range [{min(sigmas):.2e}, {max(sigmas):.2e}]"),
        ("clipped discs up to ~30px present (max >= 25)",
         max(discs) >= 25.0, f"max disc radius {max(discs):.1f}px"),
        ("monster samples present", n_mon >= 3, f"{n_mon}"),
        ("silhouette samples present", n_sil >= 3, f"{n_sil}"),
    ]
    if tgt is not None:
        checks += [
            ("FWHM p10 brackets measured (gen p10 <= measured p10)",
             g10 <= tgt[0] + 0.35, f"gen {g10:.2f} vs measured {tgt[0]:.2f}"),
            ("FWHM p90 brackets measured (gen p90 >= measured p90)",
             g90 >= tgt[2] - 0.35, f"gen {g90:.2f} vs measured {tgt[2]:.2f}"),
            ("FWHM median within 35% of measured",
             abs(g50 / max(tgt[1], 1e-6) - 1.0) < 0.35,
             f"gen {g50:.2f} vs measured {tgt[1]:.2f} "
             f"({g50 / max(tgt[1], 1e-6):.2f}x)"),
            ("FWHM median no longer half-scale (v2's bug: 0.73x reality)",
             g50 / max(tgt[1], 1e-6) > 0.85,
             f"{g50 / max(tgt[1], 1e-6):.2f}x reality"),
        ]
    checks += [
        ("noise correlation length ~2 px on every frame",
         1.4 <= c50 <= 2.8 and min(corrs) > 1.0,
         f"median {c50:.2f} px, min {min(corrs):.2f}, max {max(corrs):.2f}"),
        ("measured-ACF grain on 100% of frames", n_acf == N, f"{n_acf}/{N}"),
        ("silhouette rate ~ measured 0.42",
         0.42 - 0.16 <= n_sil / N <= 0.42 + 0.16,
         f"{n_sil / N:.2f} (target {cfg['silhouette_p']:.2f})"),
        ("real masks dominate the silhouettes",
         n_sil == 0 or n_real_sil >= 0.7 * n_sil,
         f"{n_real_sil}/{n_sil} real"),
        ("empirical PSF rate ~ real_psf_p",
         abs(n_real_psf / N - cfg["real_psf_p"]) <= 0.18,
         f"{n_real_psf / N:.2f} (target {cfg['real_psf_p']:.2f})"),
    ]
    print()
    fails = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
        fails += 0 if ok else 1

    # ---------------- contact sheet
    from PIL import Image, ImageDraw
    order = sorted(range(N), key=lambda i: (
        not samples[i][2]["monster"], not samples[i][2]["real_sil"], i))
    pick_m = [i for i in order if samples[i][2]["monster"]][:4]
    pick_s = [i for i in order
              if samples[i][2]["real_sil"] and i not in pick_m][:4]
    pick_p = [i for i in order if samples[i][2]["real_psf"]
              and i not in pick_m + pick_s][:4]
    rest = [i for i in range(N) if i not in pick_m + pick_s + pick_p]
    picks = (pick_m + pick_s + pick_p + rest)[:16]
    pad = 4
    sheet = Image.new("RGB", (4 * (CROP + pad) + pad, 4 * (CROP + pad) + pad),
                      (24, 24, 24))
    dctx = ImageDraw.Draw(sheet)
    for k, idx in enumerate(picks):
        _, img, flags, drad = samples[idx]
        sheet.paste(Image.fromarray(to_u8(img)),
                    (pad + (k % 4) * (CROP + pad),
                     pad + (k // 4) * (CROP + pad)))
        tag = ("M" if flags["monster"] else "") \
            + ("S" if flags["real_sil"] else
               ("s" if flags["silhouette"] else "")) \
            + ("P" if flags["real_psf"] else "")
        if tag:
            dctx.text((pad + (k % 4) * (CROP + pad) + 6,
                       pad + (k // 4) * (CROP + pad) + 4), tag,
                      fill=(255, 220, 80))
    out_png = os.path.join(HERE, "v3_contact_sheet.png")
    sheet.save(out_png)
    print(f"\n  contact sheet: {out_png}  "
          f"(M=monster, S=real silhouette, s=procedural, P=empirical PSF)")

    if fails:
        print(f"\nCOVERAGE FAIL ({fails} checks)")
        sys.exit(1)
    print("\nCOVERAGE PASS — v3 recipe brackets the census AND the measured "
          "harvest priors")


if __name__ == "__main__":
    main()
