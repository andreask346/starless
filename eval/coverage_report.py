# Coverage report for the v2 data recipe vs the real-frame forensic census.
#   .venv\Scripts\python.exe eval\coverage_report.py
#
# Census of the real 7008x4672 stretched, background-neutralized frame:
#   background median 0.38 | star peaks 0.19-0.68 above bg
#   noise sigma ~0.0025    | saturated flat-white discs up to ~30 px radius
#
# Generates 64 v2 samples, prints histograms of input median, star
# peak-above-bg, presented noise sigma and clipped-disc radius, and ASSERTS
# the synthetic distributions BRACKET the census. Also renders a 16-sample
# contact sheet (incl. saturated monsters + silhouettes) to
# eval/v2_contact_sheet.png.

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datagen.synth import make_pair, default_config    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 64
CROP = 256
SEED0 = 5_000_000


def disc_radius(clip_mask):
    """Equivalent radius (px) of the largest clipped blob."""
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
    """Peak values (presented units, == peak-above-bg) of local maxima of
    the target star layer."""
    s = st.mean(0, keepdim=True)[None]
    mp = F.max_pool2d(s, 5, 1, 2)
    thr = max(0.02, 4.0 * sigma_med)
    pk = s[(s == mp) & (s > thr)]
    return pk.flatten().tolist()


def hist(vals, edges, label, unit=""):
    vals = np.asarray(vals, dtype=np.float64)
    cnt, _ = np.histogram(vals, bins=edges)
    print(f"\n  {label}: n={len(vals)}  min={vals.min():.4g}  "
          f"med={np.median(vals):.4g}  max={vals.max():.4g} {unit}")
    for i, c in enumerate(cnt):
        bar = "#" * int(round(40 * c / max(cnt.max(), 1)))
        print(f"    [{edges[i]:>8.3g},{edges[i+1]:>8.3g}) {c:>4d} {bar}")


def to_u8(img):
    return (img.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            * 255).astype(np.uint8)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = default_config(2)
    print(f"v2 coverage report: {N} samples @ {CROP}px, device={device}")

    medians, sigmas, discs, peaks = [], [], [], []
    samples = []            # (index, inp, flags, disc_r)
    for i in range(N):
        seed = SEED0 + i * 331
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
        samples.append((i, inp.cpu(), ex["flags"], dr))

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

    n_mon = sum(1 for _, _, f, _ in samples if f["monster"])
    n_sil = sum(1 for _, _, f, _ in samples if f["silhouette"])
    print(f"\n  flags: {n_mon}/{N} monster samples, "
          f"{n_sil}/{N} silhouette samples")

    # ---------------- assertions: BRACKET the census
    peaks_np = np.asarray(peaks)
    checks = [
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
    print()
    fails = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
        fails += 0 if ok else 1

    # ---------------- contact sheet: 16 samples incl. monsters + silhouettes
    from PIL import Image, ImageDraw
    order = sorted(range(N), key=lambda i: (
        not samples[i][2]["monster"], not samples[i][2]["silhouette"], i))
    pick_m = [i for i in order if samples[i][2]["monster"]][:4]
    pick_s = [i for i in order
              if samples[i][2]["silhouette"] and i not in pick_m][:4]
    rest = [i for i in range(N) if i not in pick_m + pick_s]
    picks = (pick_m + pick_s + rest)[:16]
    pad = 4
    sheet = Image.new("RGB", (4 * (CROP + pad) + pad, 4 * (CROP + pad) + pad),
                      (24, 24, 24))
    dr_ctx = ImageDraw.Draw(sheet)
    for k, idx in enumerate(picks):
        _, img, flags, drad = samples[idx]
        tile = Image.fromarray(to_u8(img))
        x = pad + (k % 4) * (CROP + pad)
        y = pad + (k // 4) * (CROP + pad)
        sheet.paste(tile, (x, y))
        tag = ("M" if flags["monster"] else "") \
            + ("S" if flags["silhouette"] else "")
        if tag:
            dr_ctx.text((x + 6, y + 4), tag, fill=(255, 220, 80))
    out_png = os.path.join(HERE, "v2_contact_sheet.png")
    sheet.save(out_png)
    print(f"\n  contact sheet: {out_png} "
          f"({len(pick_m)} monster, {len(pick_s)} silhouette tiles tagged)")

    if fails:
        print(f"\nCOVERAGE FAIL ({fails} checks)")
        sys.exit(1)
    print("\nCOVERAGE PASS — v2 recipe brackets the real-frame census")


if __name__ == "__main__":
    main()
