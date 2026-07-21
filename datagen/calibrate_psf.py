# Calibrate the synthetic PSF sampler against Andreas's REAL optics.
#   .venv\Scripts\python.exe datagen\calibrate_psf.py --list starry.txt --out datagen\psf_presets.json
# Detects stars (sep) in real starry frames, fits per-frame PSF statistics
# (FWHM, elongation center->corner, wing softness proxy), and writes presets
# the training sampler mixes in — so synthetic stars look like HIS stars
# (wide lenses: the regime where StarNet/SXT are weakest).

import argparse
import json
import os
import sys

import numpy as np


def analyze(path):
    import sep
    from backgrounds import load_any
    img = load_any(path)
    lum = np.ascontiguousarray(img.mean(axis=0), dtype=np.float32)
    h, w = lum.shape
    step = 1
    if max(h, w) > 4000:                       # analysis doesn't need full res
        step = int(np.ceil(max(h, w) / 4000))
        lum = np.ascontiguousarray(lum[::step, ::step])
        h, w = lum.shape
    bkg = sep.Background(lum)
    data = lum - bkg.back()
    thr = max(5.0 * float(bkg.globalrms), 0.01,
              float(np.quantile(data[::4, ::4], 0.98)))
    objs = sep.extract(data, thr, minarea=5, deblend_cont=1.0)
    if len(objs) < 30:
        return None
    a = np.maximum(objs["a"], 1e-3)
    b = np.maximum(objs["b"], 1e-3)
    ok = (objs["peak"] < 0.95 * float(data.max())) & (b / a > 0.1)
    a, b = a[ok], b[ok]
    x, y = objs["x"][ok], objs["y"][ok]
    if len(a) < 30:
        return None
    # BUG FIX 2026-07-19: FWHM is measured on the DOWN-SAMPLED image but was
    # written out (and consumed by sample_psf_family) as full-res pixels, so
    # every preset from a >4000px frame was 1/step too narrow. On his 7008px
    # frames step=2, i.e. presets said ~1.7px when his stars measure ~3.26px
    # full-res. Half of v2's training images drew from these presets.
    fwhm = 2.3548 * np.sqrt(a * b) * step
    r = np.hypot(x - w / 2, y - h / 2) / np.hypot(w / 2, h / 2)
    center = r < 0.35
    corner = r > 0.7
    el = 1.0 - b / a
    return dict(
        source=os.path.basename(path),
        fwhm_med=round(float(np.median(fwhm)), 2),
        fwhm_center=round(float(np.median(fwhm[center])) if center.any()
                          else float(np.median(fwhm)), 2),
        fwhm_corner=round(float(np.median(fwhm[corner])) if corner.any()
                          else float(np.median(fwhm)), 2),
        elong_center=round(float(np.median(el[center])) if center.any()
                           else 0.1, 3),
        elong_corner=round(float(np.median(el[corner])) if corner.any()
                           else 0.2, 3),
        n_stars=int(len(a)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "psf_presets.json"))
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(__file__))
    presets = []
    paths = [ln.strip().lstrip("﻿") for ln in
             open(args.list, encoding="utf-8-sig")
             if ln.strip() and not ln.lstrip("﻿").startswith("#")]
    for p in paths:
        try:
            r = analyze(p)
        except Exception as e:
            print(f"SKIP {p}: {e}", flush=True)
            continue
        if r:
            presets.append(r)
            print(json.dumps(r), flush=True)
    with open(args.out, "w") as f:
        json.dump(presets, f, indent=1)
    print(f"wrote {len(presets)} presets -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
