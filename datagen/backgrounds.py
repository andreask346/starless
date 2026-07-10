# Prepare training-background tiles from starless frames.
#   .venv\Scripts\python.exe datagen\backgrounds.py --list picks.txt --out datagen\backgrounds
# Loads TIFF/FITS starless images, de-stretches toward linear (his starless
# exports are display-stretched; a gamma-family inverse gets close enough —
# the training presentation transform re-stretches randomly anyway), slices
# into 1024x1024 tiles, drops near-empty/clipped tiles, saves float32 .npy
# (3,H,W) in 0..1.

import argparse
import os
import sys

import numpy as np


def load_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(path)
        if arr.ndim == 2:
            arr = arr[..., None].repeat(3, axis=2)
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr /= 65535.0 if arr.max() > 255 else 255.0
        return np.moveaxis(arr, 2, 0)                      # (3,H,W)
    from astropy.io import fits
    with fits.open(path) as h:
        hdu = next(x for x in h if x.data is not None)
        d = hdu.data.astype(np.float32)
    if d.ndim == 2:
        d = d[None].repeat(3, axis=0)
    elif d.ndim == 3 and d.shape[0] not in (1, 3):
        d = np.moveaxis(d, -1, 0)
    if d.shape[0] == 1:
        d = d.repeat(3, axis=0)
    if d.max() > 1.5:
        d /= 65535.0
    return d


def destretch(img):
    """Heuristic inverse of a display stretch: pick gamma so the median lands
    in linear-ish territory (~0.02). Exact linearity is not required — the
    training pipeline restretches with random transforms."""
    med = float(np.median(img))
    if med <= 0.03:
        return img                                          # already linear-ish
    target = 0.02
    g = np.log(target) / np.log(max(med, 1e-4))
    g = float(np.clip(g, 1.0, 8.0))
    return np.power(np.clip(img, 0, 1), g)


def tiles_from(img, tile=1024, stride=896):
    c, h, w = img.shape
    for y in range(0, max(h - tile + 1, 1), stride):
        for x in range(0, max(w - tile + 1, 1), stride):
            t = img[:, y:y + tile, x:x + tile]
            if t.shape[1] < tile or t.shape[2] < tile:
                continue
            lum = t.mean(axis=0)
            # drop empty / hard-clipped / dead tiles
            if float(np.ptp(lum)) < 1e-4 or float((lum > 0.98).mean()) > 0.2 \
                    or float(lum.mean()) < 1e-5:
                continue
            yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True,
                    help="text file: one image path per line")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "backgrounds"))
    ap.add_argument("--max-tiles-per-image", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    paths = [ln.strip().lstrip("﻿") for ln in
             open(args.list, encoding="utf-8-sig")
             if ln.strip() and not ln.lstrip("﻿").startswith("#")]
    total = 0
    for pi, p in enumerate(paths):
        try:
            img = load_any(p)
        except Exception as e:
            print(f"SKIP {p}: {e}", flush=True)
            continue
        img = destretch(np.clip(img, 0, 1))
        n = 0
        for ti, t in enumerate(tiles_from(img)):
            if n >= args.max_tiles_per_image:
                break
            np.save(os.path.join(args.out, f"bg{pi:04d}_{ti:03d}.npy"),
                    t.astype(np.float32))
            n += 1
        total += n
        print(f"[{pi+1}/{len(paths)}] {os.path.basename(p)}: {n} tiles "
              f"(total {total})", flush=True)
    print(f"done: {total} tiles in {args.out}")


if __name__ == "__main__":
    sys.exit(main())
