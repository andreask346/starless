# EasySharp evaluation panel:
#   .venv\Scripts\python.exe eval\panel_sharp.py runs\sharp_w32\best.pt
# 1) Synthetic ground-truth: PSNR, flux-preservation error, re-blur
#    consistency, and a NO-HALLUCINATION check (deconvolving an already-sharp
#    image with a tiny PSF must be ~identity).
# 2) Real frames: measure PSF from stars, run deconv, save before/after panels.

import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "datagen"))
from blur import make_pair, conv_psf, PSF_REF, COND_DIM   # noqa: E402
from models.nafnet_sharp import build                     # noqa: E402

TORTURE = os.path.join(os.path.dirname(__file__), "torture_list.txt")


def synthetic_suite(model, device, n=40, crop=256):
    tot = dict(psnr=0.0, flux=0.0, reblur=0.0, ident=0.0)
    for i in range(n):
        inp, tgt, cond, kernel, reblur = make_pair(
            crop, crop, None, seed=61_000_000 + i, device=device)
        with torch.no_grad():
            out = model(inp[None], cond[None])[0]
        tot["psnr"] += -10 * math.log10(
            max(float(((out - tgt) ** 2).mean()), 1e-12)) / n
        tot["flux"] += abs(float(out.sum() / tgt.sum().clamp_min(1e-6))
                           - 1.0) * 100 / n
        re = conv_psf(out, kernel)
        tot["reblur"] += float((re - reblur).abs().mean()) / n
        # no-hallucination: feed an already-sharp image + near-identity PSF
        cond_id = torch.tensor([1.0, 1.0, 0.0, 0.0], device=device)
        with torch.no_grad():
            out_id = model(tgt[None], cond_id[None])[0]
        tot["ident"] += float((out_id - tgt).abs().mean()) / n
    return {k: round(v, 4) for k, v in tot.items()}


def measure_cond(lum, device):
    """Fit a rough FWHM from the frame's stars -> conditioning vector."""
    import sep
    l = np.ascontiguousarray(lum, dtype=np.float32)
    bkg = sep.Background(l)
    data = l - bkg.back()
    thr = max(5 * float(bkg.globalrms), 0.01,
              float(np.quantile(data[::4, ::4], 0.98)))
    try:
        objs = sep.extract(data, thr, minarea=5, deblend_cont=1.0)
    except Exception:
        objs = []
    if len(objs) < 20:
        return torch.tensor([1.5, 1.0, 0.1, 0.0], device=device)
    a = np.maximum(objs["a"], 1e-3)
    b = np.maximum(objs["b"], 1e-3)
    fwhm = float(np.median(2.3548 * np.sqrt(a * b)))
    elong = float(np.median(1 - b / a))
    return torch.tensor([min(fwhm / PSF_REF, 4.0), 1.0, elong, 0.0],
                        device=device)


def real_panels(model, device, out_dir, tile=512, overlap=128):
    import tifffile
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "datagen"))
    from backgrounds import load_any
    if not os.path.isfile(TORTURE):
        return []
    os.makedirs(out_dir, exist_ok=True)
    done = []
    paths = [ln.strip().lstrip("﻿") for ln in
             open(TORTURE, encoding="utf-8-sig")
             if ln.strip() and not ln.lstrip("﻿").startswith("#")]
    for p in paths:
        try:
            img = np.clip(load_any(p), 0, 1).astype(np.float32)
        except Exception as e:
            print(f"SKIP {p}: {e}", flush=True)
            continue
        c, h, w = img.shape
        if max(h, w) > 3200:
            y0, x0 = (h - 2048) // 2, (w - 2048) // 2
            img = img[:, max(y0, 0):y0 + 2048, max(x0, 0):x0 + 2048]
            c, h, w = img.shape
        cond = measure_cond(img.mean(axis=0), device)[None]
        step = tile - overlap
        pad_h = (step - (h - tile) % step) % step if h > tile else tile - h
        pad_w = (step - (w - tile) % step) % step if w > tile else tile - w
        padded = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        ph, pw = padded.shape[1:]
        win = (np.outer(np.hanning(tile), np.hanning(tile)) + 1e-4).astype(
            np.float32)
        out = np.zeros_like(padded)
        wsum = np.zeros((ph, pw), np.float32)
        with torch.no_grad():
            for y in range(0, ph - tile + 1, step):
                for x in range(0, pw - tile + 1, step):
                    t = torch.from_numpy(
                        padded[None, :, y:y + tile, x:x + tile]).to(device)
                    pred = model(t, cond)[0].cpu().numpy()
                    out[:, y:y + tile, x:x + tile] += pred * win
                    wsum[y:y + tile, x:x + tile] += win
        out /= wsum[None]
        sharp = np.clip(out[:, :h, :w], 0, 1)
        panel = np.concatenate([img, sharp], axis=2)
        name = os.path.splitext(os.path.basename(p))[0]
        o = os.path.join(out_dir, f"sharp_{name}.tif")
        tifffile.imwrite(o, (np.moveaxis(panel, 0, 2) * 65535).astype(
            np.uint16), compression="zlib")
        done.append(o)
        print(f"panel: {o}", flush=True)
    return done


def main(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ck.get("config", {})
    model = build(width=cfg.get("width", 32),
                  use_ffc=not cfg.get("no_ffc", False)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    metrics = synthetic_suite(model, device)
    run_dir = os.path.dirname(ckpt_path)
    panels = real_panels(model, device, os.path.join(run_dir, "panel"))
    metrics["panels"] = [os.path.basename(p) for p in panels]
    metrics["step"] = ck.get("step")
    json.dump(metrics, open(os.path.join(run_dir, "metrics.json"), "w"),
              indent=1)
    print("METRICS " + json.dumps(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
