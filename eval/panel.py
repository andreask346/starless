# Evaluation panel for a trained checkpoint:
#   .venv\Scripts\python.exe eval\panel.py runs\w32_v0\best.pt
# 1) Synthetic ground-truth suite: in/out-of-footprint PSNR, residual-star
#    completeness by brightness bin, star-layer leakage, recomposition dB.
# 2) Real frames: tiled inference on the torture list -> before/starless/stars
#    TIFF panels for eyeball judging (runs\<name>\panel\).
# Writes metrics JSON next to the checkpoint; prints a compact summary line
# (the queue runner puts it in the status push).

import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datagen.synth import make_pair, default_config  # noqa: E402
from models.nafnet_star import build           # noqa: E402
from eval.infer_tiles import infer_image       # noqa: E402

TORTURE = os.path.join(os.path.dirname(__file__), "torture_list.txt")


def synthetic_suite(model, device, n=40, crop=256, data_cfg=None):
    # data_cfg must match the recipe the model TRAINED on, else the metrics
    # are incomparable across runs (v1 history was scored on the v1 recipe)
    if data_cfg is None:
        data_cfg = default_config(1)
    tot = dict(psnr_in=0.0, psnr_out=0.0, leak=0.0, recomp=0.0)
    n_out = 0
    bins = {k: [0, 0] for k in ("faint", "mid", "bright")}   # found, total
    for i in range(n):
        inp, sl, st = make_pair(crop, crop, None, seed=31_000_000 + i,
                                device=device, cfg=data_cfg)
        with torch.no_grad():
            pred = model(inp[None])[0]
        starless = inp - pred
        foot = st > 1e-4
        mse_in = float(((starless - sl)[foot] ** 2).mean()) if foot.any() else 0
        tot["psnr_in"] += -10 * math.log10(max(mse_in, 1e-12)) / n
        inv = ~foot
        if inv.any():   # dense fields can put star flux under every pixel
            mse_out = float(((starless - sl)[inv] ** 2).mean())
            tot["psnr_out"] += -10 * math.log10(max(mse_out, 1e-12))
            tot["leak"] += float(pred[inv].abs().mean())
            n_out += 1
        recomp = float(((starless + pred) - inp).abs().max())
        tot["recomp"] += -20 * math.log10(max(recomp, 1e-9)) / n
        # per-star completeness: fraction of true star flux removed, by bin
        smax = float(st.max())
        for name, lo, hi in (("faint", 1e-4, 0.01), ("mid", 0.01, 0.2),
                             ("bright", 0.2, max(smax, 0.21))):
            m = (st > lo) & (st <= hi)
            if m.any():
                frac = float(pred[m].sum() / st[m].sum())
                bins[name][0] += min(frac, 1.0)
                bins[name][1] += 1
    for k in ("psnr_out", "leak"):
        tot[k] /= max(n_out, 1)
    comp = {k: round(v[0] / max(v[1], 1), 4) for k, v in bins.items()}
    return {**{k: round(v, 2) for k, v in tot.items()},
            "completeness": comp}


def real_panels(model, device, out_dir, tile=512):
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
        if max(h, w) > 3200:                       # panel crops, not full frames
            y0, x0 = (h - 2048) // 2, (w - 2048) // 2
            img = img[:, max(y0, 0):y0 + 2048, max(x0, 0):x0 + 2048]
        stars = infer_image(model, img, tile=tile, device=device)
        starless = img - stars
        panel = np.concatenate([img, starless, np.clip(stars, 0, 1)], axis=2)
        name = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(out_dir, f"panel_{name}.tif")
        tifffile.imwrite(out, (np.moveaxis(panel, 0, 2) * 65535).astype(
            np.uint16), compression="zlib")
        done.append(out)
        print(f"panel: {out}", flush=True)
    return done


def main(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ck.get("config", {})
    model = build(width=cfg.get("width", 32),
                  use_ffc=not cfg.get("no_ffc", False)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    recipe = str(cfg.get("data_recipe", "v1"))
    metrics = synthetic_suite(
        model, device, data_cfg=default_config(1 if recipe == "v1" else 2))
    metrics["data_recipe"] = recipe
    run_dir = os.path.dirname(ckpt_path)
    panels = real_panels(model, device, os.path.join(run_dir, "panel"))
    metrics["panels"] = [os.path.basename(p) for p in panels]
    metrics["step"] = ck.get("step")
    out = os.path.join(run_dir, "metrics.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=1)
    print("METRICS " + json.dumps(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
