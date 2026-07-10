# EasySharp training harness (deconvolution / sharpening).
#   .venv\Scripts\python.exe train\train_sharp.py --name sharp_w32 --width 32 --iters 200000
# Same infra as train.py (resume, bf16, eval, samples) but PSF-conditioned
# model + deconvolution losses:
#   Charbonnier(out, sharp) + FFT + re-blur consistency (anti-hallucination)
#   + gradient loss (encourage crisp edges without ringing).

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "datagen"))
sys.path.insert(0, os.path.dirname(__file__))
from blur import SharpDataset, conv_psf, make_pair   # noqa: E402
from models.nafnet_sharp import build                # noqa: E402
from trainutils import atomic_save, safe_load, NaNGuard   # noqa: E402


def charbonnier(a, b, eps=1e-3):
    return torch.sqrt((a - b) ** 2 + eps * eps).mean()


def fft_loss(a, b):
    return (torch.fft.rfft2(a, norm="ortho")
            - torch.fft.rfft2(b, norm="ortho")).abs().mean()


def grad_loss(a, b):
    ax = a[..., :, 1:] - a[..., :, :-1]
    ay = a[..., 1:, :] - a[..., :-1, :]
    bx = b[..., :, 1:] - b[..., :, :-1]
    by = b[..., 1:, :] - b[..., :-1, :]
    return (ax - bx).abs().mean() + (ay - by).abs().mean()


def reblur_consistency(out, kernel, reblur_ref):
    """Re-blurring the sharpened output with the SAME degradation PSF must
    reproduce the (noise-free) blurred input. Penalizes invented structure
    that would not survive re-blurring — the core anti-hallucination term."""
    b = out.shape[0]
    re = torch.stack([conv_psf(out[i], kernel[i]) for i in range(b)])
    return charbonnier(re, reblur_ref)


def loss_fn(out, tgt, kernel, reblur_ref):
    l_main = charbonnier(out, tgt)
    l_fft = fft_loss(out, tgt)
    l_grad = grad_loss(out, tgt)
    l_re = reblur_consistency(out, kernel, reblur_ref)
    total = l_main + 0.1 * l_fft + 0.1 * l_grad + 0.3 * l_re
    return total, dict(main=l_main.item(), fft=l_fft.item(),
                       grad=l_grad.item(), reblur=l_re.item())


@torch.no_grad()
def quick_eval(model, device, crop=256, n=16):
    model.eval()
    tot_psnr, tot_flux = 0.0, 0.0
    for i in range(n):
        inp, tgt, cond, kernel, reblur = make_pair(
            crop, crop, None, seed=40_000_000 + i, device=device)
        out = model(inp[None], cond[None])[0]
        tot_psnr += -10 * math.log10(max(F.mse_loss(out, tgt).item(), 1e-12)) / n
        # flux preservation: total signal ratio out/target (deconv must conserve)
        tot_flux += abs(float(out.sum() / tgt.sum().clamp_min(1e-6)) - 1.0) / n
    model.train()
    return tot_psnr, tot_flux * 100.0


def save_samples(model, device, out_dir, step, crop=512):
    import numpy as np
    import tifffile
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(3):
            inp, tgt, cond, k, rb = make_pair(crop, crop, None,
                                              seed=50_000_000 + i, device=device)
            out = model(inp[None], cond[None])[0]
            panel = torch.cat([inp, out.clamp(0, 1), tgt], dim=2)
            arr = (panel.permute(1, 2, 0).cpu().numpy() * 65535).astype(np.uint16)
            tifffile.imwrite(os.path.join(out_dir, f"step{step:07d}_s{i}.tif"), arr)
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=200000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ref-dir", default="")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--no-ffc", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(os.path.dirname(__file__), "..", args.out, args.name)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(run_dir, "config.json"), "w"),
              indent=1)

    model = build(width=args.width, use_ffc=not args.no_ffc).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                            betas=(0.9, 0.9))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters, eta_min=1e-6)

    start = 0
    best = -1
    ckpt = os.path.join(run_dir, "last.pt")
    if args.resume:
        ck = safe_load(ckpt, device)
        if ck is not None:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start = ck["step"]
            best = ck.get("best", -1)
            print(f"resumed {args.name} at step {start} "
                  f"(best {best:.2f})", flush=True)

    ds = SharpDataset(ref_dir=args.ref_dir or None, crop=args.crop,
                      length=10_000_000, base_index=start * args.batch)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                    pin_memory=True, persistent_workers=args.workers > 0)
    it = iter(dl)
    print(f"train {args.name}: {n_params/1e6:.1f}M params (deconv), "
          f"device={device}, bf16, batch={args.batch}@{args.crop}", flush=True)
    log_path = os.path.join(run_dir, "log.jsonl")
    t0 = time.time()
    guard = NaNGuard()
    model.train()
    for step in range(start, args.iters):
        try:
            inp, tgt, cond, kernel, reblur = next(it)
        except StopIteration:
            it = iter(dl)
            inp, tgt, cond, kernel, reblur = next(it)
        inp, tgt, cond, kernel, reblur = (
            t.to(device, non_blocking=True)
            for t in (inp, tgt, cond, kernel, reblur))
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            out = model(inp, cond)
            loss, parts = loss_fn(out, tgt, kernel, reblur)
        if not guard.check(loss):
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (step + 1) % 200 == 0:
            speed = 200 * args.batch / (time.time() - t0)
            t0 = time.time()
            rec = dict(step=step + 1, loss=round(loss.item(), 5),
                       **{k: round(v, 5) for k, v in parts.items()},
                       img_s=round(speed, 1), lr=sched.get_last_lr()[0])
            print(json.dumps(rec), flush=True)
            open(log_path, "a").write(json.dumps(rec) + "\n")
        if (step + 1) % 5000 == 0 or step + 1 == args.iters:
            psnr, flux = quick_eval(model, device, crop=args.crop)
            print(f"eval step {step+1}: PSNR {psnr:.2f} dB / flux err "
                  f"{flux:.2f}%", flush=True)
            if os.path.isfile(ckpt):
                try:
                    os.replace(ckpt, ckpt + ".prev")
                except OSError:
                    pass
            atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(),
                             sched=sched.state_dict(), step=step + 1,
                             best=best, config=vars(args)), ckpt)
            if math.isfinite(psnr) and psnr > best:
                best = psnr
                atomic_save(dict(model=model.state_dict(), step=step + 1,
                                 config=vars(args)),
                            os.path.join(run_dir, "best.pt"))
            save_samples(model, device, os.path.join(run_dir, "samples"),
                         step + 1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
