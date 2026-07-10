# Starless training harness.
#   .venv\Scripts\python.exe train\train.py --name w32_v0 --width 32 --iters 200000
# Resumable (--resume auto-finds last checkpoint), bf16 AMP, on-the-fly data,
# periodic synthetic-eval + PNG sample dumps. Designed to run unattended in a
# queue (see train/queue.py).

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
sys.path.insert(0, os.path.dirname(__file__))
from datagen.synth import PairDataset          # noqa: E402
from models.nafnet_star import build           # noqa: E402
from trainutils import atomic_save, safe_load, NaNGuard   # noqa: E402


def charbonnier(a, b, eps=1e-3):
    return torch.sqrt((a - b) ** 2 + eps * eps).mean()


def fft_loss(a, b):
    fa = torch.fft.rfft2(a, norm="ortho")
    fb = torch.fft.rfft2(b, norm="ortho")
    return (fa - fb).abs().mean()


def loss_fn(stars_pred, inp, tgt_starless, tgt_stars):
    starless_pred = inp - stars_pred
    l_sl = charbonnier(starless_pred, tgt_starless)
    l_st = charbonnier(stars_pred, tgt_stars)
    l_f = fft_loss(starless_pred, tgt_starless)
    # purity: star layer should be ~zero away from true stars
    bgmask = (tgt_stars < 1e-4).float()
    l_pure = (stars_pred * bgmask).abs().mean()
    total = l_sl + 0.5 * l_st + 0.1 * l_f + 0.5 * l_pure
    return total, dict(starless=l_sl.item(), stars=l_st.item(),
                       fft=l_f.item(), purity=l_pure.item())


@torch.no_grad()
def quick_eval(model, device, crop=256, n=16):
    from datagen.synth import make_pair
    model.eval()
    tot_in, tot_out = 0.0, 0.0
    n_in, n_out = 0, 0
    for i in range(n):
        inp, sl, st = make_pair(crop, crop, None, seed=10_000_000 + i,
                                device=device)
        pred = model(inp[None])[0]
        starless = inp - pred
        foot = st > 1e-4
        if foot.any():
            tot_in += F.mse_loss(starless[foot], sl[foot]).item()
            n_in += 1
        if (~foot).any():
            tot_out += F.mse_loss(starless[~foot], sl[~foot]).item()
            n_out += 1
    model.train()
    psnr_in = -10 * math.log10(max(tot_in / max(n_in, 1), 1e-12))
    psnr_out = -10 * math.log10(max(tot_out / max(n_out, 1), 1e-12))
    return psnr_in, psnr_out


def save_samples(model, device, out_dir, step, crop=512):
    import numpy as np
    import tifffile
    from datagen.synth import make_pair
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(3):
            inp, sl, st = make_pair(crop, crop, None, seed=20_000_000 + i,
                                    device=device)
            pred = model(inp[None])[0]
            starless = (inp - pred).clamp(0, 1)
            panel = torch.cat([inp, starless, sl, pred.clamp(0, 1)], dim=2)
            arr = (panel.permute(1, 2, 0).cpu().numpy() * 65535).astype(
                np.uint16)
            tifffile.imwrite(os.path.join(
                out_dir, f"step{step:07d}_s{i}.tif"), arr)
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=200000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bg-dir", default="")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--no-ffc", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(os.path.dirname(__file__), "..", args.out, args.name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    model = build(width=args.width, use_ffc=not args.no_ffc).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4, betas=(0.9, 0.9))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters, eta_min=1e-6)

    start = 0
    best_in = -1
    ckpt_path = os.path.join(run_dir, "last.pt")
    if args.resume:
        ck = safe_load(ckpt_path, device)
        if ck is not None:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start = ck["step"]
            best_in = ck.get("best_in", -1)     # don't clobber best.pt on retry
            print(f"resumed {args.name} at step {start} "
                  f"(best {best_in:.2f})", flush=True)

    # seed the on-the-fly data stream from the ABSOLUTE step so a resume
    # continues the stream instead of replaying the first `start` steps
    ds = PairDataset(bg_dir=args.bg_dir or None, crop=args.crop,
                     length=10_000_000, base_index=start * args.batch)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                    pin_memory=True, persistent_workers=args.workers > 0)
    it = iter(dl)

    print(f"train {args.name}: {n_params/1e6:.1f}M params, device={device}, "
          f"bf16, batch={args.batch}@{args.crop}", flush=True)
    log_path = os.path.join(run_dir, "log.jsonl")
    t0 = time.time()
    guard = NaNGuard()
    model.train()
    for step in range(start, args.iters):
        try:
            inp, sl, st = next(it)
        except StopIteration:
            it = iter(dl)
            inp, sl, st = next(it)
        inp, sl, st = (t.to(device, non_blocking=True) for t in (inp, sl, st))
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            pred = model(inp)
            loss, parts = loss_fn(pred, inp, sl, st)
        if not guard.check(loss):          # non-finite: skip, don't poison
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
                       img_s=round(speed, 1),
                       lr=sched.get_last_lr()[0])
            print(json.dumps(rec), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if (step + 1) % 5000 == 0 or step + 1 == args.iters:
            pin, pout = quick_eval(model, device, crop=args.crop)
            print(f"eval step {step+1}: PSNR in-footprint {pin:.2f} dB / "
                  f"background {pout:.2f} dB", flush=True)
            # also write eval to log.jsonl so it surfaces in STATUS.md / phone
            with open(log_path, "a") as f:
                f.write(json.dumps(dict(eval_step=step + 1,
                                        psnr_stars=round(pin, 2),
                                        psnr_bg=round(pout, 2))) + "\n")
            if os.path.isfile(ckpt_path):       # rotate for safe_load fallback
                try:
                    os.replace(ckpt_path, ckpt_path + ".prev")
                except OSError:
                    pass
            atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(),
                             sched=sched.state_dict(), step=step + 1,
                             best_in=best_in, config=vars(args)), ckpt_path)
            if math.isfinite(pin) and pin > best_in:
                best_in = pin
                atomic_save(dict(model=model.state_dict(), step=step + 1,
                                 config=vars(args)),
                            os.path.join(run_dir, "best.pt"))
            save_samples(model, device, os.path.join(run_dir, "samples"),
                         step + 1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
