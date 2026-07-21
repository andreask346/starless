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
from datagen.synth import PairDataset, default_config     # noqa: E402
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
    # Stars are only ~2-5% of pixels. An unweighted L1 lets the model win by
    # predicting ZERO stars (background dominates the mean) — and a purity
    # penalty on the 95% background makes that collapse even more attractive.
    # Fix: weight the star footprint heavily so actually removing stars is
    # what minimizes the loss. l_sl already keeps the background clean (target
    # stars are 0 there), so no separate purity term is needed.
    # Weight by star BRIGHTNESS (not a noise-level threshold): bright, visible
    # stars dominate, so removing them is what minimizes the loss.
    w = 1.0 + 300.0 * tgt_stars.clamp(0, 0.3)

    def wl1(a, b):
        # tiny eps: star pixel values are ~1e-3..0.2, so a 1e-6 floor under the
        # sqrt would swamp faint-star gradients and collapse the model to zero.
        return (torch.sqrt((a - b) ** 2 + 1e-10) * w).sum() / w.sum()

    l_sl = wl1(starless_pred, tgt_starless)      # clean background under stars
    l_st = wl1(stars_pred, tgt_stars)            # correct star flux
    l_f = fft_loss(starless_pred, tgt_starless)
    total = l_sl + l_st + 0.05 * l_f
    return total, dict(starless=l_sl.item(), stars=l_st.item(),
                       fft=l_f.item())


# ------------------------------------------------------------------ loss v2
def _box_blur(x, k):
    pad = k // 2
    return F.avg_pool2d(F.pad(x, (pad,) * 4, mode="reflect"), k, stride=1)


def _lowpass(x, f=16):
    h, w = x.shape[-2:]
    y = F.avg_pool2d(x, f, stride=f, ceil_mode=True)
    return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)


def default_loss_cfg():
    """Knobs for loss v2 (defaults tuned against the real-frame census)."""
    return dict(
        w_k=40.0,          # contrast-weight gain
        w_c1=0.5,          # local-background coefficient in the denominator
        w_c2=2.0,          # local-noise-sigma coefficient in the denominator
        w_cap=8.0,         # contrast ratio clamp
        lam_wing=1.0,      # dilated-wing term (under-predicted halos/wings)
        lam_site=0.02,     # star-site starless quality, in noise-sigma units
        lam_chroma=0.05,   # chroma-consistency at star sites
        lam_leak=2.0,      # smooth-leak (low-pass star energy off-footprint)
        foot_thresh=2e-3,  # star-footprint threshold on tgt_stars
    )


def loss_fn_v2(stars_pred, inp, tgt_starless, tgt_stars, sigma, lc):
    """v2 loss (see PLAN): the census showed the v1 brightness weight
    w = 1 + 300*st.clamp(0, .3) SATURATES at bright stars (97-100% of stars
    above FWHM 6px missed) and is blind to local contrast, so faint stars on
    bright Milky Way and monsters on dark sky got the same push.
      A. contrast-relative weight  w = 1 + k*(st / (c1*bg_loc + c2*sig)).clamp
      B. dilated-wing term: blurred/dilated tgt_stars weights halo/wing error
      C. star-site starless quality in local-noise units + chroma penalty
         (57% dark holes, 53% chroma shifts in the census)
      D. smooth-leak: low-pass energy of stars_pred outside the footprint
         (28.7% of star-mask energy was Milky Way glow leakage)."""
    starless_pred = inp - stars_pred
    sig = sigma.clamp_min(1e-4)
    bg_local = _box_blur(tgt_starless, 17).clamp_min(0)
    contrast = tgt_stars / (lc["w_c1"] * bg_local + lc["w_c2"] * sig + 1e-4)
    w = 1.0 + lc["w_k"] * contrast.clamp(0, lc["w_cap"])

    def wl1(a, b, wt):
        return (torch.sqrt((a - b) ** 2 + 1e-10) * wt).sum() / wt.sum()

    l_sl = wl1(starless_pred, tgt_starless, w)   # clean background under stars
    l_st = wl1(stars_pred, tgt_stars, w)         # correct star flux
    l_f = fft_loss(starless_pred, tgt_starless)

    # B: halo/wing coverage — weight |err| by blurred+dilated tgt_stars so
    # under-predicted wings around bright cores are penalized
    st_m = tgt_stars.amax(1, keepdim=True)
    wing_w = _box_blur(F.max_pool2d(st_m, 5, 1, 2), 9)
    l_wing = (wing_w * (stars_pred - tgt_stars).abs()).sum() \
        / (3.0 * wing_w.sum() + 1e-6)

    # star footprint (dilated)
    foot = (F.max_pool2d(st_m, 9, 1, 4) > lc["foot_thresh"]).float()
    n_foot = foot.sum() * 3.0

    # C: starless quality at star sites, scaled in local-noise-sigma units
    resid = starless_pred - tgt_starless
    l_site = ((resid.abs() / sig).clamp(max=30.0) * foot).sum() \
        / (n_foot + 1e-6)
    chroma = resid - resid.mean(1, keepdim=True)     # channel-mean removed
    l_chroma = ((chroma.abs() / sig).clamp(max=30.0) * foot).sum() \
        / (n_foot + 1e-6)

    # D: smooth-leak — low-pass star-layer energy outside the true footprint
    foot_dil = F.max_pool2d(foot, 11, 1, 5)
    outside = 1.0 - foot_dil
    l_leak = (_lowpass(stars_pred).abs() * outside).sum() \
        / (3.0 * outside.sum() + 1e-6)

    total = l_sl + l_st + 0.05 * l_f \
        + lc["lam_wing"] * l_wing + lc["lam_site"] * l_site \
        + lc["lam_chroma"] * l_chroma + lc["lam_leak"] * l_leak
    return total, dict(starless=l_sl.item(), stars=l_st.item(),
                       fft=l_f.item(), wing=l_wing.item(),
                       site=l_site.item(), chroma=l_chroma.item(),
                       leak=l_leak.item())


@torch.no_grad()
def quick_eval(model, device, crop=256, n=16, cfg=None):
    from datagen.synth import make_pair
    model.eval()
    tot_in, tot_out = 0.0, 0.0
    n_in, n_out = 0, 0
    for i in range(n):
        inp, sl, st = make_pair(crop, crop, None, seed=10_000_000 + i,
                                device=device, cfg=cfg)
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


def save_samples(model, device, out_dir, step, crop=512, cfg=None):
    import numpy as np
    import tifffile
    from datagen.synth import make_pair
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(3):
            inp, sl, st = make_pair(crop, crop, None, seed=20_000_000 + i,
                                    device=device, cfg=cfg)
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
    ap.add_argument("--init-from", default="",
                    help="warm-start from another run's checkpoint (model "
                         "weights only; fresh optimizer/schedule, step 0). "
                         "Ignored when --resume finds this run's own "
                         "checkpoint, so queue retries keep working.")
    ap.add_argument("--out", default="runs")
    # v2 data recipe + loss (defaults). v1 = the shipped recipe/loss.
    ap.add_argument("--data-recipe", choices=["v1", "v2", "v3"], default="v2")
    ap.add_argument("--loss-version", choices=["v1", "v2"], default="v2")
    lc0 = default_loss_cfg()
    for k, v in lc0.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(os.path.dirname(__file__), "..", args.out, args.name)
    os.makedirs(run_dir, exist_ok=True)
    data_cfg = default_config({"v1": 1, "v2": 2, "v3": 3}[args.data_recipe])
    loss_cfg = {k: getattr(args, k) for k in default_loss_cfg()}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dict(vars(args), data_cfg=data_cfg), f, indent=1)

    model = build(width=args.width, use_ffc=not args.no_ffc).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4, betas=(0.9, 0.9))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters, eta_min=1e-6)

    start = 0
    best_in = -1
    ckpt_path = os.path.join(run_dir, "last.pt")
    resumed = False
    if args.resume:
        ck = safe_load(ckpt_path, device)
        if ck is not None:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start = ck["step"]
            best_in = ck.get("best_in", -1)     # don't clobber best.pt on retry
            resumed = True
            print(f"resumed {args.name} at step {start} "
                  f"(best {best_in:.2f})", flush=True)
    if not resumed and args.init_from:
        # fine-tune warm start: weights only. A plain --resume of a DONE run
        # would restore a cosine schedule past T_max and the LR would CLIMB.
        ck = safe_load(args.init_from, device)
        if ck is None:
            raise SystemExit(f"--init-from unreadable: {args.init_from}")
        model.load_state_dict(ck["model"])
        print(f"warm-started {args.name} from {args.init_from} "
              f"(weights only, fresh opt/sched)", flush=True)

    # seed the on-the-fly data stream from the ABSOLUTE step so a resume
    # continues the stream instead of replaying the first `start` steps
    ds = PairDataset(bg_dir=args.bg_dir or None, crop=args.crop,
                     length=10_000_000, base_index=start * args.batch,
                     cfg=data_cfg)
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
            inp, sl, st, sig = next(it)
        except StopIteration:
            it = iter(dl)
            inp, sl, st, sig = next(it)
        inp, sl, st, sig = (t.to(device, non_blocking=True)
                            for t in (inp, sl, st, sig))
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            pred = model(inp)
            if args.loss_version == "v1":
                loss, parts = loss_fn(pred, inp, sl, st)
            else:
                loss, parts = loss_fn_v2(pred, inp, sl, st, sig, loss_cfg)
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
            pin, pout = quick_eval(model, device, crop=args.crop, cfg=data_cfg)
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
                         step + 1, cfg=data_cfg)
    print("done", flush=True)


if __name__ == "__main__":
    main()
