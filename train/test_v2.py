# Unit tests for the v2 data recipe + v2 loss.
#   .venv\Scripts\python.exe train\test_v2.py
# Covers: exact GT decomposition pre-presentation; presentation applied
# identically to input/targets; identical noise in input/target; silhouette
# occlusion (zero star energy under the mask); v2 loss finite with flowing
# gradients; the old v1 config path still runs.

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from datagen.synth import make_pair, default_config, PairDataset  # noqa: E402
from models.nafnet_star import build                              # noqa: E402
from train import loss_fn, loss_fn_v2, default_loss_cfg           # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""), flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg2 = default_config(2)
    cfg1 = default_config(1)
    print(f"device={device}")

    # ---------------- v2 sample invariants over a batch of seeds
    print("\n[1] v2 recipe invariants")
    sil_seen = monster_seen = False
    for i in range(24):
        seed = 42_000_000 + i * 101
        inp, sl, st, ex = make_pair(192, 192, None, seed=seed, device=device,
                                    cfg=cfg2, return_extras=True)
        # exact recomposition + non-negative star layer
        assert torch.allclose(inp, sl + st, atol=1e-6), f"recompose seed {seed}"
        assert st.min() >= 0, f"negative star layer seed {seed}"
        # exact GT decomposition PRE-presentation: the clip can only shrink
        # star flux toward zero, so pixelwise
        #   min(stars_lin, 0) <= sky - starless_lin <= max(stars_lin, 0)
        # (stars_lin dips below 0 by design for dark-halo/oversharpened PSFs)
        d = ex["sky"] - ex["starless_lin"]
        s = ex["stars_lin"]
        assert (d - s.clamp_min(0)).max() <= 1e-4, \
            f"clip grew star flux seed {seed}"
        assert (d - s.clamp_max(0)).min() >= -1e-4, \
            f"clip deepened star dip seed {seed}"
        # identical noise in input and target
        n_in = ex["noisy"] - ex["sky"]
        n_tg = ex["starless_noisy"] - ex["starless_lin"]
        assert torch.allclose(n_in, n_tg, atol=1e-7), f"noise differs {seed}"
        # presentation applied identically: re-apply T to both sides
        T = ex["T"]
        assert torch.equal(T(ex["noisy"]).clamp(0, 1).float(), inp), \
            f"T(input) mismatch seed {seed}"
        tsl = torch.minimum(T(ex["starless_noisy"]).clamp(0, 1),
                            T(ex["noisy"]).clamp(0, 1)).float()
        assert torch.equal(tsl, sl), f"T(target) mismatch seed {seed}"
        # sigma sane
        assert ex["sigma"].min() > 0 and torch.isfinite(ex["sigma"]).all()
        # silhouette occlusion: ZERO star energy under the hard mask, in
        # both the linear star layer and the presented star target
        if ex["flags"]["silhouette"]:
            hard = ex["mask"][0] >= 1.0     # fully-occluded (sub-feather) px
            if hard.any():
                sil_seen = True
                assert ex["stars_lin"][:, hard].abs().max() == 0, \
                    f"linear stars under silhouette seed {seed}"
                assert st[:, hard].abs().max() <= 1e-6, \
                    f"target star layer under silhouette seed {seed}"
        if ex["flags"]["monster"]:
            monster_seen = True
    check("exact recomposition inp == sl + st (24 seeds)", True)
    check("star layer >= 0 (24 seeds)", True)
    check("pre-presentation decomposition (clip only shrinks star flux)",
          True)
    check("noise identical in input and target", True)
    check("presentation applied identically to input and targets", True)
    check("silhouette occludes stars (star energy under mask == 0)", sil_seen,
          "at least one silhouette sample verified")
    check("saturated monster samples occur", monster_seen)

    # ---------------- pure-linear samples exist (presentation ~27% linear)
    print("\n[2] presentation mix")
    lin = 0
    for i in range(24):
        _, _, _, ex = make_pair(96, 96, None, seed=77_000_000 + i * 13,
                                device=device, cfg=cfg2, return_extras=True)
        x = torch.rand(3, 4, 4, device=device)
        if torch.allclose(ex["T"](x), x):
            lin += 1
    check("pure-linear fraction present", 0 < lin < 24, f"{lin}/24 linear")

    # ---------------- v1 path still runs and keeps its invariants
    print("\n[3] v1 config path")
    inp, sl, st = make_pair(192, 192, None, seed=123, device=device, cfg=cfg1)
    check("v1 make_pair runs", True)
    check("v1 recomposition exact", torch.allclose(inp, sl + st, atol=1e-6))
    check("v1 star layer >= 0", st.min() >= 0)
    ds1 = PairDataset(crop=96, length=4, cfg=cfg1)
    i1, s1, t1, g1 = ds1[0]
    check("v1 PairDataset returns 4-tuple incl. sigma",
          g1.shape == i1.shape and torch.isfinite(g1).all())

    # ---------------- loss v2: finite, gradients flow; v1 loss still runs
    print("\n[4] losses")
    torch.manual_seed(0)
    model = build(width=16).to(device)
    ds2 = PairDataset(crop=128, length=4, cfg=cfg2)
    batch = [torch.stack(ts) for ts in zip(*(ds2[i] for i in range(2)))]
    binp, bsl, bst, bsig = (t.to(device) for t in batch)
    pred = model(binp)
    loss2, parts2 = loss_fn_v2(pred, binp, bsl, bst, bsig, default_loss_cfg())
    check("loss v2 finite", torch.isfinite(loss2).item(),
          f"total={loss2.item():.4f} parts={ {k: round(v, 4) for k, v in parts2.items()} }")
    loss2.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check("loss v2 gradients flow", len(grads) > 0)
    check("loss v2 gradients finite",
          all(torch.isfinite(g).all().item() for g in grads))
    gn = sum(float(g.abs().sum()) for g in grads)
    check("loss v2 gradients non-zero", gn > 0, f"sum|grad|={gn:.3e}")
    model.zero_grad(set_to_none=True)
    pred = model(binp)
    loss1, _ = loss_fn(pred, binp, bsl, bst)
    check("loss v1 still runs and is finite", torch.isfinite(loss1).item(),
          f"total={loss1.item():.4f}")
    loss1.backward()
    check("loss v1 gradients finite",
          all(torch.isfinite(p.grad).all().item()
              for p in model.parameters() if p.grad is not None))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
