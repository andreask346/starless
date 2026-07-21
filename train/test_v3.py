# Unit tests for the v3 REAL-DATA recipe (datagen/v3 assets) + regression
# tests that v1/v2 are still BIT-EXACT.
#   .venv\Scripts\python.exe train\test_v3.py
#
# Covers everything test_v2.py covers for the v3 config (exact recomposition,
# identical noise in input/target, identical presentation, silhouette
# occlusion), plus the v3-specific invariants:
#   * empirical stamp path: finite output, unit-flux kernels, star flux
#     conserved through the empirical splat;
#   * hybrid bright path: measured core + analytic wings, monsters still
#     analytic and still engaging;
#   * measured noise: per-channel sigma map, ACF grain, noise identical in
#     input and target;
#   * real silhouette masks used, and stars fully occluded under them;
#   * graceful degradation to v2 behaviour when the assets are missing;
#   * v1 AND v2 outputs hash-identical to the pre-v3 HEAD behaviour.

import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from datagen import v3assets                                     # noqa: E402
from datagen.synth import (make_pair, default_config, PairDataset,  # noqa: E402
                           sample_psf_family, render_stars, _emp_kernel,
                           _hybrid_kernel, make_psf)
from models.nafnet_star import build                             # noqa: E402
from train import loss_fn_v2, default_loss_cfg                   # noqa: E402

PASS, FAIL = [], []

# Frozen fingerprints of v1/v2 make_pair output, captured from git HEAD
# BEFORE the v3 work started (scratchpad/v3build/ref_before.json).
REF = {
 "v1_s42000000": ["45a657f535d66068", "e6640bfe3ec64f0c", "8e17c029141854ef"],
 "v1_s42000101": ["0b179da17a92fa8e", "4662637d8be30ecb", "c48ccd5bdb1c14c5"],
 "v1_s42000202": ["fce00b6263991ec5", "5d690cc77885c09c", "19ee8e7cb194bb86"],
 "v1_s42000303": ["36cdaf6352e16d3d", "700af2b5c0a26da7", "d3e0e37e0ad384a0"],
 "v1_s42000404": ["04d26587751c306d", "793530fb446e3592", "d9689386ae8d9d2e"],
 "v1_s42000505": ["96f5fc954f6fa2de", "110e1e60a32602fd", "c80300f34002f1e1"],
 "v1_s42000606": ["938b83e51be21c4b", "4bf4ba4e338a1fd4", "584b4462b4f801f4"],
 "v1_s42000707": ["1a206daf0bf34868", "7e5a58d962c722e9", "4b6ab9a8a444adf7"],
 "v1_ds0": ["e7d93b0bbda0a39b", "7d89ab96f07c516d", "421ea33353955eee",
            "814ff431e4ec6bfb"],
 "v1_ds3": ["82e1da64f8399737", "73ae75df46b2a001", "8415196036bb5703",
            "277c1fd43c5ed6f9"],
 "v2_s42000000": ["233b212a731ef357", "2990aad8e84595b3", "602767f9a35173c5"],
 "v2_s42000101": ["f62c688d36e83988", "0ee4cc644f011cfa", "30036ecc967468cb"],
 "v2_s42000202": ["c984241c4bd23fb5", "a4d3388c374ca147", "2133bf2f26c59464"],
 "v2_s42000303": ["9c932818fd3962e0", "61821ea7db9de134", "c3b67aa806aa6dd5"],
 "v2_s42000404": ["f52678a0f4aa3a57", "5bb11a41c4c36c62", "99474be651242f2b"],
 "v2_s42000505": ["b60be6ef91daf5f9", "cb62df508dfe025f", "7ae164d25ca6f276"],
 "v2_s42000606": ["e3e79eb7dcff1444", "f39e8cdd66d23b0d", "ad56094ab551c9fa"],
 "v2_s42000707": ["2d94c13b44c9f9ca", "0a281b161b55bd72", "e6bdc59d4824d080"],
 "v2_ds0": ["f32eecaf5d45ed07", "75cf07bf2c57738a", "1925280b4c236dd4",
            "0a0ee92f49bce83a"],
 "v2_ds3": ["ac28a2b5ce3859eb", "8847d8a28073c7c8", "4e69446094d0259f",
            "4358785c3765bd0b"],
}


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""), flush=True)


def _h(t):
    return hashlib.sha256(
        t.detach().cpu().contiguous().float().numpy().tobytes()).hexdigest()[:16]


def _acf_len(x):
    """1/e correlation length (px) of a 2-D field, along the x axis."""
    y = (x - x.mean()) / x.std().clamp_min(1e-9)
    f = torch.fft.rfft2(y)
    ac = torch.fft.irfft2(f * f.conj(), s=y.shape).real
    ac = ac / ac[0, 0]
    row = ac[0, :x.shape[-1] // 2]
    below = (row < 1 / torch.e).nonzero()
    return float(below[0]) if len(below) else float(len(row))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg3 = default_config(3)
    cfg2 = default_config(2)
    cfg1 = default_config(1)
    print(f"device={device}")
    print("assets:", v3assets.summary())

    # ---------------- [1] v3 sample invariants
    print("\n[1] v3 recipe invariants")
    sil_seen = mon_seen = real_psf = real_sil = acf_seen = 0
    n = 32
    for i in range(n):
        seed = 42_000_000 + i * 101
        inp, sl, st, ex = make_pair(192, 192, None, seed=seed, device=device,
                                    cfg=cfg3, return_extras=True)
        assert torch.isfinite(inp).all() and torch.isfinite(sl).all(), seed
        assert torch.allclose(inp, sl + st, atol=1e-6), f"recompose {seed}"
        assert st.min() >= 0, f"negative star layer {seed}"
        d = ex["sky"] - ex["starless_lin"]
        s = ex["stars_lin"]
        assert (d - s.clamp_min(0)).max() <= 1e-4, f"clip grew flux {seed}"
        assert (d - s.clamp_max(0)).min() >= -1e-4, f"clip deepened dip {seed}"
        n_in = ex["noisy"] - ex["sky"]
        n_tg = ex["starless_noisy"] - ex["starless_lin"]
        assert torch.allclose(n_in, n_tg, atol=1e-7), f"noise differs {seed}"
        T = ex["T"]
        assert torch.equal(T(ex["noisy"]).clamp(0, 1).float(), inp), seed
        tsl = torch.minimum(T(ex["starless_noisy"]).clamp(0, 1),
                            T(ex["noisy"]).clamp(0, 1)).float()
        assert torch.equal(tsl, sl), f"T(target) mismatch {seed}"
        assert ex["sigma"].min() > 0 and torch.isfinite(ex["sigma"]).all()
        f = ex["flags"]
        real_psf += bool(f["real_psf"])
        real_sil += bool(f["real_sil"])
        acf_seen += bool(f["noise_acf"])
        mon_seen += bool(f["monster"])
        if f["silhouette"]:
            hard = ex["mask"][0] >= 1.0
            if hard.any():
                sil_seen += 1
                assert ex["stars_lin"][:, hard].abs().max() == 0, \
                    f"linear stars under silhouette {seed}"
                assert st[:, hard].abs().max() <= 1e-6, \
                    f"target stars under silhouette {seed}"
    check(f"v3 finite + exact recomposition inp == sl + st ({n} seeds)", True)
    check("v3 star layer >= 0", True)
    check("v3 pre-presentation decomposition (clip only shrinks star flux)",
          True)
    check("v3 noise identical in input and target", True)
    check("v3 presentation applied identically to input and targets", True)
    check("v3 silhouette occludes stars (star energy under mask == 0)",
          sil_seen > 0, f"{sil_seen} silhouette samples verified")
    check("v3 empirical-PSF path engages", real_psf > 0,
          f"{real_psf}/{n} images (real_psf_p={cfg3['real_psf_p']})")
    check("v3 real silhouette masks used", real_sil > 0, f"{real_sil}/{n}")
    check("v3 measured-ACF noise on every image", acf_seen == n,
          f"{acf_seen}/{n}")
    check("v3 saturated monster samples still occur", mon_seen > 0,
          f"{mon_seen}/{n}")

    # ---------------- [2] empirical kernels: flux normalisation
    print("\n[2] empirical kernel flux normalisation")
    A = v3assets.psf_assets()
    if A is None:
        check("psf assets present", False, "datagen/v3/psf_stamps.npz missing")
    else:
        sums, finite = [], True
        for gid in (0, A["n_groups"] // 2, A["n_groups"] - 1):
            for size in (31, 63, 127):
                for sc, el, th in ((1.0, 0.0, 0.0), (0.75, 0.0, 1.1),
                                   (1.35, 0.3, -0.6)):
                    k = _emp_kernel(A, gid, 1, size, sc, el, th, device)
                    if k is None:
                        finite = False
                        continue
                    finite &= bool(torch.isfinite(k).all())
                    sums.append(float(k.sum()))
        err = max(abs(s - 1.0) for s in sums)
        check("empirical kernels finite", finite)
        check("empirical kernels are unit flux", err < 1e-5,
              f"max |sum-1| = {err:.2e} over {len(sums)} kernels")
        # hybrid: measured core + analytic wings, still unit flux, and the
        # wings really come from the analytic PSF (more energy outside r=16)
        p = dict(fwhm_x=3.0, fwhm_y=3.0, theta=0.0, beta=2.5,
                 spike_frac=0.25, n_vanes=4, vane_width=0.01, rotation=0.3,
                 obstruction=0.2, halo_frac=0.10, halo_fwhm=18.0,
                 defocus_r=0.0, dark_halo_k=0.0, dark_halo_s=0.0)
        ana = make_psf(127, p, device)
        emp = _emp_kernel(A, 0, 1, 127, 1.0, 0.0, 0.0, device)
        hyb = _hybrid_kernel(emp, ana, 1.0, 0.0, cfg3["emp_core_frac"])
        ax = torch.arange(127, device=device, dtype=torch.float32) - 63.0
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        # beyond the 33x33 stamp's diagonal reach the empirical kernel is
        # EXACTLY zero (no halo/spike/far-wing information was harvested);
        # the hybrid must carry the analytic wings there instead.
        out = torch.hypot(xx, yy) > 24.0
        check("hybrid kernel unit flux", abs(float(hyb.sum()) - 1) < 1e-5,
              f"sum={float(hyb.sum()):.6f}")
        check("hybrid kernel takes wings from the ANALYTIC PSF",
              float(emp[out].abs().sum()) == 0.0
              and float(hyb[out].sum()) > 0.05 * float(ana[out].sum()),
              f"wing energy emp={float(emp[out].sum()):.2e} "
              f"hybrid={float(hyb[out].sum()):.2e} "
              f"analytic={float(ana[out].sum()):.2e}")
        check("hybrid core stays measured (differs from pure analytic)",
              float((hyb - ana).abs().sum()) > 1e-3)

        # star flux conservation through the empirical splat
        gen = torch.Generator().manual_seed(4242)
        fam = None
        for _ in range(200):
            f2 = sample_psf_family(gen, device, cfg3)
            if f2.get("real") is not None:
                fam = f2
                break
        if fam is None:
            check("empirical family sampled", False)
        else:
            fam["chan_shift"] = None
            pos = torch.tensor([[128.0, 128.0]], device=device)
            flux = torch.tensor([100.0], device=device)
            col = torch.ones(1, 3, device=device)
            img = render_stars(256, 256, fam, pos, flux, col, gen, device,
                               flux_scale=1e-4, cfg=cfg3)
            got = float(img[1].sum())
            check("empirical splat conserves star flux",
                  abs(got - 100.0 * 1e-4) / 1e-2 < 0.02,
                  f"expected {1e-2:.4f}, got {got:.4f}")
            check("empirical splat output finite",
                  bool(torch.isfinite(img).all()))

    # ---------------- [3] measured noise texture
    print("\n[3] measured noise")
    lens, chans = [], 0
    for i in range(8):
        _, _, _, ex = make_pair(256, 256, None, seed=61_000_000 + i * 7,
                                device=device, cfg=cfg3, return_extras=True)
        nz = ex["noise"]
        lens.append(_acf_len(nz[1].float()))
        sd = nz.flatten(1).std(dim=1)
        chans += int(sd.min() > 0)
    med_len = sorted(lens)[len(lens) // 2]
    check("generated noise is spatially correlated (~2 px)",
          1.2 <= med_len <= 3.5, f"1/e length median {med_len:.2f} px")
    check("all channels carry noise", chans == 8)

    # ---------------- [4] graceful fallback when assets are missing
    print("\n[4] graceful degradation")
    real_dir = v3assets.V3DIR
    try:
        v3assets.V3DIR = os.path.join(real_dir, "__does_not_exist__")
        v3assets._CACHE.clear()
        ok = True
        for i in range(4):
            inp, sl, st = make_pair(128, 128, None, seed=71_000_000 + i,
                                    device=device, cfg=cfg3)
            ok &= bool(torch.isfinite(inp).all())
            ok &= bool(torch.allclose(inp, sl + st, atol=1e-6))
        check("v3 falls back to v2 behaviour when assets are missing", ok)
    finally:
        v3assets.V3DIR = real_dir
        v3assets._CACHE.clear()

    # ---------------- [5] v1 / v2 bit-exact regression
    print("\n[5] v1 + v2 bit-exact vs pre-v3 HEAD")
    bad = []
    for ver, cfg in ((1, cfg1), (2, cfg2)):
        for i in range(8):
            seed = 42_000_000 + i * 101
            a, b, c = make_pair(160, 160, None, seed=seed, device="cpu",
                                cfg=cfg)
            k = f"v{ver}_s{seed}"
            if [_h(a), _h(b), _h(c)] != REF[k]:
                bad.append(k)
        ds = PairDataset(crop=96, length=8, cfg=cfg)
        for i in (0, 3):
            t = ds[i]
            k = f"v{ver}_ds{i}"
            if [_h(x) for x in t] != REF[k]:
                bad.append(k)
    check("v1 + v2 make_pair/PairDataset outputs unchanged", not bad,
          f"{len(REF)} fingerprints" if not bad else f"MISMATCH {bad}")

    # ---------------- [6] loss v2 on v3 data
    print("\n[6] loss v2 on v3 data")
    torch.manual_seed(0)
    model = build(width=16).to(device)
    ds3 = PairDataset(crop=128, length=4, cfg=cfg3)
    batch = [torch.stack(ts) for ts in zip(*(ds3[i] for i in range(2)))]
    binp, bsl, bst, bsig = (t.to(device) for t in batch)
    pred = model(binp)
    loss, parts = loss_fn_v2(pred, binp, bsl, bst, bsig, default_loss_cfg())
    check("loss v2 finite on v3 data", torch.isfinite(loss).item(),
          f"total={loss.item():.4f}")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check("gradients finite and non-zero",
          bool(grads) and all(torch.isfinite(g).all().item() for g in grads)
          and sum(float(g.abs().sum()) for g in grads) > 0)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
