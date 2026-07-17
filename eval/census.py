#!/usr/bin/env python
"""
census.py -- permanent star-removal regression harness.

Ported from the forensic scripts f1_load.py / f2_detect.py / f3_census.py /
f4_refine.py / f5_leak.py (2026-07 forensic run on Neutralised.tif, 172,542
classified stars).

Two subcommands so one detection pass serves many scored variants:

  detect  Detect stars in the INPUT image once (DoG / local-maxima), measure
          peak-above-background, flux, FWHM per star, and save every site to
          an .npz cache.

  score   Classify every cached site against a STARLESS result:
          clean / artifact (dark_hole, bright_residual, ring, +color) /
          missed.  Reports rates broken down by FWHM bins and peak bins,
          dark-hole median depth in local-noise-sigma units, residual star
          count, and (with --starmask) the smooth-leak fraction of the star
          mask.  Emits JSON (machine-comparable, includes a single scalar
          `score`) and optionally Markdown.

Usage:
  python census.py detect --input Neutralised.tif --cache sites.npz
  python census.py score  --cache sites.npz --input Neutralised.tif \
         --starless starless.fit --starmask starmask.fit \
         --label baseline --json out.json --md out.md

FITS row-order: FITS files are frequently vertically flipped relative to the
TIFF the tool consumed.  When --starmask is given, orientation is picked by
minimising |starless + starmask - input| (reconstruction check, from
f1_load.py).  Without a starmask, orientation is picked by correlating the
downsampled starless luma against the input luma (direct vs flipped).

SCORE SCALAR (documented, keep stable across variants):
  For each populated peak bin b with importance weight w_b
      w = {<0.03:0.25, 0.03-0.1:0.5, 0.1-0.2:0.75, 0.2-0.35:1.0,
           0.35-0.5:1.25, 0.5-0.65:1.5, >0.65:2.0}
  score = sum_b w_b * (clean_frac_b - 0.5*artifact_frac_b - missed_frac_b)
          / sum_b w_b
  Range [-1, +1]; brighter stars matter more (a missed or mangled bright
  star is glaring, a soft faint artifact is not).  Baseline Neutralised.tif
  scores ~= +0.088.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

VERSION = "1.0"

# Classification bins (identical to f3_census.py, plus a catch-all faint bin)
PK_BINS = [0.0, 0.032, 0.1, 0.2, 0.35, 0.5, 0.65, 2.0]
PK_LAB = ["<0.03", "0.03-0.1", "0.1-0.2", "0.2-0.35", "0.35-0.5", "0.5-0.65", ">0.65"]
PK_WEIGHT = {"<0.03": 0.25, "0.03-0.1": 0.5, "0.1-0.2": 0.75, "0.2-0.35": 1.0,
             "0.35-0.5": 1.25, "0.5-0.65": 1.5, ">0.65": 2.0}
FW_BINS = [0, 1.5, 2, 3, 4, 6, 10, 1000]
FW_LAB = ["<1.5", "1.5-2", "2-3", "3-4", "4-6", "6-10", ">10"]


# ---------------------------------------------------------------- loading --
def load_image(path):
    """Load a TIFF or FITS as float32 HxWxC in [0,1] (C=1 or 3).

    FITS planar CxHxW is converted to HxWxC.  16-bit-scaled data is divided
    by 65535.  NO orientation flip is applied here -- the input defines the
    coordinate frame; starless/starmask are oriented against it later.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fit", ".fits", ".fts"):
        from astropy.io import fits
        with fits.open(path) as h:
            a = h[0].data
        if a is None:
            raise ValueError("empty FITS primary HDU: %s" % path)
        a = np.asarray(a)
        if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[0] < a.shape[2]:
            a = np.moveaxis(a, 0, -1)
    else:
        import tifffile
        a = tifffile.imread(path)

    if a.dtype == np.uint16:
        f = a.astype(np.float32) / 65535.0
    elif a.dtype == np.uint8:
        f = a.astype(np.float32) / 255.0
    else:
        f = a.astype(np.float32)
        if f.max() > 2.0:  # 16-bit-scaled float FITS
            f = f / 65535.0
    if f.ndim == 2:
        f = f[:, :, None]
    return f


def orient_starless(inp_f, sl, sm):
    """Pick FITS orientation for starless (+ optional starmask) vs the input.

    Returns (sl, sm, info) where info records the decision and quality.
    Port of the f1_load.py reconstruction check; correlation fallback when
    no starmask is available.
    """
    luma = inp_f.mean(axis=2)
    sluma = sl.mean(axis=2)
    info = {}
    if sm is not None:
        mluma = sm.mean(axis=2)
        rec = sluma + mluma
        err_direct = float(np.abs(rec - luma).mean())
        err_flip = float(np.abs(rec[::-1] - luma).mean())
        flip = err_flip < err_direct
        info.update(method="reconstruction", recon_err_direct=err_direct,
                    recon_err_flipped=err_flip, flipped=bool(flip))
        if flip:
            sl = sl[::-1].copy()
            sm = sm[::-1].copy()
        info["recon_err"] = min(err_direct, err_flip)
        if info["recon_err"] > 0.01:
            print("WARNING: starless+starmask does not reconstruct the input "
                  "(mean |err| = %.4f). Wrong file pairing?" % info["recon_err"])
    else:
        import cv2
        H, W = luma.shape
        a = cv2.resize(luma, (max(1, W // 8), max(1, H // 8)),
                       interpolation=cv2.INTER_AREA).ravel()
        b = cv2.resize(sluma, (max(1, W // 8), max(1, H // 8)),
                       interpolation=cv2.INTER_AREA)
        cc_direct = float(np.corrcoef(a, b.ravel())[0, 1])
        cc_flip = float(np.corrcoef(a, b[::-1].ravel())[0, 1])
        flip = cc_flip > cc_direct
        info.update(method="correlation", corr_direct=cc_direct,
                    corr_flipped=cc_flip, flipped=bool(flip))
        if flip:
            sl = sl[::-1].copy()
        if max(cc_direct, cc_flip) < 0.5:
            print("WARNING: weak input/starless correlation (%.3f). "
                  "Wrong file pairing?" % max(cc_direct, cc_flip))
    return sl, sm, info


# -------------------------------------------------------------- detection --
class RRCache:
    """Cached radial-distance grids keyed by half-window radius r."""

    def __init__(self):
        self._c = {}

    def __call__(self, r):
        rr = self._c.get(r)
        if rr is None:
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            rr = np.sqrt(yy * yy + xx * xx)
            self._c[r] = rr
        return rr


def detect_stars(inp_f, edge=24, verbose=True):
    """f2_detect.py detection + per-star measurement + dedupe.

    Returns (sites dict of arrays, stats dict).
    """
    import cv2
    from scipy import ndimage

    H, W, _ = inp_f.shape
    luma = inp_f.mean(axis=2).astype(np.float32)
    t0 = time.time()

    # background: downsample x8, median filter, upsample (~120px scale)
    small = cv2.resize(luma, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    small = ndimage.median_filter(small, size=15)
    bg = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    hp = luma - bg
    if verbose:
        print("bg done %.1fs" % (time.time() - t0))

    # noise sigma: MAD of pixel-level high-pass on quiet pixels
    gs = cv2.GaussianBlur(luma, (0, 0), 1.0)
    finegrain = luma - gs
    sel = np.abs(hp) < np.percentile(np.abs(hp), 70)
    noise = float(np.median(np.abs(finegrain[sel])) * 1.4826)
    noise_hp = float(np.median(np.abs(hp[sel] - np.median(hp[sel]))) * 1.4826)
    if verbose:
        print("noise sigma (pixel): %.6f  (hp band): %.6f" % (noise, noise_hp))

    # detection: smoothed high-pass local maxima
    det = cv2.GaussianBlur(hp, (0, 0), 1.5)
    mx = ndimage.maximum_filter(det, size=7)
    thr = max(5 * noise, 4 * noise_hp)
    peaks = (det == mx) & (det > thr)
    ys, xs = np.nonzero(peaks)
    m = (ys > edge) & (ys < H - edge) & (xs > edge) & (xs < W - edge)
    ys, xs = ys[m], xs[m]
    if verbose:
        print("peaks after edge cut: %d  thr: %.5f  %.1fs"
              % (len(ys), thr, time.time() - t0))

    rrc = RRCache()

    def measure(y, x):
        r = 16
        for _ in range(3):
            y0, y1, x0, x1 = y - r, y + r + 1, x - r, x + r + 1
            if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
                return None
            cut = luma[y0:y1, x0:x1]
            cbg = bg[y0:y1, x0:x1]
            sub = cut - cbg
            rr = rrc(r)
            ann = sub[(rr > r * 0.75)]
            lb = np.median(ann)
            sub = sub - lb
            pk = sub[r, r]
            if pk <= 0:
                return None
            half = pk / 2.0
            above = sub >= half
            lab, _n = ndimage.label(above)
            comp = lab == lab[r, r]
            area = comp.sum()
            touches_edge = (comp[0, :].any() or comp[-1, :].any()
                            or comp[:, 0].any() or comp[:, -1].any())
            if touches_edge and r < 48:
                r += 16
                continue
            fwhm = 2.0 * np.sqrt(area / np.pi)
            rad = min(r, max(3, fwhm))
            disk = rr <= rad * 1.5
            flux = float(sub[disk].sum())
            sat = float(luma[y, x] >= 0.999)
            return dict(y=y, x=x, peak=float(pk), fwhm=float(fwhm), flux=flux,
                        r=r, sat=sat, rawpeak=float(luma[y, x]))
        return None

    stars = []
    for y, x in zip(ys, xs):
        s = measure(int(y), int(x))
        if s is not None and s["peak"] > thr:
            stars.append(s)
    if verbose:
        print("measured stars: %d  %.1fs" % (len(stars), time.time() - t0))

    # dedupe: peaks closer than 0.7*max(fwhm,4) keep the brighter
    stars.sort(key=lambda s: -s["peak"])
    from collections import defaultdict
    cell = 12
    grid = defaultdict(list)
    final = []
    for s in stars:
        cy, cx = s["y"] // cell, s["x"] // cell
        dup = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for t in grid[(cy + dy, cx + dx)]:
                    d = np.hypot(s["y"] - t["y"], s["x"] - t["x"])
                    if d < 0.7 * max(s["fwhm"], t["fwhm"], 4):
                        dup = True
                        break
                if dup:
                    break
            if dup:
                break
        if not dup:
            grid[(cy, cx)].append(s)
            final.append(s)
    stars = final
    if verbose:
        print("after dedupe: %d  %.1fs" % (len(stars), time.time() - t0))

    sites = dict(
        y=np.array([s["y"] for s in stars], np.int32),
        x=np.array([s["x"] for s in stars], np.int32),
        peak=np.array([s["peak"] for s in stars], np.float32),
        fwhm=np.array([s["fwhm"] for s in stars], np.float32),
        flux=np.array([s["flux"] for s in stars], np.float32),
        rawpeak=np.array([s["rawpeak"] for s in stars], np.float32),
        sat=np.array([s["sat"] for s in stars], np.uint8),
        rmeas=np.array([s["r"] for s in stars], np.int16),
    )
    stats = dict(noise=noise, noise_hp=noise_hp, thr=thr, n_sites=len(stars),
                 H=H, W=W, C=inp_f.shape[2], edge=edge)
    return sites, stats


def cmd_detect(args):
    t0 = time.time()
    inp_f = load_image(args.input)
    print("input: %s %s" % (str(inp_f.shape), args.input))
    sites, stats = detect_stars(inp_f, edge=args.edge)
    np.savez_compressed(
        args.cache,
        version=VERSION,
        input_path=os.path.abspath(args.input),
        input_shape=np.array(inp_f.shape, np.int64),
        noise=stats["noise"], noise_hp=stats["noise_hp"], thr=stats["thr"],
        edge=stats["edge"],
        **sites)
    print("cached %d sites -> %s  (%.1fs total)"
          % (stats["n_sites"], args.cache, time.time() - t0))


# ---------------------------------------------------------------- scoring --
def classify_sites(idx, cache, sluma, sl_color, noise):
    """f2_detect.py classify() port, vectorised over cached sites.

    Returns per-site record arrays (cls codes, artifact types, amplitudes).
    cls: 0=clean 1=artifact 2=missed  (-1 = skipped, too close to edge)
    """
    H, W = sluma.shape
    rrc = RRCache()
    n = len(idx)
    cls = np.full(n, -1, np.int8)
    art = [""] * n
    rmaxs = np.zeros(n, np.float32)
    rmins = np.zeros(n, np.float32)
    amps = np.zeros(n, np.float32)
    resfracs = np.zeros(n, np.float32)
    ys, xs = cache["y"], cache["x"]
    fws, pks = cache["fwhm"], cache["peak"]
    has_color = sl_color is not None and sl_color.shape[2] >= 3
    sig = 3.0 * noise

    for k, i in enumerate(idx):
        y, x = int(ys[i]), int(xs[i])
        fw = float(fws[i])
        A = float(pks[i])
        r = int(np.clip(2.0 * fw, 8, 64))
        y0, y1, x0, x1 = y - r, y + r + 1, x - r, x + r + 1
        if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
            continue
        rr = rrc(r)
        scut = sluma[y0:y1, x0:x1]
        ann = rr > 0.8 * r
        lb = np.median(scut[ann])
        res = scut - lb
        disk = rr <= max(1.5, 0.75 * fw)
        rmax = float(res[disk].max())
        rmin = float(res[disk].min())
        midann = (rr > 0.75 * fw) & (rr <= 1.6 * fw)
        if midann.any():
            ring_amp = float(res[midann].max())
            ring_min = float(res[midann].min())
        else:
            ring_amp = ring_min = 0.0
        amp = max(abs(rmax), abs(rmin), abs(ring_amp), abs(ring_min))

        if rmax > 0.5 * A and rmax > sig:
            c = 2  # missed
        elif amp > max(sig, 0.08 * A):
            c = 1  # artifact
        else:
            c = 0  # clean

        a = ""
        if c == 1:
            if rmin < -max(sig, 0.08 * A) and abs(rmin) >= rmax:
                a = "dark_hole"
            elif rmax > max(sig, 0.08 * A):
                a = "bright_residual"
            elif ring_amp > max(sig, 0.08 * A) or ring_min < -max(sig, 0.08 * A):
                a = "ring"
            else:
                a = "other"
            if has_color:
                ccut = sl_color[y0:y1, x0:x1, :]
                cres = np.array([float(np.mean(ccut[..., ch][disk])
                                       - np.median(ccut[..., ch][ann]))
                                 for ch in range(3)])
                if cres.max() - cres.min() > max(2 * noise, 0.05 * A):
                    a += "+color"

        cls[k] = c
        art[k] = a
        rmaxs[k] = rmax
        rmins[k] = rmin
        amps[k] = amp
        resfracs[k] = rmax / A if A > 0 else 0.0
    return cls, art, rmaxs, rmins, amps, resfracs


def binned_census(vals, cls, bins, labels):
    out = []
    for i, lab in enumerate(labels):
        m = (vals >= bins[i]) & (vals < bins[i + 1]) & (cls >= 0)
        n = int(m.sum())
        if n == 0:
            continue
        nc = int((cls[m] == 0).sum())
        na = int((cls[m] == 1).sum())
        nm = int((cls[m] == 2).sum())
        out.append(dict(bin=lab, n=n, clean=nc, artifact=na, missed=nm,
                        clean_pct=round(100.0 * nc / n, 1),
                        artifact_pct=round(100.0 * na / n, 1),
                        missed_pct=round(100.0 * nm / n, 1)))
    return out


def compute_score(census_peak):
    """Weighted per-peak-bin score in [-1,1]. See module docstring."""
    num = den = 0.0
    for row in census_peak:
        w = PK_WEIGHT.get(row["bin"], 1.0)
        cf = row["clean"] / row["n"]
        af = row["artifact"] / row["n"]
        mf = row["missed"] / row["n"]
        num += w * (cf - 0.5 * af - mf)
        den += w
    return num / den if den > 0 else 0.0


def smooth_leak_fraction(sm):
    """f5_leak.py: fraction of star-mask energy that is low-frequency diffuse
    glow (nebula leakage) rather than point sources.  Median filter on a 4x
    downsample kills point sources up to ~10px; what survives is smooth."""
    import cv2
    from scipy import ndimage
    H, W, _ = sm.shape
    mluma = sm.mean(axis=2).astype(np.float32)
    ms = cv2.resize(mluma, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    smooth = ndimage.median_filter(ms, size=9)
    tot = float(ms.sum())
    return dict(
        smooth_leak_fraction=float(smooth.sum() / tot) if tot > 0 else 0.0,
        smooth_mean=float(smooth.mean()),
        smooth_p99=float(np.percentile(smooth, 99)),
        smooth_max=float(smooth.max()),
        smooth_frac_gt_0p01=float((smooth > 0.01).mean()),
    )


def pctiles(a, ps):
    if len(a) == 0:
        return {str(p): None for p in ps}
    return {str(p): round(float(np.percentile(a, p)), 5) for p in ps}


def cmd_score(args):
    t0 = time.time()
    cache = np.load(args.cache, allow_pickle=False)
    n_total = len(cache["y"])
    noise = float(cache["noise"])
    thr = float(cache["thr"])
    print("cache: %d sites  noise=%.6f thr=%.5f" % (n_total, noise, thr))

    inp_f = load_image(args.input)
    H, W, _ = inp_f.shape
    cH, cW = int(cache["input_shape"][0]), int(cache["input_shape"][1])
    if (cH, cW) != (H, W):
        sys.exit("ERROR: cache was built for %dx%d input, got %dx%d"
                 % (cH, cW, H, W))

    sl = load_image(args.starless)
    sm = load_image(args.starmask) if args.starmask else None
    if sl.shape[:2] != (H, W):
        sys.exit("ERROR: starless is %s, input is %dx%d"
                 % (str(sl.shape), H, W))
    sl, sm, orient = orient_starless(inp_f, sl, sm)
    print("orientation:", orient)
    del inp_f  # frame no longer needed

    sluma = sl.mean(axis=2).astype(np.float32)
    sl_color = sl if sl.shape[2] >= 3 else None

    # deterministic subsample (same cache + seed + max-sites => same sites)
    if args.max_sites and n_total > args.max_sites:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(n_total, args.max_sites, replace=False))
        print("subsampled %d / %d sites (seed %d)"
              % (len(idx), n_total, args.seed))
    else:
        idx = np.arange(n_total)

    cls, art, rmaxs, rmins, amps, resfracs = classify_sites(
        idx, cache, sluma, sl_color, noise)
    scored = cls >= 0
    n_scored = int(scored.sum())
    print("classified %d sites  %.1fs" % (n_scored, time.time() - t0))

    nc = int((cls == 0).sum())
    na = int((cls == 1).sum())
    nm = int((cls == 2).sum())

    peaks = cache["peak"][idx]
    fwhms = cache["fwhm"][idx]
    cen_pk = binned_census(peaks, cls, PK_BINS, PK_LAB)
    cen_fw = binned_census(fwhms, cls, FW_BINS, FW_LAB)

    isa = cls == 1
    a_amps = amps[isa]
    art_types = {}
    for k in np.nonzero(isa)[0]:
        art_types[art[k]] = art_types.get(art[k], 0) + 1
    is_dh = np.array([a.startswith("dark_hole") for a in art], bool) & isa
    dh_depth = np.abs(rmins[is_dh])
    is_missed = cls == 2

    # residual stars: sites whose starless residual peak still exceeds the
    # input detection threshold (i.e. a detector would still find a star).
    n_resid = int(((rmaxs > thr) & scored).sum())
    scale = n_total / max(1, n_scored)

    result = dict(
        harness_version=VERSION,
        label=args.label,
        input=os.path.abspath(args.input),
        starless=os.path.abspath(args.starless),
        starmask=os.path.abspath(args.starmask) if args.starmask else None,
        cache=os.path.abspath(args.cache),
        orientation=orient,
        noise_sigma=noise,
        detection_thr=thr,
        n_sites_total=n_total,
        n_scored=n_scored,
        subsample=dict(max_sites=args.max_sites, seed=args.seed,
                       fraction=n_scored / n_total if n_total else 0.0),
        overall=dict(
            clean=nc, artifact=na, missed=nm,
            clean_pct=round(100.0 * nc / n_scored, 2),
            artifact_pct=round(100.0 * na / n_scored, 2),
            missed_pct=round(100.0 * nm / n_scored, 2)),
        census_peak=cen_pk,
        census_fwhm=cen_fw,
        artifact_types=art_types,
        artifact_amp_sigma=dict(
            median=round(float(np.median(a_amps) / noise), 2) if na else None,
            p75=round(float(np.percentile(a_amps, 75) / noise), 2) if na else None,
            p95=round(float(np.percentile(a_amps, 95) / noise), 2) if na else None),
        dark_hole=dict(
            count=int(is_dh.sum()),
            frac_of_artifacts=round(float(is_dh.sum() / na), 4) if na else None,
            median_depth_sigma=(round(float(np.median(dh_depth) / noise), 2)
                                if is_dh.any() else None)),
        missed=dict(
            count=nm,
            median_peak=(round(float(np.median(peaks[is_missed])), 4)
                         if nm else None),
            median_fwhm=(round(float(np.median(fwhms[is_missed])), 2)
                         if nm else None),
            resfrac_pct=pctiles(resfracs[is_missed], [25, 50, 75, 95])),
        residual_stars=dict(
            count_in_scored=n_resid,
            estimated_total=int(round(n_resid * scale)),
            definition="sites with starless residual peak (rmax) > input "
                       "detection threshold"),
        score=round(compute_score(cen_pk), 4),
        score_weights=PK_WEIGHT,
        runtime_sec=round(time.time() - t0, 1),
    )

    if sm is not None:
        result["mask_smooth_leak"] = smooth_leak_fraction(sm)

    with open(args.json, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote %s" % args.json)

    md = render_md(result)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(md)
        print("wrote %s" % args.md)
    print(md)


def render_md(r):
    L = []
    L.append("## Star-removal census: %s" % r["label"])
    L.append("")
    L.append("input `%s`" % os.path.basename(r["input"]))
    L.append("starless `%s`  score **%+.4f**" %
             (os.path.basename(r["starless"]), r["score"]))
    o = r["overall"]
    L.append("")
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append("| sites scored | %d / %d |" % (r["n_scored"], r["n_sites_total"]))
    L.append("| clean | %.2f%% (%d) |" % (o["clean_pct"], o["clean"]))
    L.append("| artifact | %.2f%% (%d) |" % (o["artifact_pct"], o["artifact"]))
    L.append("| missed | %.2f%% (%d) |" % (o["missed_pct"], o["missed"]))
    aa = r["artifact_amp_sigma"]
    if aa["median"] is not None:
        L.append("| artifact median residual | %.1f x noise sigma |" % aa["median"])
    dh = r["dark_hole"]
    if dh["frac_of_artifacts"] is not None:
        L.append("| dark holes | %.1f%% of artifacts, median depth %.1f sigma |"
                 % (100 * dh["frac_of_artifacts"], dh["median_depth_sigma"] or 0))
    rs = r["residual_stars"]
    L.append("| residual stars (est. total) | %d |" % rs["estimated_total"])
    if "mask_smooth_leak" in r:
        L.append("| mask smooth-leak fraction | %.1f%% |"
                 % (100 * r["mask_smooth_leak"]["smooth_leak_fraction"]))
    L.append("")
    L.append("### By peak (bg-subtracted, 0-1)")
    L.append("")
    L.append("| peak bin | n | clean % | artifact % | missed % |")
    L.append("|---|---|---|---|---|")
    for b in r["census_peak"]:
        L.append("| %s | %d | %.1f | %.1f | %.1f |"
                 % (b["bin"], b["n"], b["clean_pct"], b["artifact_pct"],
                    b["missed_pct"]))
    L.append("")
    L.append("### By FWHM (px)")
    L.append("")
    L.append("| FWHM bin | n | clean % | artifact % | missed % |")
    L.append("|---|---|---|---|---|")
    for b in r["census_fwhm"]:
        L.append("| %s | %d | %.1f | %.1f | %.1f |"
                 % (b["bin"], b["n"], b["clean_pct"], b["artifact_pct"],
                    b["missed_pct"]))
    L.append("")
    return "\n".join(L)


# -------------------------------------------------------------------- cli --
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="detect stars in input, cache sites")
    d.add_argument("--input", required=True, help="input TIFF/FITS (with stars)")
    d.add_argument("--cache", required=True, help="output .npz site cache")
    d.add_argument("--edge", type=int, default=24, help="edge exclusion px")
    d.set_defaults(fn=cmd_detect)

    s = sub.add_parser("score", help="score a starless result against a cache")
    s.add_argument("--cache", required=True, help=".npz site cache from detect")
    s.add_argument("--input", required=True, help="same input given to detect")
    s.add_argument("--starless", required=True, help="starless TIFF/FITS")
    s.add_argument("--starmask", default=None, help="optional star-mask TIFF/FITS")
    s.add_argument("--label", required=True, help="variant name")
    s.add_argument("--json", required=True, help="output JSON path")
    s.add_argument("--md", default=None, help="optional output Markdown path")
    s.add_argument("--max-sites", type=int, default=0,
                   help="deterministic subsample size (0 = all sites)")
    s.add_argument("--seed", type=int, default=42, help="subsample seed")
    s.set_defaults(fn=cmd_score)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
