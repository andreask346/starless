# Runtime loader for the v3 REAL-DATA assets harvested from Andreas's frames
# (datagen/v3/).  Everything here is loaded LAZILY at runtime and re-checked
# on disk, because the harvest keeps running and the assets keep growing:
#   * no counts are hardcoded — group/stamp/mask counts come from the files;
#   * a changed file (mtime/size) is picked up on the next check (>= RELOAD_S
#     seconds apart) without restarting the process;
#   * a half-written or missing file NEVER raises: the previous good copy is
#     kept, and if there never was one the getters return None so the caller
#     falls back to plain v2 behaviour.
#
# Assets (see datagen/v3/HARVEST_REPORT.md for how they were measured):
#   psf_stamps.npz    stamps (N,3,S,S) f16 + rep_stamps (G,3,S,S) f32
#                     (orientation-aligned median kernel per group) + per-stamp
#                     fwhm / ellipticity / theta / radius / snr / group_id
#   psf_index.json    G groups = (optical setup) x (center|mid|corner)
#   noise_model.json  per-channel sigma(S)=sqrt(a*S+b^2), sigma_rel, 9x9 ACF,
#                     R/G/B correlation coefficients
#   silhouettes/      binary PNG masks (255 = occluding foreground) + index

import json
import os
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V3DIR = os.path.join(HERE, "v3")
RELOAD_S = 60.0            # min seconds between disk re-checks

_LOCK = threading.Lock()
_CACHE = {}                # key -> dict(stamp=..., obj=..., checked=...)


def v3_dir():
    return V3DIR


def _stamp(paths):
    """(mtime, size) fingerprint of a list of paths; None if any is missing."""
    out = []
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            return None
        out.append((round(st.st_mtime, 3), st.st_size))
    return tuple(out)


def _cached(key, paths, build):
    """Generic lazy/reloading cache. Returns the built object or None."""
    now = time.time()
    with _LOCK:
        ent = _CACHE.get(key)
        if ent is not None and now - ent["checked"] < RELOAD_S:
            return ent["obj"]
        stamp = _stamp(paths)
        if ent is not None:
            ent["checked"] = now
            if stamp is None or stamp == ent["stamp"]:
                return ent["obj"]           # unchanged (or vanished): keep it
        try:
            obj = build() if stamp is not None else None
        except Exception:
            obj = ent["obj"] if ent is not None else None   # mid-write: keep
            stamp = ent["stamp"] if ent is not None else None
        _CACHE[key] = dict(stamp=stamp, obj=obj, checked=now)
        return obj


# ------------------------------------------------------------------ PSF
def _build_psf():
    ipath = os.path.join(V3DIR, "psf_index.json")
    npath = os.path.join(V3DIR, "psf_stamps.npz")
    idx = json.load(open(ipath, "r", encoding="utf-8"))
    z = np.load(npath, allow_pickle=False)
    rep = np.asarray(z["rep_stamps"], dtype=np.float32)      # (G,3,S,S)
    groups = idx["groups"]
    ng = min(len(groups), rep.shape[0])                      # tolerate skew
    groups = groups[:ng]
    rep = rep[:ng]
    # per-channel unit flux (harvest normalises over r<=14; renormalise exactly)
    s = rep.reshape(ng, 3, -1).sum(axis=2)
    s[np.abs(s) < 1e-12] = 1.0
    rep = rep / s[:, :, None, None]
    rep_t = torch.from_numpy(np.ascontiguousarray(rep))

    fwhm = np.asarray(z["fwhm"], dtype=np.float32)
    ell = np.asarray(z["ellipticity"], dtype=np.float32)
    gid = np.asarray(z["group_id"], dtype=np.int64)

    field = [g.get("field", "corner") for g in groups]
    g_fwhm = np.array([float(g["fwhm"]["med"]) for g in groups],
                      dtype=np.float32)
    # per-group spread: seeing/focus varies between the frames of one shoot,
    # so a group's own p10..p90 is the right prior for resampling its kernel
    g_fwhm_p10 = np.array([float(g["fwhm"].get("p10", g["fwhm"]["med"]))
                           for g in groups], dtype=np.float32)
    g_fwhm_p90 = np.array([float(g["fwhm"].get("p90", g["fwhm"]["med"]))
                           for g in groups], dtype=np.float32)
    g_ell = np.array([float(g["ellipticity"]["med"]) for g in groups],
                     dtype=np.float32)
    g_theta = np.array([float(g.get("theta_med", 0.0)) for g in groups],
                       dtype=np.float32)
    g_n = np.array([float(g.get("n", 1)) for g in groups], dtype=np.float64)

    by_field = {}
    for i, f in enumerate(field):
        by_field.setdefault(f, []).append(i)
    by_setup = {}
    for i, g in enumerate(groups):
        by_setup.setdefault(g.get("setup", "?"), {}) \
            .setdefault(field[i], []).append(i)
    setups = sorted(by_setup)
    setup_w = np.array([sum(g_n[i] for lst in by_setup[s].values()
                            for i in lst) for s in setups], dtype=np.float64)
    setup_w = setup_w / max(setup_w.sum(), 1e-9)

    # de-biased analytic pools: the harvest samples 5 windows per frame
    # (centre + 4 corners) so the corner bin is over-represented by
    # CONSTRUCTION.  Keep one pool per field bin and let the caller draw the
    # bin, instead of drawing from the pooled (corner-heavy) array.
    stamp_field = np.array([field[g] if g < ng else "corner" for g in
                            np.clip(gid, 0, ng - 1)])
    pools = {}
    for f in set(field):
        m = stamp_field == f
        if m.sum() >= 8:
            pools[f] = (fwhm[m].copy(), ell[m].copy())
    return dict(rep=rep_t, size=int(rep.shape[-1]), n_groups=ng,
                field=field, by_field=by_field, by_setup=by_setup,
                setups=setups, setup_w=setup_w,
                g_fwhm=g_fwhm, g_fwhm_p10=g_fwhm_p10, g_fwhm_p90=g_fwhm_p90,
                g_ell=g_ell, g_theta=g_theta,
                fwhm=fwhm, ell=ell, pools=pools,
                n_stamps=int(fwhm.shape[0]),
                fields=sorted(by_field))


def psf_assets():
    return _cached("psf",
                   [os.path.join(V3DIR, "psf_index.json"),
                    os.path.join(V3DIR, "psf_stamps.npz")],
                   _build_psf)


# ------------------------------------------------------------------ noise
def _acf_to_kernel(acf, ksize=9):
    """Kernel k with (k star k) == acf, i.e. white noise convolved with k has
    the MEASURED autocorrelation.  k = IFFT(sqrt(|FFT(acf)|)), normalised to
    sum(k^2) == 1 so the marginal std is preserved exactly."""
    n = 33
    pad = np.zeros((n, n), dtype=np.float64)
    s = acf.shape[0]
    o = (n - s) // 2
    pad[o:o + s, o:o + s] = acf
    pad = np.fft.ifftshift(pad)
    ps = np.real(np.fft.fft2(pad))
    ps = np.clip(ps, 0.0, None)                    # power spectrum >= 0
    k = np.real(np.fft.ifft2(np.sqrt(ps)))
    k = np.fft.fftshift(k)
    o = (n - ksize) // 2
    k = k[o:o + ksize, o:o + ksize]
    e = float(np.sqrt((k ** 2).sum()))
    return k / max(e, 1e-12)


def _q(d, key, default):
    try:
        v = float(d[key])
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _build_noise():
    p = os.path.join(V3DIR, "noise_model.json")
    nm = json.load(open(p, "r", encoding="utf-8"))
    ch = ("R", "G", "B")

    def band(sec, lo_k="p10", md_k="med", hi_k="p90", dflt=(0.0, 0.0, 0.0)):
        out = []
        for i, c in enumerate(ch):
            d = nm.get(sec, {}).get(c, {})
            out.append((_q(d, lo_k, dflt[0]), _q(d, md_k, dflt[1]),
                        _q(d, hi_k, dflt[2])))
        return out

    sig_rel = band("sigma_rel", dflt=(0.01, 0.1, 0.3))
    # sigma_rel has a fat, physically-impossible upper tail on a handful of
    # near-black sources (max 190) — clamp to a sane sky-noise range.
    sig_rel = [(max(lo, 2e-3), min(max(md, 3e-3), 0.6), min(max(hi, 5e-3), 0.8))
               for lo, md, hi in sig_rel]
    a_band = band("a", dflt=(1e-5, 3e-4, 3e-3))
    a_band = [(max(lo, 1e-7), max(md, 1e-6), max(hi, 1e-5))
              for lo, md, hi in a_band]
    rho = {k: _q(nm.get("channel_rho", {}).get(k, {}), "med", 0.3)
           for k in ("rg", "rb", "gb")}
    sp = nm.get("spatial", {})
    acf = np.asarray(sp.get("acf9", []), dtype=np.float64)
    kern = _acf_to_kernel(acf) if acf.ndim == 2 and acf.shape[0] >= 3 else None
    return dict(sigma_rel=sig_rel, a=a_band, rho=rho,
                acf=acf if acf.ndim == 2 else None,
                kernel=torch.from_numpy(kern).float()
                if kern is not None else None,
                corr_len=_q(sp, "corr_len_1_over_e_px", 1.75),
                ksigma=_q(sp, "equivalent_gauss_ksigma", 0.874),
                n_sources=int(nm.get("n_sources", 0)))


def noise_assets():
    return _cached("noise", [os.path.join(V3DIR, "noise_model.json")],
                   _build_noise)


# ------------------------------------------------------------------ silhouettes
_MASK_CACHE = {}
_MASK_ORDER = []
_MASK_CAP = 16          # ~1.4 MB each as uint8; x10 DataLoader workers


def _build_sil():
    d = os.path.join(V3DIR, "silhouettes")
    idx = json.load(open(os.path.join(d, "index.json"), "r", encoding="utf-8"))
    masks = [m for m in idx.get("masks", [])
             if os.path.isfile(os.path.join(d, m.get("file", "")))]
    if not masks:
        return None
    return dict(dir=d, masks=masks, n=len(masks),
                acceptance=_q(idx, "acceptance", 0.42),
                fg_frac=idx.get("fg_frac", {}))


def sil_assets():
    return _cached("sil", [os.path.join(V3DIR, "silhouettes", "index.json")],
                   _build_sil)


def load_mask(path):
    """Decode a silhouette PNG to a uint8 (H,W) array in {0,1}, LRU-cached
    (decoding dominates otherwise and would starve the GPU; uint8 keeps the
    per-worker cache at ~20 MB instead of ~90 MB)."""
    a = _MASK_CACHE.get(path)
    if a is not None:
        return a
    from PIL import Image
    with Image.open(path) as im:
        a = np.asarray(im.convert("L"), dtype=np.uint8)
    a = (a >= 128).astype(np.uint8)
    with _LOCK:
        _MASK_CACHE[path] = a
        _MASK_ORDER.append(path)
        while len(_MASK_ORDER) > _MASK_CAP:
            _MASK_CACHE.pop(_MASK_ORDER.pop(0), None)
    return a


# ------------------------------------------------------------------ summary
def summary():
    p, n, s = psf_assets(), noise_assets(), sil_assets()
    return dict(
        psf=None if p is None else dict(groups=p["n_groups"],
                                        stamps=p["n_stamps"],
                                        size=p["size"], fields=p["fields"],
                                        setups=len(p["setups"])),
        noise=None if n is None else dict(sources=n["n_sources"],
                                          corr_len=n["corr_len"],
                                          kernel=None if n["kernel"] is None
                                          else list(n["kernel"].shape)),
        sil=None if s is None else dict(masks=s["n"],
                                        acceptance=s["acceptance"]))


if __name__ == "__main__":
    print(json.dumps(summary(), indent=1))
