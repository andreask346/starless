# -*- coding: utf-8 -*-
# One Click Starless — AI star removal for Siril.
# Removes stars from the loaded image (or a file), returning a starless image
# and a star layer with EXACT recomposition — Subtract mode (linear images):
# starless + stars == input; Descreen mode (stretched images):
# 1-(1-starless)*(1-stars) == input (Screen blend). Auto-detects per image.
# Inference: ONNX Runtime (DirectML GPU when available, CPU fallback),
# 512px tiles, 50% overlap, Hann-feathered blending. Experimental extras
# (off by default until the v2 model): canonicalized inference, deep clean.
#
# Requires Siril >= 1.4.4 (sirilpy >= 1.0.25). Model weights: starless_*.onnx
# next to this script, or picked via the GUI.
# (c) 2026 Andreas Karaolis - GPL-3.0-or-later

import os
import sys
import time
import traceback

VERSION = "0.2.1"
MIN_SIRIL = "1.4.4"
TILE = 512
OVERLAP = 256           # 50% tile overlap: more Hann-averaged predictions per
                        # pixel; measured +1pp clean rate on the census frame
EXTRACT_MODES = ("auto", "subtract", "descreen")
STRETCH_BG = 0.05       # robust bg median above this => stretched image
CANON_TARGET_BG = 0.05  # background used when canonicalize is forced on

try:
    import sirilpy as s
    HAS_SIRILPY = True
except ImportError:
    s = None
    HAS_SIRILPY = False

import numpy as np


def _ensure_ort():
    """onnxruntime via sirilpy's ONNXHelper when available, else pip."""
    try:
        import onnxruntime  # noqa: F401
        return
    except ImportError:
        pass
    if HAS_SIRILPY:
        try:
            helper = s.ONNXHelper()
            helper.ensure_onnxruntime()
            import onnxruntime  # noqa: F401
            return
        except Exception:
            s.ensure_installed("onnxruntime-directml")
            return
    raise RuntimeError("onnxruntime not available")


_ensure_ort()
import onnxruntime as ort  # noqa: E402


# ------------------------------------------------------------- inference
def make_session(model_path, log):
    providers = []
    avail = ort.get_available_providers()
    if "DmlExecutionProvider" in avail:
        providers.append("DmlExecutionProvider")
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(model_path, providers=providers)
    log(f"model: {os.path.basename(model_path)} on {sess.get_providers()[0]}")
    return sess


def hann2d(size):
    w = np.hanning(size)
    return (np.outer(w, w) + 1e-4).astype(np.float32)


def remove_stars(sess, img, progress=None):
    """img: (3,H,W) float32 0..1 -> stars layer (3,H,W); starless = img - stars."""
    c, h, w = img.shape
    step = TILE - OVERLAP
    pad_h = (step - (h - TILE) % step) % step if h > TILE else TILE - h
    pad_w = (step - (w - TILE) % step) % step if w > TILE else TILE - w
    padded = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    ph, pw = padded.shape[1:]
    out = np.zeros_like(padded)
    weight = np.zeros((ph, pw), dtype=np.float32)
    win = hann2d(TILE)
    ys = list(range(0, ph - TILE + 1, step))
    xs = list(range(0, pw - TILE + 1, step))
    total = len(ys) * len(xs)
    k = 0
    for y in ys:
        for x in xs:
            tile = padded[None, :, y:y + TILE, x:x + TILE].astype(np.float32)
            pred = sess.run(None, {"input": tile})[0][0]
            out[:, y:y + TILE, x:x + TILE] += pred * win
            weight[y:y + TILE, x:x + TILE] += win
            k += 1
            if progress:
                progress(k / total)
    out /= weight[None]
    stars = out[:, :h, :w]
    return np.clip(np.minimum(stars, np.clip(img, 0, None)), 0, None)


# ------------------------------------------------- domain + star-layer math
def robust_bg_median(img):
    """Robust background estimate: median over a subsampled frame."""
    sub = img[:, ::4, ::4] if img.shape[1] > 2000 else img
    return float(np.median(sub))


def mtf(x, m):
    """Midtones transfer function (monotonic, fixes 0 and 1).
    The inverse of mtf(., m) is mtf(., 1 - m)."""
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return ((m - 1.0) * x) / (((2.0 * m - 1.0) * x) - m)


def solve_mtf_m(from_bg, to_bg):
    """m such that mtf(from_bg, m) == to_bg."""
    b, t = float(from_bg), float(to_bg)
    return b * (t - 1.0) / (2.0 * t * b - t - b)


def descreen_stars(inp, starless, eps=1e-6):
    """Screen-consistent star layer ('stars over black'): recombine with
    1-(1-starless)*(1-stars). The right layer for stretched images — the
    star layer no longer carries the sky background, so it can be scaled,
    blurred or recombined without dark rings or clipping."""
    x = np.clip(inp, 0.0, 1.0)
    sl = np.clip(starless, 0.0, 1.0)
    return np.clip((x - sl) / np.maximum(1.0 - sl, eps), 0.0, 1.0)


def resize_ch(img, h, w):
    """Bilinear resize of (3,H,W), numpy only."""
    _, ih, iw = img.shape
    ys = np.linspace(0, ih - 1, h, dtype=np.float32)
    xs = np.linspace(0, iw - 1, w, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.minimum(y0 + 1, ih - 1)
    x0 = np.floor(xs).astype(np.int32)
    x1 = np.minimum(x0 + 1, iw - 1)
    wy = (ys - y0)[None, :, None]
    wx = (xs - x0)[None, None, :]
    top = img[:, y0][:, :, x0] * (1 - wx) + img[:, y0][:, :, x1] * wx
    bot = img[:, y1][:, :, x0] * (1 - wx) + img[:, y1][:, :, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def extract_star_layer(sess, data, extract_mode="auto", deep_clean=False,
                       canonicalize="off", target_bg=CANON_TARGET_BG,
                       log=print, progress=None):
    """Full pipeline on (3,H,W) float32 -> (starless, stars, info).

    canonicalize/deep_clean are EXPERIMENTAL and off by default: census-
    measured on the v1 weights, inverse-MTF canonicalization amplifies
    background-adjacent model errors through the restretch gain (artifacts
    21->32-51 sigma) and deep clean's coarse passes eat Milky Way texture
    (mask leak 29->59%). Re-evaluate both after the v2 retrain; the
    machinery is kept because the mechanisms are sound, the v1 weights
    aren't. Exact recomposition is preserved in every combination."""
    bg = robust_bg_median(data)
    stretched = bg > STRETCH_BG
    mode = extract_mode if extract_mode != "auto" else (
        "descreen" if stretched else "subtract")
    canon = stretched if canonicalize == "auto" else (canonicalize == "on")
    log(f"background median {bg:.3f} -> "
        f"{'stretched' if stretched else 'linear'} image; "
        f"star layer: {mode}" + (", canonicalized inference" if canon else ""))
    info = {"bg": bg, "stretched": stretched, "mode": mode,
            "canonicalized": canon}

    if canon:
        m_c = solve_mtf_m(bg, target_bg)
        work = mtf(data, m_c)
        log(f"inverse MTF m={m_c:.3f} (background {bg:.3f} -> {target_bg:.3f})")
    else:
        m_c = None
        work = data

    def sub_prog(a, b):
        if progress is None:
            return None
        return lambda f: progress(a + (b - a) * f)

    main_end = 0.55 if deep_clean else 1.0
    stars_w = remove_stars(sess, work, progress=sub_prog(0.0, main_end))

    if deep_clean:
        _, h, w = work.shape
        # coarse passes catch big saturated stars + halos beyond one tile
        for fct, a, b in ((4, 0.55, 0.60), (2, 0.60, 0.70)):
            down = resize_ch(work, max(h // fct, 1), max(w // fct, 1))
            sd = remove_stars(sess, down, progress=sub_prog(a, b))
            stars_w = np.maximum(stars_w, resize_ch(sd, h, w))
        stars_w = np.minimum(stars_w, np.clip(work, 0, None))
        # second pass sweeps residuals the first pass left behind
        resid = remove_stars(sess, work - stars_w,
                             progress=sub_prog(0.70, 1.0))
        stars_w = np.minimum(stars_w + resid, np.clip(work, 0, None))

    starless_w = work - stars_w
    if canon:
        starless = mtf(starless_w, 1.0 - m_c)  # back to the input domain
        starless = np.minimum(starless, np.clip(data, 0.0, 1.0))
    else:
        starless = starless_w
    stars_sub = data - starless               # exact in the input domain

    if mode == "descreen":
        stars = descreen_stars(data, starless)
        rec = 1.0 - (1.0 - np.clip(starless, 0, 1)) * (1.0 - stars)
        info["recomp_err"] = float(np.abs(rec - np.clip(data, 0, 1)).max())
        info["recombine"] = "screen: 1-(1-starless)*(1-stars)"
    else:
        stars = stars_sub
        info["recomp_err"] = float(np.abs((starless + stars) - data).max())
        info["recombine"] = "add: starless + stars"
    return starless.astype(np.float32), stars.astype(np.float32), info


def default_model():
    d = os.path.dirname(os.path.abspath(__file__))
    cands = sorted(
        [f for f in os.listdir(d) if f.startswith("starless")
         and f.endswith(".onnx")], reverse=True)
    return os.path.join(d, cands[0]) if cands else ""


# ------------------------------------------------------------- siril glue
def make_siril():
    if not HAS_SIRILPY:
        return None
    try:
        iface = s.SirilInterface()
        iface.connect()
        return iface
    except Exception:
        return None


def get_loaded_image(siril):
    fit = siril.get_image(with_pixels=True)
    fit.ensure_data_type(np.float32)
    data = np.asarray(fit.data, dtype=np.float32)
    if data.ndim == 2:
        data = data[None].repeat(3, axis=0)
        mono = True
    else:
        mono = False
    return data, mono


def run_on_loaded(siril, model_path, save_starless, save_stars,
                  replace_loaded, extract_mode, deep_clean, log, prog):
    t0 = time.time()
    sess = make_session(model_path, log)
    data, mono = get_loaded_image(siril)
    log(f"removing stars from loaded image "
        f"({data.shape[2]}x{data.shape[1]})...")
    starless, stars, info = extract_star_layer(
        sess, data, extract_mode=extract_mode, deep_clean=deep_clean,
        log=log, progress=lambda f: prog(0.1 + 0.85 * f, "removing stars"))
    log(f"recomposition ({info['recombine']}) max err "
        f"{info['recomp_err']:.2e}")
    log("recombine recipe (PixelMath): "
        + ("1-(1-starless)*(1-starmask)" if info["mode"] == "descreen"
           else "starless + starmask"))

    base = os.path.splitext(siril.get_image_filename() or "image")[0]
    outputs = []
    wd = siril.get_siril_wd()
    hdr = None
    try:
        hdr = siril.get_image_fits_header(return_as="str")
    except Exception:
        pass
    for flag, name, arr in ((save_starless, "starless", starless),
                            (save_stars, "starmask", stars)):
        if not flag:
            continue
        out_arr = arr[0] if mono else arr
        p = os.path.join(wd, f"{name}_{os.path.basename(base)}.fit")
        siril.save_image_file(out_arr.astype(np.float32), header=hdr or "",
                              filename=p)
        outputs.append(p)
        log(f"saved {p}")
    if replace_loaded:
        with siril.image_lock():
            siril.undo_save_state("One Click Starless removal")
            siril.set_image_pixeldata(
                (starless[0] if mono else starless).astype(np.float32))
        log("loaded image replaced with starless result (undo available)")
    prog(1.0, "done")
    log(f"done in {time.time() - t0:.1f}s")
    return outputs


# ------------------------------------------------------------- GUI
def run_gui(siril):
    try:
        import PyQt6  # noqa: F401
    except ImportError:
        if HAS_SIRILPY:
            s.ensure_installed("PyQt6")
    from PyQt6.QtCore import QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QLineEdit, QCheckBox, QComboBox, QProgressBar, QFileDialog,
        QMessageBox)

    def log(msg):
        if siril is not None:
            try:
                siril.log(f"OneClickStarless: {msg}"[:1000])
            except Exception:
                pass
        print(f"OneClickStarless: {msg}")

    class Worker(QThread):
        progressed = pyqtSignal(float, str)
        done_sig = pyqtSignal(object)

        def __init__(self, model_path, opts):
            super().__init__()
            self.model_path = model_path
            self.opts = opts

        def run(self):
            try:
                def prog(f, m):
                    self.progressed.emit(f, m)
                    if siril is not None:
                        try:
                            siril.update_progress(m, min(max(f, 0), 1))
                        except Exception:
                            pass
                outs = run_on_loaded(siril, self.model_path, *self.opts,
                                     log, prog)
                self.done_sig.emit(outs)
            except Exception as e:
                log(f"fatal: {e}\n{traceback.format_exc()}")
                self.done_sig.emit(None)

    class Dlg(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"One Click Starless {VERSION}")
            self.setMinimumWidth(560)
            v = QVBoxLayout(self)
            v.addWidget(QLabel("<b>Model</b> (.onnx)"))
            h = QHBoxLayout()
            self.model_edit = QLineEdit(default_model())
            b = QPushButton("Browse...")
            b.clicked.connect(self.pick_model)
            h.addWidget(self.model_edit)
            h.addWidget(b)
            v.addLayout(h)
            v.addWidget(QLabel("Runs on the image currently loaded in Siril."))
            h2 = QHBoxLayout()
            h2.addWidget(QLabel("Star layer:"))
            self.mode_combo = QComboBox()
            self.mode_combo.addItems([
                "Auto (detected per image)",
                "Subtract — linear images (starless + stars)",
                "Descreen — stretched images (Screen recombine)"])
            h2.addWidget(self.mode_combo, 1)
            v.addLayout(h2)
            self.cb_deep = QCheckBox("Deep clean — extra passes for big "
                                     "stars (experimental until v2 model)")
            self.cb_deep.setChecked(False)
            v.addWidget(self.cb_deep)
            self.cb_starless = QCheckBox("Save starless FITS")
            self.cb_stars = QCheckBox("Save star layer FITS "
                                      "(exact recomposition)")
            self.cb_replace = QCheckBox("Replace loaded image with starless "
                                        "(undoable)")
            for cb in (self.cb_starless, self.cb_stars, self.cb_replace):
                cb.setChecked(True)
                v.addWidget(cb)
            self.pbar = QProgressBar()
            self.pbar.setRange(0, 1000)
            v.addWidget(self.pbar)
            self.status = QLabel("Ready.")
            self.status.setWordWrap(True)
            v.addWidget(self.status)
            h = QHBoxLayout()
            h.addStretch()
            self.run_btn = QPushButton("Remove stars")
            self.run_btn.clicked.connect(self.go)
            h.addWidget(self.run_btn)
            v.addLayout(h)

        def pick_model(self):
            p, _ = QFileDialog.getOpenFileName(self, "Starless model", "",
                                               "ONNX model (*.onnx)")
            if p:
                self.model_edit.setText(p)

        def closeEvent(self, event):
            if getattr(self, "worker", None) is not None \
                    and self.worker.isRunning():
                self.status.setText("still running - wait for it to finish")
                event.ignore()
                return
            super().closeEvent(event)

        def go(self):
            mp = self.model_edit.text().strip()
            if not os.path.isfile(mp):
                QMessageBox.warning(self, "One Click Starless",
                                    "Pick a .onnx model file.")
                return
            if siril is None or not siril.is_image_loaded():
                QMessageBox.warning(self, "One Click Starless",
                                    "Load an image in Siril first.")
                return
            self.run_btn.setEnabled(False)
            self.worker = Worker(
                mp, (self.cb_starless.isChecked(),
                     self.cb_stars.isChecked(),
                     self.cb_replace.isChecked(),
                     EXTRACT_MODES[self.mode_combo.currentIndex()],
                     self.cb_deep.isChecked()))
            self.worker.progressed.connect(
                lambda f, m: (self.pbar.setValue(int(f * 1000)),
                              self.status.setText(m)))
            self.worker.done_sig.connect(self._on_done)
            self.worker.start()

        # NOTE: not done() — that would override QDialog.done(int)
        def _on_done(self, outs):
            self.run_btn.setEnabled(True)
            self.status.setText(
                "failed - see Siril log" if outs is None
                else f"done: {len(outs)} file(s) written")

    app = QApplication.instance() or QApplication(sys.argv)
    dlg = Dlg()
    shot = os.environ.get("ONECLICK_STARLESS_SCREENSHOT")
    if shot:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(900, lambda: (dlg.grab().save(shot), app.quit()))
    dlg.show()
    app.exec()


# ------------------------------------------------------------- entry
def run_headless(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="OneClickStarless")
    ap.add_argument("--model", default=default_model())
    ap.add_argument("--image", required=True, help="FITS/TIFF to process")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--extract", choices=EXTRACT_MODES, default="auto")
    ap.add_argument("--deep-clean", dest="deep_clean", action="store_true",
                    default=False)
    ap.add_argument("--no-deep-clean", dest="deep_clean",
                    action="store_false")
    ap.add_argument("--canonicalize", choices=("auto", "on", "off"),
                    default="off")
    ap.add_argument("--target-bg", type=float, default=CANON_TARGET_BG)
    ap.add_argument("--suffix", default="",
                    help="extra tag appended to output filenames")
    args = ap.parse_args(argv)

    def log(m):
        print(f"OneClickStarless: {m}", flush=True)

    sess = make_session(args.model, log)
    ext = os.path.splitext(args.image)[1].lower()
    from astropy.io import fits as afits
    roworder = "TOP-DOWN"  # numpy row 0 = image top, matching TIFF reads
    if ext in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(args.image).astype(np.float32)
        if arr.ndim == 3:
            arr = np.moveaxis(arr, 2, 0)
        else:
            arr = arr[None].repeat(3, axis=0)
        if arr.max() > 1.5:
            arr /= 65535.0
    else:
        with afits.open(args.image) as h:
            hdu = next(x for x in h if x.data is not None)
            d = hdu.data.astype(np.float32)
            roworder = hdu.header.get("ROWORDER", "BOTTOM-UP")
        arr = d if d.ndim == 3 else d[None].repeat(3, axis=0)
        if arr.max() > 1.5:
            arr /= 65535.0
    starless, stars, info = extract_star_layer(
        sess, arr, extract_mode=args.extract, deep_clean=args.deep_clean,
        canonicalize=args.canonicalize, target_bg=args.target_bg, log=log,
        progress=lambda f: print(f"\r{f*100:.0f}%", end=""))
    print()
    log(f"recomposition ({info['recombine']}) max err "
        f"{info['recomp_err']:.2e}")
    out = args.outdir or os.path.dirname(args.image)
    os.makedirs(out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0] + args.suffix
    for name, a in (("starless", starless), ("starmask", stars)):
        hdu = afits.PrimaryHDU(a)
        hdu.header["ROWORDER"] = (roworder, "stored row order")
        hdu.header["EXTRMODE"] = (info["mode"].upper(),
                                  "star layer mode")
        hdu.writeto(os.path.join(out, f"{name}_{base}.fit"), overwrite=True)
    log(f"wrote starless_{base}.fit + starmask_{base}.fit -> {out}")
    return 0


def main():
    argv = sys.argv[1:]
    if not argv and os.environ.get("ONECLICK_STARLESS_ARGS"):
        import shlex
        argv = [a.strip('"') for a in
                shlex.split(os.environ["ONECLICK_STARLESS_ARGS"], posix=False)]
    if argv:
        sys.exit(run_headless(argv))
    siril = make_siril()
    if siril is None and not os.environ.get("ONECLICK_STARLESS_MOCK"):
        print("OneClickStarless: no Siril connection - run from Siril's Scripts menu "
              "or pass --image for headless use.")
        sys.exit(1)
    if siril is not None:
        try:
            siril.cmd("requires", MIN_SIRIL)
        except s.CommandError:
            sys.exit(1)
        except Exception:
            pass
    run_gui(siril)


if __name__ == "__main__":
    main()
