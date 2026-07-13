# -*- coding: utf-8 -*-
# EasySharp — AI deconvolution / sharpening for Siril.
# Measures the PSF from the image's own stars, then sharpens with a
# PSF-conditioned model (one model adapts to any optics). Deterministic and
# faithful: a re-blur-consistency-trained residual net, no invented detail.
#
# Requires Siril >= 1.4.4 (sirilpy >= 1.0.25). Model: easysharp_*.onnx next to
# this script. Inference: ONNX Runtime (DirectML GPU / CPU fallback), 512px
# tiles, Hann-feathered.
# (c) 2026 Andreas Karaolis - GPL-3.0-or-later

import os
import sys
import time
import traceback

VERSION = "0.1.0"
MIN_SIRIL = "1.4.4"
TILE = 512
OVERLAP = 128
PSF_REF = 2.0
COND_DIM = 4

try:
    import sirilpy as s
    HAS_SIRILPY = True
except ImportError:
    s = None
    HAS_SIRILPY = False

import numpy as np


def _ensure_deps():
    try:
        import onnxruntime  # noqa: F401
        import sep          # noqa: F401
        return
    except ImportError:
        pass
    if HAS_SIRILPY:
        try:
            s.ONNXHelper().ensure_onnxruntime()
        except Exception:
            s.ensure_installed("onnxruntime-directml")
        s.ensure_installed("sep")
        return
    raise RuntimeError("onnxruntime/sep not available")


_ensure_deps()
import onnxruntime as ort  # noqa: E402
import sep                 # noqa: E402


def make_session(model_path, log):
    avail = ort.get_available_providers()
    providers = (["DmlExecutionProvider"] if "DmlExecutionProvider" in avail
                 else []) + ["CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)
    log(f"model: {os.path.basename(model_path)} on {sess.get_providers()[0]}")
    return sess


def measure_psf(lum, log):
    """Fit FWHM + elongation from the image's stars -> conditioning vector."""
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
        log("few stars detected - using a moderate default PSF")
        return np.array([1.5, 1.0, 0.1, 0.0], np.float32), 1.5 * PSF_REF, 0.1
    a = np.maximum(objs["a"], 1e-3)
    b = np.maximum(objs["b"], 1e-3)
    ok = objs["peak"] < 0.95 * float(data.max())
    a, b = a[ok], b[ok]
    fwhm = float(np.median(2.3548 * np.sqrt(a * b)))
    elong = float(np.median(1 - b / a))
    log(f"measured PSF: FWHM {fwhm:.2f}px, elongation {elong:.2f} "
        f"({len(a)} stars)")
    return (np.array([min(fwhm / PSF_REF, 4.0), 1.0, elong, 0.0], np.float32),
            fwhm, elong)


def hann2d(size):
    w = np.hanning(size)
    return (np.outer(w, w) + 1e-4).astype(np.float32)


def sharpen(sess, img, cond, strength=1.0, progress=None):
    """img (3,H,W) 0..1 -> sharpened (3,H,W). strength scales the fwhm-ratio
    conditioning: 1.0 = measured, <1 gentler, >1 stronger."""
    c, h, w = img.shape
    cvec = cond.copy()
    cvec[0] = 1.0 + (cvec[0] - 1.0) * strength
    cvec = cvec[None].astype(np.float32)
    step = TILE - OVERLAP
    pad_h = (step - (h - TILE) % step) % step if h > TILE else TILE - h
    pad_w = (step - (w - TILE) % step) % step if w > TILE else TILE - w
    padded = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    ph, pw = padded.shape[1:]
    out = np.zeros_like(padded)
    weight = np.zeros((ph, pw), np.float32)
    win = hann2d(TILE)
    ys = list(range(0, ph - TILE + 1, step))
    xs = list(range(0, pw - TILE + 1, step))
    k, total = 0, len(ys) * len(xs)
    for y in ys:
        for x in xs:
            tile = padded[None, :, y:y + TILE, x:x + TILE].astype(np.float32)
            pred = sess.run(None, {"input": tile, "cond": cvec})[0][0]
            out[:, y:y + TILE, x:x + TILE] += pred * win
            weight[y:y + TILE, x:x + TILE] += win
            k += 1
            if progress:
                progress(k / total)
    out /= weight[None]
    return np.clip(out[:, :h, :w], 0, 1)


def default_model():
    d = os.path.dirname(os.path.abspath(__file__))
    c = sorted([f for f in os.listdir(d) if f.startswith("easysharp")
                and f.endswith(".onnx")], reverse=True)
    return os.path.join(d, c[0]) if c else ""


def make_siril():
    if not HAS_SIRILPY:
        return None
    try:
        iface = s.SirilInterface()
        iface.connect()
        return iface
    except Exception:
        return None


def run_on_loaded(siril, model_path, strength, correct_only, replace,
                  log, prog):
    t0 = time.time()
    sess = make_session(model_path, log)
    fit = siril.get_image(with_pixels=True)
    fit.ensure_data_type(np.float32)
    data = np.asarray(fit.data, dtype=np.float32)
    mono = data.ndim == 2
    if mono:
        data = data[None].repeat(3, axis=0)
    cond, fwhm, elong = measure_psf(data.mean(axis=0), log)
    if correct_only:                       # aberration correction, no sharpen
        cond[0] = 1.0
    log(f"sharpening ({data.shape[2]}x{data.shape[1]}, strength {strength})...")
    out = sharpen(sess, data, cond, strength=strength,
                  progress=lambda f: prog(0.1 + 0.85 * f, "sharpening"))
    # honesty report: flux preservation
    flux_dev = abs(float(out.sum() / max(data.sum(), 1e-6)) - 1.0) * 100
    log(f"flux change {flux_dev:.2f}% (deconvolution conserves total signal)")
    result = out[0] if mono else out
    wd = siril.get_siril_wd()
    base = os.path.splitext(siril.get_image_filename() or "image")[0]
    hdr = ""
    try:
        hdr = siril.get_image_fits_header(return_as="str") or ""
    except Exception:
        pass
    outputs = []
    p = os.path.join(wd, f"sharp_{os.path.basename(base)}.fit")
    siril.save_image_file(result.astype(np.float32), header=hdr, filename=p)
    outputs.append(p)
    log(f"saved {p}")
    if replace:
        with siril.image_lock():
            siril.undo_save_state("EasySharp deconvolution")
            siril.set_image_pixeldata(result.astype(np.float32))
        log("loaded image replaced (undo available)")
    prog(1.0, "done")
    log(f"done in {time.time() - t0:.1f}s")
    return outputs


def run_gui(siril):
    try:
        import PyQt6  # noqa: F401
    except ImportError:
        if HAS_SIRILPY:
            s.ensure_installed("PyQt6")
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QLineEdit, QCheckBox, QProgressBar, QFileDialog, QMessageBox, QSlider)

    def log(msg):
        if siril is not None:
            try:
                siril.log(f"EasySharp: {msg}"[:1000])
            except Exception:
                pass
        print(f"EasySharp: {msg}")

    class Worker(QThread):
        progressed = pyqtSignal(float, str)
        done_sig = pyqtSignal(object)

        def __init__(self, mp, opts):
            super().__init__()
            self.mp, self.opts = mp, opts

        def run(self):
            try:
                def prog(f, m):
                    self.progressed.emit(f, m)
                    if siril is not None:
                        try:
                            siril.update_progress(m, min(max(f, 0), 1))
                        except Exception:
                            pass
                outs = run_on_loaded(siril, self.mp, *self.opts, log, prog)
                self.done_sig.emit(outs)
            except Exception as e:
                log(f"fatal: {e}\n{traceback.format_exc()}")
                self.done_sig.emit(None)

    class Dlg(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"EasySharp {VERSION} - AI deconvolution")
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
            v.addWidget(QLabel("PSF is measured automatically from the "
                               "image's stars. Runs on the loaded image."))
            h = QHBoxLayout()
            h.addWidget(QLabel("Strength"))
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(0, 150)
            self.slider.setValue(100)
            self.slval = QLabel("1.00")
            self.slider.valueChanged.connect(
                lambda x: self.slval.setText(f"{x/100:.2f}"))
            h.addWidget(self.slider)
            h.addWidget(self.slval)
            v.addLayout(h)
            self.cb_correct = QCheckBox("Correct aberrations only (no sharpen)")
            self.cb_replace = QCheckBox("Replace loaded image (undoable)")
            self.cb_replace.setChecked(True)
            v.addWidget(self.cb_correct)
            v.addWidget(self.cb_replace)
            self.pbar = QProgressBar()
            self.pbar.setRange(0, 1000)
            v.addWidget(self.pbar)
            self.status = QLabel("Ready.")
            self.status.setWordWrap(True)
            v.addWidget(self.status)
            h = QHBoxLayout()
            h.addStretch()
            self.run_btn = QPushButton("Sharpen")
            self.run_btn.clicked.connect(self.go)
            h.addWidget(self.run_btn)
            v.addLayout(h)

        def pick_model(self):
            p, _ = QFileDialog.getOpenFileName(self, "EasySharp model", "",
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
                QMessageBox.warning(self, "EasySharp", "Pick a .onnx model.")
                return
            if siril is None or not siril.is_image_loaded():
                QMessageBox.warning(self, "EasySharp",
                                    "Load an image in Siril first.")
                return
            self.run_btn.setEnabled(False)
            self.worker = Worker(mp, (self.slider.value() / 100.0,
                                      self.cb_correct.isChecked(),
                                      self.cb_replace.isChecked()))
            self.worker.progressed.connect(
                lambda f, m: (self.pbar.setValue(int(f * 1000)),
                              self.status.setText(m)))
            self.worker.done_sig.connect(self._on_done)
            self.worker.start()

        def _on_done(self, outs):
            self.run_btn.setEnabled(True)
            self.status.setText("failed - see Siril log" if outs is None
                                else f"done: {len(outs)} file(s) written")

    app = QApplication.instance() or QApplication(sys.argv)
    dlg = Dlg()
    shot = os.environ.get("EASYSHARP_SCREENSHOT")
    if shot:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(900, lambda: (dlg.grab().save(shot), app.quit()))
    dlg.show()
    app.exec()


def run_headless(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="EasySharp")
    ap.add_argument("--model", default=default_model())
    ap.add_argument("--image", required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--correct-only", action="store_true")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args(argv)

    def log(m):
        print(f"EasySharp: {m}", flush=True)

    sess = make_session(args.model, log)
    ext = os.path.splitext(args.image)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(args.image).astype(np.float32)
        arr = np.moveaxis(arr, 2, 0) if arr.ndim == 3 else arr[None].repeat(3, 0)
        if arr.max() > 1.5:
            arr /= 65535.0
    else:
        from astropy.io import fits as afits
        with afits.open(args.image) as h:
            d = next(x for x in h if x.data is not None).data.astype(np.float32)
        arr = d if d.ndim == 3 else d[None].repeat(3, axis=0)
        if arr.max() > 1.5:
            arr /= 65535.0
    cond, _, _ = measure_psf(arr.mean(axis=0), log)
    if args.correct_only:
        cond[0] = 1.0
    out = sharpen(sess, arr, cond, strength=args.strength,
                  progress=lambda f: print(f"\r{f*100:.0f}%", end=""))
    print()
    from astropy.io import fits as afits
    base = os.path.splitext(os.path.basename(args.image))[0]
    d = args.outdir or os.path.dirname(args.image)
    os.makedirs(d, exist_ok=True)
    afits.PrimaryHDU(out).writeto(os.path.join(d, f"sharp_{base}.fit"),
                                  overwrite=True)
    log(f"wrote sharp_{base}.fit -> {d}")
    return 0


def main():
    argv = sys.argv[1:]
    if not argv and os.environ.get("EASYSHARP_ARGS"):
        import shlex
        argv = [a.strip('"') for a in
                shlex.split(os.environ["EASYSHARP_ARGS"], posix=False)]
    if argv:
        sys.exit(run_headless(argv))
    siril = make_siril()
    if siril is None and not os.environ.get("EASYSHARP_MOCK"):
        print("EasySharp: no Siril connection - run from Scripts menu or pass "
              "--image for headless use.")
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
