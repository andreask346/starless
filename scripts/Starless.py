# -*- coding: utf-8 -*-
# Starless — AI star removal for Siril.
# Removes stars from the loaded image (or a file), returning a starless image
# and a star layer with EXACT recomposition: starless + stars == input.
# Inference: ONNX Runtime (DirectML GPU when available, CPU fallback),
# 512px tiles with Hann-feathered blending.
#
# Requires Siril >= 1.4.4 (sirilpy >= 1.0.25). Model weights: starless_*.onnx
# next to this script, or picked via the GUI.
# (c) 2026 Andreas Karaolis - GPL-3.0-or-later

import os
import sys
import time
import traceback

VERSION = "0.1.0"
MIN_SIRIL = "1.4.4"
TILE = 512
OVERLAP = 128

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
                  replace_loaded, log, prog):
    t0 = time.time()
    sess = make_session(model_path, log)
    data, mono = get_loaded_image(siril)
    log(f"removing stars from loaded image "
        f"({data.shape[2]}x{data.shape[1]})...")
    stars = remove_stars(sess, data,
                         progress=lambda f: prog(0.1 + 0.8 * f,
                                                 "removing stars"))
    starless = data - stars
    # exact-recomposition + honesty check
    recomp = float(np.abs((starless + stars) - data).max())
    log(f"recomposition max err {recomp:.2e} (exact by construction)")

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
            siril.undo_save_state("Starless star removal")
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
        QLineEdit, QCheckBox, QProgressBar, QFileDialog, QMessageBox)

    def log(msg):
        if siril is not None:
            try:
                siril.log(f"Starless: {msg}"[:1000])
            except Exception:
                pass
        print(f"Starless: {msg}")

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
            self.setWindowTitle(f"Starless {VERSION} - AI star removal")
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
            self.cb_starless = QCheckBox("Save starless FITS")
            self.cb_stars = QCheckBox("Save star layer FITS "
                                      "(starless + stars = original, exact)")
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
                QMessageBox.warning(self, "Starless",
                                    "Pick a .onnx model file.")
                return
            if siril is None or not siril.is_image_loaded():
                QMessageBox.warning(self, "Starless",
                                    "Load an image in Siril first.")
                return
            self.run_btn.setEnabled(False)
            self.worker = Worker(mp, (self.cb_starless.isChecked(),
                                      self.cb_stars.isChecked(),
                                      self.cb_replace.isChecked()))
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
    shot = os.environ.get("STARLESS_SCREENSHOT")
    if shot:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(900, lambda: (dlg.grab().save(shot), app.quit()))
    dlg.show()
    app.exec()


# ------------------------------------------------------------- entry
def run_headless(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="Starless")
    ap.add_argument("--model", default=default_model())
    ap.add_argument("--image", required=True, help="FITS/TIFF to process")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args(argv)

    def log(m):
        print(f"Starless: {m}", flush=True)

    sess = make_session(args.model, log)
    ext = os.path.splitext(args.image)[1].lower()
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
        from astropy.io import fits as afits
        with afits.open(args.image) as h:
            d = next(x for x in h if x.data is not None).data.astype(np.float32)
        arr = d if d.ndim == 3 else d[None].repeat(3, axis=0)
        if arr.max() > 1.5:
            arr /= 65535.0
    stars = remove_stars(sess, arr,
                         progress=lambda f: print(f"\r{f*100:.0f}%", end=""))
    print()
    starless = arr - stars
    out = args.outdir or os.path.dirname(args.image)
    base = os.path.splitext(os.path.basename(args.image))[0]
    from astropy.io import fits as afits
    afits.PrimaryHDU(starless).writeto(
        os.path.join(out, f"starless_{base}.fit"), overwrite=True)
    afits.PrimaryHDU(stars).writeto(
        os.path.join(out, f"starmask_{base}.fit"), overwrite=True)
    log(f"wrote starless_{base}.fit + starmask_{base}.fit -> {out}")
    return 0


def main():
    argv = sys.argv[1:]
    if not argv and os.environ.get("STARLESS_ARGS"):
        import shlex
        argv = [a.strip('"') for a in
                shlex.split(os.environ["STARLESS_ARGS"], posix=False)]
    if argv:
        sys.exit(run_headless(argv))
    siril = make_siril()
    if siril is None and not os.environ.get("STARLESS_MOCK"):
        print("Starless: no Siril connection - run from Siril's Scripts menu "
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
