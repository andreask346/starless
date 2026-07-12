# Assemble the deliverable once training finishes:
#  - install the best ONNX models next to the Siril scripts (so Starless.py /
#    EasySharp.py work out of the box), preferring the w64 ship models
#  - render every real-frame before/after panel to viewable JPGs
#  - write results/RESULTS.md (metrics table + how to try the tools)
#
#   .venv\Scripts\python.exe results\package.py

import glob
import json
import os
import shutil

import numpy as np
import tifffile
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, "scripts")
OUT = os.path.join(ROOT, "results")
os.makedirs(os.path.join(OUT, "panels"), exist_ok=True)


def newest_onnx(run):
    c = glob.glob(os.path.join(ROOT, "runs", run, "*.onnx"))
    return c[0] if c else None


def install_models():
    installed = []
    # star: prefer w64 ship, fall back to w32
    for run, dest in (("star_w64_ship", "starless_w64.onnx"),
                      ("star_w32", "starless_w32.onnx")):
        o = newest_onnx(run)
        if o:
            shutil.copy(o, os.path.join(SCRIPTS, dest))
            installed.append(dest)
    for run, dest in (("sharp_w64_ship", "easysharp_w64.onnx"),
                      ("sharp_w32", "easysharp_w32.onnx")):
        o = newest_onnx(run)
        if o:
            shutil.copy(o, os.path.join(SCRIPTS, dest))
            installed.append(dest)
    return installed


def render_panels():
    imgs = []
    for run in ("star_w64_ship", "sharp_w64_ship"):
        for p in sorted(glob.glob(os.path.join(ROOT, "runs", run, "panel",
                                               "*.tif"))):
            a = tifffile.imread(p).astype(np.float32)
            a = a / a.max() if a.max() > 1.5 else a
            a = np.clip(a, 0, 1) ** 0.5
            name = f"{run}__{os.path.splitext(os.path.basename(p))[0]}.jpg"
            w = 1800
            im = Image.fromarray((a * 255).astype("uint8"))
            im = im.resize((w, int(im.height * w / im.width)))
            im.save(os.path.join(OUT, "panels", name), quality=85)
            imgs.append((run, name))
    return imgs


def metrics(run):
    p = os.path.join(ROOT, "runs", run, "metrics.json")
    if os.path.isfile(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}
    return {}


def write_results(installed, panels):
    sm = metrics("star_w64_ship")
    sm32 = metrics("star_w32")
    dm = metrics("sharp_w64_ship")
    dm32 = metrics("sharp_w32")
    lines = ["# Starless + EasySharp — training results\n",
             "Trained locally on the RTX 4090. Two Siril AI tools:\n",
             "- **Starless** — star removal (starless + recomposable stars layer)",
             "- **EasySharp** — PSF-aware deconvolution / sharpening\n",
             "## Star removal (completeness = fraction of star flux removed)\n",
             "| model | faint | mid | bright | PSNR |",
             "|---|---|---|---|---|"]
    for name, m in (("w32 baseline", sm32), ("w64 SHIP", sm)):
        c = m.get("completeness", {})
        lines.append(f"| {name} | {c.get('faint','-')} | {c.get('mid','-')} "
                     f"| {c.get('bright','-')} | {m.get('psnr_in','-')} |")
    lines += ["\n## Deconvolution (flux err %, no-hallucination identity)\n",
              "| model | PSNR | flux_err | identity |",
              "|---|---|---|---|"]
    for name, m in (("w32 baseline", dm32), ("w64 SHIP", dm)):
        lines.append(f"| {name} | {m.get('psnr','-')} | {m.get('flux','-')} "
                     f"| {m.get('ident','-')} |")
    lines += ["\n## Installed models", ""]
    for i in installed:
        lines.append(f"- `scripts/{i}`")
    lines += ["\nThe Siril scripts auto-load the newest matching .onnx next to "
              "them, so Starless/EasySharp work out of the box.\n",
              "## Real-frame before/after panels\n",
              "In `results/panels/` — star panels are input | starless | "
              "stars; sharp panels are input | sharpened.\n",
              "**Honest note:** on real frames the star model removes bright/"
              "medium stars cleanly; some faint residuals remain (a mix of "
              "missed faint stars and sensor noise). Judge against StarNet on "
              "the same frame; the next iteration is tuning the synthetic "
              "faint-star/noise recipe to match your sensor.\n"]
    for run, name in panels:
        lines.append(f"### {name}\n\n![{name}](panels/{name})\n")
    with open(os.path.join(OUT, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    installed = install_models()
    panels = render_panels()
    write_results(installed, panels)
    print("installed:", installed)
    print("panels:", len(panels))
    print("wrote results/RESULTS.md")


if __name__ == "__main__":
    main()
