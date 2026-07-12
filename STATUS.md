# Starless + EasySharp training status

updated 2026-07-12 10:13

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## sharp_w32: ok (3.8 h)
- export: ok
- eval: {"psnr": 54.9395, "flux": 0.0389, "reblur": 0.001, "ident": 0.0007, "panels": ["sharp_DS.tif", "sharp_Stacked.tif", "sharp_7IV01626.tif", "sharp_7IV01627.tif", "sharp_(1)_20260520113330.tif", "sharp_(10)_20260520113355.tif"], "step": 80000}
## RUNNING: star_w64_ship (2026-07-12 10:13)
{"step": 165800, "loss": 0.01526, "starless": 0.00758, "stars": 0.00758, "fft": 0.00212, "img_s": 51.1, "lr": 2.2058782099497525e-05}
{"step": 166000, "loss": 0.00989, "starless": 0.00491, "stars": 0.00491, "fft": 0.00156, "img_s": 51.0, "lr": 2.181906696291171e-05}
{"step": 166200, "loss": 0.01924, "starless": 0.00953, "stars": 0.00953, "fft": 0.00339, "img_s": 51.0, "lr": 2.1580621855184324e-05}

latest sample: `runs/star_w64_ship/samples/step0165000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
