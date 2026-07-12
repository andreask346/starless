# Starless + EasySharp training status

updated 2026-07-12 17:11

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## sharp_w32: ok (3.8 h)
- export: ok
- eval: {"psnr": 54.9395, "flux": 0.0389, "reblur": 0.001, "ident": 0.0007, "panels": ["sharp_DS.tif", "sharp_Stacked.tif", "sharp_7IV01626.tif", "sharp_7IV01627.tif", "sharp_(1)_20260520113330.tif", "sharp_(10)_20260520113355.tif"], "step": 80000}
## star_w64_ship: ok (17.5 h)
- export: ok
- eval: {"psnr_in": 57.73, "psnr_out": NaN, "leak": NaN, "recomp": 151.87, "completeness": {"faint": 0.9316, "mid": 0.9958, "bright": 0.9862}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## RUNNING: sharp_w64_ship (2026-07-12 17:11)
{"step": 41400, "loss": 0.00465, "main": 0.00325, "fft": 0.0024, "grad": 0.00266, "reblur": 0.00296, "img_s": 46.5, "lr": 0.00026948650382881595}
{"step": 41600, "loss": 0.00239, "main": 0.00162, "fft": 0.00124, "grad": 0.0019, "reblur": 0.0015, "img_s": 46.5, "lr": 0.0002692015645978558}
{"step": 41800, "loss": 0.00255, "main": 0.00176, "fft": 0.00155, "grad": 0.00193, "reblur": 0.00148, "img_s": 46.4, "lr": 0.00026891545383037464}

latest sample: `runs/sharp_w64_ship/samples/step0040000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
