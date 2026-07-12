# Starless + EasySharp training status

updated 2026-07-12 23:12

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
## RUNNING: sharp_w64_ship (2026-07-12 23:12)
{"step": 104000, "loss": 0.00339, "main": 0.00207, "fft": 0.00213, "grad": 0.00176, "reblur": 0.00309, "img_s": 46.5, "lr": 0.00014111281733037383}
{"step": 104200, "loss": 0.00206, "main": 0.00142, "fft": 0.001, "grad": 0.00144, "reblur": 0.0013, "img_s": 46.5, "lr": 0.00014064412310627497}
{"step": 104400, "loss": 0.00249, "main": 0.00168, "fft": 0.00129, "grad": 0.00157, "reblur": 0.00176, "img_s": 46.4, "lr": 0.0001401755261557023}

latest sample: `runs/sharp_w64_ship/samples/step0100000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
