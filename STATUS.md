# Starless + EasySharp training status

updated 2026-07-13 03:42

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
## RUNNING: sharp_w64_ship (2026-07-13 03:42)
{"step": 151000, "loss": 0.00245, "main": 0.00169, "fft": 0.00143, "grad": 0.00166, "reblur": 0.00152, "img_s": 46.3, "lr": 4.3140118484407524e-05}
{"step": 151200, "loss": 0.00276, "main": 0.0019, "fft": 0.00179, "grad": 0.0024, "reblur": 0.00145, "img_s": 46.4, "lr": 4.2813800779262146e-05}
{"step": 151400, "loss": 0.00188, "main": 0.00131, "fft": 0.0007, "grad": 0.00137, "reblur": 0.00121, "img_s": 46.2, "lr": 4.248854589342842e-05}

latest sample: `runs/sharp_w64_ship/samples/step0150000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
