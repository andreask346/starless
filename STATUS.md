# Starless + EasySharp training status

updated 2026-07-13 08:13

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
## RUNNING: sharp_w64_ship (2026-07-13 08:13)
{"step": 198000, "loss": 0.0028, "main": 0.00188, "fft": 0.00174, "grad": 0.00215, "reblur": 0.00177, "img_s": 46.5, "lr": 1.073769225323158e-06}
{"step": 198200, "loss": 0.00205, "main": 0.00141, "fft": 0.0009, "grad": 0.00145, "reblur": 0.00135, "img_s": 46.2, "lr": 1.059754006286674e-06}
{"step": 198400, "loss": 0.00204, "main": 0.0014, "fft": 0.00094, "grad": 0.00158, "reblur": 0.00131, "img_s": 46.3, "lr": 1.0472137021466793e-06}

latest sample: `runs/sharp_w64_ship/samples/step0195000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
