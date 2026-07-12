# Starless + EasySharp training status

updated 2026-07-12 21:42

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
## RUNNING: sharp_w64_ship (2026-07-12 21:42)
{"step": 88400, "loss": 0.00233, "main": 0.00158, "fft": 0.00131, "grad": 0.00177, "reblur": 0.00149, "img_s": 46.6, "lr": 0.00017759026166226478}
{"step": 88600, "loss": 0.00225, "main": 0.00155, "fft": 0.00116, "grad": 0.00192, "reblur": 0.00132, "img_s": 46.5, "lr": 0.0001771282358945071}
{"step": 88800, "loss": 0.0027, "main": 0.00182, "fft": 0.00182, "grad": 0.00198, "reblur": 0.00167, "img_s": 46.5, "lr": 0.0001766659473168116}

latest sample: `runs/sharp_w64_ship/samples/step0085000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
