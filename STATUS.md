# Starless + EasySharp training status

updated 2026-07-13 01:42

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
## RUNNING: sharp_w64_ship (2026-07-13 01:42)
{"step": 130200, "loss": 0.00219, "main": 0.0015, "fft": 0.00106, "grad": 0.00168, "reblur": 0.00138, "img_s": 45.2, "lr": 8.221027856702759e-05}
{"step": 130400, "loss": 0.00197, "main": 0.00138, "fft": 0.00088, "grad": 0.00134, "reblur": 0.00121, "img_s": 46.3, "lr": 8.17928108370929e-05}
{"step": 130600, "loss": 0.00244, "main": 0.00165, "fft": 0.00143, "grad": 0.00201, "reblur": 0.0015, "img_s": 46.3, "lr": 8.137602121937695e-05}

latest sample: `runs/sharp_w64_ship/samples/step0130000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
