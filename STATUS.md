# Starless + EasySharp training status

updated 2026-07-12 20:42

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
## RUNNING: sharp_w64_ship (2026-07-12 20:42)
{"step": 78000, "loss": 0.00256, "main": 0.00172, "fft": 0.00143, "grad": 0.0023, "reblur": 0.00156, "img_s": 46.4, "lr": 0.0002011413190766793}
{"step": 78200, "loss": 0.00282, "main": 0.00191, "fft": 0.00179, "grad": 0.00214, "reblur": 0.00174, "img_s": 46.5, "lr": 0.000200699168214199}
{"step": 78400, "loss": 0.00211, "main": 0.00148, "fft": 0.00096, "grad": 0.00146, "reblur": 0.0013, "img_s": 46.5, "lr": 0.00020025652190619497}

latest sample: `runs/sharp_w64_ship/samples/step0075000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
