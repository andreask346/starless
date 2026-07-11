# Starless + EasySharp training status

updated 2026-07-11 20:42

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## sharp_w32: ok (3.8 h)
- export: ok
- eval: {"psnr": 54.9395, "flux": 0.0389, "reblur": 0.001, "ident": 0.0007, "panels": ["sharp_DS.tif", "sharp_Stacked.tif", "sharp_7IV01626.tif", "sharp_7IV01627.tif", "sharp_(1)_20260520113330.tif", "sharp_(10)_20260520113355.tif"], "step": 80000}
## RUNNING: star_w64_ship (2026-07-11 20:42)
{"step": 10800, "loss": 0.02341, "starless": 0.01164, "stars": 0.01164, "fft": 0.00268, "img_s": 50.9, "lr": 0.00029785386697477734}
{"step": 11000, "loss": 0.02369, "starless": 0.01171, "stars": 0.01171, "fft": 0.00554, "img_s": 50.9, "lr": 0.00029777384426014073}
{"step": 11200, "loss": 0.01541, "starless": 0.00763, "stars": 0.00763, "fft": 0.00299, "img_s": 50.9, "lr": 0.0002976923680121179}

latest sample: `runs/star_w64_ship/samples/step0010000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
