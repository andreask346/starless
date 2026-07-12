# Starless + EasySharp training status

updated 2026-07-12 12:44

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## sharp_w32: ok (3.8 h)
- export: ok
- eval: {"psnr": 54.9395, "flux": 0.0389, "reblur": 0.001, "ident": 0.0007, "panels": ["sharp_DS.tif", "sharp_Stacked.tif", "sharp_7IV01626.tif", "sharp_7IV01627.tif", "sharp_(1)_20260520113330.tif", "sharp_(10)_20260520113355.tif"], "step": 80000}
## RUNNING: star_w64_ship (2026-07-12 12:44)
{"step": 194400, "loss": 0.0252, "starless": 0.01253, "stars": 0.01253, "fft": 0.00259, "img_s": 50.8, "lr": 1.5780254330959542e-06}
{"step": 194600, "loss": 0.00394, "starless": 0.00194, "stars": 0.00194, "fft": 0.00105, "img_s": 50.8, "lr": 1.5374994961336645e-06}
{"step": 194800, "loss": 0.01276, "starless": 0.00633, "stars": 0.00633, "fft": 0.00195, "img_s": 50.8, "lr": 1.4984437589127441e-06}

latest sample: `runs/star_w64_ship/samples/step0190000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
