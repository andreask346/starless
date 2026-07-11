# Starless + EasySharp training status

updated 2026-07-11 23:12

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## sharp_w32: ok (3.8 h)
- export: ok
- eval: {"psnr": 54.9395, "flux": 0.0389, "reblur": 0.001, "ident": 0.0007, "panels": ["sharp_DS.tif", "sharp_Stacked.tif", "sharp_7IV01626.tif", "sharp_7IV01627.tif", "sharp_(1)_20260520113330.tif", "sharp_(10)_20260520113355.tif"], "step": 80000}
## RUNNING: star_w64_ship (2026-07-11 23:12)
{"step": 39800, "loss": 0.01288, "starless": 0.00638, "stars": 0.00638, "fft": 0.00234, "img_s": 51.1, "lr": 0.00027172350733444407}
{"step": 40000, "loss": 0.03229, "starless": 0.01605, "stars": 0.01605, "fft": 0.00388, "img_s": 51.1, "lr": 0.0002714480406590584}
{"eval_step": 40000, "psnr_stars": 41.55, "psnr_bg": 69.54}

latest sample: `runs/star_w64_ship/samples/step0040000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
