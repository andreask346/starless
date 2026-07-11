# Starless + EasySharp training status

updated 2026-07-11 22:42

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## sharp_w32: ok (3.8 h)
- export: ok
- eval: {"psnr": 54.9395, "flux": 0.0389, "reblur": 0.001, "ident": 0.0007, "panels": ["sharp_DS.tif", "sharp_Stacked.tif", "sharp_7IV01626.tif", "sharp_7IV01627.tif", "sharp_(1)_20260520113330.tif", "sharp_(10)_20260520113355.tif"], "step": 80000}
## RUNNING: star_w64_ship (2026-07-11 22:42)
{"step": 33800, "loss": 0.0196, "starless": 0.00974, "stars": 0.00974, "fft": 0.00228, "img_s": 51.1, "lr": 0.00027941937814482143}
{"step": 34000, "loss": 0.01188, "starless": 0.0059, "stars": 0.0059, "fft": 0.00183, "img_s": 51.1, "lr": 0.00027918093303709406}
{"step": 34200, "loss": 0.0169, "starless": 0.00838, "stars": 0.00838, "fft": 0.00259, "img_s": 51.1, "lr": 0.00027894121790050834}

latest sample: `runs/star_w64_ship/samples/step0030000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
