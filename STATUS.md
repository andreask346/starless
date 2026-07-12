# Starless + EasySharp training status

updated 2026-07-12 12:14

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## sharp_w32: ok (3.8 h)
- export: ok
- eval: {"psnr": 54.9395, "flux": 0.0389, "reblur": 0.001, "ident": 0.0007, "panels": ["sharp_DS.tif", "sharp_Stacked.tif", "sharp_7IV01626.tif", "sharp_7IV01627.tif", "sharp_(1)_20260520113330.tif", "sharp_(10)_20260520113355.tif"], "step": 80000}
## RUNNING: star_w64_ship (2026-07-12 12:14)
{"step": 188800, "loss": 0.0092, "starless": 0.00456, "stars": 0.00456, "fft": 0.00161, "img_s": 50.8, "lr": 3.307631987884508e-06}
{"step": 189000, "loss": 0.01032, "starless": 0.00513, "stars": 0.00513, "fft": 0.00118, "img_s": 50.8, "lr": 3.2261557398620273e-06}
{"step": 189200, "loss": 0.0074, "starless": 0.00366, "stars": 0.00366, "fft": 0.00175, "img_s": 50.9, "lr": 3.146133025225526e-06}

latest sample: `runs/star_w64_ship/samples/step0185000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
