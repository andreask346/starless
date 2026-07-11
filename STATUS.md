# Starless + EasySharp training status

updated 2026-07-11 22:12

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## sharp_w32: ok (3.8 h)
- export: ok
- eval: {"psnr": 54.9395, "flux": 0.0389, "reblur": 0.001, "ident": 0.0007, "panels": ["sharp_DS.tif", "sharp_Stacked.tif", "sharp_7IV01626.tif", "sharp_7IV01627.tif", "sharp_(1)_20260520113330.tif", "sharp_(10)_20260520113355.tif"], "step": 80000}
## RUNNING: star_w64_ship (2026-07-11 22:12)
{"step": 28000, "loss": 0.01145, "starless": 0.00566, "stars": 0.00566, "fft": 0.00244, "img_s": 51.1, "lr": 0.00028577164434367253}
{"step": 28200, "loss": 0.01333, "starless": 0.00662, "stars": 0.00662, "fft": 0.0019, "img_s": 51.0, "lr": 0.0002855710021827416}
{"step": 28400, "loss": 0.00417, "starless": 0.00206, "stars": 0.00206, "fft": 0.00095, "img_s": 51.1, "lr": 0.00028536902692554903}

latest sample: `runs/star_w64_ship/samples/step0025000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
