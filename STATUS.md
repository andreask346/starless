# Starless + EasySharp training status

updated 2026-07-12 09:43

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## sharp_w32: ok (3.8 h)
- export: ok
- eval: {"psnr": 54.9395, "flux": 0.0389, "reblur": 0.001, "ident": 0.0007, "panels": ["sharp_DS.tif", "sharp_Stacked.tif", "sharp_7IV01626.tif", "sharp_7IV01627.tif", "sharp_(1)_20260520113330.tif", "sharp_(10)_20260520113355.tif"], "step": 80000}
## RUNNING: star_w64_ship (2026-07-12 09:43)
{"eval_step": 160000, "psnr_stars": 43.76, "psnr_bg": 67.59}
{"step": 160200, "loss": 0.01474, "starless": 0.00732, "stars": 0.00732, "fft": 0.00179, "img_s": 49.1, "lr": 2.9276492665562018e-05}
{"step": 160400, "loss": 0.00528, "starless": 0.00262, "stars": 0.00262, "fft": 0.00076, "img_s": 51.2, "lr": 2.9002222417254252e-05}

latest sample: `runs/star_w64_ship/samples/step0160000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
