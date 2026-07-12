# Starless + EasySharp training status

updated 2026-07-12 07:43

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## sharp_w32: ok (3.8 h)
- export: ok
- eval: {"psnr": 54.9395, "flux": 0.0389, "reblur": 0.001, "ident": 0.0007, "panels": ["sharp_DS.tif", "sharp_Stacked.tif", "sharp_7IV01626.tif", "sharp_7IV01627.tif", "sharp_(1)_20260520113330.tif", "sharp_(10)_20260520113355.tif"], "step": 80000}
## RUNNING: star_w64_ship (2026-07-12 07:43)
{"step": 137200, "loss": 0.0041, "starless": 0.00203, "stars": 0.00203, "fft": 0.0008, "img_s": 50.9, "lr": 6.802894234105406e-05}
{"step": 137400, "loss": 0.02429, "starless": 0.01206, "stars": 0.01206, "fft": 0.00357, "img_s": 51.0, "lr": 6.763760992886871e-05}
{"step": 137600, "loss": 0.00781, "starless": 0.00387, "stars": 0.00387, "fft": 0.00144, "img_s": 51.0, "lr": 6.724709533502023e-05}

latest sample: `runs/star_w64_ship/samples/step0135000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
