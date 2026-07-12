# Starless + EasySharp training status

updated 2026-07-12 09:13

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## sharp_w32: ok (3.8 h)
- export: ok
- eval: {"psnr": 54.9395, "flux": 0.0389, "reblur": 0.001, "ident": 0.0007, "panels": ["sharp_DS.tif", "sharp_Stacked.tif", "sharp_7IV01626.tif", "sharp_7IV01627.tif", "sharp_(1)_20260520113330.tif", "sharp_(10)_20260520113355.tif"], "step": 80000}
## RUNNING: star_w64_ship (2026-07-12 09:13)
{"step": 154400, "loss": 0.03763, "starless": 0.01867, "stars": 0.01867, "fft": 0.00599, "img_s": 51.0, "lr": 3.7739418579956116e-05}
{"step": 154600, "loss": 0.03453, "starless": 0.01716, "stars": 0.01716, "fft": 0.00407, "img_s": 51.0, "lr": 3.743159815240853e-05}
{"step": 154800, "loss": 0.03042, "starless": 0.01509, "stars": 0.01509, "fft": 0.00471, "img_s": 51.0, "lr": 3.7124893664339665e-05}

latest sample: `runs/star_w64_ship/samples/step0150000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
