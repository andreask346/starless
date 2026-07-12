# Starless + EasySharp training status

updated 2026-07-12 07:13

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## sharp_w32: ok (3.8 h)
- export: ok
- eval: {"psnr": 54.9395, "flux": 0.0389, "reblur": 0.001, "ident": 0.0007, "panels": ["sharp_DS.tif", "sharp_Stacked.tif", "sharp_7IV01626.tif", "sharp_7IV01627.tif", "sharp_(1)_20260520113330.tif", "sharp_(10)_20260520113355.tif"], "step": 80000}
## RUNNING: star_w64_ship (2026-07-12 07:13)
{"step": 131400, "loss": 0.00961, "starless": 0.00476, "stars": 0.00476, "fft": 0.00171, "img_s": 51.0, "lr": 7.971572604036434e-05}
{"step": 131600, "loss": 0.01519, "starless": 0.00754, "stars": 0.00754, "fft": 0.00213, "img_s": 51.0, "lr": 7.930238855164802e-05}
{"step": 131800, "loss": 0.00542, "starless": 0.00268, "stars": 0.00268, "fft": 0.00137, "img_s": 51.0, "lr": 7.888975375461318e-05}

latest sample: `runs/star_w64_ship/samples/step0130000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
