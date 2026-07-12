# Starless + EasySharp training status

updated 2026-07-12 10:43

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## sharp_w32: ok (3.8 h)
- export: ok
- eval: {"psnr": 54.9395, "flux": 0.0389, "reblur": 0.001, "ident": 0.0007, "panels": ["sharp_DS.tif", "sharp_Stacked.tif", "sharp_7IV01626.tif", "sharp_7IV01627.tif", "sharp_(1)_20260520113330.tif", "sharp_(10)_20260520113355.tif"], "step": 80000}
## RUNNING: star_w64_ship (2026-07-12 10:43)
{"step": 171600, "loss": 0.00436, "starless": 0.00215, "stars": 0.00215, "fft": 0.00093, "img_s": 50.8, "lr": 1.563097307445396e-05}
{"step": 171800, "loss": 0.00982, "starless": 0.00484, "stars": 0.00484, "fft": 0.00254, "img_s": 50.8, "lr": 1.54289978172614e-05}
{"step": 172000, "loss": 0.01678, "starless": 0.00834, "stars": 0.00834, "fft": 0.00204, "img_s": 50.9, "lr": 1.5228355656330093e-05}

latest sample: `runs/star_w64_ship/samples/step0170000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
