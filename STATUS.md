# Starless + EasySharp training status

updated 2026-07-12 01:43

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## sharp_w32: ok (3.8 h)
- export: ok
- eval: {"psnr": 54.9395, "flux": 0.0389, "reblur": 0.001, "ident": 0.0007, "panels": ["sharp_DS.tif", "sharp_Stacked.tif", "sharp_7IV01626.tif", "sharp_7IV01627.tif", "sharp_(1)_20260520113330.tif", "sharp_(10)_20260520113355.tif"], "step": 80000}
## RUNNING: star_w64_ship (2026-07-12 01:43)
{"step": 68200, "loss": 0.00869, "starless": 0.0043, "stars": 0.0043, "fft": 0.00163, "img_s": 51.0, "lr": 0.00022211024624540154}
{"step": 68400, "loss": 0.00888, "starless": 0.0044, "stars": 0.0044, "fft": 0.0016, "img_s": 51.0, "lr": 0.00022169761144836644}
{"step": 68600, "loss": 0.02295, "starless": 0.01139, "stars": 0.01139, "fft": 0.0036, "img_s": 51.1, "lr": 0.00022128427395965005}

latest sample: `runs/star_w64_ship/samples/step0065000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
