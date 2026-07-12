# Starless + EasySharp training status

updated 2026-07-12 03:13

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## sharp_w32: ok (3.8 h)
- export: ok
- eval: {"psnr": 54.9395, "flux": 0.0389, "reblur": 0.001, "ident": 0.0007, "panels": ["sharp_DS.tif", "sharp_Stacked.tif", "sharp_7IV01626.tif", "sharp_7IV01627.tif", "sharp_(1)_20260520113330.tif", "sharp_(10)_20260520113355.tif"], "step": 80000}
## RUNNING: star_w64_ship (2026-07-12 03:13)
{"step": 85400, "loss": 0.01919, "starless": 0.00951, "stars": 0.00951, "fft": 0.00329, "img_s": 51.1, "lr": 0.0001844860167391809}
{"step": 85600, "loss": 0.05649, "starless": 0.02806, "stars": 0.02806, "fft": 0.00751, "img_s": 51.2, "lr": 0.0001840284787619398}
{"step": 85800, "loss": 0.00737, "starless": 0.00365, "stars": 0.00365, "fft": 0.0012, "img_s": 51.1, "lr": 0.00018357060987214956}

latest sample: `runs/star_w64_ship/samples/step0085000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
