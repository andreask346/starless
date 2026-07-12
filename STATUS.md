# Starless + EasySharp training status

updated 2026-07-12 13:41

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## sharp_w32: ok (3.8 h)
- export: ok
- eval: {"psnr": 54.9395, "flux": 0.0389, "reblur": 0.001, "ident": 0.0007, "panels": ["sharp_DS.tif", "sharp_Stacked.tif", "sharp_7IV01626.tif", "sharp_7IV01627.tif", "sharp_(1)_20260520113330.tif", "sharp_(10)_20260520113355.tif"], "step": 80000}
## star_w64_ship: ok (17.5 h)
- export: ok
- eval: {"psnr_in": 57.73, "psnr_out": NaN, "leak": NaN, "recomp": 151.87, "completeness": {"faint": 0.9316, "mid": 0.9958, "bright": 0.9862}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## RUNNING: sharp_w64_ship (2026-07-12 13:41)
{"step": 4800, "loss": 0.00479, "main": 0.00323, "fft": 0.003, "grad": 0.00364, "reblur": 0.00297, "img_s": 46.5, "lr": 0.00029957525558896205}
{"step": 5000, "loss": 0.00745, "main": 0.00559, "fft": 0.00286, "grad": 0.00345, "reblur": 0.00412, "img_s": 46.2, "lr": 0.00029953914139310293}
{"eval_step": 5000, "psnr": 52.01, "flux_err": 0.31}

latest sample: `runs/sharp_w64_ship/samples/step0005000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
