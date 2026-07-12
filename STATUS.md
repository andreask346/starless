# Starless + EasySharp training status

updated 2026-07-12 15:11

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
## RUNNING: sharp_w64_ship (2026-07-12 15:11)
{"step": 20400, "loss": 0.00341, "main": 0.00225, "fft": 0.00207, "grad": 0.00246, "reblur": 0.00237, "img_s": 46.6, "lr": 0.00029238987367605164}
{"step": 20600, "loss": 0.00473, "main": 0.00297, "fft": 0.00228, "grad": 0.00293, "reblur": 0.00413, "img_s": 46.6, "lr": 0.00029224123460076546}
{"step": 20800, "loss": 0.00383, "main": 0.00264, "fft": 0.00179, "grad": 0.0022, "reblur": 0.00262, "img_s": 46.5, "lr": 0.0002920911965967174}

latest sample: `runs/sharp_w64_ship/samples/step0020000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
