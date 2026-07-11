# Starless + EasySharp training status

updated 2026-07-11 18:54

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## RUNNING: sharp_w32 (2026-07-11 18:54)
{"step": 63000, "loss": 0.00209, "main": 0.00144, "fft": 0.00097, "grad": 0.00177, "reblur": 0.00125, "img_s": 94.3, "lr": 3.3095118833328805e-05}
{"step": 63200, "loss": 0.00231, "main": 0.00159, "fft": 0.0012, "grad": 0.00183, "reblur": 0.0014, "img_s": 94.2, "lr": 3.237182564983444e-05}
{"step": 63400, "loss": 0.00226, "main": 0.00158, "fft": 0.00112, "grad": 0.0015, "reblur": 0.00141, "img_s": 94.2, "lr": 3.165581916856746e-05}

latest sample: `runs/sharp_w32/samples/step0060000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
