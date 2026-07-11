# Starless + EasySharp training status

updated 2026-07-11 17:54

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## RUNNING: sharp_w32 (2026-07-11 17:54)
{"step": 41600, "loss": 0.00246, "main": 0.00166, "fft": 0.00136, "grad": 0.00203, "reblur": 0.00152, "img_s": 95.7, "lr": 0.00014111281733036554}
{"step": 41800, "loss": 0.0026, "main": 0.00179, "fft": 0.00166, "grad": 0.00203, "reblur": 0.00147, "img_s": 95.7, "lr": 0.0001399412656032881}
{"step": 42000, "loss": 0.00249, "main": 0.0017, "fft": 0.00141, "grad": 0.00225, "reblur": 0.00143, "img_s": 95.4, "lr": 0.00013877036518868474}

latest sample: `runs/sharp_w32/samples/step0040000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
