# Starless + EasySharp training status

updated 2026-07-11 19:42

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## RUNNING: sharp_w32 (2026-07-11 19:42)
{"step": 79800, "loss": 0.00265, "main": 0.00178, "fft": 0.00167, "grad": 0.00183, "reblur": 0.00174, "img_s": 93.8, "lr": 1.004610932103946e-06}
{"step": 80000, "loss": 0.00234, "main": 0.0016, "fft": 0.00133, "grad": 0.00192, "reblur": 0.00141, "img_s": 93.8, "lr": 1e-06}
{"eval_step": 80000, "psnr": 58.54, "flux_err": 0.05}

latest sample: `runs/sharp_w32/samples/step0080000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
