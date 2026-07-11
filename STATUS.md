# Starless + EasySharp training status

updated 2026-07-11 19:24

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## RUNNING: sharp_w32 (2026-07-11 19:24)
{"step": 73600, "loss": 0.00233, "main": 0.0016, "fft": 0.00141, "grad": 0.00181, "reblur": 0.00139, "img_s": 93.4, "lr": 5.69681741126971e-06}
{"step": 73800, "loss": 0.00271, "main": 0.00184, "fft": 0.00182, "grad": 0.00203, "reblur": 0.00163, "img_s": 93.9, "lr": 5.409282216452945e-06}
{"step": 74000, "loss": 0.00249, "main": 0.0017, "fft": 0.00141, "grad": 0.00207, "reblur": 0.00149, "img_s": 93.4, "lr": 5.130696900547385e-06}

latest sample: `runs/sharp_w32/samples/step0070000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
