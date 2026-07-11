# Starless + EasySharp training status

updated 2026-07-11 16:24

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## RUNNING: sharp_w32 (2026-07-11 16:24)
{"eval_step": 10000, "psnr": 52.83, "flux_err": 0.15}
{"step": 10200, "loss": 0.00321, "main": 0.00219, "fft": 0.00227, "grad": 0.00247, "reblur": 0.00179, "img_s": 90.8, "lr": 0.000288166399281083}
{"step": 10400, "loss": 0.0036, "main": 0.00247, "fft": 0.00254, "grad": 0.00293, "reblur": 0.00195, "img_s": 93.2, "lr": 0.0002877043165397547}

latest sample: `runs/sharp_w32/samples/step0010000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
