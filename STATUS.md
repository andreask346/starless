# Starless + EasySharp training status

updated 2026-07-11 18:24

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## RUNNING: sharp_w32 (2026-07-11 18:24)
{"step": 52400, "loss": 0.00215, "main": 0.00147, "fft": 0.00115, "grad": 0.00166, "reblur": 0.00133, "img_s": 95.4, "lr": 8.054449276804396e-05}
{"step": 52600, "loss": 0.00222, "main": 0.00151, "fft": 0.00119, "grad": 0.0016, "reblur": 0.00146, "img_s": 95.8, "lr": 7.950896971434585e-05}
{"step": 52800, "loss": 0.00312, "main": 0.00206, "fft": 0.00165, "grad": 0.00228, "reblur": 0.00225, "img_s": 95.7, "lr": 7.847782572179323e-05}

latest sample: `runs/sharp_w32/samples/step0050000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
