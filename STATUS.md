# Starless + EasySharp training status

updated 2026-07-11 17:24

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## RUNNING: sharp_w32 (2026-07-11 17:24)
{"step": 31000, "loss": 0.00253, "main": 0.00174, "fft": 0.00158, "grad": 0.00191, "reblur": 0.00147, "img_s": 93.5, "lr": 0.00020224450003308354}
{"step": 31200, "loss": 0.00258, "main": 0.00176, "fft": 0.00139, "grad": 0.00192, "reblur": 0.00164, "img_s": 93.4, "lr": 0.00020114131907666941}
{"step": 31400, "loss": 0.00233, "main": 0.00159, "fft": 0.00132, "grad": 0.00166, "reblur": 0.00146, "img_s": 93.8, "lr": 0.00020003501432515244}

latest sample: `runs/sharp_w32/samples/step0030000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
