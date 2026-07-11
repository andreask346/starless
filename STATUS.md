# Starless + EasySharp training status

updated 2026-07-11 16:54

queue of 4 experiments
## star_w32: ok (5.6 h)
- export: ok
- eval: {"psnr_in": 55.55, "psnr_out": NaN, "leak": NaN, "recomp": 153.21, "completeness": {"faint": 0.8964, "mid": 0.9758, "bright": 0.9724}, "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10)_20260520113355.tif"],
## RUNNING: sharp_w32 (2026-07-11 16:54)
{"step": 20400, "loss": 0.00354, "main": 0.00233, "fft": 0.00229, "grad": 0.00265, "reblur": 0.00237, "img_s": 93.2, "lr": 0.0002545389630905488}
{"step": 20600, "loss": 0.00486, "main": 0.00309, "fft": 0.00241, "grad": 0.00305, "reblur": 0.00408, "img_s": 93.5, "lr": 0.00025369256041504583}
{"step": 20800, "loss": 0.00357, "main": 0.00253, "fft": 0.00196, "grad": 0.00242, "reblur": 0.00202, "img_s": 93.1, "lr": 0.0002528397923363362}

latest sample: `runs/sharp_w32/samples/step0020000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
