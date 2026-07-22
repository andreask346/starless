# Starless + EasySharp training status

updated 2026-07-22 05:30

queue of 2 experiments
## star_v3_e1: ok (5.3 h)
- export: ok
- eval: {"psnr_in": 41.31, "psnr_out": 70.74, "leak": 0.0, "recomp": 150.97, "completeness": {"faint": 0.8535, "mid": 0.9553, "bright": 0.9833}, "data_recipe": "v3", "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10
## RUNNING: star_v3_e2_512 (2026-07-22 05:30)
{"eval_step": 5000, "psnr_stars": 31.92, "psnr_bg": 32.3}
{"step": 5200, "loss": 0.07885, "starless": 0.01294, "stars": 0.01294, "fft": 0.00385, "wing": 0.0131, "site": 0.93432, "chroma": 0.41342, "leak": 0.00015, "img_s": 12.0, "lr": 0.0002528397923363389}
{"step": 5400, "loss": 0.04216, "starless": 0.0035, "stars": 0.0035, "fft": 0.00113, "wing": 0.00938, "site": 0.60741, "chroma": 0.27059, "leak": 2e-05, "img_s": 12.5, "lr": 0.000249366123865886}

latest sample: `runs/star_v3_e2_512/samples/step0005000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
