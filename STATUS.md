# Starless + EasySharp training status

updated 2026-07-22 06:47

queue of 2 experiments
## star_v3_e1: ok (5.3 h)
- export: ok
- eval: {"psnr_in": 41.31, "psnr_out": 70.74, "leak": 0.0, "recomp": 150.97, "completeness": {"faint": 0.8535, "mid": 0.9553, "bright": 0.9833}, "data_recipe": "v3", "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10
## RUNNING: star_v3_e2_512 (2026-07-22 06:47)
{"step": 19800, "loss": 0.03759, "starless": 0.00474, "stars": 0.00474, "fft": 0.00838, "wing": 0.01733, "site": 0.21914, "chroma": 0.1052, "leak": 0.00036, "img_s": 12.5, "lr": 1.0737692253231292e-06}
{"step": 20000, "loss": 0.05134, "starless": 0.00707, "stars": 0.00707, "fft": 0.00123, "wing": 0.00887, "site": 0.58299, "chroma": 0.32972, "leak": 6e-05, "img_s": 12.5, "lr": 1e-06}
{"eval_step": 20000, "psnr_stars": 35.88, "psnr_bg": 36.04}

latest sample: `runs/star_v3_e2_512/samples/step0020000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
