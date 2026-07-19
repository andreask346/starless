# Starless + EasySharp training status

updated 2026-07-19 09:55

queue of 3 experiments
## star_v2_e1: ok (9.1 h)
- export: ok
- eval: {"psnr_in": 42.07, "psnr_out": 72.94, "leak": 0.0, "recomp": 150.82, "completeness": {"faint": 0.8221, "mid": 0.9625, "bright": 0.9903}, "data_recipe": "v2", "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10
## RUNNING: star_v2_e4_512 (2026-07-19 09:55)
{"step": 14800, "loss": 0.06493, "starless": 0.01321, "stars": 0.01321, "fft": 0.01331, "wing": 0.02972, "site": 0.21345, "chroma": 0.07241, "leak": 0.00012, "img_s": 10.7, "lr": 4.816020766366168e-05}
{"step": 15000, "loss": 0.09512, "starless": 0.02589, "stars": 0.02589, "fft": 0.00967, "wing": 0.02619, "site": 0.28567, "chroma": 0.20398, "leak": 0.00038, "img_s": 8.0, "lr": 4.478753621261181e-05}
{"eval_step": 15000, "psnr_stars": 41.37, "psnr_bg": 58.79}

latest sample: `runs/star_v2_e4_512/samples/step0015000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
