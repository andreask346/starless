# Starless + EasySharp training status

updated 2026-07-22 06:30

queue of 2 experiments
## star_v3_e1: ok (5.3 h)
- export: ok
- eval: {"psnr_in": 41.31, "psnr_out": 70.74, "leak": 0.0, "recomp": 150.97, "completeness": {"faint": 0.8535, "mid": 0.9553, "bright": 0.9833}, "data_recipe": "v3", "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10
## RUNNING: star_v3_e2_512 (2026-07-22 06:30)
{"step": 16400, "loss": 0.04611, "starless": 0.0081, "stars": 0.0081, "fft": 0.00396, "wing": 0.02317, "site": 0.1528, "chroma": 0.06354, "leak": 0.00015, "img_s": 12.5, "lr": 2.4272975137448996e-05}
{"step": 16600, "loss": 0.06822, "starless": 0.00888, "stars": 0.00888, "fft": 0.01589, "wing": 0.02331, "site": 0.56912, "chroma": 0.28366, "leak": 0.0004, "img_s": 12.5, "lr": 2.181906696291068e-05}
{"step": 16800, "loss": 0.14846, "starless": 0.01955, "stars": 0.01955, "fft": 0.00341, "wing": 0.02985, "site": 2.31139, "chroma": 0.65827, "leak": 0.0001, "img_s": 12.5, "lr": 1.9492151333442606e-05}

latest sample: `runs/star_v3_e2_512/samples/step0015000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
