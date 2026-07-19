# Starless + EasySharp training status

updated 2026-07-19 08:55

queue of 3 experiments
## star_v2_e1: ok (9.1 h)
- export: ok
- eval: {"psnr_in": 42.07, "psnr_out": 72.94, "leak": 0.0, "recomp": 150.82, "completeness": {"faint": 0.8221, "mid": 0.9625, "bright": 0.9903}, "data_recipe": "v2", "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10
## RUNNING: star_v2_e4_512 (2026-07-19 08:55)
{"step": 5000, "loss": 0.07873, "starless": 0.01748, "stars": 0.01748, "fft": 0.00428, "wing": 0.01817, "site": 0.60298, "chroma": 0.26523, "leak": 4e-05, "img_s": 12.4, "lr": 0.0002562124637873884}
{"eval_step": 5000, "psnr_stars": 39.17, "psnr_bg": 20.02}
{"step": 5200, "loss": 0.10401, "starless": 0.02758, "stars": 0.02758, "fft": 0.01491, "wing": 0.03715, "site": 0.20611, "chroma": 0.10318, "leak": 0.00084, "img_s": 11.9, "lr": 0.0002528397923363389}

latest sample: `runs/star_v2_e4_512/samples/step0005000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
