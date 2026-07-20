# Starless + EasySharp training status

updated 2026-07-20 18:12

queue of 3 experiments
## star_v2_e1: done earlier (skipped)
## star_v2_e4_512: done earlier (skipped)
## RUNNING: star_v2_w32 (2026-07-20 18:12)
{"step": 50000, "loss": 0.07614, "starless": 0.01382, "stars": 0.01382, "fft": 0.00993, "wing": 0.02944, "site": 0.41774, "chroma": 0.20073, "leak": 9e-05, "img_s": 78.4, "lr": 2.1029202134226906e-05}
{"eval_step": 50000, "psnr_stars": 31.86, "psnr_bg": 56.44}
{"step": 50200, "loss": 0.12383, "starless": 0.02161, "stars": 0.02161, "fft": 0.02177, "wing": 0.06366, "site": 0.34291, "chroma": 0.17268, "leak": 0.00018, "img_s": 76.5, "lr": 2.0253535237532112e-05}

latest sample: `runs/star_v2_w32/samples/step0050000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
