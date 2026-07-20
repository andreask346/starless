# Starless + EasySharp training status

updated 2026-07-20 18:51

queue of 3 experiments
## star_v2_e1: done earlier (skipped)
## star_v2_e4_512: done earlier (skipped)
## RUNNING: star_v2_w32 (2026-07-20 18:51)
{"step": 59800, "loss": 0.16138, "starless": 0.02089, "stars": 0.02089, "fft": 0.02535, "wing": 0.09831, "site": 0.43069, "chroma": 0.21731, "leak": 0.00027, "img_s": 57.4, "lr": 1.0081971798559359e-06}
{"step": 60000, "loss": 0.08625, "starless": 0.01574, "stars": 0.01574, "fft": 0.00866, "wing": 0.03931, "site": 0.34613, "chroma": 0.15042, "leak": 0.00029, "img_s": 54.7, "lr": 1e-06}
{"eval_step": 60000, "psnr_stars": 32.2, "psnr_bg": 55.67}

latest sample: `runs/star_v2_w32/samples/step0060000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
