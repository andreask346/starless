# Starless + EasySharp training status

updated 2026-07-22 06:00

queue of 2 experiments
## star_v3_e1: ok (5.3 h)
- export: ok
- eval: {"psnr_in": 41.31, "psnr_out": 70.74, "leak": 0.0, "recomp": 150.97, "completeness": {"faint": 0.8535, "mid": 0.9553, "bright": 0.9833}, "data_recipe": "v3", "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10
## RUNNING: star_v3_e2_512 (2026-07-22 06:00)
{"step": 10800, "loss": 0.07813, "starless": 0.00765, "stars": 0.00765, "fft": 0.00564, "wing": 0.01627, "site": 1.18864, "chroma": 0.43356, "leak": 0.00041, "img_s": 12.5, "lr": 0.00013176268158213684}
{"step": 11000, "loss": 0.07974, "starless": 0.00909, "stars": 0.00909, "fft": 0.00301, "wing": 0.00947, "site": 1.16193, "chroma": 0.56138, "leak": 0.00032, "img_s": 12.5, "lr": 0.0001271130474764859}
{"step": 11200, "loss": 0.06784, "starless": 0.00872, "stars": 0.00872, "fft": 0.00191, "wing": 0.0106, "site": 1.04309, "chroma": 0.36813, "leak": 0.00023, "img_s": 12.5, "lr": 0.00012248649346943493}

latest sample: `runs/star_v3_e2_512/samples/step0010000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
