# Starless + EasySharp training status

updated 2026-07-19 09:25

queue of 3 experiments
## star_v2_e1: ok (9.1 h)
- export: ok
- eval: {"psnr_in": 42.07, "psnr_out": 72.94, "leak": 0.0, "recomp": 150.82, "completeness": {"faint": 0.8221, "mid": 0.9625, "bright": 0.9903}, "data_recipe": "v2", "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10
## RUNNING: star_v2_e4_512 (2026-07-19 09:25)
{"step": 10400, "loss": 0.08275, "starless": 0.01151, "stars": 0.01151, "fft": 0.01531, "wing": 0.04267, "site": 0.32509, "chroma": 0.1684, "leak": 0.00069, "img_s": 12.5, "lr": 0.000141112817330368}
{"step": 10600, "loss": 0.04392, "starless": 0.01096, "stars": 0.01096, "fft": 0.00272, "wing": 0.01125, "site": 0.24298, "chroma": 0.11242, "leak": 7e-05, "img_s": 12.5, "lr": 0.00013643080715888245}
{"step": 10800, "loss": 0.06191, "starless": 0.01463, "stars": 0.01463, "fft": 0.00248, "wing": 0.01613, "site": 0.39637, "chroma": 0.16413, "leak": 0.00013, "img_s": 12.5, "lr": 0.00013176268158213684}

latest sample: `runs/star_v2_e4_512/samples/step0010000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
