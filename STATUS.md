# Starless + EasySharp training status

updated 2026-07-19 10:58

queue of 3 experiments
## star_v2_e1: done earlier (skipped)
## star_v2_e4_512: ok (0.5 h)
- export: ok
- eval: {"psnr_in": 41.76, "psnr_out": 69.37, "leak": 0.0, "recomp": 150.36, "completeness": {"faint": 0.8525, "mid": 0.9634, "bright": 0.9898}, "data_recipe": "v2", "panels": ["panel_DS.tif", "panel_Stacked.tif", "panel_7IV01626.tif", "panel_7IV01627.tif", "panel_(1)_20260520113330.tif", "panel_(10
## RUNNING: star_v2_w32 (2026-07-19 10:58)
{"step": 3800, "loss": 0.11283, "starless": 0.02407, "stars": 0.02407, "fft": 0.0047, "wing": 0.02856, "site": 0.78772, "chroma": 0.39607, "leak": 0.00017, "img_s": 34.3, "lr": 0.00029705054060594915}
{"step": 4000, "loss": 0.20011, "starless": 0.03021, "stars": 0.03021, "fft": 0.01537, "wing": 0.09703, "site": 0.96688, "chroma": 0.43903, "leak": 0.0003, "img_s": 36.8, "lr": 0.0002967330663097055}
{"step": 4200, "loss": 0.18943, "starless": 0.03293, "stars": 0.03293, "fft": 0.01458, "wing": 0.05519, "site": 1.26344, "chroma": 0.84229, "leak": 0.00014, "img_s": 35.2, "lr": 0.0002963995559098447}

latest sample: `runs/star_v2_e4_512/samples/step0020000_s2.tif`

![progress](progress.jpg)

_panel = input | model output | ground truth (left to right)_
