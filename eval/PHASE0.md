# Phase-0 inference-side sweep — 2026-07-17

Held-out test frame: `Neutralised.tif` (7008x4672 stretched, background-
neutralized Cygnus nightscape; median 0.381, noise sigma 0.0025). Scored with
`eval/census.py` against the 172,542-site cache; `score` = peak-weighted
`clean - 0.5*artifact - missed`. Model: `star_w64_ship` weights, unchanged.

| variant | score | clean % | artifact % | missed % | artifact med (sigma) | mask leak % | notes |
|---|---|---|---|---|---|---|---|
| **ov256 — 512 tile, 50% overlap** | **+0.0991** | **42.83** | 53.96 | 3.21 | 21.0 | 28.4 | **SHIPPED as 0.2.1 default** |
| t256 — 256 tile (train crop), 25% ov | +0.0961 | 42.56 | 54.10 | 3.34 | 21.1 | 28.4 | FFC spectrum match: real but smaller win |
| t256ov — 256 tile, 50% overlap | +0.0932 | 42.37 | 54.29 | 3.34 | 21.1 | 28.4 | wins do not stack |
| baseline — 512 tile, 25% overlap (v0.1) | +0.0878 | 41.85 | 54.94 | 3.21 | 21.1 | 28.7 | what Andreas ran |
| canon10 — inverse-MTF to bg 0.10 | -0.4382 | 8.00 | 85.50 | 6.50 | 32.2 | 28.3 | REFUTED |
| deep512 — multi-scale + 2nd pass | -0.5133 | 1.60 | 96.37 | 2.03 | 51.1 | 59.4 | REFUTED (v1 weights) |
| deep256 — same at 256 tile | -0.5149 | 1.52 | 96.44 | 2.04 | 51.3 | 59.1 | REFUTED (v1 weights) |
| canon05 — inverse-MTF to bg 0.05 | -0.5577 | 2.14 | 85.53 | 12.32 | 41.4 | 26.2 | REFUTED |
| canon02 — inverse-MTF to bg 0.02 | -0.6794 | 0.44 | 67.62 | 31.94 | 51.3 | 21.3 | REFUTED |

## Competitor benchmark — StarXTerminator AI11 (CLI 1.0.2, defaults, RTX 4090, 13 s)

Scored 2026-07-17 on the identical census. THE BAR FOR v2:

| tool | score | clean % | artifact % | missed % | artifact med (sigma) | mask leak % |
|---|---|---|---|---|---|---|
| SXT AI11 | +0.8944 | 97.82 | 1.89 | 0.29 | 7.8 | **42.4** |
| OneClick v1 (ov256) | +0.0991 | 42.83 | 53.96 | 3.21 | 21.0 | **28.4** |

Where SXT is beatable on this frame: monsters (FWHM>10px: 95.1% missed for
SXT too; 6-10px: 50.5% missed), faint bin (peak 0.03-0.1: only 44.4% clean),
diffuse-glow theft (42.4% of its star mask is low-frequency background vs
our 28.4% — consistent with the earlier finding that our starless preserves
nebula texture better), and recomposition (SXT starless+stars deviates up
to 6.5e-4 from the input; ours is exact to float precision by construction).
Everywhere else SXT is far ahead — that is the v2 target.

## v2 retrain results (fine-tuned from ship ckpt on clean 3264-plate library)

Scored on the identical census frame. Progress toward the SXT bar:

| model | score | clean % | artifact % | missed % | artifact med (sigma) | mask leak % |
|---|---|---|---|---|---|---|
| SXT AI11 (target) | +0.894 | 97.82 | 1.89 | 0.29 | 7.8 | 42.4 |
| **v3 E2** (real PSF/noise/silhouettes, 2026-07-23) | **+0.709** | 90.67 | 9.09 | 0.24 | 10.3 | - |
| v3 E1 (60k @256, real data) | +0.703 | 89.76 | 9.99 | 0.26 | 11.4 | - |
| **v2 E4** (E1 + 20k @512 FFC-adapt) | +0.508 | 76.01 | 23.36 | 0.63 | 16.1 | 33.1 |
| **v2 E1** (60k @256, 2026-07-19) | +0.491 | 74.44 | 24.86 | 0.71 | 16.6 | 34.9 |
| **v2 w32** (light model, 60k) | +0.202 | 48.63 | 50.38 | 0.99 | 19.0 | 29.3 |
| v1 best (w64, ov256) | +0.099 | 42.83 | 53.96 | 3.21 | 21.0 | 28.4 |

LADDER COMPLETE 2026-07-19. Ship model = **star_v2_e4_512** (+0.508).
Note the v2 *light* w32 model (+0.202) beats the old v1 *full-width* w64
model (+0.099) outright - the data-recipe/loss fix outweighs 2x model
width, which is the cleanest evidence that the v1 failure was never a
capacity problem.

E1->E4 gained only +0.017 (clean 74.4->76.0%): the crop-512 FFC adaptation
helps marginally but the DATA RECIPE plateaued ~76% clean / +0.51. The v3
data levers closed most of that gap - see below.

## v3 result (real empirical data + the 2x PSF unit-bug fix)

v3 = empirical PSF stamps (55787 real stars -> 251 groups), measured noise
ACF, 379 real silhouettes, and the calibrate_psf FWHM 2x fix. Fine-tuned
from v2 E4. Census on the same held-out frame:

  SXT AI11  +0.894  97.8% clean  1.9% artifact  0.29% missed  7.8 sigma
  v3 E2     +0.709  90.7%        9.1%           0.24%         10.3    <- SHIP
  v2 E4     +0.508  76.0%        23.4%          0.63%         16.1

v3 closed ~2/3 of the v2->SXT gap (+0.508 -> +0.709) and BEATS SXT on
missed stars (0.24% vs 0.29%). Remaining gap to SXT is artifact rate
(9.1% vs 1.9%) and severity (10.3 vs 7.8 sigma) + residual faint stars
(2973 vs ~900). The rendered star FWHM went from 0.73x reality (v2, the
unit bug) to 1.06x (v3) - the single biggest lever. Further gains: extend
empirical stamps past 33px (capture bright-star wings/halos empirically),
push the loss harder on the artifact/faint bins, harvest more sources.

E1 alone closed ~half the v1->SXT gap. Big wins vs v1: clean 42.8->74.4%,
artifact 54->25%, missed 3.2->0.71%, artifact severity 21->16.6 sigma. Still
behind SXT on clean/artifact rate. Remaining weak bins after E1: faint peaks
(0.03-0.1: 0% clean), FWHM 3-6px (46/21% clean), and FWHM 6-10px (76% missed
vs SXT 50%) - though FWHM>10px v2 already edges SXT (90% vs 95% missed). Mask
leak rose slightly (28->35%) but stays below SXT's 42%. E4 (crop-512 FFC
adaptation) + w32 rungs pending; expected to help the mid-FWHM + artifact
bins. Data recipe still has headroom (real PSFs, measured noise) for a v3.

## Conclusions

1. **Canonicalization (inverse-MTF to a "linear-like" domain) is empirically
   harmful on the v1 weights** — the restretch back to the input domain
   amplifies every background-adjacent model error by the local stretch
   gain (artifacts 21 -> 32/41/51 sigma, monotone in gain), and the
   compressed star peaks fall below the model's sensitivity (missed up to
   32%). The v1 model's sweet spot IS the stretched domain (it trained with
   `presentation()` augmentation). Kept as `--canonicalize on` for
   re-testing after the v2 retrain; default off.
2. **Deep clean (4x/2x coarse passes + residual sweep) is harmful on the v1
   weights** — at coarse scale, Milky Way granulation reads as stars, so the
   coarse passes strip diffuse signal (mask leak 29 -> 59%) and re-running a
   model whose dominant failure is over-subtraction digs the dark holes
   deeper (21 -> 51 sigma). Missed rate DOES drop (3.2 -> 2.0%) — the
   mechanism works — so re-test after v2 fixes the over-subtraction.
3. **50% tile overlap is the only free win** (+1pp clean, -1pp artifact,
   ~4x tiles => ~45 s for a 33 MP frame on the RTX 4090). Shipped.
4. **256-tile export helps but less than overlap**, and does not stack with
   it. The FFC 256-train/512-infer mismatch is real; the right fix is the
   E4 crop-512 adaptation fine-tune in the retrain plan, not a smaller tile.
5. The failure-bin structure (97-100% missed above FWHM 6px, 0% clean below
   peak 0.1, 55% artifact rate) is unchanged by every inference-side lever —
   confirming the diagnosis: **these are training-data/loss gaps (data
   recipe v2) and cannot be rescued at inference time.**

Cross-check: census `detect`+`score` reproduced the independent forensic
analysis to the decimal on the baseline (see README_census.md), and the
orientation check verified the 0.2.x headless FITS writer is ROWORDER-
consistent (recon err 1.5e-08, no flip) where the v0.1 Siril-path outputs
needed a flip.
