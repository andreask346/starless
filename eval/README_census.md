# census.py — star-removal regression harness

Permanent, parametrized port of the July-2026 forensic analysis (f1–f5) that
classified 172,542 stars on a real failing frame. One detection pass on an
input frame serves any number of scored starless variants, so model/version
comparisons are apples-to-apples on the exact same star sites.

Python: `C:\Users\User\Documents\For Claude\starless\.venv\Scripts\python.exe`
(needs numpy, tifffile, astropy, scipy, opencv).

## Usage

Detect once per input frame (slow-ish, minutes — caches every star site):

```
python eval\census.py detect ^
  --input "path\to\Input.tif" ^
  --cache "eval\sites_input.npz"
```

Score each starless variant against the cache (fast, ~1–3 min):

```
python eval\census.py score ^
  --cache "eval\sites_input.npz" ^
  --input "path\to\Input.tif" ^
  --starless "path\to\starless_variant.fit" ^
  --starmask "path\to\starmask_variant.fit" ^
  --label variant_name ^
  --json "eval\census_variant_name.json" ^
  --md "eval\census_variant_name.md"
```

`--starmask` is optional but enables the reconstruction-based FITS
orientation check and the mask smooth-leak metric. Without it, orientation
falls back to a correlation check. Inputs/outputs may be TIFF or FITS; FITS
vertical flip vs TIFF is handled automatically.

`--max-sites N --seed 42` takes a deterministic subsample (same cache + seed
+ N ⇒ identical sites every run) if a full pass is too slow. Keep the same
N/seed across every variant you compare.

## Metric definitions

Detection (from the input, cached): DoG-style — background = 8× downsample +
median filter; high-pass smoothed σ=1.5; local maxima above
`thr = max(5·σ_pixel, 4·σ_hp)`; per star: peak above local background, FWHM
from the half-max component area, flux in a 1.5·FWHM disk; grid dedupe.
`σ_pixel` (noise sigma) = MAD of the pixel-level high-pass on quiet pixels.

Per-site classification against the starless (residual = starless − local
annulus median, measured in a disk of 0.75·FWHM + a ring to 1.6·FWHM):

- **missed** — residual peak > 50 % of the original star peak (and > 3σ):
  the star is still there.
- **artifact** — residual amplitude > max(3σ, 8 % of star peak). Subtypes:
  `dark_hole` (negative crater dominates), `bright_residual` (positive mush),
  `ring`, `other`; `+color` appended when the residual chroma spread exceeds
  max(2σ, 5 % of peak).
- **clean** — anything else.

Aggregates:

- rates broken down by peak bins and FWHM bins,
- **artifact median residual** in units of noise sigma,
- **dark holes**: fraction of artifacts + median crater depth (σ units),
- **residual stars**: sites whose starless residual peak still exceeds the
  input detection threshold (a detector would re-find the star),
- **mask smooth-leak fraction** (needs `--starmask`): fraction of star-mask
  energy that is low-frequency diffuse glow (nebula stolen into the mask)
  rather than point sources — median filter on a 4× downsample kills point
  sources, whatever survives is leak.

### Scalar `score` (higher = better, range −1 … +1)

For each populated peak bin `b` with importance weight
`w = {<0.03: 0.25, 0.03–0.1: 0.5, 0.1–0.2: 0.75, 0.2–0.35: 1.0, 0.35–0.5: 1.25, 0.5–0.65: 1.5, >0.65: 2.0}`:

```
score = Σ_b w_b · (clean_frac_b − 0.5·artifact_frac_b − missed_frac_b) / Σ_b w_b
```

Bright stars weigh more (a missed or cratered bright star is glaring). The
JSON also carries every count/rate so any other objective can be recomputed
without re-running.

## Baseline: StarNet on Neutralised.tif (2026-07-17)

172,542 sites, full pass (no subsample), score runtime ~13 s classification /
< 1 min wall including FITS loads. FITS pair was vertically flipped vs the
TIFF; reconstruction check picked it up (recon err 1.6e-8 after flip).

| metric | baseline value |
|---|---|
| scalar `score` | **+0.0878** |
| clean | 41.85 % (72,211) |
| artifact | 54.94 % (94,798) |
| missed | 3.21 % (5,533) |
| artifact median residual | 21.1 × noise sigma (p75 26.2, p95 46.8) |
| dark holes | 57.2 % of artifacts, median depth 20.7 sigma |
| residual stars (rmax > thr) | 43,402 |
| mask smooth-leak fraction | 28.7 % |
| noise sigma / detection thr | 0.002525 / 0.03213 |

By peak (bg-subtracted):

| peak bin | n | clean % | artifact % | missed % |
|---|---|---|---|---|
| 0.03–0.1 | 2,059 | 0.0 | 53.1 | 46.9 |
| 0.1–0.2 | 7,737 | 3.0 | 88.0 | 9.0 |
| 0.2–0.35 | 25,519 | 33.6 | 65.7 | 0.7 |
| 0.35–0.5 | 40,769 | 42.6 | 55.5 | 1.9 |
| 0.5–0.65 | 78,486 | 43.3 | 53.6 | 3.2 |
| >0.65 | 17,972 | 67.3 | 30.4 | 2.3 |

By FWHM:

| FWHM bin | n | clean % | artifact % | missed % |
|---|---|---|---|---|
| <1.5 px | 18,954 | 74.6 | 25.4 | 0.0 |
| 1.5–2 | 86,593 | 51.5 | 48.4 | 0.1 |
| 2–3 | 56,049 | 23.9 | 74.2 | 1.9 |
| 3–4 | 8,054 | 0.6 | 68.3 | 31.0 |
| 4–6 | 2,538 | 0.0 | 38.4 | 61.6 |
| 6–10 | 273 | 0.0 | 2.6 | 97.4 |
| >10 | 81 | 0.0 | 1.2 | 98.8 |

Files: cache `eval\sites_neutralised.npz`, JSON
`eval\census_baseline_neutralised.json`.

## Held-out TEST data — do not train on it

`E:\Tenerife_&_La_Palma_June_2026\Photos\La_Palma\Pared De Roberto Path
Cygnus\PI\Neutralised.tif` (and its starless/starmask pair) is **permanent
held-out TEST data**. Never include it, crops of it, or derived tiles in any
training or validation set. It exists solely to score variants.
