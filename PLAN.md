# StarForge (working name) — Build Plan

**Two open AI tools for Siril, one training platform, built to beat the commercial incumbents:**

- **Tool A — star removal** (vs StarNet++ 2.5, StarXTerminator): starless image + a *mathematically exact* recomposable star layer.
- **Tool B — AI deconvolution** (vs BlurXTerminator): PSF-aware sharpening + aberration correction with *provably* no invented detail.

Research basis: [docs/research-digest.md](docs/research-digest.md) — 16-agent verified corpus (StarNet's actual source code read line-by-line; Croman's public methodology reconstructed from talks/forums; SOTA restoration ML; licensing verified).

## Why this is winnable

The incumbents have concrete, documented weaknesses we can attack:

1. **Star-layer purity** — StarNet's head can only *subtract* (dark holes by construction) and runs BatchNorm in training mode at inference (tile artifacts). SXT leaks nebulosity into its star layer. Our head predicts a non-negative star image S with `starless = input − S`, so `starless + stars` rebuilds the original **exactly** — a claim no competitor can make.
2. **Stretched-data fragility** — SXT struggles on GHS/arcsinh-stretched images (Croman's own FAQ). We train with identical random stretches applied to input+target pairs.
3. **Wide-field lens PSFs** — both incumbents are weakest on lens-aberrated, big-FWHM stars — *exactly Andreas's 14–35mm nightscape niche*. Our synthetic data generator makes lens PSFs (coma, astigmatism, cross-screen spikes) a first-class citizen.
4. **Verifiability** — BXT emits no evidence. We emit per-tile PSF maps, residual maps, photometry-preservation reports (EasyReg DNA).
5. **Frozen incumbents** — SXT's core AI unchanged since Sep 2022; StarNet since 2022 (2.5 was repackaging). A 4090 + fast iteration beats a frozen model.

**Legal note:** training on StarNet/SXT outputs is banned (RC-Astro EULA; StarNet weights CC BY-NC-SA) — and would cap quality at theirs anyway. Our data is 100% synthetic injection over openly-licensed backgrounds. Clean room.

## Architecture (from the digest)

**Shared backbone:** NAFNet-class U-Net (norm-free, no BatchNorm pathology), width-32 for experiments → width-64 to ship; 4–5 scales with FFC (Fourier) bottleneck blocks for image-wide receptive field — this is what kills the "huge halo/spike left behind" failure mode (fallback if FFC won't export to ONNX: 31×31 large-kernel convs).

- **Tool A head:** non-negative star residual S; losses: Charbonnier on starless *and* star layer + FFT frequency loss + star-layer purity penalty. Second small stage: noise-matched infill under star cores (SXT's killer feature, absent from every free tool). No GAN.
- **Tool B:** same body, PSF-conditioned (SFT layers fed a fitted PSF from **EasyReg's own star detector** — per-region Moffat fits, so one model adapts to any optics); losses: Charbonnier + re-blur consistency `‖PSF⊗output − input‖` + flux conservation. Validation: photometry error <2% (BXT measured ≈2–4%), hallucination maps, synthetic round-trips.

**Training data:** one GalSim-based injection engine — pupil-FFT diffraction spikes (refractor/Newtonian/lens cross-screen), Zernike aberrations, Moffat wings + aureole + reflection halos, saturation bleed, Gaia-anchored star colors/densities (sparse fields → Milky Way core), realistic post-stack noise, random linear/stretched presentation. Backgrounds: 1–5k openly licensed starless plates (Legacy Surveys, DECaPS2, ESA Hubble/Webb) + real starless frames (his PI outputs as *backgrounds*, not labels) — generated on the fly, effectively infinite pairs.

**Deployment:** PyTorch → static-shape 512×512 ONNX (opset ≤20) → sirilpy's built-in ONNXHelper (DirectML on Windows, CPU fallback) → PyQt6 Siril script like EasyReg, Hann-feathered tile blending. Estimated ~3–10 s per 30MP frame on the 4090. Weights on GitHub Releases/HF, MIT/Apache-2.0.

## Evaluation gates ("better" = measured, not vibes)

- Synthetic ground truth: in-star-footprint vs out-of-footprint metrics separately; residual-star completeness by magnitude/FWHM bin; star-layer background leakage.
- **Recomposition round-trip ≥60 dB** and idempotence (running twice changes nothing).
- 20–30 image torture panel (dense Milky Way, bright spiky star, globulars, small galaxies that must survive, his nightscapes) blind A/B vs StarNet 2.5.3, SXT, Cosmic Clarity Dark Star, Zenith.
- Tool B: flux preservation, centroid stability, power-spectrum honesty, Hubble-blink comparisons.

## Phased roadmap (all training local on the 4090)

| Phase | What | Time |
|---|---|---|
| 1 | Data engine + evaluation harness (the moat — worth half the total effort) | build days |
| 2 | Tool A v0: width-32 overnight trains (6–10 h each), iterate data recipe vs torture panel | ~1–2 weeks of iterations |
| 3 | Tool A ship: width-64 (~2 days train) + noise-infill stage + Siril script + report | days |
| 4 | Tool B v0: PSF conditioning on the same platform | ~1–2 weeks of iterations |
| 5 | Tool B ship + docs + release | days |

The machine trains overnight/weekends; iterations are recipe changes, not code rewrites. Realistic path to "beats StarNet clearly, competitive-to-better vs SXT" on Tool A; Tool B's differentiator is verifiable honesty + lens-PSF handling.

## Open decisions (Andreas)

1. **Names** — working name "StarForge" for the platform. Tool names? (e.g. EasyStar / EasySharp to match EasyReg; or StarForge / SharpForge; your call — veto welcome, cheap to change before the repo goes up.)
2. **GPU time** — phases 2–4 run multi-hour/overnight training on your 4090 repeatedly. OK to own the box at night?
3. **Go/no-go per phase** — Phase 1+2 first, judge Tool A v0 results on your own images, then commit onward.
