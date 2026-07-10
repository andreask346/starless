# StarForge Research Digest

**Date:** 2026-07-10. Distilled from a 16-agent research + fact-check corpus (8 research topics, 8 verification passes).
**Scope:** Two open AI tools for Siril sharing one training/deployment platform:
**Tool A** — star removal (beat StarNet++ 2.5 / StarXTerminator): starless image + truly recomposable star layer.
**Tool B** — AI deconvolution/sharpening (beat BlurXTerminator): PSF-aware, aberration correction, no invented detail.
**Hardware:** RTX 4090 24GB, Ryzen 9 7950X, 128GB RAM, Windows. **Delivery:** Siril Python script (sirilpy) + ONNX Runtime (DirectML/CUDA), EasyReg-style.

Facts marked **[verified]** were checked against primary sources by the fact-check pass. Facts marked **[unverified]** are forum/talk-sourced and plausible but not primary-confirmed. All verifier corrections are applied inline and listed in §8.

---

## 1. StarNet — architecture & weaknesses

### 1.1 Original StarNet v1 (nekitmm/starnet, Nikita Misiura, 2018-19) — read from source [verified]

Repo: https://github.com/nekitmm/starnet — **code MIT; pre-trained weights CC BY-NC-SA 4.0 (non-commercial, share-alike — do NOT fine-tune/distill/convert for a distributed tool)**. Weights checkpoint ~700 MB (includes Adam state). Local clone made during research (model.py, train.py, transform.py, starnet_utils.py, starnet_v1_TF2.py).

**Generator** (model.py, verified verbatim): pix2pix-style conditional-GAN U-Net.
- 8 encoder convs, all 4x4, stride (2,2), padding 'same', filters `[64,128,256,512,512,512,512,512]`; LeakyReLU(0.2).
- 8 decoder transposed convs `[512,512,512,512,256,128,64,3]` with skip concatenations (`tf.concat([layers[-1], layers[k]], axis=3)`, deconv_i paired with encoder layers 6,5,4,3,2,1,0); ReLU.
- BatchNorm (axis=3, eps=1e-5, momentum=0.1, gamma init N(1.0,0.02)) on every layer except encoder layer 0 and the final output; Xavier init.
- **Subtract-only residual head** (the defining defect): `rectified = tf.nn.relu(deconvolved); output = tf.subtract(input, rectified)` — the net predicts non-negative "star flux" to subtract. It can only darken, never brighten → over-subtraction dark holes/rings are architecturally baked in and unfixable in place. ("~54M params" is an estimate, not source-confirmed.)

**Discriminator** (doubles as perceptual feature extractor — no VGG anywhere): 9 conv layers 3x3, filters `[32,64,64,128,128,256,256,256,8]`, alternating stride 1 'same' / stride 2 'valid', BN+LeakyReLU(0.2) (no BN on layer 0), then Dense(1)+sigmoid. Returns 8 intermediate activations p1..p8 used as perceptual features (Perceptual Adversarial Networks, arXiv:1706.09138).

**Losses** (TF1, model.py, verified verbatim):
- `gen_loss = 0.01*GAN + 0.1*p1 + 1*(p2..p7) + 10*p8 + 10*L1`
- `discrim_loss = mean(-(log(D_real+1e-8) + log(1-D_fake+1e-8)))`
- Adam, default `lr=[2e-4, 2e-4]`; starnet.py staged decay [G,D]: [2e-4,5e-5]→[2e-5,5e-6]→[2e-6,5e-7]→[2e-7,5e-8]. Gen train op has a control dependency on disc train op (alternating in one step). Note the effective weighting: GAN term is tiny — L1 dominates. The GAN was a light texture prior even for StarNet.

**TF2 rewrite** (starnet_v1_TF2.py, in-repo, verified): `StarNet(mode='RGB'|'Greyscale', window_size=512, stride=256, lr=1e-4, batch_size=1)`; `dis_optimizer = Adam(lr/4)`; `gen_loss = 0.1*GAN + 0.1*p1 + 10*(p2..p8) + 100*L1`; weights `{name}_G_{mode}.h5`; warm_up option trains y=x identity first.

**Training regime:** batch size 1 (~2GB VRAM), 1000 steps/epoch, WINDOW 256 (TF1) / 512 (TF2). Inputs = **8-bit stretched TIFF pairs** normalized [0,1]→[-1,1] (`x*2-1`).
**Critical hidden defect: BatchNorm is hard-coded `training=True` everywhere, including at inference** — the model normalizes each tile with that tile's own statistics, making output depend on tile content/position. Plausible root cause of tile seams and blotchy inconsistency. Any faithful re-export of v1 weights must replicate batch-stat behavior.

**Training data:** private hand-made starless pairs from Misiura's FSQ106 + QSI683 wsg-8 refractor. README (verified): "the only difference between two images should be stars"; "quality of star removal by the net will never be better than that in your training set"; dataset never released ("This is one part I'd like to keep for myself for now").

**Augmentation** (starnet_utils.py): random 256-crop; arbitrary rotation p=0.33 (bicubic); rescale 0.5–2.0x p=0.33; H/V flips p=0.5; rot90 p=0.5; grayscale p=0.1; per-channel brightness offset p=0.7 (`X[:,:,ch] += offset*(1-X[:,:,ch])`); **random RGB channel permutation p=0.7** (forces color-agnostic star detection — keep this idea).

**Tiled inference** (transform.py): 256x256 tiles, default stride 64 (v1), offset=(WINDOW-stride)/2; **only the central stride×stride crop of each tile is hard-replaced — zero feathering** → seams mitigated only by shrinking stride (more compute). Edge padding = tiled copies of the image's own edge strips (not reflection). Bug: published TF1 transform.py feeds [0,1] tiles although training used [-1,1]; only the TF2 transform maps `x*2-1` and back.

### 1.2 StarNet v2 → 2.5.x evolution

- **v2** (announced CloudyNights 2022-01-24, closed source): same declared family ("convolutional residual net with encoder-decoder architecture and with L1, Adversarial and Perceptual losses"), retrained on a much larger set "with the push towards star dense images"; works at any processing stage (linear or stretched); Windows GUI; PixInsight weights `StarNet2_weights.pb`. Evidence (TF2 defaults, min-image 512, default stride 256) suggests a 512-window/256-stride model — **inference, not documented fact**.
- **2.5.x** (starnetastro.com relaunch; release-notes page, verified): **2.5.0** (2026-05-26; macOS build shows 05-27) — migrated TensorFlow → self-contained **ONNX Runtime** packages (Win/Linux/Intel Mac) + native CoreML (Apple Silicon); added `--unscreen` star-layer output; weights `StarNet2_weights.onnx`. **2.5.1** (2026-05-30) — optional CUDA for Windows PI modules (ORT 1.26.0, CUDA 12.9.1, cuDNN 9.23). **2.5.2** (2026-06-06) — improved highlight protection; fixed star-layer artifacts in saturated regions. **2.5.3** (2026-06-27) — DirectML acceleration with CPU fallback; "Protect highlights" toggle on by default (`-d` disables). Site also hosts DeepSNR 1.2.2. Freeware, closed weights, no public EULA ("© 2026 Mikita Misiura. All rights reserved.").
- **CLI 2.5.3** (verified verbatim vs official docs): `starnet2 [-dehqu] [--version] [-m <mask>] [-n <starlayer>] [-o <starless>] [-s <stride>] [-w <weights>] -i <input>`. Stride even, 2–512 (default 256 per Siril docs — Siril: "the StarNet developer recommends not to change this"); `-u` = 2x intermediate upsample for tight stars (Siril docs say 4x time; official docs just say "higher runtime cost"); min image 512x512; 8/16-bit integer TIFF/PNG only — **float and alpha rejected even in 2.5.3** (easy differentiator: accept float FITS natively); 16-bit output default.
- Siril drives the CLI: optional MTF pre-stretch + exact inverse for linear data; star mask = original − starless (post-hoc — inherits every background error into the star layer); requires libtiff. Legacy C StarNet interface deprecated, removed in Siril 1.6 — Python scripts are the official path.

### 1.3 Documented failure modes (all sourced)

| Failure | Source |
|---|---|
| Huge stars + halos left behind | README ("leaving only really huge ones"); AstroBin consensus SXT "much better, especially with big stars and halos" |
| Residual small stars in dense fields | README tip 5 ("feed output to the net again") |
| Diffraction spikes poorly removed | README tip 2 (refractor-only training); astronomy.com (neither tool does spikes well) |
| Dark blobs/holes, blotchy remnants | CN "Removing Starnet++ artefacts" (topic 926195); PI forum "Starnet leaves visible artifacts behind" (threads/16019) |
| Radiating "strange star patterns" on tight bright stars | AstroBin topic 81844; acknowledged by Misiura; mitigated by `-u` at 4x compute |
| Stride/tile seams | starnet.py comment "the smaller it gets, the less artefacts"; users drop 128→64 |
| Saturated-core star-layer artifacts | explicitly fixed in 2.5.2/2.5.3 release notes (any benchmark must state exact 2.5.x version + toggle state) |
| Fails on heavily processed / unusual PSFs | README tip 1 |
| Slow CPU inference, ~3GB RAM peak | astrojolo.com; GPU under-utilized even in 2.5.x because tiling/merge run on CPU (official docs admit for RTX 5090) |
| Nebulosity damage — small galaxies/faint knots removed (land in star layer) | Linda's Astronomy Adventures; astrojolo |

**Attack surface summary:** (1) data ceiling (hand-made pairs); (2) BN-at-inference + hard tile replacement; (3) subtract-only head; (4) 8-bit/stretched lineage, still rejects float; (5) CPU-bound tiling.

---

## 2. StarXTerminator (SXT, Russell Croman / RC-Astro)

### 2.1 Current state [verified]

- PixInsight module **2.4.12 (30 Apr 2026)** but the network is **still AI11 (15 Sep 2022)** — ~4 years without a new AI. Photoshop plugin 2.0.6 (25 Nov 2022). Price **$49.95** (recently reduced from $59.95 — some cached pages still show the old price), permanent license, 3 machines, one activation covers PixInsight+Photoshop/Affinity+CLI. Free fully-functional trial. TensorFlow-based plugins.
- Three AI11 variants: **full** (best, ≥6GB GPU RAM), **lite** (~4x less memory), **lite.nonoise** (skips noise matching, much faster).
- **June–July 2026 distribution shift [verified]:** free stand-alone `rc-astro` CLI (beta 24 Jun 2026; v0.9.9 2 Jul; v0.9.10 9 Jul 2026) runs SXT/BXT/NXT without PixInsight. Existing licenses honored free; SIRIL10 coupon for new buyers. **Already integrated into Siril 1.4.4** (17 Jun 2026) via official PyQt6 scripts + Seti Astro Suite Pro. GPU: **Windows = any Direct3D-12 GPU (DirectML — AMD/Intel/NVIDIA); Linux = NVIDIA-only (CUDA/cuDNN); macOS = CoreML, macOS 14+; CPU fallback.** `rc-astro sxt --benchmark-all` picks fastest device. I/O: TIF/FITS/XISF/PNG. HISTORY cards written to FITS/XISF (added in 0.9.10). NXT2 Photoshop beta ships onnxruntime.dll — RC-Astro itself is migrating toward ONNX-Runtime-style deployment, validating our stack.

### 2.2 Methodology (as publicly known)

- Per-tile convolutional network. AI11 processes **1024x1024 tiles** (v1.x used 512), default **20% overlap with feathering**, optional **"Large Tile Overlap" 50% mode, ~3x slower** (for huge stars, dense fields, M42-class contrast) [50%/3x verified in AI11 release notes]. Script param `P.overlap` 0.05–0.75. A leaked TensorFlow OOM trace shows node `starX/dd1_xconv/Conv2D` allocating `[1,48,1024,1024]` (48 channels at full tile res, NCHW) **[unverified — forum-sourced]**. Croman: "The AI11 network is actually smaller than AI10" — receptive field, not params, is what handles JWST spikes/giant halos **[unverified quote]**. Croman's stated param counts (AIC 2022 talk): SXT AI11 = 21M (~85MB), NXT = 24M, StarNet++ v1 = 54M, StarNet v2 = 30M, Topaz Denoise = 14M **[unverified — talk-sourced]**.
- **Noise matching is a SEPARATE module** (PixInsight/CLI only) from the starless network: synthesizes replacement noise matched to surroundings "in spatial frequency and color content" so star voids are undetectable. Rationale: saturated cores have NO noise, halos have TOO MUCH (Poisson) — plain inpainting leaves smooth blobs with noisy rings. First shipped AI6 (Nov 2021), revamped AI11. This is the single most praised differentiator vs StarNet.
- **Training data = the moat.** Croman (CN, Sept 2021): "a starless image has to have *no* other differences from the original other than the absence of stars… requires very careful hand editing. Otherwise the network will learn to replicate the mistakes." Pairs from his decades-deep multi-rig library + user-submitted failure cases; "the network is still training – takes days on a fast GPU" (2021); NXT "cooked" 6+ weeks; SXT AI11 = 2 weeks on 2 high-end GPUs **[talk-sourced]**. Demo images held out of training (anti-memorization).
- AI history: AI2 (12 Sep 2021) initial → AI3 wide-field/crowded galaxies → AI5 mono narrowband → AI6 noise matching → AI7 revamped arch + galaxy retention (Dec 2021) → AI8 retrained from scratch: bright stars, crowded fields, galaxy clusters, H-II (Mar 2022) → AI11 (Sep 2022) all-new arch, trained from scratch incl. **JWST + Hubble data** [verified]: long spikes, glare detail recovery, comet-tracked misregistered stars, faint-fuzzy retention, lensed-galaxy-vs-spike disambiguation.
- **Stretch-domain design [verified verbatim]:** SXT trained ONLY on MTF-stretched images; linear input is auto-MTF-stretched internally, "then precisely reverses it after processing." **GHS/arcsinh-stretched star profiles are "indistinguishable from small elliptical galaxies" and defeat removal** — documented FAQ weakness, directly exploitable (modern Siril users use GHS).
- **Unscreen math [verified]:** stretched extraction `stars = (original − starless)/(1 − starless)`; v2.0.3 adjusted "to avoid saturated pixels when one or more channels in the original image are clipped." Recombine = screen: `~((~starless)*(~stars))` = `combine(starless, stars, op_screen())`. Linear: **plain subtraction extraction (best star color)**, screen after stretching both.
- **Mask invariant [verified]:** v2.2.0 (Sep 2023) fixed masks so protected regions are excluded from BOTH the starless AND stars-only outputs (starless+stars ≡ original). Took them 2 years — bake this in from day one. Also propagates STF, astrometric solution, FITS headers, ICC profile onto the stars image.
- Design envelope [verified]: stellar **FWHM < 8 px** (3–4 ideal; >6 px → downsample 2x first); trained for only minor aberrations (light coma, field curvature, guiding error, RGB misregistration); AVX/AVX2 CPU required.

### 2.3 Why it wins / remaining weaknesses

Wins vs StarNet (community consensus: Alan Dyer/CN review, AstroBin threads, Utah Desert Remote): big bright stars + halos; spike removal post-AI11; fewer dark holes/mottling; **noise-matched voids**; faint-fuzzy retention; native linear handling; masks/batch/speed (RTX 4090 ~30 s on drizzled 26MP; RTX 3060 ~5 s; CPU 10 min–1h11m on 70MP).

Exploitable weaknesses:
1. **Stars-layer purity** (Wei-Hao Wang, AstroBin): faint non-stellar features leak into the extracted stars layer around bright stars (visible when the difference layer is stretched) — corrupts separate stars/nebula processing. Nobody optimizes the stars layer explicitly.
2. **GHS/arcsinh fragility** (documented, above).
3. **FWHM >8 px / oversampled data cliff**; aberrated corner stars (coma) not removed — **fast wide-angle lens nightscapes (14–35mm, f/1.4–2.8) are the documented weak domain and Andreas's exact niche**.
4. Removes small elliptical galaxies, comets, faint nebulosity near stars; "smudges" needing cleanup on Ha data; AI11 "too aggressive in nebula regions" vs AI10 (user reports).
5. Frozen AI since Sep 2022.

CLI surface (template): `rc-astro sxt input.fit --stars --unscreen --overlap 0.2 --device dml -o outdir --depth 32F --ml-version 0`; stars sidecar `<input>-stars.<ext>`; `rc-astro sxt --json` returns machine-readable param schema (schemaVersion 4) **[schema/NDJSON details from README-DEVS.txt, unverified publicly]**; install `C:\Program Files\RC-Astro\CLI`. Siril's official wrapper `StarXTerminator.py` (Cyril Richard, siril-scripts, GPL-3.0) handles FITS row-order flip (Siril bottom-up vs rc-astro top-down), native bit-depth exchange, stars sidecar, license gating — the proven shape for any Siril star tool.

**Legal red line [verified verbatim]:** the RC Astro Software License Agreement prohibits using "the Software, its outputs, or any data generated through use of the Software" to "create, train, fine-tune, test, benchmark for replication purposes, distill, validate, improve, or otherwise develop any machine learning model… intended to replicate, emulate, compete with…", explicitly including "the creation of training datasets or paired input/output datasets derived from the Software's operation." Safe harbor: "lawful independent development of competing technologies… developed without use of the Software." **SXT pseudo-labels are contractually radioactive. Zero teacher models anywhere.**

---

## 3. BlurXTerminator (BXT)

### 3.1 Facts [core verified]

First beta 8 Dec 2022; **$99.95**; PixInsight + CLI only — "Because deconvolution inherently requires linear image data, BlurXTerminator will not be made available for general photography applications such as Photoshop" [verified verbatim]. Bundle (BXT+NXT+SXT) $189.85. Version arc: 1.0.0 (Dec 2022) → 1.1.1 luminance-only (Jan 2023) → 2.0/**AI4 (14 Dec 2023, still the latest AI mid-2026)** → 2.1.5 (30 Apr 2026, Apple-silicon universal). AI: 1 (Dec 2022), 2 (Jan 2023), 3-beta2 (Aug 2023, never final), 4 (Dec 2023).

### 3.2 Training-on-synthetic-degradations methodology — the template to copy

From "The Mathematics of BlurXTerminator" (rc-astro.com, 5 Jan 2024) [verified; note the article's notation is `g = f*h + n`, i.e. g=captured, h=PSF]:
- Captured image = ideal image `f` convolved with PSF + noise. Decompose `f = f_stars + f_nonstellar`.
- **Training target is NOT the pristine image** — it is the ideal image re-convolved with a **smaller, parameterized, non-aberrated output PSF g′** (separately parameterized for stellar/nonstellar), **plus the ORIGINAL noise n** (noise kept in target — this is why outputs blend seamlessly; the tool does not denoise).
- Loss: `e = f_s*g_s′ + f_ns*g_ns′ + n − F[f*g+n, W]` — pure pixel-reconstruction loss. "This is in fact precisely how BlurXTerminator's convolutional network is trained" [verbatim]. The Moffat(sigma,beta) characterization of g′ is an interpretation — the article only says "parameterized" (diameter + other characteristics).
- Because g′ is a training-time parameter, the user sliders are conditioning inputs: the net is a **family of learned deconvolvers with controllable output PSF**, not an iterative optimizer. This makes the task well-posed and stable.
- Inputs are ground-truth sharp images synthetically degraded on the fly: parameterized PSFs + simulated aberrations + added noise. Aberration list (AI4): 1st+2nd-order coma & astigmatism, trefoil, defocus/field curvature, longitudinal+lateral chromatic aberration, motion blur (guiding), per-channel seeing/scatter variation, 2x-drizzle artifacts; trained up to ~9x star radius **[forum-sourced]**.
- **Correction (verifier):** the "trained on Hubble/JWST" claim applies to **NONSTELLAR detail only** — the docs explicitly state BXT "has not been trained on stellar profiles from instruments such as the Hubble and James Webb space telescopes" (why it fails on stars in actual HST/JWST frames). The stellar model is trained separately.
- Training set "effectively infinite": pairs generated on the fly, each seen ~20–50 times then discarded; user submissions used only as held-out TEST set **[talk-sourced]**. TensorFlow, Adam, MAE-style loss.
- **Correction (verifier):** the widely-quoted "SSIM/perceptual losses strictly avoided" phrase could NOT be located in any primary source. The confirmed framing: NXT "trained using purely discriminative (non-generative) techniques"; generative methods rejected because "they lack a rigorous closed-form loss function"; goal is recovering detail "without fabricating detail that does not in fact exist."

### 3.3 Operation

- **Blind LOCAL deconvolution [verified]:** 512x512 tiles, 20% overlap, processed semi-independently; per-tile PSF inferred from the stars in that tile (handles non-stationary PSFs — corners vs center). **Star-poor tiles fall back to inferring PSF from nonstellar texture and over-sharpen smooth areas** — the documented failure mode; remedy = manual PSF Diameter.
- Controls: Sharpen Stars 0–0.70 (AI4; 0.50 max AI2) = FWHM reduction; Adjust Star Halos −0.50..+0.50; Sharpen Nonstellar 0–1.0 (default 0.9) = fraction of PSF removed; manual PSF Diameter 0–8 px (CLI: PSF radius max 4 px = 8 px diameter, confirmed via 0.9.10 changelog); Correct Only; Luminance Only.
- **Correct Only mode** = make the local PSF azimuthally symmetric (round) + channel-consistent without shrinking it — equals Automatic PSF with all sharpening at 0; applied to nonstellar features too. PixInsight's own team (Conejero/Peris) runs Correct-Only on the MARS photometric database "because it improves the photometry in aberrated stars" — the strongest third-party fidelity endorsement.
- **AI4 processes linear data directly [verified]** (AI1–3 did internal MTF stretch + inverse, which wrecked flux: star flux 2.24 → 0.582 under AI2 vs 2.24 → 2.29 under AI4; second star 0.164 → 0.0449 AI2 vs 0.160 AI4 — ~2–4% error is the bar to beat). AI4 hard-requires linear input. Trained to conserve flux and maintain PSF centering (both imperfect per Croman). Sensor-clipping (saturated-star) training claim **[unverified]**.
- Sampling doctrine: 3–4 px FWHM well-sampled; >6–8 px = 2x oversampled → downsample first (Moffat FWHM 8 px attenuates Nyquist by ~140 dB); undersampled (<2–3 px) → drizzle 2x then deconvolve with doubled PSF diameter.
- Workflow: linear data post-integration → BXT Correct-Only → re-plate-solve (centering shifts astrometry) → SPCC → BXT sharpen → SXT → stretch → NXT. Never noise-reduce before BXT; never mix narrowband channels before BXT.
- AI1's cross-channel halo equalization shifted star colors → rolled back in AI2, re-attempted properly in AI4 + Luminance-Only escape hatch. Distinguishing per-channel scatter from real star color is near-impossible — same trap awaits us.

### 3.4 Hallucination debate + rebuttals

- Croman's rebuttals: (1) generative vs reconstructive is a property of **training**, not architecture; (2) pixel-reconstruction loss only; (3) universal approximation ("deconvolution is just a mathematical function"); (4) counterfactual: "to train a tool to paste in missing detail, we would train it by REMOVING detail and showing it what to put back — that's not how BXT is trained."
- Fiercest critique (Frank Freestar8n, CN, Apr 2025), unrefuted on its own terms: **BXT never emits the blur kernel it found, so it cannot be verified as formal deconvolution** ("AI conjuring… the input image is just a prompt"). Michael Covington (AI professional): "not deconvolution in the sense of finding a kernel… it finds a good approximation to what deconvolution would compute."
- No verified case of invented detail has been produced. Documented artifact classes: "plastic worms" in galaxy arms at high nonstellar sharpening; aberrated stars converted to fake close doubles; Airy-ring collapse; merged star-forming regions; drizzle pixel-pairing artifacts (fixed AI4); dark halos with Sharpen Stars high + Adjust Halos low; featureless voids around clipped stars at long FL.
- Croman's admitted limits (tech manual): centroids not perfectly maintained, flux not perfectly conserved, **shift-invariance NOT guaranteed**, results "may not be perfectly faithful to reality in all cases."
- **An open tool wins this argument by emitting per-tile PSF estimates + a residual map — nobody does this yet.** Croman's own validation approach: compare outputs of amateur data against actual HST/JWST imagery of the same field.

CLI: `rc-astro bxt image.tif --sharpen-stars 0.5 --sharpen-nonstellar 0.5` (--ss, --ash, --nsr 0–8, --ansr, --sn, --correct-only, --engine, --ml-version 0, --overlap [0,0.5], --depth {8U,16U,32F,64F}, wildcards, `rc-astro download-models`).

---

## 4. State-of-the-art ML

**No dedicated peer-reviewed paper on amateur-astrophoto star removal exists (2024–2026).** Closest academic prior art: **Madarász et al. 2025 (arXiv:2502.18345)** — Herschel compact-source removal [verified verbatim vs full text]: partial-convolution U-Net, 11 PCDown + 11 PCUp blocks, 96x96 patches, batch 64, Adam lr 5e-3 + ReduceLROnPlateau(patience 5, factor 0.5); trained by injecting synthetic PSFs at known fluxes (~72,570 sources, 51 flux levels 10–40,000 mJy, 25–210 per field) into real backgrounds; loss `L = 2·L1_valid + 6·L1_hole + 120·(L_style_out + L_style_comp, VGG-16 Gram) + 0.4·L_SSIM + L_source(flux)`; per-image normalization; SNR-adaptive masks; **validated with aperture photometry + power-spectrum checks** (no injected artifacts). Verifier correction: flux recovery is NOT "~1%" — Table 5 standard-star flux ratios span ~0.91–1.07 (±7–9% individual). Open MIT code: github.com/xMediMadix/compact-source-removal.

### 4.1 Architecture candidates (param counts / inference cost)

| Model | Params | Cost @256² | Notes |
|---|---|---|---|
| **NAFNet width32** | paper gives no count; ~29M per third-party measurements — **do not repeat "17.1M"** | ~16 GMACs | SIDD width32 = **39.97 dB**; pure convs + SimpleGate + SCA + LayerNorm, activation-free; **cleanest ONNX export** |
| NAFNet width64 | ~68M (third-party) | ~65 GMACs | SIDD **40.30 dB** (the headline number belongs to width64, not width32) |
| Restormer | 26.1M (third-party tables, arXiv:2310.11881-class) | 141 GFLOPs | channel-wise attention (MDTA) + GDFN; 4–9x NAFNet cost; used for galaxy restoration (arXiv:2404.00102) |
| MIMO-UNet+ | 16.1M | — | multi-scale in/out + **FFT reconstruction loss (MSFR)** |
| Uformer-B | 50.9M | — | not a better fit |
| HAT | 20.8M | — | SR-specific, wrong shape |
| MambaIRv2 (CVPR 2025) | < Restormer | linear global RF | beats Restormer by ≤0.29 dB but **ONNX-export nightmare — avoid** |

NAFNet configs (verified vs repo YAMLs — **corrected**): SIDD *denoising* = width 32/64, enc `[2,2,4,8]`, mid 12, dec `[2,2,2,2]` (36 blocks); GoPro *deblurring* (task closest to star removal) = width 32 or 64, enc `[1,1,1,28]`, mid 1, dec `[1,1,1,1]` (36 blocks). Optimizer is **AdamW** (betas [0.9,0.9], wd 0 for SIDD / 1e-3 GoPro), lr 1e-3 → **eta_min 1e-7** TrueCosineAnnealingLR, gt_size 256, total_iter 200K (GoPro/ablations) or 400K (SIDD apps), batch 4–8 per GPU (x8 GPUs = 32–64 effective), PSNRLoss. Block = LayerNorm → 1x1 → 3x3 depthwise → SimpleGate (channel split, elementwise multiply) → SCA (global avg pool → 1x1 → multiply) → 1x1.

Restormer progressive learning (verified): 300K iters; (patch, total batch): (128²,64) → 92K:(160²,40) → 156K:(192²,32) → 204K:(256²,16) → 240K:(320²,8) → 276K:(384²,8); AdamW (0.9,0.999) wd 1e-4; lr 3e-4→1e-6 cosine; L1 loss.

NTIRE 2026 denoising winners' recipe (arXiv:2606.16031): Restormer/HAT/MambaIR-class 15–50M params; L1/Charbonnier + frequency losses (SWT wavelet, FFT); progressive patches 128→256→512; AdamW cosine 1–2e-4; TLC; 8x geometric TTA; tiles 128–512 px, overlaps 32–256 px, Gaussian/Hann blending. **Zero GANs, zero diffusion in the fidelity track.** Data mattered more than architecture (+3.2 dB from data alone).

**Diffusion is disqualified** for layer separation: perception-over-distortion tradeoff, texture hallucination (IJCV 2025 survey, arXiv:2308.09388; NTIRE RAIM arXiv:2405.09923), stochastic outputs break the starless+stars contract, 10–100x inference cost.

### 4.2 Receptive field for huge halos (300–600 px)

- Effective receptive fields are far smaller than theoretical; a tile smaller than the halo makes the star unidentifiable *in principle*. Inference tiles must exceed the largest halo diameter → **512–1024 px tiles**, 4–5 downsampling scales (at 1/16–1/32 scale a 500 px halo is 16–31 px).
- **LaMa-style Fast Fourier Convolution (FFC) blocks in the bottleneck** (arXiv:2109.07161, verified): image-wide receptive field in every block; trained at 256 generalizes to ~2K. ONNX: needs opset 17+ DFT op or custom FFT wrapper — test export early; fallback = 31x31 depthwise large-kernel bottleneck convs (RepLKNet-style).
- Avoid dilated convs for long-range context (gridding artifacts on smooth nebulosity) — prefer strided downsampling/FFC.
- **TLC** (arXiv:2112.04491, verified): global ops (SCA's global avg pool) see training patches but full tiles at inference → documented PSNR drop; convert to local windows at test time (github.com/megvii-research/TLC, `NAFNetLocal`, base_size = training patch). TLC's unfold logic is **hostile to ONNX export** — either bake it out (train/fine-tune at the deployment tile size, e.g. 512) or reimplement export-safely. Tile inference at ≈ train-crop scale largely removes the need.
- Cheap insurance for monster halos: **two-pass scheme** — coarse pass on 4x-downsampled full image estimates the large-scale halo/spike field, fed as an extra input channel to the fine tiled pass (MIMO-UNet multi-scale input pattern).

### 4.3 Layer-separation constraints (Tool A)

- **Predict the STARS residual S, not the starless image**: `starless = input − S` with S ≥ 0 enforced by the head (ReLU/softplus). Recomposition is exact by construction; all error lands in one interpretable layer. Direct prior art in deraining decomposition networks (arXiv:2005.09228, arXiv:1804.02688, ACM Computing Surveys deraining survey). Never properly applied to star removal — Siril currently derives the star mask by post-hoc subtraction, inheriting StarNet's background errors.
- Do separation in **LINEAR space** where star flux is additive. For stretched data the exact algebra is screen/unscreen: `stars = (starry − starless)/(1 − starless)`; recompose `starless + stars − starless·stars`. Unscreen explodes near saturation (starless→1) — epsilon-guard (StarNet shipped exactly this bug until 2.5.2).
- Optional auxiliary soft star-mask output to focus the loss.
- Loss (fidelity phase): **Charbonnier (ε≈1e-3) on starless AND on the star layer** + FFT/L1 frequency loss (MIMO MSFR, weight ~0.1) to kill blotches + SSIM or VGG-Gram style term for background continuation under saturated cores (where L1-only produces the infamous smudges) + **explicit stars-layer purity penalty** (non-stellar content in the residual) — targets SXT's weakest axis. NO adversarial loss. Optional short LPIPS fine-tune at the end. Validate with power-spectrum comparison (Herschel protocol).

### 4.4 Deconv (Tool B): PSF-conditioned networks

- **Non-blind, PSF-as-input prior art:** USRNet (CVPR 2020, cszn/USRNet, verified) — deep-unfolded HQS-MAP: data module D (closed-form FFT deconvolution given kernel k + noise σ, no trainable params) + prior module P (ResUNet denoiser) + hyper-param module H; one model, any kernel. **SFTMD/IKC** (CVPR 2019) — kernel PCA-embedding stretched to a per-pixel map, injected via Spatial Feature Transform layers (`F′ = γ⊙F + β`); IKC insight: kernel mismatch has a recognizable signature (over-sharpening ring artifacts if PSF too big, over-smoothing if too small) → **jitter the conditioning PSF ±10–20% during training** so star-fit inaccuracy degrades gracefully. DWDN (NeurIPS 2020) = feature-space Wiener with explicit PSF, adapted to astronomy in arXiv:2210.01666 (evaluated on color/ellipticity/orientation recovery).
- **Space-variant PSF:** append PSF map/params as extra input channels per tile (arXiv:2503.00156 practice); classical spatially-varying PSF interpolation = PSFEx/Piff low-order polynomial fields (review: Liaudat, Starck, Kilbinger 2023, arXiv:2306.07996). **A smooth polynomial PSF field sampled per overlapping tile beats BXT's per-tile-only inference on smoothness** (BXT-style per-tile PSF can create tile-boundary discontinuities in strongly aberrated fields).
- Blind vs non-blind: standalone U-Net blind decon beats Tikhonov at low SNR and saturates past ~5k training images (Campagne, arXiv:2601.08666, Jan 2026 — but results are on 48x48 GalSim galaxy cutouts; **PSF conditioning remains the safer bet** for full frames with saturated stars/gradients).
- **Hallucination prevention:** learned-RL hybrids bake in physics (RLN, Nature Methods 2022, ~16k params; LUCYD arXiv:2307.07998; DELAD; IJCV 2024 deep-RL for saturated deblurring). Deep decon nets systematically **suppress faint flux (photometric bias)** even without visible artifacts — Akhaury et al. (arXiv:2211.09597; arXiv:2405.07842, A&A 2024) needed an explicit wavelet-domain debiasing step; U-Net best at flux preservation, Learnlet minimal-hallucination. Euclid+Rubin multiband decon (arXiv:2502.17177) trains with **PSF re-convolution consistency + flux-conservation loss terms**. **CHEM** (arXiv:2512.09806, Dec 2025): conformalized quantile regression over wavelet/shearlet coefficients yields distribution-free hallucination maps; gives approximation-theoretic reasons U-shaped nets are hallucination-prone → counter with residual prediction + re-blur data consistency + no GAN/diffusion.
- Tool B loss stack: `Charbonnier + λ1·||PSF ⊗ output − input||₁ (re-blur consistency) + λ2·flux-conservation (regional sum preservation) + optional multi-scale gradient loss`; NO adversarial. Adopt BXT's target-PSF formulation: map (image, source PSF, target FWHM) → image at target PSF — gives a continuous principled strength slider and a well-posed task.
- Validation: PSNR/SSIM on synthetic round-trips + star aperture-photometry error (<1–2% flux target; beat BXT's ~2–4%) + centroid/shift-invariance tests + CHEM conformal hallucination maps + Hubble-reference blink tests.

### 4.5 Tiled inference seam handling

Consensus recipe: 512 px tiles (1024 on GPU), 64–128 px overlap, **Hann/Gaussian window feathering** accumulated into numerator/denominator buffers; seam elimination needs BOTH overlap-feathering AND statistical consistency (TLC or tile size = training patch size); 8x geometric TTA quality mode ≈ +0.1–0.3 dB. Alternative proven stitcher (CosmicClarity): drop 16 px tile borders, weight-accumulate, reflect-pad to multiples of 32.

---

## 5. Training data strategy

### 5.1 Precedents

- **StarNet path (manual pairs):** quality ceiling = the hand-edit; dataset private. Croman calls hand-editing his most expensive step and biggest error source ("hand edit for hours and hope you get it right; the network will faithfully learn all of your mistakes").
- **nox (github.com/charvey2718/nox, MIT code AND weights) [verified]:** StarNet-architecture clone trained on **fully synthetic** pairs — stars injected onto NON-astronomical backgrounds (SIDD Small 160 pairs + RENOIR 120 pairs, camera-denoising photos); 5,000 images is the *recommended* generation count (count behind released weights unstated). Cross-optics generalization is the author's stated **hypothesis** ("whether I have succeeded remains to be proved"), and "StarNet-equivalent quality" his aspiration, not a benchmark. Lesson: synthetic injection works mechanically, background diversity matters, but **synthetic-only never dethroned anyone — train hybrid**.
- **BXT path (synthetic degradations of clean ground truth):** the proven-at-scale template — see §3.2.

### 5.2 Synthetic star injection pipeline (Tool A ground truth generator = benchmark generator)

**PSF rendering — GalSim v2.8.x (verified API):**
`galsim.OpticalPSF(lam=<nm>, diam=<m>, defocus/astig1/astig2/coma1/coma2/trefoil1/trefoil2/spher=<waves> or aberrations=[Noll array], annular_zernike=bool, obscuration=<0..1>, nstruts=N (default 0), strut_thick=<frac, default 0.05>, strut_angle=<Angle>, pupil_plane_im=<arbitrary pupil image/array/file>, pupil_angle=, oversampling=1.5, pad_factor=1.5)` — **real diffraction spikes fall out of the FFT of the pupil**; `pupil_plane_im` supports arbitrary masks (Bahtinov, lens cross-screens, aperture blades). Atmosphere: `galsim.Kolmogorov(fwhm=|lam+r0)`, `galsim.VonKarman(lam=, r0=, L0=25.0)`. Analytic: `galsim.Moffat(beta=, fwhm=|scale_radius=, trunc=)`, `galsim.Airy(lam=, diam=, obscuration=)`. Combine: `galsim.Convolve([optical, atmospheric])`.
Lighter path: photutils 2.x `make_model_params / make_model_image / make_noise_image / apply_poisson_noise` (older names `make_random_models_table`/`make_model_sources_image` renamed — pin version) + astropy `Moffat2D(amplitude, x_0, y_0, gamma, alpha)`; FWHM = `2·gamma·sqrt(2^(1/alpha) − 1)`.

**PSF realism beyond Moffat:** real bright-star profiles = Moffat wings + **SkyMaker-style diffusion "aureole"** (Bertin 2009 — wings beyond a few FWHM creating the background glow) + internal-reflection halos (annuli, often color-fringed, sometimes offset). Halos from reflections/haze/CA are a documented SXT failure — synthesize them deliberately or inherit the failure.

**Star statistics — Gaia-anchored:** density ~few 10³ stars/deg² at high |b| to ~10⁶/deg² near the plane (Gaia EDR3 validation vs TRILEGAL); BP−RP→Teff polynomials (arXiv:2106.03882, σ 60–80K) → blackbody RGB. Train across the full density range or fail in Cygnus/Carina; all-dense training biases toward eating nebular knots.

**nox recipe as skeleton (MIT, GenerateStars.py, exact params):** 2500 small Gaussian2D stars (stddev = FWHM0/2.35 × U[1/1.1,1.1], 10% ellipticity jitter) + 300 Moffat2D (alpha∈[2,10]) + 20 large bloomed (FWHM triangular up to 15 px); amplitudes 10^triangular(0,0,1)−1; temps triangular(800,800,30000)K → kelvin_to_rgb + saturation U[0,1]; chromatic fringing = luminance-dependent hue shift; spikes = 2 perpendicular elongated Gaussians + blur σ∈[0.5,1.0]; Poisson noise gain∈[0.1,2]; midtones stretch m∈[0.01,0.1]; screen compositing. **Upgrade each piece:** GalSim pupil-FFT spikes per rig (refractor = no spikes + halos; Newtonian 4-vane; 3-vane; Bahtinov; lens cross-screen), aureole wings, saturation clipping at full-well AFTER profile build + bleed trails, offset reflection halos, per-channel PSF scale (CA).

**Compositing order matters:** stars are additive photons in LINEAR space. Correct pipeline: inject in linear → shot/read noise → optional resampling chain (drizzle/Lanczos-3/bilinear/bicubic — Lanczos negative-lobe ringing is a real domain gap that measurably degraded SXT vs StarNet2) → identical random invertible stretch (MTF/arcsinh/GHS) applied to input AND target. Noise BEFORE stretch. Fluxes spanning 5+ decades; FWHM 1–12 px; densities 100 → >30k stars/frame; emit linear + stretched variants.

**On-the-fly generation in the dataloader** (fresh star field per background crop per sample) — the 7950X/128GB pre-renders PSF banks per "optical rig" (a few hundred rigs × FWHM/aberration grid), so per-sample injection is cheap convolve+paste. Each pair seen ≤~50 times (BXT's "effectively infinite" discipline).

**Real-star-stamp track (teacher-free):** harvest isolated star stamps (+ local background subtraction) from sparse high-latitude real frames and paste onto starless backgrounds — captures true PSFs/halos/bleed without touching any teacher model (crude precedent: code2k13/starreduction).

### 5.3 Starless backgrounds + licensing [verified]

| Source | License | Notes |
|---|---|---|
| ESA/Hubble + ESA/Webb outreach mosaics | **CC BY 4.0** (esahubble.org/copyright) | credit required; full-res TIFFs |
| NASA/STScI (hubblesite/MAST) | public domain / "no copyright asserted" | courtesy credit expected |
| DESI Legacy Surveys (DECaLS/BASS/MzLS) | **CC BY 4.0**, credit "Legacy Surveys / D. Lang (Perimeter Institute)" (legible, unaltered, not disassociated) | cutout service at legacysurvey.org |
| DECaPS2 (nebula-rich galactic plane) | public (decaps.skymaps.info, max 512 px cutouts, 0.262″/px; NOIRLab Astro Data Lab) | covers the plane |
| **DSS2** | **© AURA 1994, per-plate holders, commercial use PROHIBITED — AVOID** | the obvious-looking trap |
| AstroBin uploads | users retain full copyright — **scraping is not licensed**; per-author consent or community data call only | |
| Procedural | multi-octave Perlin/fBm nebulae | proven diversity filler |
| Natural images (SIDD/RENOIR) | dataset licenses | prevent nebula-shape overfitting; blend, don't substitute |

Background QC is the single most quality-critical dataset property: run photutils DAOStarFinder over every candidate crop; reject/inpaint crops with point sources (survey mosaics are already star-processed/stretched differently — screen for residual stars).

**Pseudo-label ethics/ceiling:** hard NO on both incumbents — RC-Astro EULA explicitly bans it (§2.3); StarNet weights CC BY-NC-SA (NC + SA taint); strategically it caps quality at teacher+artifacts. Dark Star (MIT incl. released weights) is the one legally-open baseline; courteous to confirm intent with Franklin Marek before deriving.

### 5.4 Dataset sizes + 4090 training time

- Scale: nox competitive from ~480 backgrounds; NAFNet SIDD SOTA from 320 scene pairs cropped to 256 over 400K iters. **Target 1–5k distinct clean backgrounds + on-the-fly injection = effectively unlimited pairs.** Blind-decon U-Net saturated past ~5k images (arXiv:2601.08666).
- 4090 estimates (derived, not published): NAFNet-width32 @256, 200K iters ≈ **6–10 h (overnight)**; width64 @256, 400K iters ≈ **1.5–2 days** (fwd+bwd ≈ 3x fwd FLOPs; ~0.25–0.4 s/iter at 40–60 sustained bf16 TFLOPS); 512 px fine-tune stage (batch 8–16, 50–100K iters) ≈ **1–2 days**. 24 GB fits width64 batch 8–16 @512² with AMP/grad-checkpointing; width32 @256 batch 32+.
- Cadence: **weekend per full experiment → 5–10 serious data-recipe iterations/month.** Data-recipe iteration is where the quality lives (Croman: "training data and the loss function are the crown jewels — the network architecture itself is somewhat secondary"). Croman's own budgets: SXT AI11 = 2 weeks on 2 GPUs; NXT ~2 months — one 4090 ≈ that 2022 budget.

---

## 6. Deployment

### 6.1 sirilpy ONNXHelper — exact API [verified against gitlab.com/free-astro/siril master `python_module/sirilpy/gpuhelper.py`]

Siril 1.4.2 (2026-02-18), sirilpy API 1.1.10. Classes: `ONNXHelper`, `TorchHelper`, `JaxHelper`, `detect_gpu_capabilities()`.

```python
oh = sirilpy.ONNXHelper()
oh.ensure_onnxruntime()          # detect + pip-install correct wheel per platform
providers = oh.get_execution_providers_ordered(ai_gpu_acceleration=True, force_check=False)
result, session = oh.run(session, model_path, output_names, input_feed,
                         run_options=None, return_first_output=False)
```
- Backend selection: **Windows + NVIDIA/AMD/Intel-iGPU → 'directml' (onnxruntime-directml); Windows + Intel Arc → 'openvino'; Linux + NVIDIA → onnxruntime-gpu (CUDA); Apple Silicon → CoreML; ROCm → CPU.** CUDA is **never auto-selected on Windows** — Siril docs: onnxruntime-gpu's CUDA/cuDNN dependency is "too fragile to be reliable" for auto-install. Your 4090 users run **DmlExecutionProvider**.
- `get_execution_providers_ordered()` empirically validates each EP against CPU reference output on a tiny generated model (rtol 1e-3/atol 1e-5; relaxed 5e-2/1e-3 under TF32) and JSON-caches the working list.
- `run()` catches **ANY** exception, rebuilds a CPUExecutionProvider session, re-runs, returns `(result, session)` — a coding bug can masquerade as "GPU unavailable" and run 100x slower; validate inputs before the hot loop and log the actual provider.
- No VRAM detection; no DirectML device_id selection (adapter 0 = primary display) — expose an override on multi-GPU boxes.
- Image I/O: `siril.get_image_pixeldata()` (uint16 or float32; (H,W) mono or (C,H,W) planar), `get_image_fits_header()`, `undo_save_state(msg)`, `with siril.image_lock(): siril.set_image_pixeldata(out)`, `save_image_file()`, sequences via `get_seq_frame_pixeldata(i)/set_seq_frame_pixeldata(i, arr, prefix=)/create_new_seq(prefix)`, models dir `siril.get_siril_userdatadir()`. GUI standard: **PyQt6 (tkinter deprecated)**; deps via `sirilpy.ensure_installed(...)` (no == version pins — cross-script conflicts).

### 6.2 onnxruntime EPs on the 4090 / fp16 / speed

- Versions (July 2026): mainline ORT 1.27.0 (2026-06-15); **onnxruntime-directml 1.24.4 (2026-03-17) — lags ~3 releases, caps at ONNX opset 20 (DirectML 1.15.2)** → export opset 17–20, test against onnxruntime-directml specifically. DirectML is in "sustained engineering" (Microsoft Learn, verified); forward path = WinML + NVIDIA TensorRT-RTX EP (claims up to 2x DirectML **[unverified marketing]**) — keep provider handling generic via ONNXHelper.
- DirectML EP requirements [verified vs official docs]: no memory-pattern optimization (`enable_mem_pattern=False`), `ORT_SEQUENTIAL`, one thread per session, **static shapes strongly preferred** ("works most efficiently when tensor shapes are known at session creation"; AddFreeDimensionOverrideByName workaround). → **Export FIXED 512x512 input (optionally +256), pad tiles to size** — also sidesteps every dynamo-exporter dynamic-shape bug.
- Session pattern (CosmicClarity_Native.py): `so = ort.SessionOptions(); so.graph_optimization_level = ORT_ENABLE_ALL; so.execution_mode = ORT_SEQUENTIAL; ort.InferenceSession(path, sess_options=so, providers=providers)`; warm up with a dummy 512x512 run. (Note: that script actually leaves enable_mem_pattern=True.)
- fp16: `onnxconverter_common.float16.convert_float_to_float16(model, keep_io_types=True, op_block_list=[...])` or `auto_convert_mixed_precision(model, feed_dict, rtol, atol, keep_io_types=True)`; halves file size (~35 MB vs ~65–70 MB fp32 for a width32-class model). **Risks:** overflow/underflow in LayerNorm stats and SCA global pooling; RC-Astro's fp16 CoreML path caused star-core posterization. Verify on real linear astro tiles incl. saturated stars; ship fp32 default, fp16 opt-in.
- Speed anchors: DirectML ≈ 2.1x slower than CUDA on identical hardware (SAM benchmark: 780 vs 370 ms, ORT issue #19352 **[unverified]**); Cosmic Clarity users: ~10 s GPU vs 15–30 min CPU full frame. Estimate for 30MP @512 tiles/64 overlap (~150 tiles, width32-class ~64 GMACs/tile): **~3–10 s on 4090 DirectML; 1–5+ min on 7950X CPU** (derived estimates). Batch tiles on GPU / pipeline tile prep — StarNet's tiling+merge is CPU-bound and caps its GPU utilization.

### 6.3 PyTorch → ONNX pitfalls

- PyTorch 2.13.0 stable (2026-07-08); Windows CUDA wheels cu126/cu130/cu132 (cu128 last in 2.11); 4090 = sm_89, bf16-native; `pip install torch --index-url https://download.pytorch.org/whl/cu130`. torch.compile works on Windows (Triton official Windows support; woct0rdho/triton-windows archived 2026-02-18 after upstreaming; CUDA ≥12.8).
- **torch.onnx.export defaults to dynamo=True since PyTorch 2.9** [verified]; `dynamic_shapes=` replaces `dynamic_axes=`; `dynamo=False` legacy fallback remains. Known killers: `F.interpolate`/Resize with dynamic output sizes (pytorch #124884, #149826), antialiased resize `_upsample_bilinear2d_aa` (#148840). **NAFNet avoids all of it** (PixelShuffle → DepthToSpace, stride-2 convs). If adding upsampling, use scale_factor, never size=.
- Export: `torch.onnx.export(model, (torch.randn(1,3,512,512),), 'model_512.onnx', opset_version=17)` static; parity check `np.allclose(torch_out, ort_out, atol=1e-4)` on real astro tiles incl. saturated stars. FFC blocks need opset 17+ DFT — test export before committing to the architecture.
- Training loop modernization: bf16 autocast, channels_last, `torch.set_float32_matmul_precision('high')`, torch.compile, EMA weights, DataLoader spawn on Windows (`persistent_workers=True`, `if __name__=='__main__'` guard).

### 6.4 Codebases + licenses [verified]

- **NAFNet** (megvii-research): LICENSE = MIT (megvii code) + Apache-2.0 (BasicSR portions) in one file — GitHub API shows NOASSERTION but it is permissive; keep both notices. Frozen (last push Jul 2024), pins ancient BasicSR — **vendor the ~300-line arch file, don't pip-install the repo**.
- BasicSR (XPixelGroup): Apache-2.0, dormant (Jul 2024). **neosr**: Apache-2.0, actively maintained modern successor. Restormer (swz30): MIT, pushed Oct 2025. For a single-task project, a lean custom training script borrowing BasicSR pieces is lower-friction than any framework.
- Reference scripts: `processing/CosmicClarity_Native.py` (native ONNX template: PyQt6+QThread, HF models.zip download via plain requests streaming to `{siril_userdata}/cosmic_clarity/`, 256 px chunks/64 px overlap, 16 px border-ignore weighted stitching, reflect-pad to multiples of 32, 512 warmup) and `processing/StarNet.py` (star-layer math: unlinked MTF autostretch, `AS_DEFAULT_SHADOWS_CLIPPING=-2.80`, `AS_DEFAULT_TARGET_BACKGROUND=0.25`, MAD_NORM=1.4826, pseudo-inverse MTF; four star-layer modes: mask/unscreen/descreen/subtract; `subtract = clip(orig−starless,0,1)`, `descreen = clip((orig−starless)/(1−starless) where |1−starless|≥1e-6, 0, 1)`). siril-scripts repo: MIT default with per-file headers taking precedence (official wrappers are GPL-3.0-or-later); submission = plain GitLab MR, no membership.
- **Model distribution:** GitHub Releases (2 GiB/file, free bandwidth, 1000 assets) + Hugging Face mirror (`https://huggingface.co/<user>/<repo>/resolve/main/...`, plain requests, no huggingface_hub dep; anonymous rate limit 3,000 resolve req/5 min). A ~35–70 MB ONNX vs StarNet's ~700 MB TF checkpoint is itself a headline. License split: code MIT (NAFNet notices kept), weights Apache-2.0 or CC-BY-4.0 **with an explicit statement that outputs are unrestricted** — a clean permissive story is a selling point vs freeware-closed StarNet and proprietary SXT.

---

## 7. Competition & evaluation

### 7.1 The field (mid-2026)

| Tool | Status | License | Key facts |
|---|---|---|---|
| **StarNet2 2.5.3** (Misiura) | free, closed weights | v1 code MIT / v1 weights CC BY-NC-SA; v2+ closed | §1; ONNX/DirectML/CoreML since 2.5.0; unscreen output; highlight protection |
| **StarXTerminator AI11** (RC-Astro) | $49.95; free CLI since Jun 2026, **now natively scripted inside Siril 1.4.4** | proprietary + anti-distillation EULA | §2; reputation leader big stars/spikes/noise-matched fills |
| **Cosmic Clarity "Dark Star" v2.1c** (Seti Astro / Franklin Marek, 2025-06-29) | free | **MIT incl. released .pth AND .onnx weights** (github.com/setiastro/cosmicclarity) | the one legally-open baseline; official Siril script exists |
| **SyQon Starless "Zenith" v1.3** (Dec 2025/Jan 2026) | free, **Siril-exclusive license** | closed | NOT encoder-decoder: native-resolution processing, progressive local+global aggregation; ~2x slower by design; commercial **Axiom** line (Axiom2 in StarLess v1.6 Feb 2026 "often surpassing SXT"; Axiom3 mid-2026) **[Axiom versions unverified — syqon.it unreachable]**. Verifier: SetiAstro Suite Pro has its OWN SyQon-supplied model — only PixInsight is locked out |
| **GraXpert 3.1.0rc** | free, open | — | **NO star removal.** AI BGE + denoise + two decon models. Verifier: timeline is 3.1.0rc1 = **Nov 2024** (object decon), rc2 = **Jan 2025** (stellar decon); models stellar-1.0.0 / object-1.0.1 published **2025-01-02**; no stable 3.1.0 (latest stable 3.0.2, May 2024). Decon knobs: strength 0–1, image FWHM 0–10 px only |
| Cosmic Clarity **sharpen v6.5** (Tool-B rival) | free | MIT | SharpeningCNN = naive 10-layer plain conv net 3→16→32→64→128→256→128→64→32→16→3, all 3x3, ReLU, sigmoid, NO skips/norm/attention/downsampling (verified from source); PSF "handled" by 4 fixed-radius nonstellar models (radius 1/2/4/8, 3.1 MB each; deep variants 7.9 MB) + 1 stellar |
| nox (2023), code2k13/starreduction, Straton, PixelMath star-reduction scripts | abandonware/minor | MIT/various | cautionary tales |

**Dark Star architecture** (read from setiastrocosmicclarity_darkstar.py, verified with correction): DarkStarCNN U-Net — 5 encoder stages Conv(3→16→32→64→128→256)+ReLU, dilation=2 at stages 3 and 5, encoder ResBlocks 3/3/2/2/1 (ResBlock = Conv3x3-ReLU-Conv3x3-**ReLU** + skip); mirrored decoder with skip concats, **decoder ResBlocks 2/2/3/3/2 (NOT 2 each — verifier correction)**; head Conv(16→3)+Sigmoid; optional RefinementCNN, 8 conv layers, 96ch, dilations 1,2,4,8,8,4,2,1, stage-1 frozen. Tiling: 512 px chunks, 64 px (12.5%) overlap, linspace-ramp blending, 5 px median border pad, adaptive 0/16 px stitch crop; ONNX session force-overrides chunk/overlap to the model's fixed dims. Auto-stretch when median<0.125 (unlinked per-channel, target 0.25, exact inverse). Star extraction: additive `orig−starless` and unscreen `(orig−starless)/(1−starless)`. Backends: Windows PyTorch CUDA → ONNX DirectML → CPU; Mac CUDA→MPS→CPU. Dual mono/color models auto-selected by np.allclose RGB-homogeneity check.

**What separates winners in published comparisons:** (a) big-bright-star handling (core mounds, halo remnants, spikes); (b) fill quality (StarNet worms/blotches vs SXT smudges vs Zenith clean-but-denoised plastic risk); (c) preservation (SXT eats faint structure near stars + small ellipticals + comets; Zenith keeps tiny galaxies — its headline praise; early Zenith erased dim low-contrast nebulosity, "total fail" on bright star pairs with spikes); (d) star-layer purity for recomposition; (e) completeness on tiny stars/dense fields; (f) speed. **All incumbents are telescope-centric — wide-angle lens nightscapes with corner coma/astigmatism/chromatic halos at 14–35mm are the documented gap.**

### 7.2 Measurable evaluation protocol (automatable)

**Layer 1 — synthetic ground truth scorecard.** Build verified-starless backgrounds (multi-tool consensus + manual QC + synthetic fBm nebulae); inject catalog-driven stars with the §5.2 generator (Moffat β 2.5–4.5, FWHM 1–12 px sweep, elliptical/coma-distorted PSFs increasing with field radius, saturated cores, screen-composited halos + chromatic fringes, 4/6-vane + hex spike templates, densities to Milky Way core; noise after compositing; linear + stretched variants). Score:
- Starless MAE/PSNR/SSIM **split in-footprint (fill quality) vs out-of-footprint (do-no-harm — should be ~zero change; directly targets SXT's near-star edits and Zenith's incidental denoising)**. Whole-image PSNR is misleading (most pixels untouched).
- Removal completeness = DAOStarFinder/SEP residual detections on (output − GT background), matched to the injection catalog, binned by magnitude and FWHM.
- Star-layer metrics: per-star **Flux Error** `FE = (1/N)Σ|v_gt,i − v_pred,i|` via identical elliptical-aperture photometry (pattern from the STAR benchmark, arXiv:2507.16385 — verified: 70 HST wide-fields, 54,738 flux-consistent pairs; note it is a **super-resolution** benchmark, cite as a metric pattern only); color-ratio error; background leakage = p95 of star layer outside footprints; halo-capture fraction in annuli.
- Power-spectrum comparison (Herschel protocol) to prove no injected structure.

**Layer 2 — recomposition round-trip.** `screen(starless_out, stars_out)` (or add, linear) vs original must sit at/below quantization error — **target ≥60 dB PSNR on 16-bit**. Plus **idempotence**: the tool run on its own starless output must be a no-op. Exact-by-construction with the residual head — a marketing point no competitor can claim. Mask invariant: protected regions excluded from BOTH outputs.

**Layer 3 — blind A/B on a fixed torture panel.** Auto-crop 1:1 patches at the 5 brightest stars, highest-gradient nebula sites, catalog-matched small galaxies, densest star region; randomized presentation; Bradley-Terry/Elo aggregation; no-reference companion = SEP residual counts + PSF-matched-filter residual energy. Panel (20–30 images): dense MW-core nightscape at 14–20mm; Antares/Rho Oph halo field; Vega/Sirius with lens-blade spikes; M13 globular (Omega Cen = hardest case); M31 + foreground stars; NGC 6960 star carpet; IC 434 narrowband; comet + tail; undersampled FWHM≈1.2 px; oversampled FWHM 8–12 px (SXT's documented cliff); f/1.4–1.8 corner-coma frames; small-galaxy "must-NOT-remove" fields (object-wise flux retention ≥98% — comets/ellipticals/HII knots are where reviewers decide winners).
Record competitor versions in every run: SXT **AI 11** (plugin version is meaningless), StarNet2 **2.5.3** (protect-highlights on AND off), Dark Star **v2.1c**, Zenith **v1.3** — May–Jun 2026 StarNet releases invalidate all older published comparisons. Never judge star residuals with STF autostretch alone (Croman's "flashlight into your telescope" warning) — but expect reviewers to do it, so make faint residuals genuinely near-zero.

**Tool B gates:** synthetic degrade→restore round-trip PSNR/SSIM; aperture-photometry preservation <1–2% flux error (vs BXT AI4's ~2–4%); centroid stability; shift-invariance test; CHEM conformal hallucination maps; Hubble-reference blink test on M51/M42; **publish per-tile PSF estimates + residual maps** (the verifiability nobody offers); classical baseline = Siril RL (`rl [-loadpsf=] [-alpha=3000] [-iters=10] [-tv|-fh] [-mul]`, `sb`, `wiener`, `makepsf blind [-l0|-si] | stars [-sym -ks=] | manual {-gaussian|-moffat|-disc|-airy}`; PSF-from-stars fits Gaussian/Moffat on linear data, star amplitude 0.07–0.7).

---

## 8. Verification corrections (all applied inline above)

1. **NAFNet config conflation:** enc [2,2,4,8]/mid 12/dec [2,2,2,2] is the SIDD *denoising* config; the GoPro *deblurring* config is enc [1,1,1,28]/mid 1/dec [1,1,1,1]. Optimizer is **AdamW** (not Adam); eta_min **1e-7** (not 1e-6); batch figures are per-GPU×8. **SIDD 40.30 dB belongs to width64 (~65 GMACs); width32 = 39.97 dB.** The "17.1M params width32" figure is unconfirmed (third-party ~29M) — dropped.
2. **Herschel paper flux claim:** no "~1%" accuracy stated; Table 5 flux ratios ~0.91–1.07 (±7–9% individual). Qualitative conclusion stands.
3. **GraXpert timeline was off by one year and inverted:** 3.1.0rc1 = Nov 2024 (OBJECT decon), rc2 = Jan 2025 (STELLAR decon); models published 2025-01-02; still no stable 3.1.0. GraXpert decon has been in the field ~18 months, not brand-new.
4. **BXT Hubble/JWST training scope:** nonstellar detail only — stellar profiles explicitly NOT trained from HST/JWST; stellar model trained separately.
5. **"SSIM/perceptual losses strictly avoided":** quote not found in any primary source — use the confirmed "purely discriminative (non-generative)" framing; Moffat(sigma,beta) output-PSF parameterization is interpretation, not stated.
6. **SyQon exclusivity nuance:** only the Zenith model is Siril-exclusive; Seti Astro Suite Pro ships its own SyQon-supplied starless model. Only PixInsight is locked out.
7. **Dark Star decoder:** ResBlocks 2/2/3/3/2 per stage (not 2 each); ResBlock has a trailing ReLU.
8. **RC-Astro CLI GPU support:** "any D3D12 GPU incl. AMD/Intel" is **Windows-only**; Linux currently NVIDIA-only; macOS = CoreML (macOS 14+). HISTORY cards only added in 0.9.10 itself.
9. **sirilpy backend map:** Windows → DirectML for NVIDIA/AMD/Intel-iGPU but **Intel Arc → OpenVINO**; CUDA never auto-selected on Windows.
10. **SXT price:** $49.95 confirmed on current product/store pages (two verifiers); one verifier found $59.95 via cached FAQ excerpts — treat $49.95 as current, recently reduced.
11. **BXT math notation:** the article writes g = f*h + n (g = captured image, h = PSF) — reversed from some research notes.
12. **STAR benchmark (arXiv:2507.16385)** is a super-resolution benchmark, not star removal — cite only as a Flux-Error metric pattern.
13. **Unverified-but-plausible bucket** (forum/talk-sourced; do not present as fact): SXT 1024-tile size + `starX/dd1_xconv` OOM trace + "AI11 smaller than AI10"; param counts 21M/24M/54M/30M; 2-week/2-GPU training; "effectively infinite"/20–50-reuse quotes; BXT ~9x-radius aberration limit; sensor-clipping training; `--json` schemaVersion 4 / NDJSON v3; TensorRT-RTX "2x DirectML"; SAM DML-vs-CUDA 780/370 ms; all RTX 4090 wall-clock estimates (derived); StarNet `-u` "4x time" (Siril docs, not official); default stride 256 (Siril docs); StarNet 2.5.0 date 05-26 vs 05-27 (macOS build); Cosmic Clarity "v6.5 AI3.5 Dec 2025" version label; Zenith "requires stretched input"; nox "StarNet-equivalent" quality (author's aspiration).
14. Minor: StarNet v1 licensing framing sharpened everywhere — **code MIT, weights CC BY-NC-SA** (clean-room retrain is fine; weight reuse/distillation is not). CosmicClarity_Native.py actually sets enable_mem_pattern=True (does not follow DML's DisableMemPattern guidance).

---

## 9. Design implications — the winning blueprint

### 9.1 Tool A — star removal ("StarForge Stars")

- **Backbone:** NAFNet-style U-Net (start width32/36-block; width64 "quality" model later), 4–5 downsampling scales, 2–4 FFC (LaMa) blocks in the bottleneck for image-wide receptive field on 300–600 px halos (fallback: 31x31 depthwise large-kernel bottleneck if FFT export blocks). GroupNorm/LayerNorm or norm-free — never BatchNorm. No TLC in the shipped graph — fine-tune at 512 so train stats match the deployment tile.
- **Head:** predict non-negative star residual S (ReLU/softplus); `starless = input − S` — exact recomposition by construction; optional auxiliary soft star mask. Signed-residual variant only if halo-flare correction demands brightening.
- **Domain:** train natively in linear float (clipped-star augmentation) + random invertible MTF/arcsinh/**GHS** stretch augmentation so one model serves both workflows (kills SXT's GHS hole and StarNet's stretched-only lineage). Per-image invertible normalization, recorded and inverted at output. Accept float FITS (StarNet still rejects float).
- **Losses:** Charbonnier on starless AND star layer + FFT frequency loss (~0.1) + style/SSIM term for background continuation under saturated cores + stars-layer purity penalty. No GAN. Power-spectrum validation.
- **Second stage (the SXT-killer feature):** separate noise-synthesis pass filling star voids with noise matched to local spatial-frequency + per-channel statistics — absent from every free tool; any perceptual/LPIPS spice lives ONLY here, never on the structure path.
- **Differentiation targets:** stars-layer purity; GHS/arcsinh robustness; FWHM 1–12 px + lens-aberrated corner PSFs (nightscape niche — Andreas's 14–35mm f/1.4–2.8 domain, where every incumbent trained on telescope data); highlight protection default-on; auto linear/stretched detection (median<0.125) with hard-enforced matched extraction/recombination pairs (subtract↔add linear, unscreen↔screen stretched, epsilon-guarded).

### 9.2 Tool B — deconvolution ("StarForge Sharp")

- **Backbone:** same NAFNet-class body (5–30M params), residual prediction toward the sharp image, **PSF-conditioned via SFT layers** fed fitted Moffat params (FWHM, ellipticity, angle, beta) or a rendered PSF thumbnail per tile — one model for all optics (vs Cosmic Clarity's 4 fixed radii and GraXpert's scalar FWHM).
- **PSF estimation:** reuse the EasyReg star detector; fit per-star Moffat on LINEAR data (exclude saturated, amplitude >0.7); fit a low-order 2D polynomial PSF field (PSFEx/Piff pattern) across the frame; sample per overlapping tile and blend — smoother than BXT's per-tile inference.
- **Task formulation:** BXT's target-PSF trick — (image, source PSF, target FWHM) → image at target PSF; noise kept in target; separate stellar/nonstellar target-PSF maps over the star mask (synergy with Tool A). Correct-Only mode = symmetric+channel-consistent PSF at same size. Jitter conditioning PSF ±10–20% in training (IKC lesson).
- **Losses:** Charbonnier + re-blur PSF-consistency + flux conservation; optional wavelet debiasing pass; no GAN/diffusion.
- **Trust weapons:** emit per-tile PSF estimates + residual map (answers Freestar8n's "no kernel = not verifiable" critique that Croman never addressed); publish flux/centroid/shift-invariance numbers and CHEM hallucination maps; open training code.

### 9.3 Shared platform

- **Data engine:** one GalSim-based degradation/injection engine serves both tools (star injection for A = PSF/aberration synthesis for B); on-the-fly generation; 1–5k CC-BY/PD backgrounds (Legacy Surveys, DECaPS2, ESA/Hubble+Webb, NASA/STScI) + procedural + community-donated with explicit licenses + real-star-stamp library; DAOStarFinder QC on every background; **zero teacher models** (RC-Astro EULA + StarNet NC + quality ceiling).
- **Training:** PyTorch 2.13/cu130, bf16 autocast, channels_last, torch.compile, EMA; width32 @256 overnight screening runs → width64 promote → 512 fine-tune; weekend cadence per experiment; lean custom loop (or neosr).
- **Inference/delivery:** static-shape ONNX (512x512, opset 17–20), fp32 default + validated fp16 option; sirilpy ONNXHelper for install/providers/run (provider-agnostic — survives the WinML/TensorRT-RTX transition); PyQt6 + QThread GUI; tiles 512/≥64 px overlap with Hann or border-ignore stitching; GPU warmup; models on HF + GitHub Releases mirror; siril-scripts MR; FITS headers/WCS/ICC/HISTORY propagation; Siril row-order handling per official StarXTerminator.py; sequence support; failure-crop submission flywheel.
- **Benchmark as a product:** publish the full evaluation suite (§7.2) with pinned competitor versions — nobody in this space has automated, reproducible numbers, and the benchmark seeds community data donation.
- **Licensing:** code MIT (keep NAFNet MIT+Apache notices), weights Apache-2.0/CC-BY-4.0 with outputs explicitly unrestricted.

### 9.4 The five most exploitable commercial-tool weaknesses

1. **Stars-layer purity** (SXT leaks nebulosity into the stars layer; StarNet's is derived by subtraction) → residual head + purity loss + round-trip/idempotence gates.
2. **GHS/arcsinh stretch fragility** (SXT trained MTF-only; profiles become "small elliptical galaxies") → stretch-family augmentation.
3. **Lens-aberrated wide-field PSFs + FWHM >8 px** (every incumbent telescope-centric; SXT FAQ admits corner-coma failure) → nightscape-first training envelope, FWHM 1–12 px, Zernike aberrations.
4. **No verifiability anywhere** (BXT emits no kernel; all closed) → emit PSF estimates + residual maps + published metrics + open weights.
5. **Frozen/closed incumbents** (SXT AI11 unchanged since Sep 2022; StarNet closed since v2) + noise-matched fills absent from all free tools → continuous open retraining flywheel + noise-synthesis stage.
