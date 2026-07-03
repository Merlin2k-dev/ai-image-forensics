# Phase 3 — Step 1 DATA SCOUTING: per-generator-labeled 2026 sources (2026-06-29)

**Objective:** find FREE datasets covering the newest 2026 generators (Nano Banana, Qwen, Z-Image,
GPT-Image, Imagen-4, Seedream, FLUX.2, ...) that satisfy ALL iron constraints **AND carry
PER-GENERATOR LABELS** — the requirement NTIRE lacked (no labels) and DFLIP3K was poisoned on
(AI-art reals). Metadata/paper scouting only; **every candidate sampled-data-verified (pixels, not
cards)** via HF streaming + the AC-9 FFT native-ness check. No bulk downloads, no build, no training.

**Method:** four sampled sweeps — (1) OpenFake test/OOD audit, (2) NTIRE/OpenSDI label re-check,
(3) fresh HF sweep, (4) Kaggle/off-HF sweep. Resolution measured by streaming 20-40 images/category
through PIL; native-ness via a 2D-FFT radial power ratio P(0.9*Nyq)/P(0.5*Nyq) on a native-pixel
center-512 crop (the test that caught/cleared DFLIP3K).

---

## HEADLINE FINDING (reverses the "decomposition unobtainable free" read)

**A clean, per-generator-labeled, 2026-frontier surface DOES exist for free:
`ComplexDataLab/OpenFake` core *test/OOD* split, restricted to its DOCCI reals.** This was
overlooked because the prior sweep optimized for a cross-corpus *held-out* and treated OpenFake as
the (LAION) *training* source — but OpenFake's OWN test split ships (a) a per-generator `model`
column, (b) genuine 2026 frontier generators NOT in training, and (c) a high-res NON-LAION real
corpus (DOCCI). It unblocks the per-generator decomposition that NTIRE structurally could not
provide.

---

## SCREENING TABLE (all sampled-verified unless noted)

| Source | Per-gen labels | 2026 gens | Reals: corpus - SAMPLED native short-side - spectral | AC-1 | Verdict |
|---|---|---|---|---|---|
| **OpenFake `core/test` (OOD)** `ComplexDataLab/OpenFake` | **YES** (`model` col, joins to images) | **YES** — z-image-turbo, flux.2-klein, gpt-image-1.5, gpt-image-2, **nano-banana-pro**, midjourney-7, seedream-v5, ideogram-2, recraft-v3, ernie, lumina, aurora (+video: sora-2/veo-3/wan) | **DOCCI** (non-LAION) 359/**1536**/3024 native-verified (ratio 0.055); imagenet 196/375/678 (<512, EXCLUDE) | distributional (pool split, same standard as the training data) | **USABLE** — DOCCI-reals-only, crop both -> 512. **PRIMARY.** |
| **AIGenImages2026** `pthan12/AIGenImages2026` | **YES** (per-gen) | **YES** — FLUX.2/pro/dev/max, Z-Image, **Gemini-3-Pro-Img**, GPT-Img-1.5, Seedream-4.5, Imagen-4, Firefly-5, Ideogram-3, MJ-v7 | News photos (NewsAPI), native <=16.78 MP (claimed; min 0.06 MP) - **PUBLIC, not gated** | CLIP same-source pairing (strong) | **STRONG 2nd (cross-corpus).** Public `.tar.gz`; NOT yet pixel-sampled. Small N (~5.4k gen). |
| ImageAttributionBench `multiitsuki/ImageAttributionBench` | YES (`model`, 24 gens) | YES — FLUX2_KLEIN, Z_IMAGE, gemini, grok3, mid-6.0, dalle3, ideogram, hidream, kling... | COCO 240/427/640, ImageNet 314/375/600, FFHQ/CelebA/LSUN **256** — all genuine but **<512** | content-aligned, NOT pixel-pairs; **res asymmetry (reals 256-640 vs fakes ~1024)** | **REJECT for a clean measure** (sub-512 reals + res shortcut). Salvage: fakes paired w/ external hi-res reals only. |
| NTIRE 2026 `deepfakesMSU/NTIRE-*` | **NO** (binary `label` only; all 3 splits re-verified — no model/gen field anywhere) | YES (unlabeled mix) | genuine photos, >=512 (med 864), native-verified | n/a | **REJECT** — no per-gen labels (blocker re-confirmed). Conflated-only. |
| OpenSDI `nebula/OpenSDI_test`, `OpenSDIDplus` | YES (sd15/sd2/sd3/sdxl/flux partitions) | **NO** (2023-24 SD-family only) | COCO 640x480, **24/25 <512** | **FAIL** — 0 real/fake ID overlap; reals 427px COCO stapled to ~1024px fakes (res+corpus confound) | **REJECT** (old gens + AC-1 fail). [the prior "megalith reals" note corrected: the test reals are COCO.] |
| DFLIP3K `nebula/DFLIP3K` | YES (excellent, all headliners) | YES | **POISONED** — reals are Civitai AI-art, not photos | degenerate | **REJECT** (the real class is synthetic). Fakes-only salvage. |
| TrueFake (GDrive, off-HF) | YES (per-gen partitions) | **NO** (caps FLUX.1/SD3) | FFHQ 1024 + FORLAB camera (non-LAION) | **FAIL** (no same-source pairing) | REJECT for frontier; FLUX.1/SD3-era cross-check only. |
| CommGen15 (ModelScope) | YES (per-gen folders) | partial (commercial) | COCO uncropped - **fakes 320x320** | weak | **REJECT** — 320px fakes. |
| DeepGuardDB / IEEE-AIGID-v2 (IEEE-DataPort) | partial | NO | paywalled / resized-512 | paired | REJECT — subscription wall + pre-2026 + reals resized. |
| Kaggle `alessandrasala79/ai-vs-human` | **unverifiable** (no API; JS card) | unknown | Shutterstock, unstated | paired | REJECT — per-gen labels unverifiable; likely binary-only. |

---

## OPENFAKE TEST/OOD — full evidence (PRIMARY surface)

**Schema:** `['image','prompt','label','model','type','release_date']` — `model` IS the
per-generator label, joins to every image. 13 test files.

**Counts (test split, 45,699 real / 45,699 fake):**
- Reals: **docci 14,847** (high-res non-LAION) + imagenet 30,852 (mostly <512).
- Fakes (per-gen): z-image-turbo 12,634 - flux.2-klein-9b 8,249 - illustrious 6,694 -
  gpt-image-1.5 5,573 - midjourney-7 3,586 - veo-3 2,169 (video) - wan-video-2.5 1,174 (video) -
  recraft-v3 1,000 - ernie-image-turbo 687 - sora-2 557 (video) - lumina 543 - gpt-image-2 474 -
  **nano-banana-pro 386** - seedream-v5.0 372 - ernie-image 315 - ideogram-2.0 282 - recraft-v2
  282 - aurora 282 - frames 250 - halfmoon 190.

**Sampled resolution + native-ness (short-side min/med/max - P(.9N)/P(.5N)):**
- docci (real) 359/**1536**/3024 - 0.055 — genuine hi-res photos, native (the low ratio = a
  512-window of a large image, NOT an upscale collapse).
- imagenet (real) 196/375/678 - 0.140 — sub-512, **exclude**.
- z-image 832/1024/3328 - flux.2 832/1024/1024 - gpt-image-1.5 720/1024/2160 - nano-banana-pro
  416/1076/2488 - midjourney-7 640/896/1024 - seedream-v5 1440/1728/2048 - gpt-image-2
  1024/1024/1536 - sora-2 720/1024/2560. All RGB, hi-res native.

**Why it's clean:** DOCCI reals (med 1536) and frontier fakes (med ~1024) are BOTH >=512 =>
center-crop both down to the V1 512 substrate (AC-9: crop, never upscale; a few DOCCI <512
excluded) => **no real/fake resolution asymmetry** (the exact shortcut that kills
ImageAttributionBench/OpenSDI). DOCCI is **non-LAION** => cross-corpus on the real side vs V1's
LAION training. The frontier gens are **cross-LAB** (Google nano-banana, OpenAI gpt-image,
ByteDance seedream, Z-Image, ideogram, recraft != BFL/Stability).

**Caveats (honest):**
1. **N is thin on the rarest frontier gens** (nano-banana-pro 386, seedream 372, ideogram 282,
   gpt-image-2 474, sora-2 557): per-generator AUC will have wide CIs — adequate for a
   measurement, not for training a per-gen model. The big gens (z-image 12.6k, flux.2 8.2k,
   gpt-image-1.5 5.5k, mj-7 3.6k) are plentiful.
2. **Video gens** (veo-3, wan-video-2.5, sora-2) are frames — exclude or flag separately (video
   compression != still-image substrate).
3. **Pipeline coupling partially shared:** the test is the OpenFake *family* (same
   generation/curation pipeline as V1's training), so this is cross-corpus (DOCCI != LAION) +
   cross-lab (gen labs) but NOT cross-*dataset-pipeline*. AIGenImages2026 (news reals,
   independent build) is the complementary cross-pipeline check.
4. **Substrate:** raw test images must pass through the frozen V1 `preprocess()` (512
   center-crop, Q75 4:2:0) before measurement — same harness as P3.2/P3.3;
   guilty-until-audited still applies.

---

## VERDICT

**The per-generator-label + native->=512-real + 2026-frontier intersection IS obtainable for
free** — contrary to the prior "decomposition unobtainable" read — via **OpenFake test/OOD
restricted to DOCCI reals**, with **AIGenImages2026** (public, news reals, independent pipeline)
as a strong second cross-corpus surface. This directly enables Step 2: measure frozen V1 transfer
to EACH 2026 generator individually (the cross-LAB per-generator number I had never been able to
compute). Honesty notes: thin N on the rarest gens (wide CIs), and full cross-dataset-pipeline
independence still wants the AIGenImages2026 second surface. Nothing here requires a paid API or
GPU.

**HOLD:** I paused here to make the go/no-go call — approve OpenFake-test/OOD (DOCCI-reals) as
the Step-2 per-generator measurement surface (+/- AIGenImages2026 as the 2nd cross-corpus
surface) before any download/build/audit.
