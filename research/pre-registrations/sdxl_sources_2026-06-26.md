# SDXL Test-Set Sources for Phase-2 Transfer Measurement — Research Findings

**Date:** 2026-06-26
**Scope:** Evidence only. No downloads, no decisions here. Find SDXL-generated images paired
with REAL images of defensible *shared provenance* (anti-confound rule AC-1: single-source
reals, never proxy). The final dataset choice is made separately.

---

## The core problem restated

GenImage gave Phase 1 an easy life: ImageNet reals and fakes-from-ImageNet-classes shared
provenance. SDXL is text-to-image and ships **no bundled reals**. So every candidate below
must be judged on one make-or-break axis: **do the reals and the SDXL fakes share an
underlying corpus / caption distribution, and can the *non-semantic* nuisance variables
(resolution history, compression history, format) be harmonized so they don't leak class
identity?**

The literature already documents exactly the confound I worry about. The "Dual Data
Alignment" paper (arXiv 2505.14359) states plainly that in standard AIGC detection sets
**"real images are JPEG-encoded and vary in size, whereas synthetic images are uniformly
PNG-encoded and fixed in size,"** and that synthetic images carry "disproportionately strong
high-frequency details" while JPEG strips them from reals — a frequency/format leak a
detector can exploit instead of learning real forensic artifacts. This is the trap. Treat
any candidate whose reals and fakes differ in format/size as confounded **until I harmonize
it myself**.

---

## Candidate 1 — DRCT-2M (Diffusion Reconstruction Contrastive Training)

1. **Source & access** — Paper: "DRCT: Diffusion Reconstruction Contrastive Training
   towards Universal Detection of Diffusion Generated Images" (ICML 2024). Code:
   https://github.com/beibuwandeluori/DRCT . Dataset released on **ModelScope** (linked from
   the repo). OpenReview PDF: https://openreview.net/pdf/6a65ad38d0c82d1a5b968eef583f28efb1c0a6bd .
2. **License** — Research use (academic); inherits MSCOCO terms for the real half. Confirm on
   ModelScope before use.
3. **SDXL specifics** — Strong. DRCT-2M spans 16 SD model types and **explicitly includes
   SDXL, SDXL-refiner, SDXL-Turbo, LCM-SDXL, SDXL-Ctrl, and SDXL-DR** (text-to-image and
   image-to-image). ~236k images total with an equal real/fake split; SDXL is one of the
   better-represented modern families here.
4. **The reals — AC-1 verdict: STRONG.** Reals are **MSCOCO** photographs. Fakes are
   generated from the **captions of those same MSCOCO images** (text-to-image input = the
   caption of a real MSCOCO photo). So real and fake share the scene/caption distribution by
   construction — this is the "deliberately built fakes against a known real corpus" ideal.
   Shared provenance is genuine, not bolted-on.
5. **Native resolution** — *Mismatch risk.* SDXL generates natively at ~**1024x1024**; MSCOCO
   reals are variable and typically much smaller (~640x480). The crop_fraction / FOV confound
   I predicted is present and **must be harmonized by me** (common crop+resize pipeline).
6. **Compression / format** — *Mismatch risk.* MSCOCO reals are **JPEG** (variable QF);
   generated images are typically stored **PNG**. This is exactly the format/frequency leak
   from Dual Data Alignment. Must re-encode to a common format/QF before measuring.
7. **Known issues** — The Dual Data Alignment work (2505.14359) uses DRCT precisely to
   demonstrate format/frequency and semantic bias; its DRCT-DR variants exist to *mitigate*
   the JPEG-vs-PNG leak by JPEG-compressing reconstructed fakes. Read this as confirmation
   that DRCT-2M's raw real/fake pairing is confounded on format until harmonized.

---

## Candidate 2 — AI-Generated Image Detection Dataset v2 (IEEE DataPort, "10k+60k paired")

1. **Source & access** — IEEE DataPort:
   https://ieee-dataport.org/documents/ai-generated-image-detection-dataset-v2-10k60k-paired-real-and-synthetic-images
   (~24 GB; login required). Appears in 2025 detection literature (e.g. Co-Spy, CVPR 2025).
2. **License** — **Non-commercial research use only** (inherits ImageNet + COCO restrictions).
3. **SDXL specifics** — Contains **SDXL** as one of six generators (with SD-v1.5,
   FLUX.1-schnell, Kandinsky 2.2, PixArt-Sigma, Stable Cascade). 60k synthetic total =>
   ~**10k SDXL** images. Variant labeled "SDXL" (base); confirm the exact checkpoint on the page.
4. **The reals — AC-1 verdict: STRONG.** 10k reals from **COCO + ImageNet-1k**. Each real
   image has **six synthetic counterparts sharing a single BLIP-2 caption** => explicit
   per-image semantic pairing between the real and its SDXL fake. Same-source by design.
5. **Native resolution — HARMONIZED.** Everything standardized to **512x512** for both
   classes. *Pro:* removes the FOV/crop confound for free. *Con:* SDXL is downsampled
   1024 -> 512, which **attenuates the native 1024 generation artifacts** my frozen detector
   may rely on — so this measures transfer to *downscaled* SDXL, not native SDXL.
6. **Compression / format — HARMONIZED.** Canonical pipeline: EXIF-transpose -> RGB -> crop ->
   resize -> **JPEG equalization** -> **PNG export**, applied to *both* classes. This is the
   only candidate that explicitly neutralizes the Dual-Data-Alignment format/frequency leak.
7. **Known issues** — The harmonization is the headline feature; the main caveat is the 512
   downscale changing the SDXL artifact signature (a confound of a different kind — it makes
   the test "easier"/different from native deployment). Provenance and per-image caption
   pairing are well documented.

---

## Candidate 3 — Synthbuster (+ RAISE-1k as the reals)

1. **Source & access** — Zenodo: https://zenodo.org/records/10066460 (12.4 GB). Paper:
   "Synthbuster: Towards Detection of Diffusion Model Generated Images" (IEEE OJSP 2023),
   vera.ai project page. Reals (RAISE-1k) downloaded separately:
   http://loki.disi.unitn.it/RAISE/download.html .
2. **License** — **CC-BY-NC-SA-4.0** (non-commercial, share-alike) for Synthbuster. RAISE has
   its own (research) terms.
3. **SDXL specifics** — Contains **Stable Diffusion XL**, 1,000 images (1k per model x 9
   models: SD1.3/1.4/2, SDXL, DALL-E 2, DALL-E 3, Firefly, Midjourney v5, GLIDE). Only
   **~1,000 SDXL images** — small for a statistically tight transfer estimate.
4. **The reals — AC-1 verdict: MODERATE (shared *scene*, not shared capture).** Fakes were
   produced from prompts **derived from each RAISE-1k image** (Midjourney /describe +
   CLIP-interrogator captions, hand-edited for photorealism). So the SDXL fakes track the
   RAISE scene distribution loosely, and the authors **explicitly recommend pairing against
   RAISE-1k reals**. The provenance link is real but looser than DRCT/v2 (the caption is a
   re-description, not COCO's ground-truth caption; persons/artist names were removed,
   shifting content).
5. **Native resolution — SEVERE MISMATCH.** RAISE-1k are **raw, uncompressed, very
   high-resolution** camera photos (~4288x2848). SDXL fakes are ~**1024x1024**. Synthbuster's
   own images range widely by model (Firefly >2000x1700, MJ ~1300x800). This is the **largest
   resolution gap of any candidate** — a wide-open FOV/crop confound and a documented
   train/eval resolution problem (SIDBench, TextureCrop both flag detectors mishandling these
   high-res reals). Requires aggressive, careful harmonization.
6. **Compression / format** — Both sides effectively **lossless** ("none of the images
   suffered JPEG compression or resampling"; RAISE is raw/TIFF). *Good:* no JPEG-vs-PNG leak.
   *But:* lossless-vs-lossless at wildly different resolutions still leaks via resampling
   statistics.
7. **Known issues** — Widely used as a *clean-source* benchmark precisely because it's
   uncompressed, but the high-res-real / 1024-fake gap is a known confound; small SDXL N.

---

## Candidate 4 — Community Forensics

1. **Source & access** — Paper arXiv 2411.04125 (2024). Code/page:
   https://github.com/JeongsooP/Community-Forensics . Full 1.1 TB; 278 GB reduced; 206 GB eval.
2. **License** — Research; aggregates many sub-dataset licenses (check per-source).
3. **SDXL specifics** — 4,803 generators / ~2.7M images; the eval set (26k from 21 held-out
   models) includes **LCM-LoRA variants of SDXL** and other SDXL-derived models. SDXL present
   but mixed among thousands of generators, often as LoRA/derivatives rather than base SDXL.
4. **The reals — AC-1 verdict: WEAK (foreign reals, bolted-on).** Reals are a **grab-bag of
   different corpora** — LAION, ImageNet, COCO, FFHQ, CelebA, MetFaces, AFHQ, Forchheim,
   IMD2020, Landscapes-HQ, VISION — **not** matched per-fake to a shared source. The real and
   SDXL distributions differ in source, camera, and content. High AC-1 risk: a detector can
   separate classes on corpus identity, not generation artifacts.
5. **Native resolution** — Heterogeneous across both classes; no harmonization guarantee.
6. **Compression / format** — Mixed (JPEG reals from web corpora vs generated PNG); the classic
   format leak likely present and uncontrolled.
7. **Known issues** — Designed for *training diversity / generalization*, not for clean
   single-source transfer measurement. Great breadth, poor provenance control. Use as a
   stress/aux set, not the primary AC-1 measurement.

---

## Candidate 5 — WildFake

1. **Source & access** — arXiv 2402.11843 (2024). 2.55M fakes from 23 generators, hierarchical
   (cross-generator / architecture / weights / time).
2. **License** — Research; verify.
3. **SDXL specifics** — Includes SDXL among 23 generators, plus many community/user-trained
   checkpoints. SDXL present but the provenance of individual fakes (scraped from open-source
   sites) is heterogeneous.
4. **The reals — AC-1 verdict: WEAK.** Fakes largely **scraped from the web** with reals from
   separate sources; no per-image shared-corpus pairing. Provenance murky.
5. **Native resolution** — Heterogeneous, uncontrolled.
6. **Compression / format** — Web-scraped => mixed/unknown compression on the fake side too;
   uncontrolled.
7. **Known issues** — Excellent for diversity/robustness; not a controlled-provenance source.

---

## Candidate 6 — ImagiNet

1. **Source & access** — arXiv 2407.20020 (2024), "ImagiNet: A Multi-Content Dataset for
   Generalizable Synthetic Image Detection via Contrastive Learning." ~100k images, 8
   generators, 4 content types (photos, paintings, faces, misc).
2. **License** — Research; verify.
3. **SDXL specifics** — SDXL is among the generators; counts modest (8 generators over 100k).
4. **The reals — AC-1 verdict: MODERATE-to-WEAK.** Reals are curated with "distinct
   provenance," balanced by content type, but **not** built as per-image same-caption pairs
   with the SDXL fakes. Better-curated than Community Forensics/WildFake, weaker than
   DRCT/v2/Synthbuster on shared-source.
5. **Native resolution** — Mixed across content types; not harmonized to fakes.
6. **Compression / format** — Mixed; not explicitly equalized.
7. **Known issues** — The literature notes the general "lack of semantic alignment" failing
   (real/fake not sharing the same caption) applies here — content bias can creep in.

---

## Candidate 7 — TrueFake (2025)

1. **Source & access** — arXiv 2504.20658 (IJCNN 2025). Repo:
   https://github.com/MMLab-unitn/TrueFake-IJCNN25 . 600k images.
2. **License** — Research; verify.
3. **SDXL specifics** — Includes **SDXL** (with SD1.5/2.1/3, FLUX.1, StyleGAN1/2/3). 100k real
   / 500k fake.
4. **The reals — AC-1 verdict: WEAK for the scene-distribution goal.** Reals from **FFHQ +
   FORLAB** (faces / camera-acquisition sets), not caption-matched to the SDXL fakes. Faces-
   heavy, foreign-corpus reals.
5. **Native resolution** — Varies; faces (FFHQ 1024) vs scenes.
6. **Compression / format** — **Notable strength on a different axis:** a 180k subset is
   re-shared through Facebook / X / Telegram, giving **realistic social-network JPEG
   recompression** for both classes (PSNR/SSIM measured). Useful for compression-robustness
   stress testing, not for clean AC-1.
7. **Known issues** — Strong for "in-the-wild / post-social-network" realism; weak for
   shared-source transfer measurement.

---

## Candidate 8 — DiffusionDB — **NOT USABLE for AC-1 (no reals)**

- arXiv 2210.14896; https://huggingface.co/datasets/poloclub/diffusiondb . 14M Stable
  Diffusion (v1.x) images + 1.8M prompts scraped from the official SD Discord. **Fakes-only,
  no real-image counterpart**, and it predates SDXL (SD1.x). Documented here only to record that
  it does **not** solve the same-source-reals problem and contains no SDXL. Skip for Phase 2.

---

## Summary table

| Dataset | SDXL present? (variant / approx N) | Reals source | AC-1 (shared source) | Native res (real vs SDXL) | Format/compression | One-line verdict |
|---|---|---|---|---|---|---|
| **DRCT-2M** | Yes — SDXL, refiner, Turbo, LCM, Ctrl, DR (large) | MSCOCO | **STRONG** (fakes from MSCOCO captions) | ~640 JPEG vs ~1024 PNG | mismatched, **harmonize myself** | Best provenance + big SDXL; I must equalize res+format |
| **AI-Gen Detect v2 (IEEE DataPort)** | Yes — SDXL base (~10k) | COCO + ImageNet | **STRONG** (per-image BLIP-2 caption) | both 512 | **both PNG, JPEG-equalized** | Cleanest confound control; but the 512 downscale alters SDXL artifacts |
| **Synthbuster + RAISE-1k** | Yes — SDXL base (~1,000) | RAISE-1k (raw) | MODERATE (scene via re-caption) | ~4288 raw vs ~1024 | both lossless | Clean format, but extreme res gap + tiny SDXL N |
| **Community Forensics** | Yes — SDXL/LCM-LoRA (mixed) | LAION/ImageNet/COCO/FFHQ/... | **WEAK** (foreign mix) | heterogeneous | mixed/uncontrolled | Diversity, not provenance |
| **WildFake** | Yes — SDXL among 23 | foreign / scraped | **WEAK** | heterogeneous | mixed/unknown | Robustness aux only |
| **ImagiNet** | Yes — SDXL among 8 | curated multi-content | MODERATE-WEAK | mixed | mixed | Better curation, no caption pairing |
| **TrueFake** | Yes — SDXL | FFHQ + FORLAB | **WEAK** for scenes | varies | **social-network JPEG subset** | Compression-realism stress set |
| **DiffusionDB** | **No (SD1.x only)** | **none** | **N/A** | n/a | n/a | Fakes-only, no SDXL — skip |

## Ranking by how well each solves the same-source-reals (AC-1) problem

1. **AI-Gen Detect v2** — strongest *combined* AC-1 + confound control (per-image caption
   pairing AND res/format already harmonized). Caveat: 512 downscale != native SDXL.
2. **DRCT-2M** — strongest *provenance* (MSCOCO caption -> fake) with large native-1024 SDXL,
   but I inherit the JPEG/PNG + resolution confound and must harmonize it myself.
3. **Synthbuster + RAISE-1k** — defensible scene-level shared source and clean lossless
   format, but the raw-high-res-real vs 1024-fake gap is the worst of any candidate and SDXL N
   is only ~1,000.
4. **ImagiNet** — moderate curation, no per-image caption pairing.
5. **TrueFake** — foreign reals; keep as a *compression-robustness* stress set.
6. **Community Forensics / WildFake** — foreign/mixed reals; AC-1 risk; diversity aux only.
7. **DiffusionDB** — does not qualify (no reals, no SDXL).

## Cross-cutting confound notes (literature)

- **The format/frequency leak is real and documented:** Dual Data Alignment (arXiv 2505.14359) —
  reals JPEG + variable-size vs fakes PNG + fixed-size lets detectors cheat on compression, not
  artifacts. Any candidate without harmonization (DRCT-2M, Community Forensics, WildFake) is
  confounded until I equalize format/QF and resolution myself.
- **Resolution/FOV mismatch** is the predicted #1 confound and is present everywhere except
  the pre-harmonized v2 set (and is *worst* in Synthbuster/RAISE). Plan a shared
  crop+resize+re-encode pipeline regardless of choice.
- **Semantic/content bias:** several papers (Dual Data Alignment; ImagiNet/Fake2M critiques)
  note that without a *shared caption per pair*, real and fake differ in content — only
  DRCT-2M and AI-Gen Detect v2 give true per-image caption pairing.

## Sources

- DRCT (ICML 2024): https://openreview.net/pdf/6a65ad38d0c82d1a5b968eef583f28efb1c0a6bd ;
  code https://github.com/beibuwandeluori/DRCT
- AI-Generated Image Detection Dataset v2 (IEEE DataPort):
  https://ieee-dataport.org/documents/ai-generated-image-detection-dataset-v2-10k60k-paired-real-and-synthetic-images
- Synthbuster (Zenodo): https://zenodo.org/records/10066460 ; vera.ai:
  https://www.veraai.eu/posts/dataset-synthbuster-towards-detection-of-diffusion-model-generated-images ;
  RAISE: http://loki.disi.unitn.it/RAISE/download.html
- Community Forensics: https://arxiv.org/abs/2411.04125 ;
  https://github.com/JeongsooP/Community-Forensics
- WildFake: https://arxiv.org/abs/2402.11843
- ImagiNet: https://arxiv.org/html/2407.20020v1
- TrueFake: https://arxiv.org/abs/2504.20658 ; https://github.com/MMLab-unitn/TrueFake-IJCNN25
- DiffusionDB: https://arxiv.org/abs/2210.14896 ; https://huggingface.co/datasets/poloclub/diffusiondb
- Dual Data Alignment (confound analysis): https://arxiv.org/html/2505.14359v6
- Co-Spy (CVPR 2025, uses v2-style sets): https://arxiv.org/abs/2503.18286
- SIDBench (resolution/bias eval): https://arxiv.org/pdf/2404.18552 ;
  TextureCrop: https://arxiv.org/pdf/2407.15500 ;
  Bias-Free Training Paradigm: https://arxiv.org/html/2412.17671
