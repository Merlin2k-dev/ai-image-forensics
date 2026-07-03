# Phase 3 free-source verification sweep — consolidated (2026-06-29)

**Question:** Can I assemble a FREE, confound-clean **cross-corpus + cross-lab held-out** to test V1
(trained on OpenFake = LAION reals + FLUX.1-dev/SD3.5 fakes)? **ANSWER: YES — via NTIRE 2026 (data-verified).**

The 3 bars: (1) frontier gens [Nano Banana/Qwen/Z-Image/recent FLUX-SD3.5, not 2022-era]; (2) reals
>=512 native from a **non-LAION** corpus; (3) AC-1 same-source (genuinely related, not foreign stapling).
Note: a set clearing only (2)+(3) on FLUX/SD3.5 still tests the gap V1 couldn't close.

## TOP CANDIDATES (ranked; V = data-verified by sampling, ~ = by card)

| Rank | Source | Generators (year) | Reals: corpus + ACTUAL res | Bar1 | Bar2 | Bar3 | License | Access |
|---|---|---|---|---|---|---|---|---|
| **1** | **NTIRE 2026** `deepfakesMSU/NTIRE-RobustAIGenDetection-{train,val,test-public}` | **Nano Banana/Pro/2, Qwen-Image, Z-Image, GPT-Image, Seedream, Kling, ImageGen-4, FLUX.2, FLUX.1, SD3.5** (2024-26) | CC12M+CommonPool+RedCaps (non-LAION); **V 25/25 reals >=512** (min-dim 528-3024, mean 1242) | V PASS | V PASS | ~PARTIAL (distributional, not 1:1; semantically aligned, NOT foreign-stapled) | CC-BY-4.0 | HF, public, ungated |
| 2 | `nebula/OpenSDI_test` / `OpenSDIDplus` | FLUX.1, SD3, SDXL (2024) | megalith-10m mountain photos; V >=512 (768-1024) non-LAION | V | V | X FAIL (global-synth: megalith reals stapled to COCO-caption FLUX -> foreign) | CC-BY-SA-4.0 | HF |
| 3 | `ComplexDataLab/OpenFake` core/**test** (OpenFake's OWN OOD split) | Z-Image, FLUX.2-klein, Sora-2, seedream-v5, nano-banana-pro, GPT-image-2 (2026 OOD) | DOCCI V 2048x1536 (non-LAION) + ImageNet (mixed <512) | V | ~partial | ~topic-matched | CC-BY-NC-4.0 | HF |
| 4 | `nebula/DFLIP3K` | FLUX.1-dev, SD3.5-L, Z-Image, Nano-Banana-Pro, Qwen, GPT-Image-1, Imagen-4 (24-26) | Civitai user photos; V ~512 min (borderline, possibly resized; provenance uncertain) | V | ~borderline | ~per-checkpoint | CC-BY-NC-4.0 | HF |
| — | **SynthBuster + RAISE** (Zenodo) | 9 gens 2021-23 (DALL-E2/3, MJv5, SDXL, Glide) | RAISE camera TIFF 12-16MP V lossless non-LAION | X (old) | V | ~partial | CC-BY-NC-SA | Zenodo — **= the lossless MIXED-Q anchor (Req 3)** |

## REJECTED (auditable — clears <2 bars)
- **AC-1 but reals <512:** `nebula/FakeCOCO` (FLUX/SD3, COCO ~375-500px); `bitmind/*___FLUX.1-dev` (all 256px); `bitmind/MS-COCO-unique___FLUX.1-dev` (256).
- **Fakes-only (no reals):** `Lunahera/genimagepp`, `ash12321/flux-1-dev-generated-10k`, `SaunakSS/nano-banana-synthetic-images-1500`, `lrzjason/ZImageTurboGen`, `selfishgene/sfhq-t2i`.
- **2022-era gens (bar1 fail):** GenImage (`ENSTA-U2IS/GenImage`, ImageNet non-LAION but old + often <512), `TheKernel01/AIGC-Detection-Benchmark`, `sywang/CNNDetection` (ProGAN/LSUN), AIGCDetect_testset.
- **RealHD:** NOT released (GitHub "Coming soon!"); would fail bar1 anyway (only BFL+Stability = V1's training labs). BLOCKED.

## KAGGLE: nothing clean (no API -> card-verified only; the ecosystem skews 2022-23/low-res)
Best = `alessandrasala79/ai-vs-human-generated-dataset` (Shutterstock reals, non-LAION): all 3 bars
TENTATIVE/unverified + competition-terms license. Not relied upon. Off-Kaggle leads it flagged
(DeepGuardDB/IEEE-DataPort, MS-COCOAI, TrueFake/FFHQ) are HF/other — TrueFake (FLUX/SD + FFHQ 1024) is a
possible bar-1+2 cross-check worth a later look; not needed given NTIRE.

## VERDICT
**YES — the independence gap is closeable FOR FREE.** NTIRE 2026 is the standout and is **data-verified**
(released, >=512 reals sampled, non-LAION, genuine cross-LAB frontier gens incl Nano Banana). It uniquely
provides BOTH axes the B-Free datum conflated: cross-CORPUS (CC12M/etc != LAION) AND cross-LAB (Google/
Alibaba/Zhipu/OpenAI != BFL/Stability). It also contains FLUX.1/SD3.5 -> lets me DECOMPOSE the question:
V1 on NTIRE-FLUX/SD3.5 = cross-corpus/same-lab; V1 on NTIRE-NanoBanana/Qwen/Z = cross-corpus/cross-lab.
**Bar 3 caveat:** distributional AC-1 (not 1:1) — the same standard as the OpenFake training data; verify via
the P3.2 VQDM content-match audit (guilty-until-audited). SynthBuster/RAISE = the lossless mixed-Q anchor.
**No paid path needed.** Treadmill avoided; the gap is testable.
