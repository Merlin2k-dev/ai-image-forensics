# U1 data-acquisition proposal — FLUX / SD3.5 (+ SD2.1) for the Phase-2 retrain — 2026-06-27
PAPER-ONLY research. No downloads, no generation. Recorded before the U1 go/no-go decision on any pull.

## Why this exists
The frozen Phase-1 fails on modern gens (held-out: FLUX 0.14, SD3.5 0.30). Fix = put FLUX/SD3.5 (and SD2.1) into TRAIN. **B-Free is SPENT** (its FLUX/SD3.5 was the held-out test). I need FRESH modern-gen data for TRAIN + DEV-VAL + a new locked held-out, de-confounded. The binding risk I flagged: **pipeline-signature confound** — self-generated fakes carry my sampler/steps/guidance/format, so the model may learn "my pipeline," not "FLUX/SD3.5." -> **independently-built data strongly preferred over self-generation.**

Screening rule (widened, from the resolution screen): PASS = **AC-1 same-source** (caption- or pixel-paired, not foreign/proxy reals) AND **reals-native >= fakes-native** (crop reals down, no upscale). High-res reals are good.

---

## PART 1 — SOURCE HUNT (preferred path)

### VERDICT
**Partial YES.** One large, untouched, independently-built set has BOTH FLUX-dev and SD3.5 at training scale — **OpenFake** — but its reals are a web-scraped LAION pool (caption-seeded, JPEG, not pixel-paired, not reliably >=1024). **A perfectly-clean high-res *same-source* FLUX/SD3.5 set (the B-Free/RAISE niche) does NOT exist fresh — B-Free was that one and it is spent.** So: use OpenFake as the independent training/transfer backbone WITH de-confounding (filter reals >=1024 + match compression), and treat its weaker AC-1/compression honestly.

### Candidates screened

| Dataset | FLUX / SD3.5? | N (per gen) | Reals (source / native res) | reals >= fakes? | AC-1 | Compression | Independence / settings | License | Host | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **OpenFake** (ComplexDataLab) | FLUX.1-dev **144,788**; SD3.5 **139,114**; FLUX1.1-pro 29,923; FLUX-schnell 36,084; SDXL 186,666; SD1.5/2.1 variants | **large** (>=100k FLUX-dev & SD3.5) | LAION-400M, filtered >=512px (~3M pool) | **Partial** — fakes ~1MP(~1024); LAION reals >=512 but NOT reliably >=1024 -> **filter to >=1024 subset** to satisfy the rule | caption-seeded (Qwen2.5-VL prompts from real imgs seed gen) — same-source at caption level, **not pixel-paired**; reals are a pool, not 1:1 | **reals web-JPEG (variable QF)** vs gen fakes -> **compression mismatch risk** (must match) | independent (McGill/Mila group); per-model **native params** (varied, not one fixed pipeline) — good for robustness | mixed: SD3.5/FLUX-dev/SDXL/SD2.1 = Community/Non-commercial (research OK); MJ/GPT/Imagen = Non-compete | HuggingFace `ComplexDataLab/OpenFake` (Parquet/CSV, streaming) | **LEAD — use with de-confound** |
| **REVEAL-Bench++** | FLUX, FLUX2, Z-Image, Qwen-Image, SDv3.5 | small (bench/test-scale) | TBD | TBD | TBD | TBD | independent, very recent (Nov 2025) | TBD | arXiv 2511.23158 | **transfer-check candidate** (verify N/reals) |
| **MiraGe** built set | FLUX.1-dev + SD3.5 | **1,000 each** (thin) | MS-COCO captions; reals <=640 | **NO** (640 < 1024) | caption-paired | COCO JPEG | independent (single COCO protocol) | research | per paper (2508.01525) | **REJECT** (thin N + reals<fakes) |
| **AI-GenBench** | gens 2017-2024 (FLUX/SD3.5 unclear) | n/a | **union of existing datasets** | mixed | mixed/aggregated | mixed | **aggregation of existing sets** -> overlap/provenance risk, not fresh-independent | per-source | GitHub MI-BioLab | **REJECT** as "untouched independent" |
| **TGIF2-FR** | FLUX (inpainting) | — | MS-COCO <=1024 | — | local **inpainting sub-region**, not whole-image gen | JPEG/WEBP tested | independent | research | arXiv 2603.28613 | **REJECT** (local forgery, not whole-image) |
| **TrueFake** | SD3 / FLUX.1 + SDXL | mid | FFHQ + FORLAB (**foreign, unpaired**) | n/a | **AC-1 violation** | social-net recompressed | independent | research | arXiv 2504.20658 | **REJECT** (foreign reals) |
| **B-Free extended-Synthbuster** | FLUX + SD3.5 on RAISE | 1k each | RAISE -> ~1024^2 | YES | caption-paired, clean | lossless | independent, clean | GRIP-UNINA | grip.unina.it | **SPENT** (do not reuse) |

### Source-hunt bottom line
- **OpenFake** is the only independent set giving FLUX-dev **and** SD3.5 at TRAIN scale (>100k each) plus SD2.1/SDXL — enough for TRAIN + DEV + a new locked held-out. It is independent (their settings, not mine) and uses **varied native generation params** across models, which directly helps against the pipeline-signature risk.
- **But** its reals are not the clean high-res same-source kind ideally wanted: LAION web-JPEG, caption-seeded (not pixel-paired), resolution >=512 (not >=1024). **De-confounding required before training:** (a) keep only LAION reals with native min-side >=1024 and center-crop to the fake working size (no upscale, AC-9 clean); (b) **match compression** — re-encode reals and fakes to a common format/QF (or decode both to a common lossless working buffer) so JPEG-history isn't the shortcut; (c) build the held-out from OpenFake generators/prompts **disjoint** from train (and ideally hold out one FLUX variant entirely, e.g. train FLUX-dev, test FLUX-pro/schnell, to stress the signature).
- A pristine fresh high-res *same-source-paired* FLUX/SD3.5 set is **not available** (B-Free filled that niche and is spent). If that AC-1 quality is needed, the fallback (Part 2) — generate FLUX/SD3.5 from native-RAISE-non-1k captions — reproduces B-Free's clean construction on FRESH RAISE images I haven't used.

---

## PART 2 — API-GENERATION FALLBACK (costed out regardless; also fills the clean-AC-1 gap)

### Providers, cost/image, checkpoints, license (June 2026)
| Model (checkpoint) | Cheapest provider seen | $/image (~1MP) | Output license for research/training | Notes |
|---|---|---|---|---|
| **FLUX.1-dev** | fal.ai | **~$0.025** | FLUX **non-commercial** license — research/training-for-research **OK**, no commercial deployment | most-used FLUX checkpoint; matches OpenFake/B-Free |
| **FLUX.1-schnell** | Together | **~$0.0027** | **Apache-2.0** (commercial OK) | cheapest; distilled (few-step) — different artifact profile, good for diversity |
| **FLUX.1.1-pro** | API-only (BFL/fal/Together) | ~$0.04-0.055 | commercial OK (paid API) | closed; adds variant diversity for held-out |
| **SD3.5-large** | fal / Together | **~$0.008-0.015** | Stability **Community License** — free <$1M annual revenue; research OK; outputs user-owned | primary SD3.5 |
| **SD3.5-large-turbo / medium** | fal / Together | ~$0.006-0.01 | same Community License | variant diversity |
| **SD3 (legacy)** | Together | ~$0.0019 | Community License | optional |

Providers to spread across for backend diversity: **fal.ai, Replicate, Together, HF Inference** (fal generally 30-50% cheaper than Replicate). Using >=2 providers per model reduces single-backend signature.

### Best high-res real corpora (>=1024 native, research-licensable, FRESH)
| Corpus | N | Native res | Format | License | Note |
|---|---|---|---|---|---|
| **RAISE (full, minus the 1k B-Free used)** | ~7,100 unused (8,156 total - 1k) | ~**4288x2848** | TIFF + NEF (lossless) | non-commercial research (cite Dang-Nguyen 2015) | **reachable** (loki.disi.unitn.it, confirmed by a prior probe); the *other* RAISE images are untouched -> cleanest AC-1 path for self-generation from their captions |
| **DIV2K** | 1,000 | ~2K (e.g. 2040x1356) | PNG (lossless) | research/eval (NTIRE) | small but pristine |
| **Flickr2K** | 2,650 | ~2K | PNG | research | combine w/ DIV2K = DF2K ~3,650 |
| **Unsplash-Lite** | 25,000 | high-res (varied, mostly >=1024) | JPEG (high-Q) | Unsplash License (free use) | large, but JPEG (match compression) |

Recommended real base for self-gen: **unused RAISE images** (lossless, ~4288, never used by me or B-Free) -> re-caption via the Synthbuster/B-Free protocol (Midjourney-describe/CLIP-interrogator or a VLM), generate FLUX/SD3.5 from those captions, then derive reals by **center-crop native RAISE -> working size** (no upscale, lossless both sides). This recreates B-Free-grade clean AC-1 on FRESH data.

### Pipeline-signature MITIGATION (mandatory if I generate)
Randomize per image so no single signature dominates:
- **Steps:** FLUX-dev 20-50; SD3.5 20-40 (sample uniformly).
- **Guidance/CFG:** FLUX 2.5-5.0; SD3.5 3.5-7.0.
- **Seed:** random per image.
- **Scheduler/sampler:** vary where the provider exposes it.
- **Resolution/aspect:** mix 1024^2, 1152x896, 896x1152, 1216x832, etc. (then unify by center-crop).
- **Checkpoint variants:** FLUX-dev + schnell + (some) pro; SD3.5-large + large-turbo + medium.
- **Providers:** split each model across >=2 (fal + Replicate/Together).
- **Format:** save **lossless PNG**, then apply the SAME compression policy to reals and fakes (match QF) so format/JPEG-history is not learnable.

### Independent secondary transfer-check set (validates beyond my settings)
Hold out, NEVER train on: a small slice of **OpenFake** FLUX/SD3.5 (someone else's settings) and/or **REVEAL-Bench++** FLUX/SD3.5. If the detector trained on my self-gen (or OpenFake-train) holds up on this independent slice, the pipeline-signature risk is controlled. This is the decisive test I want.

### Total cost estimate (target ~2-4k per generator)
Take **3,000 FLUX + 3,000 SD3.5** (+ optional 2,000 SD2.1):
- FLUX.1-dev 3,000 x $0.025 = **$75** (or schnell 3,000 x $0.0027 = $8).
- SD3.5-large 3,000 x $0.012 = **$36**.
- SD2.1 3,000 x ~$0.003 = **$9** (optional).
- Add ~30% for variant/provider spread + regen of rejects ~ **+$35**.
- **~$120-160 total** for 6-9k images. Even pushing to 4k/gen across pro variants + multi-provider stays **< ~$300**. Cost is negligible; the real cost is curation/de-confounding time.

---

## RECOMMENDATION
1. **Primary (independent, preferred):** ingest **OpenFake** FLUX.1-dev + SD3.5 (+ SD2.1/SDXL) for TRAIN/DEV, WITH de-confounding (reals -> >=1024 LAION subset center-cropped; compression matched; held-out disjoint by generator-variant + prompt). Pros: independent settings, huge N, varied native params. Cons: caption-not-pixel AC-1, LAION web-JPEG reals.
2. **Clean-AC-1 complement / fallback:** if OpenFake's real side is too dirty, **self-generate FLUX/SD3.5 from FRESH (unused) RAISE captions** with the full settings-variation plan above (~$120-160), giving B-Free-grade clean pairing on data I haven't spent.
3. **Mandatory transfer gate:** lock an **independent** FLUX/SD3.5 slice (OpenFake-holdout and/or REVEAL-Bench++) as the de-confounded held-out to prove I caught "FLUX/SD3.5," not "my pipeline."

Open items to verify at pull-time (not blockers): OpenFake exact SD3.5 checkpoint (large vs medium) + FLUX-dev settings logged in its metadata; REVEAL-Bench++ N + real source; OpenFake per-model license line items for the variants actually used. NO data pulled — proposal only.
