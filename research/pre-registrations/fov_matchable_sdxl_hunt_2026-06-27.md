# FOV-matchable SDXL dataset hunt — 2026-06-27

**Question (narrow):** Find a dataset that has ALL of:
1. SDXL-generated **whole-image** fakes (variant ideally documented; base-1.0 preferred).
2. **Same-source reals (AC-1)** — reals from the same corpus/provenance as the fakes; pixel-paired (img2img / reconstruction) is ideal.
3. Reals at **~1024^2 NATIVE resolution** (so a 256 center-crop ~ crop_fraction 0.0625, matching SDXL output FOV). **Binding constraint.**
4. Compression-clean (lossless / matched-known), secondary.

**Result: NOT FOUND.** No dataset cleanly satisfies #1+#2+#3 together. The structural reason holds: same-source corpora pair SDXL fakes with reals that are either ~640px (COCO-derived) or ~4000px+ (RAW sets), never the ~1024 native band. The only ~1024^2 native reals (FFHQ) are foreign/proxy to the SDXL fakes (AC-1 violation). This is a time-boxed sweep, not exhaustive.

---

## Searches / sources actually checked
- DRCT-2M composition + resolution: OpenReview PDF `https://openreview.net/pdf/6a65ad38d0c82d1a5b968eef583f28efb1c0a6bd`, repo `https://github.com/beibuwandeluori/DRCT`. (T2I + I2I + DR variants incl. `SDXL`, `SDXL-refiner`, `SDXL-DR`; reals = MSCOCO; classifier input 224x224.)
- TGIF / TGIF2 (text-guided inpainting forgery): `https://arxiv.org/html/2407.11566v2`, `https://arxiv.org/abs/2603.28613`, dataset pages `https://media.idlab.ugent.be/tgif-dataset`, `https://media.idlab.ugent.be/tgif2-dataset`.
- FFHQ + SDXL face sets: SFHQ-T2I `https://github.com/SelfishGene/SFHQ-T2I-dataset`, "130K Real vs Fake Face" `https://gts.ai/...`, `https://huggingface.co/datasets/Birchlabs/sdxl-latents-ffhq`.
- TrueFake (IJCNN 2025): `https://arxiv.org/html/2504.20658v1`, `https://github.com/MMLab-unitn/TrueFake-IJCNN25`.
- "Bias-Free Training Paradigm" (SDXL fakes generated from RAISE & COCO, tested vs same-dataset reals): `https://arxiv.org/pdf/2412.17671`.
- IEEE DataPort "AI-Generated Image Detection Dataset v2" (10k COCO/ImageNet reals + 60k SDXL): `https://ieee-dataport.org/documents/ai-generated-image-detection-dataset-v2-...`.
- General sweeps: "SDXL vs real 1024 forensic benchmark", "img2img reconstruction DIV2K/Unsplash/Flickr2K 1024 paired", "2025 forensic benchmark control resolution/FOV", "ForensicHub / FakeShield". No additional qualifying candidate surfaced.

## Near-misses and exactly why each fails

### 1. DRCT-2M — `SDXL-DR` (diffusion reconstruction). Closest in STRUCTURE; fails on resolution.
- **#1 SDXL fake:** YES, whole-image. Variants documented: `SDXL`, `SDXL-refiner`, `SDXL-DR` (and SDXL-Turbo/-Ctrl/-LCM). `SDXL-DR` reconstructs a specific real MSCOCO image -> pixel-paired whole-image SDXL.
- **#2 same-source:** YES — reals are MSCOCO; SDXL-DR is the real image passed through SDXL (ideal pairing).
- **#3 ~1024^2 reals:** **FAILS.** Reals are MSCOCO (~640px max side). This IS the exact wall already hit (SDXL T2I fakes 1024^2 vs MSCOCO reals <=640 -> crop_fraction collinear with label). SDXL-DR doesn't escape it: the paired real is still the ~640 COCO image.
- **#4 compression:** MSCOCO are JPEG; not lossless.
- *Verdict:* perfect pairing, wrong resolution. The same confound the project already documented.

### 2. TGIF / TGIF2 — same-source 512-1024px MS-COCO reals, but the fakes are LOCAL inpainting.
- **SDXL variant DOCUMENTED:** `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` (the inpainting checkpoint, **not** base-1.0). SDXL outputs saved at 1024p.
- **#2 same-source:** YES — authentic images are MS-COCO val2017; manipulated variants derive from them (24 variants/image; not aligned pixel-pairs but the same source is available).
- **#3 reals resolution:** PARTIAL/CLOSER — "collect images with dimensionality of up to 1024 pixels (the largest publicly available resolution)"; excluded any dim <512px. So reals span ~512-1024px largest side, variable aspect — **not** square 1024^2 native, but far better than COCO-640.
- **#1 whole-image SDXL fake:** **FAILS for whole-image detection.** The forgery is text-guided **inpainting**: "the model regenerates the entire image, yet only visibly changing the inpainted area." Only the masked region is synthetic; most pixels are the original photo. So as an "SDXL fake vs real" whole-image sample it's mostly real pixels — not a clean generation-signal target, and the FOV of the synthetic content is a sub-region.
- **#4 compression:** base images from Flickr (JPEG); they explicitly test JPEG/WEBP QF80/60 as degradations.
- *Verdict:* best same-source ~1024-ish reals found, but the fake is local inpainting, not whole-image SDXL, and the reals aren't 1024^2 native-square. Usable only if the task is reframed to localized-forgery detection.

### 3. TrueFake (IJCNN 2025) & FFHQ+SDXL face sets — 1024^2 native reals, but FOREIGN/proxy reals.
- TrueFake reals = **FFHQ + FORLAB**; fakes = SD1.5/2.1/3/**SDXL** (two-stage = base+refiner)/FLUX. FFHQ is 1024^2 native — satisfies #3 for the real side.
- **#2 same-source:** **FAILS.** The SDXL images are independent text-to-image generations, **not** conditioned on / paired with FFHQ photos. FFHQ reals are a foreign proxy real set bolted onto SDXL fakes (exactly the AC-1 violation to avoid). Domain (faces) may not even match the SDXL gens.
- SFHQ-T2I (122k 1024^2 SDXL/Flux/DALL-E faces) is **synthetic-only** (no real pairing). The "130K real vs fake" set uses FFHQ reals downscaled to **256px** (loses native res AND foreign-source).
- *Verdict:* right real resolution, wrong provenance. The confound trades FOV-collinearity for source-collinearity.

### 4. "Bias-Free Training Paradigm" (2412.17671) — SDXL fakes from RAISE & COCO vs same-dataset reals.
- Same-source by construction (SDXL fakes generated from RAISE and from COCO; tested vs reals of the same dataset). But RAISE reals ~4288px (**too high**) and COCO reals ~640px (**too low**). Neither hits ~1024^2. The same wall, both directions.

### 5. Synthbuster (prior task) — RAISE-1k reals.
- Reals are RAISE-1k RAW (~4288px). **Too high** for a FOV match to 1024^2 SDXL crops; also the SDXL variant is undocumented (see the prior report). Fails #3.

### 6. IEEE DataPort AIGID v2 — 10k COCO/ImageNet reals + 60k SDXL.
- Reals low-res (COCO/ImageNet), T2I only, foreign-ish pairing. Fails #3.

---

## Bottom line
**NOT-FOUND.** No dataset gives whole-image SDXL fakes + same-source reals at ~1024^2 native resolution. The ~1024 native band intersected with same-source pairing is empty in what was checked.

- If **same-source pairing** is non-negotiable: closest is **DRCT-2M SDXL-DR** (pixel-paired whole-image SDXL reconstruction, documented variant) — but reals are MSCOCO ~640, i.e. the identical FOV confound, just paired. No resolution gain.
- If a **reframe to localized-forgery** is acceptable: **TGIF/TGIF2** gives same-source MS-COCO reals at 512-1024px with a documented SDXL inpainting checkpoint — but the SDXL content is a sub-region, not the whole image, and reals aren't square 1024^2.
- If **provenance can be relaxed** (proxy reals): **FFHQ-based SDXL face sets / TrueFake** give true 1024^2 native reals — but violate AC-1 (foreign, unpaired reals; faces domain).

The honest conclusion is the expected one: a clean same-source SDXL-vs-real corpus in the ~1024^2 native band does not appear to exist in current public forensic benchmarks.
