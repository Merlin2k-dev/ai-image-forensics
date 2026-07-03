# Generator x resolution screen — 2026-06-27 (RE-RANKED, widened criterion) — PAPER/DOC ONLY, no downloads

## Widened pass rule
PASS requires BOTH:
1. **AC-1 same-source** — reals share corpus / are semantically- or pixel-paired with the fakes (HARD; reject foreign/proxy reals).
2. **Common native size without upscaling = reals-native >= fakes-native** (HARD). If reals are BIGGER, center-crop them DOWN to the fakes' native size (real pixels, no interpolation, AC-9 clean) -> both share a working native size -> a fixed 256-crop has matched crop_fraction -> FOV controlled.
   - **Reals >= fakes = PASS** (crop reals down). **Only reject = reals SMALLER than fakes** (forces upscaling / leaves an uncontrollable FOV gap — the SDXL-1024-vs-COCO-640 case).
Then rank by: AC-1 -> reals>=fakes -> compression (lossless/matched best) -> generator modernity.

Real-corpus native res: RAISE ~4288 (uncompressed RAW/TIFF); FFHQ = 1024^2 (PNG, lossless); MS-COCO <=640 long side (JPEG, variable); ImageNet ~470x400 typical (JPEG, high variance, many <512).

---

## *** Key result: SDXL (and FLUX / SD3.5) are VIABLE AGAIN ***
The SDXL failure was the **DRCT-2M/MSCOCO pairing** (1024 fakes vs <=640 reals), not the generator. Under the widened rule, any same-source set with **reals >= 1024** rehabilitates SDXL:
- **Synthbuster** (RAISE-1k reals ~4288 >= SDXL 1024) — caption-paired same corpus, both lossless -> SDXL PASSES.
- **B-Free / extended-Synthbuster** (GRIP-UNINA + Google DeepMind, CVPR 2025) — adds **FLUX and SD3.5** to the RAISE-conditioned Synthbuster generators, AND ships an SD2.1 **self-conditioned reconstruction** training set (pixel-content-aligned, strongest AC-1). RAISE reals >= all fakes -> SDXL/FLUX/SD3.5 all pass. Source: `https://grip-unina.github.io/B-Free/`, paper `https://arxiv.org/abs/2412.17671`.

---

## PASS list (AC-1 clean AND reals >= fakes), ranked

| # | Generator (dataset) | Fake native res | Real corpus + native res | reals >= fakes? | AC-1 status | Compression | Modernity |
|---|---|---|---|---|---|---|---|
| 1 | **SD3.5 / FLUX.1 (B-Free extended Synthbuster)** | 1024^2 | RAISE-1k ~4288 | **YES** (crop 4288->1024) | same-source, RAISE-caption-paired (Synthbuster protocol) | RAISE lossless + fake PNG (verify) | **highest — 2024 DiT models** |
| 2 | **SDXL (Synthbuster / B-Free ext.)** | 1024^2 | RAISE-1k ~4288 | **YES** | same-source, RAISE-caption-paired | RAISE lossless + PNG (Zenodo: "no JPEG/resampling") | 2023 (variant undocumented — see prior note) |
| 3 | **SD2.1 self-conditioned recon (B-Free training set)** | 512^2 (reals resized -> 512) | COCO + RAISE, used @512 | **YES** (equal, no upscale) | **strongest AC-1: pixel/content-conditioned from the specific real image** | mixed source (COCO JPEG / RAISE lossless) + recon PNG | method 2024; base SD2.1 (2022) |
| 4 | **DALL-E 3 / Midjourney v5 / Firefly (Synthbuster)** | <=1024^2 | RAISE-1k ~4288 | **YES** | same-source, RAISE-caption-paired | RAISE lossless + PNG | 2023 (closed flagships) |
| 5 | **SDXL-Turbo / SD-Turbo / LCM-SDv1.5 (DRCT-2M)** | 512^2 | MS-COCO <=640 | **YES** (crop 640->512) | same-source, COCO caption/recon-paired | COCO JPEG / fake likely PNG -> **mismatch, verify** | 2023-24 distilled |
| 6 | **SD1.4 / SD1.5 / SD2 (DRCT-2M)** | 512^2 | MS-COCO <=640 | **YES** | same-source COCO | JPEG/PNG mismatch (verify) | 2022 |
| 7 | **StyleGAN3 (ffhq-1024) + FFHQ** | 1024^2 | FFHQ 1024^2 | **YES** (equal) | same-source *distributional* (trained on FFHQ; not pixel-paired) | **both lossless PNG — cleanest AC-4** | 2021 GAN; faces only |
| 8 | **ADM / GLIDE / VQDM (GenImage)** | 256^2 | ImageNet ~470 | **YES** (crop -> 256) | same-source ImageNet class-paired | both JPEG | 2021-22 (older) |

**Conditional (resolution caveat):** **GenImage SD1.4/1.5/Wukong @512** — ImageNet reals are highly variable and **many are <512**, so those images can't be cropped to 512 without upscaling. Passes only on the subset of ImageNet reals >=512; otherwise downgrade. Note before use.

**Recommended targets for "modern + viable + clean":**
- **Rows 1-2 (B-Free extended Synthbuster: SD3.5 / FLUX / SDXL on RAISE):** newest generators, AC-1 caption-paired, reals hugely exceed fakes, both lossless. Best modernity+resolution+compression combo. **This is the path that revives SDXL and adds FLUX/SD3.5.**
- **Row 3 (B-Free SD2.1 self-conditioned recon):** cleanest AC-1 of all (pixel-content-conditioned from the exact real image), zero resolution gap; ideal if a content-paired control is wanted over a newer generator.
- **Row 5 (DRCT-2M Turbo/LCM @512):** already-paired same corpus, modern distilled gens; watch the JPEG/PNG compression mismatch.

**Verify at pull-time (logistics, not screening rejects):** (a) B-Free extended-Synthbuster exact generator list + that FLUX/SD3.5 use RAISE conditioning at <=1024 (the page says "extended dataset also includes FLUX and SD3.5"); (b) the Synthbuster Zenodo download is currently rate-limited (12GB zip) — access logistics, not a screening fail; (c) the Synthbuster SDXL checkpoint/refiner remains undocumented (see `fov_matchable_sdxl_hunt` / sdxl_sources notes); (d) DRCT fake storage format vs COCO-JPEG reals.

---

## REJECTED (narrower reasons under the widened rule)

| Generator (dataset) | Reject reason |
|---|---|
| **SDXL / SDXL-refiner / -DR / -Ctrl / LCM-SDXL (DRCT-2M, MSCOCO)** | reals <=640 **SMALLER** than 1024 fakes -> upscaling required -> uncontrollable FOV gap (the original wall). *NB: SDXL itself is fine elsewhere — see PASS rows 1-2.* |
| **FLUX.1 / SD3 (GenImage++, ImageNet)** | 1024 fakes vs ImageNet ~470 reals -> reals smaller -> reject. *Use the B-Free/RAISE-conditioned FLUX/SD3.5 instead (PASS row 1).* |
| **Midjourney v5/v6 (GenImage, ImageNet)** | 1024 vs ~470 -> reals smaller. (MJ v5 via **Synthbuster/RAISE** passes — row 4.) |
| **PixArt-Sigma/alpha, Stable Cascade, Kandinsky, Playground v2.5 (ImageNet/COCO-anchored)** | 1024 native vs sub-700 reals -> reals smaller. (Would pass if paired with RAISE/FFHQ-scale same-source reals — none found yet.) |
| **TrueFake (FFHQ/FORLAB reals + SDXL/SD3/FLUX fakes)** | AC-1 violation: fakes are independent T2I, not paired to FFHQ; foreign proxy reals. (Resolution would pass — fails on AC-1.) |
| **SFHQ-T2I / "130K real-vs-fake faces"** | synthetic-only (no paired real) OR FFHQ reals downscaled to 256 + foreign-source -> AC-1/res fail. |
| **DiffusionFace (FFHQ/CelebA + diffusion faces)** | mixed: T2I-from-foreign portions violate AC-1; only img2img-edited-from-FFHQ portions could qualify (1024 same-source) — **unverified**, flag for a focused check if a face domain is wanted. |
| **TGIF / TGIF2 (MS-COCO, SDXL-inpainting)** | the fake is a local inpainting sub-region, not whole-image generation -> not a clean gen-signal target. |
| **B-Free COCO branch with 1024-output fakes vs native COCO 640** | reals smaller than fakes -> reject that branch; use the RAISE branch (reals >= fakes) or the 512-recon training set. |

---

## Bottom line
Widening the rule flips the high-res same-source corpora to PASS and **revives SDXL** (and newly admits **FLUX / SD3.5**) via **Synthbuster + B-Free's extended-Synthbuster on RAISE-1k** (reals >= fakes, caption-paired, both lossless). The cleanest AC-1 control is **B-Free's SD2.1 self-conditioned reconstructions** (pixel-aligned, 512=512). DRCT-2M is still usable for **<=512 modern generators** (Turbo/LCM) by cropping COCO-640 -> 512, but its SDXL-1024 variants still reject. The only hard rejects now are **reals-smaller-than-fakes** (no croppable-down option) and **AC-1 violations** (FFHQ/TrueFake foreign reals, TGIF local-inpainting).

---

## Reachability (2026-06-27) — host + HTTP probe (probes only, no data pulled)
Network note: **Synthbuster's Zenodo record returns 403 from my network** (record/API/homepage — reconfirmed from a prior task). B-Free does **NOT** depend on Zenodo — it self-hosts on **`grip.unina.it`** (Univ. Naples Federico II) + GitHub. All probes below are HEAD / 1-byte-range (`Range: bytes=0-0`).

| Item | Host | File / endpoint | Probe result | Verdict |
|---|---|---|---|---|
| **(1) FLUX + SD3.5 fakes** (extended Synthbuster) | `www.grip.unina.it` | `/download/prog/B-Free/extended_synthbuster/sd3_flux.zip` (3.44 GB) | **HTTP 206**, `content-range: bytes 0-0/3436462649`, `accept-ranges: bytes`, `application/zip` | **REACHABLE** |
| **(1) paired RAISE reals** | `www.grip.unina.it` | `.../extended_synthbuster/real_RAISE_1k.zip` (1.65 GB) | **HTTP 206**, `0-0/1646333296`, range OK | **REACHABLE** |
| (1) also present | `www.grip.unina.it` | `.../extended_synthbuster/latent-diffusion.zip` + `checksum.txt` (MD5s) | dir listing HTTP 200; checksum.txt HTTP 200 | **REACHABLE** |
| **(2) SD2.1 self-conditioned recon** (training set) | `www.grip.unina.it` | `/download/prog/B-Free/training_data/COCO_real_512.zip` (22.0 GB) + `SD2.1_selfconditioned.zip`, `SD2.1_inpainted_*`, `masks_and_bbox.zip` | **HTTP 206**, `content-range: bytes 0-0/22039405628`, range OK | **REACHABLE** |
| Code / README / file manifest | GitHub (`raw.githubusercontent.com`, `api.github.com`, `github.com`) | repo `grip-unina/B-Free`, `training_data/README.md` | HTTP 200 | **REACHABLE** |

**License (both):** Copyright (c) 2025 GRIP-UNINA, all rights reserved; "used, reproduced and modified only for informational and nonprofit purposes" (LICENSE.txt in the package). Research/nonprofit OK; confirm before any redistribution.

### Per-item answers + corrections to the screen
- **(1) FLUX / SD3.5 — REACHABLE** from `grip.unina.it` (HTTP 206, range-capable), no Zenodo needed. Mechanism: direct HTTPS zip (`sd3_flux.zip` = FLUX+SD3.5 fakes; `real_RAISE_1k.zip` = paired reals). **Confirmed RAISE-conditioned** ("fake images are generated using captions extracted from real images ... using captions from RAISE"). **Resolution correction:** the README says the RAISE reals were **resized so area ~ 1024x1024**, NOT kept at native ~4288. So this packaged set is **reals ~ 1024^2 vs FLUX/SD3.5 1024^2 -> reals >= fakes by equality** (no upscale) — still PASS, but it's *equal*, not reals >> fakes. Both lossless-packaged (RAISE source uncompressed; verify the fake format inside the zip). Generators: 1K FLUX + 1K SD3.5 (+ latent-diffusion).
- **(2) SD2.1 self-conditioned reconstruction — REACHABLE** from `grip.unina.it` (HTTP 206, 22 GB, range-capable). Mechanism: direct HTTPS zips. **Pixel-content-aligned confirmed:** reals = MS-COCO "largest central crop resized to **512x512**"; fakes = SD2.1 generated from that same crop via the SD2.1 inpainting code (self-conditioned reconstructions + same-/diff-category inpaints). Working resolution **512x512**, reals = fakes (no upscale). Real source COCO (CC-licensed subset), 51,517 real / 309,102 generated.
- **SDXL reachability caveat (correction to PASS rows 1-2):** the B-Free extended set ships **FLUX, SD3.5, latent-diffusion only — NOT SDXL.** SDXL on RAISE exists **only in the original Synthbuster**, whose sole host is **Zenodo = blocked from my network.** So under this network: **SDXL-on-RAISE is currently UNREACHABLE** (screening-viable but host-blocked); **FLUX/SD3.5-on-RAISE and SD2.1-recon are REACHABLE.** If SDXL specifically is wanted, options are: find a non-Zenodo Synthbuster mirror, or generate SDXL-from-RAISE-captions in-house following the Synthbuster/B-Free protocol.

**Net:** the two B-Free candidates I would pick from (FLUX/SD3.5 extended-Synthbuster; SD2.1 self-conditioned recon) are both **REACHABLE from `grip.unina.it` via range-capable HTTPS zips** — no Zenodo dependency. SDXL-on-RAISE remains blocked (Zenodo-only).

### native RAISE-1k reachability (2026-06-27) — for the Row 1 AC-9 fix path
**Verdict: REACHABLE** from my network. Native RAISE-1k is NOT Zenodo-hosted; it lives on the original Univ. Trento server, which serves with range support.
- **Host:** `loki.disi.unitn.it/RAISE/` — HTTP **200** (http and https), Apache, `accept-ranges: bytes`. (`mmlab.disi.unitn.it/RAISE/` 301-redirects to an https variant; the primary `loki` host is the live one.)
- **Mechanism:** the RAISE-1k "package" is a **CSV manifest of direct per-image URLs**, not a monolithic archive. `GET /RAISE/getFile.php?p=1k` -> HTTP **200**, `Content-Disposition: attachment; filename="RAISE_1k.csv.zip"`, `application/octet-stream` — served **without an enforced form/cookie** (the `confirm.php?package=1k` agreement page exists but the getFile endpoint returns the manifest directly). The CSV lists, per image, direct TIFF (and NEF) URLs on the same host. Probed a representative direct binary on the host (`/RAISE/Flat-field/Flat-field.zip`): HTTP **206**, `content-range: bytes 0-0/5381509030`, `accept-ranges: bytes` -> confirms the host streams large binaries with range requests (so per-image TIFFs are pullable the same way). *No image bytes pulled — status/headers only.*
- **Formats:** uncompressed **TIFF** + camera-native **NEF (RAW)** per image (the CSV provides both URLs); high-res — so there IS a decodable real >=1024 on the real side. (Nikon D7000/D90/D40 capture -> typical native ~**4288x2848**; some D40 frames ~3008x2000.) A 1024 **center-crop of the native TIFF** needs no interpolation (AC-9 clean).
- **Native resolution:** ~**4288x2848** (>>1024), confirming the Row-1 fix is geometrically valid: re-derive reals by center-cropping native RAISE TIFFs to 1024, preserving the Synthbuster/B-Free caption pairing.
- **License:** non-commercial research/educational use only; cite Dang-Nguyen et al., ACM MMSys 2015. (Mirror note: a partial **Kaggle** "RAISE-TIFF Uncompressed x300" exists but is a 300-image subset, not the full 1k — the primary `loki` host is reachable and preferred.)

**Caveat to verify at pull-time:** B-Free's `real_RAISE_1k.zip` reals are RAISE **resized to ~1024^2 area**; if I re-derive from native RAISE instead, I must re-match the **exact same 1000 RAISE image IDs** used by Synthbuster/B-Free (the CSV `File` column gives the IDs) so the caption pairing and the FLUX/SD3.5 fake set stay aligned. The native-RAISE host exposes every ID, so this is doable.
