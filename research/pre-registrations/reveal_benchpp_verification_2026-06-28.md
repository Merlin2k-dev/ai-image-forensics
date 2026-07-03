# REVEAL-Bench++ usability verification (U3.3 staging) — 2026-06-28

**Question:** Is REVEAL-Bench++ (arXiv 2511.23158) usable as the PRIMARY independent frontier
cross-check (a DIFFERENT lab's FLUX/SD3.5, to prove I caught "FLUX/SD3.5" and not "OpenFake's
pipeline")?

**VERDICT: UNUSABLE — recorded as an open fork requiring an explicit decision before U3.3.**

## Findings
- Paper: Cao et al., "REVEAL," arXiv 2511.23158 (v1 Nov 2025, v2 Apr 2026).
- REVEAL-Bench++ test set: 10,000 imgs = 5 generators x 2,000 (1,000 real + 1,000 AI each).
  Generators: FLUX (variant UNVERIFIED — dev vs schnell not stated), FLUX2, SDv3.5, Z-Image, Qwen-Image.
- **Blocking (primary): NOT RELEASED.** No HuggingFace repo, no GitHub, no project page as of 2026-06-28
  (7 months after v1). The paper says "data and codes will be released" (future tense). No download path exists.
- **Blocking (secondary, if released): resolution unspecified for the ++ test set.** The REVEAL *training*
  set policy is 50% >=512 / 30% 384-512 / 20% <384. If ++ follows it, up to 50% would be <512 on a side ->
  cannot center-crop to 512 without upscaling (violates AC-9 no-upscale). Unconfirmed but cannot be ruled out.
- Tertiary: FLUX variant unconfirmed as FLUX.1-dev; real-image source/provenance unstated; license unstated.
- No alternative within the paper/group fills the frontier-FLUX/SD3.5 role (it also evals on GenImage, which
  lacks FLUX.1-dev/SD3.5).

## Implication for U3.3
The designated PRIMARY pipeline-independence check is not obtainable. Remaining cross-checks are:
- OpenFake-internal: FINAL-TEST, fluxpro, fluxschnell (share OpenFake's generation pipeline -> cannot
  separate "caught FLUX" from "caught OpenFake's FLUX pipeline").
- **B-Free SD2.1 (cross-lab, cross-corpus): the ONLY genuine generation-source-independence signal available**
  — but SD-lineage only, NOT the FLUX/SD3.5 frontier.
So the frontier pipeline-independence question cannot be fully answered at U3.3. This is a documented
limitation (pull-if-it-existed was option 1a; it does not exist).
