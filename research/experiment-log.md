# Experiment log — modern-generator transfer and the V2/V3/V4 line

Chronological record of every experiment and model-affecting decision. Dates, numbers,
and verdicts are exactly as recorded at the time. "Frozen-prediction" means: fit only on
the training split, predict on a disjoint surface, never fresh cross-validation.
"One-shot" surfaces are locked test sets read exactly once and then marked spent.
Where an entry says a result was independently re-derived, that means I re-checked the
evidence in a second, separate pass (fresh code path, fresh eyes on the logs) before
banking the verdict.

---

## Step 1 — 2026-06-27 — SDXL transfer: CHARACTERIZATION (no clean number obtainable)

**What was measured:** Transfer of the frozen Phase 1 white-box detector to **SDXL**, on a
confound-audited DRCT-2M set (same-source MSCOCO val2017 reals + SDXL-base fakes; 4,007
id-matched pairs/class; frozen `preprocess()` path unchanged).

**Result:** **No confound-clean transfer number is obtainable on available data.**
Audit-split classes are ~99% separable, but the separation is **confound-dominated, not
generation detection.**

**Per-axis confound audit (audit split, 1,500/class; AC-8 — SDXL is the single generator here):**
- **Content-complexity (VQDM):** unmatched 0.9945 -> within-entropy-bin 0.9946 — **PASS**
  (not a composition confound; caption-level pairing was not the problem).
- **Compression:** metadata probe raw 1.000 -> post-`preprocess()` 0.962, entirely
  `file_size` (a model-EXCLUDED feature). Model-facing compression bounded at
  **0.65 / +0.12** (real-vs-real diagnostic). Bounded, does not reach the decision.
- **FOV/zoom:** matched-zoom **n_real = 0**, `crop_fraction` disjoint (fakes 0.0625 vs
  reals >= 0.16) — **STRUCTURALLY UNCONTROLLABLE** (perfectly collinear with class).

**Confound checks (AC-1..AC-9):** AC-1 same-source PASS; AC-3/AC-9 256 crop no-upscale
PASS; AC-4 compression measured + bounded (model-facing); AC-5 split-first PASS; AC-6 test
set UNSPENT (G3 not run); AC-7 no test tuning PASS; AC-8 single-generator; **FOV axis
FAIL-to-control (structural).**

**Decision:** Do NOT run G3 — a high-but-confounded number is not worth the sacred
one-shot. The DRCT test split remains unlocked/unspent. Reporting "cannot be cleanly
measured" is the honest result, backed by a documented NOT-FOUND source hunt for a
FOV-matchable same-source SDXL set (see
`pre-registrations/fov_matchable_sdxl_hunt_2026-06-27.md`).

**Conclusion after an independent re-check of the evidence:** the audit-split separability
is real but attributable to FOV + compressibility, not SDXL generation; it cannot be
de-confounded on DRCT (FOV collinear) or on any available same-source SDXL dataset
(~1024^2 native same-source reals do not exist). The trustworthy conclusion is
**"transfer cannot be cleanly measured"**, not "transfer is high/low."

**Contribution:** Confound-clean white-box transfer measurement to high-native-resolution
modern generators is structurally blocked — the same property (high res, no vintage
tells) drives both the detection difficulty and the measurement wall (FOV collinearity
from missing matched-resolution same-source reals).

**Next:** Step 2 reframed: ship Phase 1 with this documented modern-generator limitation;
do NOT retrain on DRCT (STANDING RULE); any extension still needs a compression-clean,
FOV-matchable same-source SDXL source that does not currently exist. Held pending a final check.

---

## Step 2 — 2026-06-27 — MULTI-GENERATOR TRANSFER (AUDIT PHASE): apparent GENUINE generalization — HYPOTHESIS REVERSAL

**STATUS WARNING: these are AUDIT-SPLIT regime indicators (fresh-CV on the 31 frozen
features), NOT the held-out result. The official numbers come from the G3 batch (the
frozen-MODEL run on the LOCKED test sets), which had NOT been run at the time of this
entry and was kept as a deliberate, separate step. The audit numbers had
predicted well so far, but G3 on untouched locked data is what counts — especially
because this entry overturns a prior finding.**

**What was measured:** Transfer of the frozen Phase 1 white-box LR to four MODERN
generators on resolution-matched, AC-1-clean datasets selected via the up-front resolution
screen (reals >= fakes, crop down — never up). Per-generator (AC-8), full confound audit
on each audit split.

**Audit-split regime (clean, post all confound control):**

| Generator | Dataset | Lineage | Clean regime | Confound state |
|---|---|---|---|---|
| SD-Turbo | DRCT-2M @512 | SD | ~0.99 | FOV/content clean; compression-inflated (bounded 0.65, model-excluded file_size) |
| SD2.1-recon | B-Free pixel-aligned @512 | SD | ~0.79 | AIRTIGHT (lossless, pixel-aligned, FOV=0 by construction; resampling <= 0.567) |
| FLUX | B-Free + native-RAISE @1024 | NOVEL (rectified-flow) | ~0.99 | AIRTIGHT after native-RAISE fix |
| SD3.5 | B-Free + native-RAISE @1024 | NOVEL (MMDiT) | ~0.98 | AIRTIGHT after native-RAISE fix |

**Determination:** apparent **GENUINE generalization** — the frozen white-box LR, trained
only on 2022-era SD generators, separates all four modern generators (incl. the 2024
frontier FLUX/SD3.5) at high AUC with every classical confound removed. **Both prior
predictions FALSIFIED:** (i) "white-box degrades on modern generators" (vintage-tell
hypothesis); (ii) the narrower "SD-lineage-only transfers, novel degrades."

**Strongest evidence it's real (not residual confound):** the native-RAISE FOV fix made
FLUX/SD3.5 RISE (0.93 -> 0.98), not fall. Had 0.93 been FOV inflation, removing the
confound would have collapsed it toward chance; instead it rose, so the confound was
DEFLATING the true signal. A confound-fix that *strengthens* a result is hard to fake.
(Matched-zoom n: 0 -> 498; metadata RAW 0.998 -> 0.65.)

**TWO CAVEATS HELD FIRMLY (not spin):**
1. **Audit-split, not held-out.** These are feature-CV regime indicators. The G3
   frozen-MODEL run on the locked test sets is the validating result and has not happened.
2. **This REVERSES the Phase-1 vintage-tell / Midjourney-wall conclusion.** Not a
   contradiction (MJ-in-GenImage was a different, FOV-limited measurement — see Step 1's
   FOV structural wall), but a result that overturns a prior finding earns maximal
   scrutiny. G3 on untouched locked data validates the reversal.

**Proposed mechanism (HYPOTHESIS, not proven fact):** diffusion + flow models share
VAE-decoder upsampling and frequency/noise fingerprints that persist across architectures;
the white-box frequency/noise/texture features capture them, so detection generalizes
SD1.x -> SD2.1 -> SD-Turbo -> SD3.5 -> FLUX. Stated as a plausible explanation only.

**Confound checks (per leg):** AC-1 same-source (SD-Turbo/SD2.1 caption/pixel-paired;
FLUX/SD3.5 RAISE-caption-paired) PASS; AC-3/AC-9 crop-not-upscale PASS (native-RAISE fix
= pure square center-crop); AC-4 compression (none for B-Free lossless legs;
bounded + model-excluded for DRCT-Turbo); AC-5 split-first PASS; AC-6 test sets
LOCKED + UNSPENT (G3 not run); AC-7 no test tuning PASS; AC-8 per-generator. FOV:
controllable on all four (SD-Turbo overlap; SD2.1 cf=const; FLUX/SD3.5 matched post
native-RAISE).

**Next:** all four test sets locked + staged; then the G3 batch under a deliberate
authorization step; then the held-out per-generator table; then finalize this entry with
the G3 numbers.

---

## Step 2 — 2026-06-27 — G3 HELD-OUT FINAL (sacred one-shot, test sets SPENT): the audit reversal was a PROXY ARTIFACT

**This SUPERSEDES the audit-phase "genuine generalization" reading above.** The held-out
frozen-MODEL run is the result that counts; it reverses the audit-split picture decisively.

**OFFICIAL HELD-OUT TABLE (frozen Phase-1 `predict_image`, run ONCE; run-once sentinel
`data/G3_COMPLETED.json` with per-CSV SHA-256; harness `evaluation/g3_batch.py`
dummy-verified in an independent pass before the run; executed after a deliberate
authorization step):**

| Generator | N/class | Held-out AUC | Bootstrap 95% CI | Acc@0.5* | Audit regime (proxy) | Gap (held-audit) |
|---|---|---|---|---|---|---|
| SD-Turbo (SD-lineage) | 2507 | **0.828** | [0.816, 0.839] | 0.698 | 0.994 | **-0.166** |
| SD2.1-recon (SD-lineage) | 2500 | **0.554** | [0.538, 0.569] | 0.528 | 0.786 | **-0.232** |
| SD3.5 (MMDiT, novel) | 500 | **0.301** | [0.270, 0.334] | 0.400 | 0.984 | **-0.683** |
| FLUX (rectified-flow, novel) | 500 | **0.144** | [0.120, 0.170] | 0.200 | 0.986 | **-0.842** |

*Acc@0.5 = 0.5 default; Phase 1 set no operating threshold (`CONFIG.VERDICT_THRESHOLD_*` = None).

**Determination (flat, zero spin): the frozen Phase-1 white-box LR does NOT generalize to
modern generators.** Transfer degrades with architectural distance from Phase-1's SD1.x:
SD-Turbo (closest, distilled SD2.1) retains **partial** transfer (0.83, itself down from
the 0.877 in-distribution); SD2.1-reconstruction is **near chance** (0.55); the novel
architectures FLUX (0.14) and SD3.5 (0.30) are **ANTI-correlated** — below 0.5, the
model's tells point the wrong way (it scores the high-res real photos as *more* AI than
the smooth modern fakes). The original Phase-1 vintage-tell/degradation hypothesis is
**CONFIRMED**; the strong "SD-lineage transfers / novel degrades" prediction is partially
borne out in ORDERING (SD-Turbo > SD2.1 > SD3.5 > FLUX) but at far lower levels than the
audit implied, with novel archs anti-correlated rather than merely degraded.

**Why the audit was wrong (methodological finding — own it):** the audit-split "regime"
was a **fresh-CV AUC on the 31 frozen features** — it measured whether *some* classifier
can separate the classes in that feature space (separability-in-principle: yes, ~0.98).
The held-out is the **fixed Phase-1 coefficients**, whose decision direction does not
align with modern generators' feature differences (and is inverted for FLUX/SD3.5).
**Fresh-CV feature-separability is NOT a valid proxy for frozen-model transfer** (gaps
-0.17 to -0.84). Had the audit used the frozen model's `predict_image` on the non-sacred
audit split (allowed) instead of a fresh CV, the degradation would have been visible
pre-G3. **The sacred held-out one-shot is irreplaceable — it caught a false
"generalization" before it became a claim.** This is the central methodological lesson of
Phase 2.

**Confound checks:** all four legs were confound-audited (FOV/resampling/compression/
content controlled per leg; FLUX/SD3.5 via the native-RAISE square-1024 fix). The held-out
failure is a genuine model-transfer failure, not a residual confound. AC-6 test sets SPENT
(one-shot consumed, sentinel locked).

**Disposition (Step-2 -> ship):** ship the Phase-1 model with an HONEST, MEASURED
modern-generator scope: reliable on its 2022 generators (held-out 0.877); **partial on
SD-Turbo (~0.83); fails on SD2.1-recon (~chance) and is anti-correlated on the 2024
frontier (FLUX/SD3.5).** State supported generators explicitly. Do NOT claim
modern-generator coverage. Do NOT retrain on the confounded DRCT set (STANDING RULE).
Phase 1 stays frozen.

---

## U3.1 — 2026-06-28 — Modern-model data infra: census + pull + locked splits + frozen-prediction harness (INFRA STATE, not an eval result)

**What was done:** Built the U3.1 infrastructure for the SEPARATE modern model
(OpenFake-512, real=LAION vs fake=FLUX.1-dev + SD3.5). No features, no training, no
held-out reads — this entry records state, per U3.0's gated order. Reproducibility fix
confirmed in place: build scripts live in-repo
(`scripts/{census,modern_pull,build_splits,verify_pull,verify_harness}.py`, committed)
and run from a fresh, dedicated environment.

**Census (STOP-GATE — a binding pre-set constraint; reals bind):** LAION >= 512 pool in
OpenFake estimated **~269,391** images (sample across 20 dataset files: laion_frac mean
0.2661 +/- 0.0140 -> TOTAL_LAION ~613,752 +/- 14,180; >= 512 yield 0.439 +/- 0.011).
**Stop-gate CLEARED** — the pool is healthy, not thin; supports TRAIN + DEV-VAL + a stable
few-hundred-per-class locked FINAL-TEST.

**Final pull counts (on-disk):** laion **6000/6000** OK, flux **2500/2500** OK, sd35
**2500/2500** OK, fluxschnell **1000/1000** OK, fluxpro **841/1000** — **841 ACCEPTED as
final** (fluxpro is a bonus cross-variant held-out check, not a training class or primary
gate; 841/class is plenty for a stable AUC; not worth re-engaging the throttled, flaky
download source for 159 non-critical images).

**Locked splits (seed 42, content-hash global dedup, per-generator AC-8;
`scripts/build_splits.py`):**

| Split | Total | Real (laion) | Fake by gen | Perms |
|---|---|---|---|---|
| TRAIN | 6000 | 3000 | flux 1500 + sd35 1500 | rw (dev) |
| DEV-VAL | 2000 | 1000 | flux 500 + sd35 500 | rw (dev) |
| FINAL-TEST | 2000 | 1000 | flux 500 + sd35 500 | **read-only LOCKED** |
| XCHECK_fluxpro | 1841 | 1000 | fluxpro 841 | **read-only LOCKED** |
| XCHECK_fluxschnell | 2000 | 1000 | fluxschnell 1000 | **read-only LOCKED** |

Cross-check real pool = 1000 leftover LAION (disjoint from core reals; shared across the
two xchecks by design — reals are never trained on, so not a leak). **Disjointness
asserted and PASSED:** TRAIN/DEV-VAL/FINAL-TEST mutually content-hash disjoint; no
intra-split dup hashes; xcheck-reals disjoint from all core. FINAL-TEST = 500/class **per
generator** — above the stable-AUC floor. SD-lineage held-out = the pre-existing 47,517
unused B-Free SD2.1 (separate, hash-disjoint, not rebuilt here).

**Frozen-prediction harness DUMMY-VERIFIED (`evaluation/harness.py`,
`scripts/verify_harness.py`):** exit 0, "HARNESS OK", all five checks pass —
(1) separable -> AUC 0.981; (2) pure noise -> AUC 0.500 (no spurious signal manufactured);
(3) **frozen != CV sign-flip -> AUC 0.014** (a feature separable per-split but sign-flipped
across splits anti-correlates under the FITTED direction — structurally reproduces the
FLUX 0.14 artifact that fresh-CV hid at 0.98; the 0.98 -> 0.14 mistake CANNOT recur through
this harness); (4) train/eval content-hash overlap -> AssertionError fires; (5) missing
content_hash col -> ValueError fires. The harness exposes exactly ONE primitive
`evaluate(features, train_split, eval_split)` (fit LR on TRAIN only, predict on disjoint
eval, return AUC) — no fresh-CV-on-features code path exists.

**Disposition:** U3.1 COMPLETE. **HOLD at the U3.1 -> U3.2 gate** for a review of the
locked split sizes + harness dummy-verification output. On approval, U3.2 begins with
**VAE-decoder fingerprints** as the first feature family through the frozen-prediction
held-out gate (TRAIN -> DEV-VAL), per U3.0 non-negotiable (2). No feature work until that
approval.

---

## U3.2 — Family 1/N — 2026-06-28 — VAE-decoder fingerprints — CONDITIONAL ACCEPT

**What changed:** First fresh feature family for the separate modern model (no
grandfathering). 18 white-box features (pure NumPy/SciPy) computed on the **512 raw
substrate** (single Q75 4:2:0, uniform both classes) keyed by content_hash: F1 chroma
period asymmetry, F2 cross-difference anti-diagonal SNR (JPEG-axis-nulling), F3
Gaussian-residual grid SNR, F4 azimuthal radial spectrum deviation/slope, F5 DCT grid
excess — all at periods 8 AND 16 (FLUX/SD3.5 VAE period is **16px**, SD2.1/SDXL 8px;
arXiv:2510.05633). Workflow: literature -> recipes -> implementation -> independent
evaluation pass.

**Eval primitive:** frozen-prediction ONLY (`evaluation/harness.py` `evaluate()`, fit LR
on TRAIN, predict disjoint DEV-VAL). NO fresh-CV. DEV-VAL = dev set (soft-overfit by
selection), NOT the final number.

**Per-generator frozen-prediction DEV-VAL AUC (AC-8; C=1.0, cw=None; 95% CI seed 42):**

| Eval | AUC | 95% CI | vs frozen-2022 G3 |
|---|---|---|---|
| **FLUX** | **0.705** | [0.678, 0.732] | 0.14 -> genuine jump above chance |
| **SD3.5** | **0.618** | [0.588, 0.647] | 0.30 -> above chance |
| Combined | 0.662 | [0.638, 0.684] | — |

Both CIs exclude 0.50. Stable across C in {0.1, 1, 10} and class_weight None/balanced
(nothing knife-edge).

**Confound axis audit (designated axis = compression + FOV):**
- **FOV:** controlled at 512 (uniform square crop, cf ~0.25 both classes; U2 record). PASS.
- **Compression real-vs-real bound:** features ARE compression-sensitive (Q75-vs-Q60
  real-vs-real separability 0.694; Q75-vs-Q90 0.524) — BUT re-compressing reals shifts the
  frozen classifier's fake-prob by -0.037 (Q60) / -0.001 (Q90), i.e. *toward real*, while
  the real -> fake gap is +0.103 (FLUX) / +0.043 (SD3.5). Compression is ANTI-aligned with
  the learned fake direction and cannot manufacture the class gap. **Compression confound
  controlled.**
- **Mechanism localization:** NOT a content-energy/magnitude proxy (magnitudes-only ~
  chance 0.556; dropping them costs nothing). SD3.5 is on-mechanism (grid-only 0.613).
  **FLUX rides f4 broadband spectral slope, NOT the named period-8/16 grid (grid-only FLUX
  0.627)** — so the "VAE fingerprint" label overstates FLUX's mechanism (it's
  broadband-spectral for FLUX, grid for SD3.5).

**Confound checks:** AC-1 same-source (LAION-caption-seeded) PASS; AC-3/AC-9 512 crop
no-upscale PASS; AC-4 uniform Q75 4:2:0 both classes PASS; AC-5 split-before-stats
(TRAIN-only fit, harness scaler on train) PASS; AC-6 FINAL-TEST + xchecks
LOCKED/untouched PASS; AC-7 no test tuning (DEV-VAL only) PASS; AC-8 per-generator PASS.

**Verdict after the independent evaluation pass — CONDITIONAL ACCEPT:** the signal is real
and confound-controlled, but **modest (0.62-0.71)** — a weak ensemble contributor, not
standalone; the FLUX lift is broadband-spectral, not grid. **Accept as a candidate
contributor** into the family set, re-understood as broadband(FLUX) + grid(SD3.5) mixture.
**Upgrade/confirm conditions (U3.3):** both generators must clear chance on the locked
FINAL-TEST + cross-checks. **Reject if:** either generator collapses on FINAL-TEST, or the
f4 FLUX signal proves compression/content-driven on any NON-uniform-Q surface (production
realism).

**Next recommended step:** proceed to Family 2 — **camera-sensor-absence**
(PRNU/demosaicing, attacks from the REAL side; confound axis = compression + real-corpus)
— same mini-cycle. The VAE family's FINAL-TEST behavior is deferred to the U3.3 one-shot;
do NOT consult cross-checks per-family. Implementation:
`pipeline/features/vae_decoder.py`, `scripts/extract_features.py`.

---

## U3.2 — Family 2/N — 2026-06-28 — Camera-sensor-absence — CONDITIONAL ACCEPT (load-bearing PAIR only)

**What changed:** Second candidate family, scored INCREMENTALLY over the accepted VAE set
(per a pre-committed guardrail). 9 white-box features (pure NumPy/SciPy, luma-domain) on
the SAME 512 raw substrate (no new pull — features are a different computation on the same
pixels): s_rho (cross-channel HF correlation), s_nyq_ratio/_col (luma 2px-Bayer Nyquist),
s_noise_cov/_maxmed (local-noise-variance stationarity), s_nlf_slope/_intercept
(shot-noise NLF), s_resid_kurtosis/_skew (residual moments). I flagged the brutal
Q75-4:2:0 substrate up front (single-image PRNU + RGB-2px CFA DESTROYED, correctly
excluded).

**Accepted-set baseline:** VAE-18 alone — FLUX 0.7054, SD3.5 0.6177 (frozen-prediction
DEV-VAL).

**INCREMENTAL transfer, per-generator (AC-8; frozen-prediction DEV-VAL; paired-bootstrap
lift CI seed 42):**

| Eval | VAE-only | VAE+sensor(9) | sensor-only | Incremental lift [95% CI] |
|---|---|---|---|---|
| **FLUX** | 0.7054 | 0.7305 | 0.6257 | **+0.0254 [+0.007, +0.044]** — REAL |
| **SD3.5** | 0.6177 | 0.6796 | 0.6306 | **+0.0618 [+0.041, +0.082]** — REAL |

Lift CI excludes 0 on BOTH generators; stable across C in {0.1, 1, 10}.

**Independence (decisive parsimony finding):** the family reduces to TWO load-bearing,
low-VAE-correlation features — **`s_noise_cov`** carries FLUX (+0.023, rho=0.37 w/ VAE)
and **`s_resid_skew`** carries SD3.5 (+0.036, rho=0.14). The pair reproduces 92%/68% of
the lift. The other 7 add ~0: `s_nyq_ratio` (rho=0.57 w/ VAE grid) is REDUNDANT with VAE
and fires INVERTED (higher on FLUX 1.48 than reals 0.94 -> NOT camera CFA; a FLUX HF tell
already captured) — mechanistically mislabeled, non-contributing.

**Confound axis audit (compression + real-corpus):**
- **Compression:** the lift SURVIVES dropping the explosive `s_noise_maxmed` (~1e11 on
  reals = JPEG flat-block artifact, NOT camera physics — DROP, conditioning hazard).
  Real-vs-real Q60/Q90 re-encode moves the sensor classifier's fake-prob AWAY from fake
  (-0.012) or negligibly (+0.005) vs the real -> fake gap (+0.048/+0.043) -> compression
  anti-aligned/orthogonal, cannot manufacture the gap. **Controlled.**
- **Real-corpus:** CANNOT be fully resolved without a 2nd real corpus (deferred to U3.3
  cross-checks). Risk-4 noted (FLUX/SD3.5 trained on LAION may have learned ISP-like
  traces). Re-labeled honestly as a **noise-structure + residual-skew tell, NOT a
  camera-CFA/Nyquist detector.**

**Confound checks:** AC-1/3/4/5/6/7/8 same status as Family 1 (uniform Q75 both classes;
TRAIN-only fit; FINAL-TEST + xchecks LOCKED/untouched — mtime verified; per-generator).

**Verdict after the independent evaluation pass — CONDITIONAL ACCEPT (pair only).**
ACCEPTED INTO THE FEATURE SET: **VAE-18 + {s_noise_cov, s_resid_skew}** (20 features). The
other 7 sensor features are DROPPED as non-contributing dev residue (respects the
explainability cap; avoids carrying confound-prone/redundant features). Conditions to bank
at U3.3: both lifts confirmed on locked FINAL-TEST; the two drivers validated on a 2nd
real corpus (cross-checks). The lift holds on both generators but via two independent
single-feature tells, not one mechanism.

**Accepted feature set now (CONDITIONAL, banked at U3.3): VAE-18 + {s_noise_cov,
s_resid_skew} = 20.** Combined DEV-VAL (this set): FLUX ~0.728, SD3.5 ~0.660. The next
family's incremental is measured over THIS 20.

**Next recommended step:** Family 3 — **Benford / block-DCT** (PRIOR-RISK FLAG: rejected
in Phase 1 for compression contamination -> EXTRA compression-axis scrutiny, no free
pass), then polynomial/interaction (explainability-cap watch). Same mini-cycle;
cross-checks stay locked for U3.3. Implementation: `pipeline/features/sensor_absence.py`.

---

## U3.2 — Family 3/N — 2026-06-28 — Benford / block-DCT — REJECT (extends the Phase-1 rejection, new reason)

**What changed:** Third candidate (PRIOR-RISK FLAG: rejected in Phase 1 for compression
contamination), scored INCREMENTALLY over the accepted 20-set (VAE-18 + {s_noise_cov,
s_resid_skew}). 8 white-box scalar features on the same 512 raw substrate (no new pull):
b_img_midband_kl (compression-prone control), b_resid_kl/_chi2 (median-residual Benford),
b_genben_mse/_alpha (generalized Benford fit), b_dct_kurtosis/_skew (mid-band moments),
b_resid_ovlp_chi2 (Wiener-residual fine-scale). Framing going in: the naive image-domain
version stays confounded; residual/generalized variants are materially different — test
them.

**INCREMENTAL transfer, per-generator (AC-8; frozen-prediction DEV-VAL; paired-bootstrap
lift CI seed 42):**

| Eval | 20-set | 20+benford(8) | benford-only(8) | Incremental lift [95% CI] |
|---|---|---|---|---|
| **FLUX** | 0.7284 | 0.7629 | 0.6857 | +0.0346 [+0.018, +0.052] — REAL but FLUX-only |
| **SD3.5** | 0.6599 | 0.6674 | 0.5637 | **+0.0075 [-0.013, +0.028] — NOT real (CI spans 0)** |

Stable across C in {0.1, 1, 10}. **AC-8 FAILS: lift on FLUX only, within noise on SD3.5.**

**Compression-axis audit (extra scrutiny — the Phase-1 confound did NOT recur this time):**
- The image-domain compression-prone trio (b_img_midband_kl, b_dct_kurtosis, b_dct_skew)
  carries ZERO lift (FLUX -0.0015, SD3.5 -0.0009) -> the old confound is absent.
- Real-vs-real Q60/Q90 re-encode moves the benford classifier AWAY from fake
  (-0.078/-0.021) vs the real -> fake gap (+0.081/+0.028) -> compression cannot manufacture
  the gap. The lone FLUX contributor `b_genben_alpha` is compression-INVARIANT
  (real-vs-real separability ~chance 0.47/0.51).
- Smoothness: fakes smoother (resid-energy 5.4 vs 7.0); most benford features are a
  smoothness meter.

**Redundancy / attribution:** the entire FLUX lift is ONE feature, `b_genben_alpha`
(+0.0348 alone ~ the full 8-set lift; rho=0.18 with the 20-set). The feared
`b_dct_skew` ~ `s_resid_skew` overlap is DISPROVEN (rho=-0.083), BUT b_dct_kurtosis is
redundant with accepted s_noise_cov (rho=0.76) and b_resid_ovlp_chi2 (rho=0.59) — the thin
SD3.5 signal rides those redundant/smoothness features. No subset lifts BOTH generators.

**Confound checks:** AC-1/3/4/5/6/7/8 — FINAL-TEST + xchecks LOCKED/untouched (mtime
verified); per-generator.

**Verdict after the independent evaluation pass — REJECT.** Unlike the sensor family (a
reduced pair lifted both generators), no honest benford subset clears both: the only
clean/compression-robust/non-redundant signal (`b_genben_alpha`) is FLUX-only; the SD3.5
contribution is redundant + smoothness-driven. Adding 8 features that lift one generator
and re-measure the accepted set on the other is the selection-set-fitting trap. **Accepted
set UNCHANGED: VAE-18 + {s_noise_cov, s_resid_skew} = 20.** Honest carve-out (kept OUT for
now): `b_genben_alpha` may be reconsidered later as a FLUX-only weak tell ONLY if it
survives the locked FINAL-TEST on FLUX + a 2nd real corpus. Extending the Phase-1
rejection (for a NEW reason — AC-8/redundancy, not compression) is a valid honest outcome;
no pass was manufactured.

**Next recommended step:** Family 4 — **polynomial / interaction terms** over the accepted
20-set, under the EXPLAINABILITY HARD CAP (low-hundreds total; each term individually
justified; a blanket degree-N blow-up is a trade-off call I make explicitly, not
automatic). Same mini-cycle. Implementation: `pipeline/features/benford_dct.py`.

---

## U3.2 — Family 4/N — 2026-06-28 — Polynomial / interaction terms (BOUNDED) — REJECT

**What changed:** Fourth candidate, under the EXPLAINABILITY HARD CAP. 7
individually-justified pairwise interactions (products of accepted features;
grid/spectral x noise-structure; content-energy magnitudes deliberately excluded),
computed from the existing feature tables (NO image access). Scored incrementally over the
accepted 20-set. Special rule set in advance: if the lift needs MANY terms / grows with
complexity -> REJECT (that diffuseness IS the linear-model-forced-past-reach signal).

**INCREMENTAL transfer, per-generator (AC-8; frozen-prediction DEV-VAL; paired-bootstrap
CI seed 42):**

| Eval | 20-set | 20+poly(7) | poly-only(7) | Incremental lift [95% CI] |
|---|---|---|---|---|
| **FLUX** | 0.7284 | 0.7229 | 0.6712 | **-0.0055 [-0.011, -0.000] — HURTS FLUX** |
| **SD3.5** | 0.6599 | 0.6685 | 0.6352 | +0.0085 [+0.004, +0.014] — SD3.5 only |

Stable across C in {0.1, 1, 10}. **AC-8 FAILS: SD3.5-only, FLUX actively regressed.**

**Concentrated-vs-diffuse (special guardrail):** the lift is CONCENTRATED, not diffuse —
so it does NOT fail by term-count ballooning (cap technically satisfied). Forward
selection: the SD3.5 lift is carried by a SINGLE term `p_cdsnr16_x_skew` (+0.0074
standalone; steps 2-7 non-significant). The spec'd "star" cross-term `p_slope_x_skew`
(FLUX-slope x SD3.5-skew) is noise-to-NEGATIVE on both generators. No single term is
positive-real on both.

**Damning redundancy:** the lone SD3.5 carrier `p_cdsnr16_x_skew` is Spearman rho=**0.991**
with its parent `s_resid_skew` — already in the 20-set. It is a monotone rescaling of an
existing feature, NOT a new grid x skew mechanism; the +0.0074 is re-feeding the LR a
near-duplicate (selection-overfit signature), regularization-stable but mechanism-free.
`p_slope_x_skew` likewise rho=-0.993 with s_resid_skew.

**Confound checks:** AC-5 TRAIN-only fit; AC-6 FINAL-TEST + xchecks LOCKED/untouched (no
image/CSV access at all — poly reads only feature parquets); AC-8 per-generator.

**Verdict after the independent evaluation pass — REJECT** on two independent fatal gates:
(1) AC-8 (SD3.5-only, FLUX hurt); (2) the SD3.5 lift re-expresses an existing parent
(rho=0.991). Terms worth carrying: NONE. **Accepted set UNCHANGED: VAE-18 + {s_noise_cov,
s_resid_skew} = 20.** Honest expected outcome — interactions do not beat the wall; no pass
manufactured. **The principled candidate set (VAE, sensor, Benford, polynomials) is now
EXHAUSTED.**

**Next step:** STAGE toward U3.3 (decision: do NOT fish for more families). Stage the
one-shot on the final accepted 20-set: confirm FINAL-TEST + all cross-checks
(REVEAL-Bench++, fluxpro, fluxschnell, B-Free SD2.1) locked + ready; review the staging
state before the final authorization. Implementation: `pipeline/features/interactions.py`.

---

## U3.2 — Family 5/N — 2026-06-28 — Inversion-residual (2nd-tier) — CONDITIONAL (FLUX-targeted carve-out, NOT banked)

**What changed:** Second-tier candidate (white-box fixed-operator proxy for "one
reverse/denoising step": AI sits closer to a natural-image-prior manifold -> smaller
residual). 4 features on the 512 raw substrate: iv_tv_curv_energy (TV mean-curvature field
energy = direct reverse-step residual magnitude), iv_tv_curv_kurt (curvature-field
sparsity), iv_wav_noise_sigma (MAD noise floor), iv_wav_subthresh_kurt (a-trous
sub-threshold noise-Gaussianity). I scoped out the proposed cross-difference/upsampling
variants (F2/F4) as redundant-by-construction with the accepted VAE f2_cd_snr. pywt
absent -> scipy-only a-trous (documented). Scored incrementally over the accepted 20-set.

**INCREMENTAL transfer, per-generator (AC-8; frozen-prediction DEV-VAL; paired-bootstrap
CI seed 42):**

| Eval | 20-set | 20+inv(4) | inv-only(4) | Incremental lift [95% CI] |
|---|---|---|---|---|
| **FLUX** | 0.7284 | 0.8079 | 0.8061 | **+0.0791 [+0.061, +0.098]** — large, real |
| **SD3.5** | 0.6599 | 0.6862 | 0.5882 | +0.0263 [+0.005, +0.048] — barely; rides one redundant feat |

Stable across C in {0.1, 1, 10}.

**Architecture-specificity (the mechanism's key risk — flagged in advance):** STRONG
asymmetry. inv-only AUC 0.806 FLUX vs 0.588 SD3.5 (~chance); the lift is 3x larger on
FLUX. Every single feature's standalone SD3.5 lift CI includes 0 — FLUX-targeted, NOT
generation-general.

**Compression axis: CONTROLLED.** Q60 re-encode moves reals' fake-prob AWAY from fake
(-0.052) vs the real -> fake gap +0.105 -> anti-aligned. Q75-vs-Q60 real-vs-real
separability 0.61 (the features read compression level) but the direction is opposite the
fake axis. The FLUX lift is multi-axis (magnitude pair +0.068 AND structural pair +0.032,
both significant).

**Redundancy:** no |rho| > 0.7, BUT iv_wav_noise_sigma rho=-0.677 with accepted
s_noise_cov, and the SD3.5 pass exists ONLY when iv_wav_noise_sigma is included (drop it
-> TV-pair SD3.5 lift +0.016 [-0.005, +0.038], CI spans 0). The "both-generators" claim
rides on the most-redundant, compression-prone feature — the selection-set-fitting trap.
iv_wav_subthresh_kurt is dead on both (drop).

**Verdict — CONDITIONAL.** Disposition: **NOT banked into the both-generators 20-set**
(consistent with the strict AC-8 rule that sank Benford/poly — the only independent signal
is FLUX-only; the SD3.5 pass is redundant-driven). **PARKED as a strong FLUX-targeted
carve-out = the TV pair {iv_tv_curv_energy, iv_tv_curv_kurt}** (large +0.079 FLUX gain,
independent, compression-controlled, zero SD3.5 cost). DROP iv_wav_subthresh_kurt (dead)
and iv_wav_noise_sigma (redundant/fragile). **DECISION HELD FOR THE AUTHORIZATION GATE:**
final model = 20-set (strict, both-gen-clean) OR 20 + TV pair = 22 (FLUX-targeted, helps
the worst generator at no SD3.5 cost; the FLUX-pipeline-vs-general risk is tested at
U3.3). **The accepted both-generators set remains 20** for Family 6's baseline.
Implementation: `pipeline/features/inversion_residual.py`.

---

## U3.2 — Family 6/N — 2026-06-28 — Local-correlation / neighborhood-consistency (2nd-tier) — CONDITIONAL HOLD

**What changed:** Final candidate (2nd-tier). Mechanism: camera capture is spatially
uniform -> stationary local pixel-correlation; VAE/flow decoders -> non-stationary. 5
features = SPATIAL STD across 32x32 tiles of local correlation stats: lc_svlac_h/_v (lag-1
autocorr of green HP residual), lc_markov_d_std (autocorr of residual differences),
lc_xchcorr_rg_std/_gb_std (cross-channel R-G / G-B residual correlation — a genuinely NEW
axis; cross-channel s_rho existed in the sensor family but was DROPPED, so nothing
accepted covers it). Scored incrementally over the accepted 20-set.

**INCREMENTAL transfer, per-generator (AC-8; frozen-prediction DEV-VAL; paired-bootstrap
CI seed 42):**

| Eval | 20-set | 20+lc(5) | Incremental lift [95% CI] |
|---|---|---|---|
| **FLUX** | 0.7284 | 0.7902 | **+0.0618 [+0.041, +0.083]** |
| **SD3.5** | 0.6599 | 0.6951 | **+0.0352 [+0.013, +0.059]** |

Both CIs exclude 0; C-stable. **First family to clear AC-8 (both generators) on the
headline.**

**Carrier (AC-8):** the cross-channel pair {lc_xchcorr_rg_std, lc_xchcorr_gb_std} is the
ONLY both-generator carrier (reproduces ~the whole lift). lc_markov_d_std lifts FLUX
(+0.037) but is DEAD on SD3.5; SVLAC h/v tiny + FLUX-only. So AC-8 rests entirely on the
cross-channel axis.

**Confound audit — compression NOT cleanly controlled (the decisive finding):**
real-vs-real Q90 4:4:4 (chroma subsampling REMOVED) moves reals SAME-DIRECTION toward fake
(+0.032 prob; ~24% of the FLUX gap, ~43% of the SD3.5 gap). Mechanism: LAION web-JPEG
reals arrive already chroma-decimated; fresh fakes arrive full-chroma then subsampled once
-> cross-channel std is partly a CHROMA-HISTORY confound (strictly worse than the sensor
family's anti-aligned audit). BUT a substantial fake gap SURVIVES chroma-matched
equalization (FLUX +0.055, SD3.5 +0.028) -> not a pure artifact. Compression-history and
the real-corpus (ISP/demosaicing) risk are the SAME mechanism on the SAME load-bearing
feature — UNRESOLVABLE on DEV.

**Redundancy:** clean (max |rho| ~0.43 vs accepted, top s_noise_cov). Cross-channel is a
genuinely new axis.

**Verdict — CONDITIONAL HOLD.** Disposition: **NOT banked into the 20-set.** It's the
strongest both-generators candidate but its load-bearing cross-channel feature carries an
UNRESOLVED compression/real-corpus confound (~25-45% of the gap is chroma-history).
**PARKED: cross-channel pair {lc_xchcorr_rg_std, lc_xchcorr_gb_std}** with a HARD GATE —
adopt ONLY if the gap SURVIVES on the B-Free SD2.1 cross-check (different reals) with
chroma/compression equalized at U3.3; if it collapses on different reals -> REJECT as a
corpus/compression artifact. Drop svlac/markov (fail AC-8). **Accepted both-gen set
remains 20.**

**FEATURE SEARCH COMPLETE (decision: stop after Families 5 & 6).** Four families tested
earlier (VAE accept, sensor cond-accept-pair, Benford reject, poly reject) + two 2nd-tier
(inversion FLUX-carve-out, localcorr cross-channel-carve-out). **Firm accepted set = 20.**
Two PARKED carve-outs for the U3.3 one-shot to resolve on sacred data: (i) inversion TV
pair (FLUX-targeted), (ii) localcorr cross-channel pair (both-gen but chroma-confound-gated
on B-Free). Implementation: `pipeline/features/local_corr.py`.

---

## U3.3 — HELD-OUT FINAL (sacred one-shot, surfaces SPENT) — 2026-06-28 — modern model beats 0.14/0.30; cross-CORPUS transfer FAILS

**Authorized deliberately (same routing as G3); run-once, sentinel `data/U33_COMPLETED.json`
(per-CSV SHA-256). Harness `evaluation/u33_oneshot.py` dummy-verified HARNESS OK before
the run. Frozen-prediction only (LR C=1.0, fit on TRAIN, predict on DISJOINT locked
surfaces). 4 PRE-REGISTERED variants (no fishing). Surfaces NOW SPENT — no re-run.**

**OFFICIAL HELD-OUT TABLE (AUC, bootstrap 95% CI seed 42):**

| Variant | FINAL-TEST FLUX | FINAL-TEST SD3.5 | xcheck fluxpro | xcheck fluxschnell | B-Free SD2.1 (cross-lab) |
|---|---|---|---|---|---|
| **V0_base20** (PRIMARY, 20) | **0.7374** [.711,.765] | **0.6757** [.650,.703] | 0.7213 | 0.6578 | **0.5466** [.532,.563] |
| V1_inv22 (+TV pair) | 0.8167 [.794,.839] | 0.7068 [.681,.733] | 0.7960 | 0.7976 | 0.5470 |
| V2_lc22 (+cross-chan) | 0.7973 [.774,.820] | 0.7147 [.689,.741] | 0.7572 | 0.7646 | 0.5431 |
| V3_all24 (+both) | **0.8190** [.797,.840] | **0.7235** [.698,.750] | 0.7814 | 0.8008 | 0.5460 |

**DEV-VAL -> FINAL-TEST gap (V0):** FLUX 0.728 -> 0.737 (+0.009), SD3.5 0.660 -> 0.676
(+0.016). **FINAL >= DEV on both — NO overfit gap.** The frozen-prediction gate held; the
0.98 -> 0.14-type proxy collapse did NOT recur. Slightly above the recorded flat
expectation (FLUX high-0.6s / SD3.5 ~0.6).

**Determination (flat):**
1. **Substantial beat on OpenFake FLUX/SD3.5 vs the frozen-2022 baseline (0.14 / 0.30):**
   V0 FLUX 0.737 / SD3.5 0.676; best variant V3 FLUX 0.819 / SD3.5 0.724. The
   2022 -> modern wall is PARTIALLY beaten on the frontier targets — a real, measured
   improvement, not 0.85.
2. **Both parked carve-outs SURVIVED FINAL-TEST** (lifted both generators on held-out; not
   DEV-overfits). The TV pair (V1) = strongest independence evidence (cross-variant
   fluxpro 0.72 -> 0.80, fluxschnell 0.66 -> 0.80). V3 best raw on the frontier.
3. **Cross-VARIANT (fluxpro/fluxschnell) = positive partial-independence:** the model
   transfers to DIFFERENT FLUX variants (V0 0.66-0.72; V3 0.78-0.80). Pure FLUX-dev
   pipeline-memorization would collapse here; it didn't. BUT same lab / same LAION reals
   -> NOT cross-lab.
4. **Cross-LAB / cross-CORPUS (B-Free SD2.1) = NEAR CHANCE (0.546), ALL variants.** On a
   different lab's SD-lineage generator vs different reals, the model barely beats chance.
   The cross-channel carve-out's HARD GATE FAILED: V2 (0.543) ~ V0 (0.547) on B-Free ->
   its OpenFake gain did NOT survive on different reals (consistent with the
   chroma-history/corpus-coupling concern; B-Free conflates different-corpus with
   different-generator so it cannot fully isolate the two).

**FIRST-CLASS LIMITATION — frontier pipeline-independence UNVERIFIED (state in every
result and in the product):** The substantial FINAL-TEST numbers are on
OPENFAKE-GENERATED FLUX/SD3.5 vs OpenFake's LAION reals. **FINAL-TEST cannot distinguish
"we detect FLUX/SD3.5" from "we detect OpenFake's FLUX/SD3.5 pipeline + corpus"** — it
shares the training pipeline. The PRIMARY frontier independence check (REVEAL-Bench++) is
unobtainable (unreleased; its non-existence is itself the recurring Phase-2
data-availability finding — see
`pre-registrations/reveal_benchpp_verification_2026-06-28.md`). The ONE cross-corpus data
point available (B-Free SD2.1) shows POOR transfer (0.546) -> detection is substantially
OpenFake-corpus-coupled. Partial independence evidence that IS non-zero: cross-VARIANT
transfer (fluxpro/fluxschnell), mechanistically-motivated white-box features (not opaque
fingerprints). Honest framing: **"substantial lift on OpenFake-generated FLUX/SD3.5
(0.14 -> 0.74 base / up to 0.82 augmented), with cross-variant + (negative) cross-lineage
transfer reads, but full frontier pipeline-independence UNVERIFIED pending an independent
high-res FLUX/SD3.5 dataset that does not currently exist publicly."**

**Confound checks:** AC-5 TRAIN-only fit; AC-6 surfaces locked + SPENT (sentinel,
one-shot); AC-7 no test-tuning (4 variants pre-registered before the run); AC-8
per-generator throughout. Substrate uniform 512 Q75 4:2:0 across all surfaces (B-Free
harmonized identically).

**SHIP RECOMMENDATION (confirmed):** ship **V1_inv22 (20 + TV pair)** as the modern model
— best independence evidence (cross-variant transfer), TV pair mechanistically principled
and FINAL-TEST-confirmed, FLUX 0.817 / SD3.5 0.707. (V3 is +0.002-0.016 better on raw
FINAL-TEST but adds the cross-channel pair whose generation-generality is UNVERIFIED — its
B-Free gate failed; do not ship it as generation-general.) Conservative alternative:
V0_base20 (no carve-outs, FLUX 0.737 / SD3.5 0.676). Either ships WITH the first-class
frontier-independence limitation above. **U3.4 = ship decision + product scope.**

---

## P3.3 — CROSS-CORPUS HELD-OUT (sacred one-shot, NTIRE surface SPENT) — 2026-06-29 — V1 transfers ABOVE chance cross-corpus; triangulation complete

**Authorized deliberately (same routing as G3/U3.3); run-once, sentinel
`data/P33_COMPLETED.json` (held-out + frozen-model SHA-256). Harness
`evaluation/p33_oneshot.py` dummy-verified HARNESS OK (29/29; sign-flip -> 0.0000 proved
frozen-predict-only). FROZEN V1 loaded from `frozen_models/v1_inv22/model.joblib` and
PREDICTED only — no refit, no fresh-CV. NTIRE held-out SPENT.**

**MAIN RESULT — conflated cross-corpus frozen-prediction AUC (1500 non-LAION reals vs
1250 frontier-mix fakes): 0.7385 [0.7199, 0.7565]** (bootstrap 2000, seed 42). **V1
transfers ABOVE chance on the non-LAION frontier mix.**

**TRIANGULATION COMPLETE (frozen V1, three cells):**

| Regime | Source | AUC |
|---|---|---|
| same-corpus (LAION) - same-lab | OpenFake FLUX/SD3.5 (U3.3) | 0.82 / 0.71 |
| **cross-corpus (non-LAION) - frontier-MIX** | **NTIRE (this one-shot)** | **0.7385 [0.72, 0.76]** |
| cross-corpus - cross-lab - SD-lineage | B-Free SD2.1 | 0.547 |

**Mixed-Q sweep (additional re-encode on the Q75 substrate) — production-robustness,
CLOSES the U3-audit mixed-Q caveat:** Q40 0.6935 - Q60 0.7457 - Q75 0.7386 - Q90 0.7358 -
lossless 0.7385. **V1 is robust to additional eval-time compression** — stable 0.74-0.75
across Q60-lossless, only a modest drop to 0.69 under heavy Q40. (Caveat: the held-out
already carries Q75 history; the sweep measures further-compression robustness, not
arbitrary-Q from scratch.)

**Determination (flat):** V1 retains SUBSTANTIAL cross-corpus signal on a non-LAION
frontier mix (0.74) — markedly better than the lone prior cross-corpus datum (B-Free
0.547). This REVISES the pessimistic B-Free read: the 0.547 was likely driven by SD2.1
being an OLD SD-lineage generator V1's modern features don't catch (+ B-Free specifics),
NOT a fundamental corpus-coupling. The independence gap is **PARTIALLY CLOSED**: aggregate
cross-corpus transfer to frontier generators is decent and compression-robust.

**HONEST CAVEATS (no rounding up):**
1. **CONFLATED / cross-LAB independence still UNPROVEN.** NTIRE fakes are an unlabeled MIX
   that includes some SAME-lab frontier (FLUX.2/SD3.5) alongside cross-lab (Nano
   Banana/Qwen/Z). The 0.74 is "cross-corpus, MIXED-lab" — cross-LAB transfer cannot be
   isolated, and it cannot be ruled out that same-lab frontier fakes carry the signal
   while a pure cross-lab generator (e.g. Nano Banana) is weaker. Per-generator
   decomposition remains unobtainable on free data (NTIRE no labels; DFLIP3K poisoned).
2. **Above-chance != deployment-grade.** 0.74 is meaningfully above chance, not a 0.9
   detector.
3. Mixed-Q closed only relative to a Q75-history substrate.

**Confound checks:** the P3.2 audit cleared the held-out (VQDM content-match
0.799 -> 0.796 within-bin; metadata 0.50 excl file_size; FOV 512 uniform; compression
uniform Q75 4:2:0; substrate byte-matches V1 training). AC-6 surface SPENT (one-shot,
sentinel). Frozen-prediction only.

**Implication for the banked real-side work:** the "real-side NECESSARY" case is WEAKER
than B-Free implied (V1 is NOT near-chance cross-corpus) -> real-side features are now an
OPTIONAL improvement / cross-LAB-proof lever, NOT a rescue. The decomposition limitation
(can't prove per-lab independence) persists and is the remaining open axis. V1 stays
FROZEN.

---

## P3 feature-prototyping (real-side, cross-corpus gate) — DEV gate selects CANDIDATES only; the locked FINAL one-shot is the only number that counts (NTIRE-DEV is soft-overfit)

### P3-cand09 — 2026-06-29 — Fourier phase-spectrum (7 feats) — REJECT

Verified lead (Li CVPR2026; phase JPEG-robust, real-side). Gated vs accepted-22,
frozen-prediction.
- **PRIMARY cross-corpus DEV (fit OpenFake-TRAIN / predict NTIRE-audit, conflated):**
  22 = 0.7300 [.711,.749] (sanity-matches P3.3 0.7385) -> 22+phase7 = 0.7207; phase7-only
  0.6130. **LIFT -0.0093 [-0.013, -0.006], P(>0)=0 — phase DEGRADES transfer, CI excludes
  0 on the wrong side.** Stable across C. No subset rescues.
- **AC-8 no-regression (OpenFake DEV-VAL):** FLUX REGRESSES -0.0097 (0.808 -> 0.798);
  SD3.5 +0.012 -> same-corpus FLUX regression (the suspect flag). FAIL.
- **Independence:** entropy P1 DEAD (saturated ~1.834, standalone ~0.51); circvar
  anti-predictive (0.45); p_phaseonly_ncc near-redundant (rho=0.693 w/ f2_cd_peak_k8) and
  hurts; phasecong weak + overlaps inversion. FAIL.
- **Compression axis:** PASS — phase IS ~JPEG-robust as claimed (Q75-vs-Q90 0.496,
  recompression orthogonal to the fake direction). The theory held; the discriminative
  signal just isn't there on this substrate/data.

**Verdict: REJECT** (needs all 4; only compression-control passed). Honest negative —
phase carries no bankable cross-corpus signal here and degrades both conflated NTIRE-DEV
and same-corpus FLUX. Accepted set stays 22. Implementation:
`pipeline/features/phase_spectrum.py`.

### P3-cand01 — 2026-06-29 — Color-distribution (8 feats) — REJECT (corpus-coupled: same-corpus gain, zero cross-corpus lift)

CoDA-corroborated, EXTRA scrutiny (color = the most confound-prone axis here). Gated vs
accepted-22, frozen-prediction.
- **PRIMARY cross-corpus DEV (fit OpenFake-LAION-TRAIN / predict NTIRE-CC12M-DEV):**
  22 = 0.7300 [.711,.749]; 22+color8 = 0.7299; color8-only = 0.5531 [.531,.575]. **LIFT
  -0.0002 [-0.013, +0.012] — straddles 0.** Stable across C. FAIL.
- **The TELL — per-gen no-regression (OpenFake DEV-VAL):** FLUX 0.808 -> 0.831 (+0.023),
  SD3.5 0.676 -> 0.707 (+0.031) — a REAL same-corpus (LAION) gain that does NOT transfer
  cross-corpus. Same-corpus gain + zero cross-corpus lift = CORPUS COUPLING (exactly what
  an OpenFake-only gate would have falsely passed; the cross-corpus gate caught it).
- **Corpus/content audit:** partially exonerated (direction CONSISTENT across both real
  corpora, not a single-corpus sign-flip; JPEG-robust PASS) BUT within content-matched
  NTIRE bins color8-only is near-chance (0.51-0.62) and adding to 22 goes mixed/negative
  -> content/corpus coupling, the signal doesn't persist within content bins.
- **Independence:** novel (max |rho|=0.51) but moot. The 22 already ABSORB the ~3-pt
  cross-corpus color signal that exists.

**Verdict: REJECT** — the marginal cross-corpus lift is exactly zero; the same-corpus
improvement is corpus-coupled. Accepted set stays 22. (Methodological win: the
cross-corpus gate caught a feature that a same-corpus gate would have accepted —
validates the gate design.) Implementation: `pipeline/features/color_distribution.py`.

### P3-cand3 — 2026-06-29 — White-balance / illuminant-consistency (6 feats) — REJECT (real signal, but already captured by VAE Cb-chroma)

Unresearched real-side axis; per-tile illuminant consistency = the principled bet. Gated
vs accepted-22, frozen-prediction.
- **PRIMARY cross-corpus DEV:** 22 = 0.7300; 22+wb6 = 0.7314. **LIFT +0.0013
  [-0.006, +0.009] — straddles 0.** Consistency-only (std_r/b) lift +0.0032
  [-0.003, +0.010] also straddles 0. C-stable. FAIL.
- **Best candidate yet, but:** wb6-only cross-corpus 0.596 [.573,.617] (a real
  ~10pt-above-chance signal, > color's 0.553), carried by the real-side consistency pair
  wb_tile_illum_std_r/b (+chroma_spread); cast features DEAD (grayworld 0.53, shadesofgray
  0.50).
- **Independence:** the s_noise_cov worry CLEARED (rho ~0.10) — consistency is NOT the
  spatial-noise axis. BUT moderately REDUNDANT with the VAE Cb-chroma features (f1_Cb_box,
  rho up to 0.647) -> that redundancy is exactly why the incremental lift is ~0. **The 22
  already absorb the illuminant-consistency signal** (the f1_Cb chroma features, kept in
  the VAE family, were doing this work).
- **Content:** doesn't vanish within bins (more content-robust than color) but
  marginal-over-22 is content-dependent (negative in low/mid-entropy bins). Compression
  PASS (JPEG-robust). Corpus: cross>same NOT reproduced under frozen prediction (same
  0.629 >= cross 0.596); per-gen FLUX +0.010 / SD3.5 +0.017 same-corpus (corpus-coupling
  tell).

**Verdict: REJECT** — the consistency mechanism is real, content-robust, independent of
s_noise_cov, but already captured by the VAE Cb-chroma features -> zero incremental
cross-corpus lift. Accepted set stays 22. Implementation:
`pipeline/features/white_balance.py`.

### P3-cand04 — 2026-06-29 — Spectral-fractal self-similarity (4 feats, fake-side cross-lab-targeted) — REJECT (hard)

Gated vs accepted-22, frozen-prediction. Zero of four gates pass.
- **PRIMARY cross-corpus DEV:** 22 = 0.7300 -> 22+fr4 = 0.7084. **LIFT -0.0215
  [-0.027, -0.016] — ENTIRELY NEGATIVE.** fractal4-only = 0.512 (~chance): raw class means
  move AI>real on both corpora but the FITTED direction does not transfer (the Phase-2
  proxy failure mode). C-stable.
- **AC-8:** FAIL — FLUX -0.0149 (regression), SD3.5 +0.0206 (opposite-sign even
  same-corpus).
- **Independence:** FAIL — the carriers (sf_selfsim_2x, sf_radial_octave) are
  most-correlated with VAE f4_spectral_slope (rho -0.68/-0.64); the lift over the
  VAE-spectral subset = -0.0298 (absorbed-and-degrades, worse than white-balance).
  Internally collinear (selfsim_2x <-> resid_2x rho=0.70).
- **Content:** FAIL — within-bin ~chance (0.46-0.56), marginal-over-22 negative in EVERY
  bin; sf_radial_octave rises monotonically with complexity = a 1/f/content tracker.
  **JPEG-grid:** partial leak (Q60-vs-Q90 fractal-feat CV 0.65 despite on-axis zeroing).
  **Cross-lab thesis FAILS:** same-corpus 0.598 -> cross-corpus 0.512 (chance).

**Verdict: REJECT.** Accepted set stays 22.

---

## P3 FEATURE WORK — NET CONCLUSION (2026-06-29): all 4 principled candidates REJECTED; V1's 22 are complete on this substrate

Tested the full pre-filtered shortlist through the cross-corpus gate (frozen-prediction
DEV lift + AC-8 + rho-independence + confound audit), one at a time, no batch-adding:

| Candidate | cross-corpus DEV lift over 22 | Why rejected |
|---|---|---|
| 09 phase-spectrum | -0.0093 [-0.013, -0.006] | degrades + FLUX regression; entropy dead (saturated), ncc redundant; (compression-robust yes, but no signal) |
| 01 color-distribution | -0.0002 [-0.013, +0.012] | corpus-coupled: same-corpus +0.02/+0.03 doesn't transfer; fails within-content-bins |
| 03 white-balance | +0.0013 [-0.006, +0.009] | real illuminant-consistency signal but ALREADY ABSORBED by VAE Cb-chroma (rho=0.647) |
| 04 spectral-fractal | -0.0215 [-0.027, -0.016] | degrades + FLUX regression; absorbed-by/worse-than VAE f4 (rho -0.68); content/1f-coupled; cross-lab thesis fails |

**Pattern:** every principled real-side / cross-lab axis is either (a) content/
corpus-coupled (color), (b) already captured by the 22 (white-balance, fractal — both
collide with VAE chroma/spectral features), or (c) substrate-killed + redundant (phase
entropy saturated at Q75). **No candidate adds bankable incremental cross-corpus signal
over V1's 22.** None reached a FINAL one-shot (all failed the DEV gate) -> no locked FINAL
pull spent. **Accepted set FINAL = 22 (V1 unchanged, frozen).**

**Disposition:** the white-box 22-feature V1 is complete on the 512-Q75 substrate;
cross-corpus transfer (0.7385, P3.3) is not improvable by these white-box real-side
features. **The cross-LAB per-generator decomposition remains the open axis** (blocked by
absence of free labeled cross-lab data — NTIRE no labels, DFLIP3K poisoned).
**Recommendation: ship V1 + pivot to the product**, with cross-lab attribution documented
as the standing limitation. (Remaining low-EV options not pursued unless I decide
otherwise: the green/luma-only CFA hatch; reconsidering the Q75-4:2:0 substrate — but 3/4
deaths were redundancy/corpus, not substrate, so the substrate-reconsider lever is weak.)

---

## P3.4 — 2026-06-29 — PER-GENERATOR 2026-frontier transfer MEASUREMENT (the decomposition NTIRE couldn't give): V1 coverage is FLUX-lineage-only cross-pipeline

**STATUS: DEV-side MEASUREMENT on NON-LOCKED surfaces (frozen-V1 predict-only, no refit).
Not a sacred one-shot — no surface spent. The cross-pipeline control surface is what makes
this the honest read.**

**What was measured:** Frozen V1 (22 feats, `frozen_models/v1_inv22/model.joblib`)
per-generator AUC+CI on TWO confound-clean per-generator-labeled 2026-frontier surfaces,
built byte-identically to V1's training substrate (512 center-crop, Q75 4:2:0; video gens
veo-3/wan-2.5/sora-2 EXCLUDED). Step-1 scouting found both free (see
`pre-registrations/phase3_pergen_scout_2026-06-29.md`):
1. **OpenFake-test/OOD, DOCCI reals** (SAME generation/curation pipeline as V1 train;
   DOCCI cross-CORPUS vs LAION). 4000 reals + 14,504 fakes / 17 image gens.
2. **AIGenImages2026** (`pthan12/AIGenImages2026`, INDEPENDENT build: NewsAPI news-photo
   reals != DOCCI != LAION -> cross-PIPELINE + cross-CORPUS + cross-LAB control). 5224
   reals + 5436 fakes / 19 gens (~305 ea).

**CONFOUND AUDIT — both surfaces PASS all 5 axes (`scripts/audit_surface.py`, same gates
as P3.2):** FOV all-512 PASS; compression uniform Q75 4:2:0, luma+chroma quant tables
identical PASS; substrate 0/12 laion-mismatch (byte-matches V1 train) PASS; metadata probe
**0.500 excl file_size** (0 non-constant cols — substrate harmonization leaves zero
provenance signal; file_size, model-excluded, gives only 0.52) PASS; VQDM content-match
holds within entropy/size bins (OpenFake 0.876 -> 0.887/0.895; AIGen 0.635 -> 0.639/0.652
— rises, no collapse) PASS. The audit fresh-CV is a CONFOUND ESTIMATOR, NOT the frozen-V1
number.

**OFFICIAL per-generator frozen-V1 transfer (AUC, bootstrap 95% CI seed 42):**

| | OpenFake-test/DOCCI (same-pipeline) | AIGenImages2026 (cross-pipeline) |
|---|---|---|
| **POOLED** | **0.666** [.657,.677] | **0.560** [.549,.571] |
| FLUX-lineage | flux.2-klein 0.619 (N2000) | flux_dev 0.747 - flux-pro1.1 0.699 - flux-2 0.653 - z-image 0.628 |
| z-image-turbo | **0.784** (N2000) | **0.628** |
| midjourney v7 | **0.653** (N2000) | **0.404** (anti) |
| nano-banana / gemini-2.5-flash | nano-banana-pro 0.653 (N60) | gemini-25-flash 0.429 (anti) |
| GPT-Image (OpenAI) | gpt-image-1.5 **0.535** (N2000) - gpt-image-2 0.536 | gpt-image-1.5 0.465 - gpt-image-1 0.386 (anti) |
| ByteDance Seedream | seedream-v5.0 0.514 (N311) | seedream-v4.5 0.758 (diff VERSION) |
| Imagen-4 | — | imagen4 0.497 (chance) |
| other strong | lumina 0.852 - illustrious(SDXL-ft) 0.748 | hidream 0.654 - ideogram-v3 0.624 - firefly 0.619 |

(OpenFake CIs +/-.01-.06; AIGen all THIN-N ~305 -> +/-.03; nano-banana-pro N60 -> +/-.06.)

**DETERMINATION (flat): frozen V1's apparent 2026 coverage is SUBSTANTIALLY
PIPELINE-COUPLED.** The same-model cross-surface FLIPS are the proof: z-image
0.78 -> 0.63, midjourney-v7 0.65 -> **0.40**, nano-banana family 0.65 -> **0.43**,
gpt-image-1.5 0.54 -> 0.47. On the independent (AIGen) corpus, POOLED drops 0.67 -> 0.56
and the frontier OpenAI/Google/Midjourney gens go **anti-correlated (<0.5)** — V1's tells
point the wrong way (the same failure mode as the original 2022 FLUX 0.14). **What HOLDS
cross-pipeline: FLUX-lineage only** (flux_dev/pro/2 0.65-0.75 — in-distribution
architecture) + a few diffusion gens (hidream, ideogram-v3, z-image ~0.62). **Robust GAP
confirmed on BOTH surfaces: GPT-Image (OpenAI) ~chance/anti (0.54 same-pipe ->
0.39-0.47 cross-pipe).** The cross-pipeline control was DECISIVE — OpenFake-alone would
have overstated coverage (0.65-0.78 on midjourney/z-image/nano-banana that collapse to
0.40-0.63 cross-pipeline).

**Why this is the honest read NTIRE couldn't give:** NTIRE had no per-gen labels
(conflated 0.7385). Now I can name WHICH 2026 gens V1 catches (FLUX-lineage) and which it
misses (OpenAI/Google/Midjourney/ByteDance on independent data). Consistent with the
banked P3 conclusion: V1's white-box features are diffusion/VAE-fingerprint-bound;
cross-LAB frontier coverage is the systemic open weakness, not a per-generator patch.

**Caveats (no rounding):** (1) AIGen per-gen N~305 -> wider CIs (but the cross-pipeline
collapse is broad/consistent, not noise). (2) seedream differs by VERSION across surfaces
(v5.0 vs v4.5) — not a clean same-model flip. (3) AIGen reals are news photos (their own
distribution); the audit clears them as confound-clean, so the collapse is genuine
transfer failure, not a new confound. (4) nano-banana-pro N60 on OpenFake (326/386 were
type=video, correctly excluded). (5) 1 AIGen row dropped for a NaN feature
(s_resid_skew/iv_tv_curv_kurt undefined on a degenerate image) — negligible.

**DISPOSITION — HOLD (measure-before-decide, Step-2 done; do NOT train/ship yet):** the
measured read is that V1 reliably covers FLUX-lineage 2026 gens and **does not** robustly
cover the cross-lab frontier (OpenAI/Google/Midjourney) on independent data. The fork:
(A) SHIP V1 with this now-MEASURED per-generator + cross-lab scope stated ("FLUX-lineage
frontier yes; GPT-Image/Gemini/MJ no") -> product; OR (B) attempt EXTENSION — trainable
only on the well-sampled OpenFake gens (z-image/flux.2/gpt-image-1.5 @2000), but the
cross-pipeline collapse WARNS that training on OpenFake-pipeline data may not transfer
cross-corpus either (would need the AIGen surface as the held-out gate, and per-gen N~305
there is thin for a locked one-shot). Step-3 feature work stays gated: only a measured gap
on a gen with genuinely different artifacts, one candidate at a time through the
cross-pipeline gate — NOT a blind re-run of the 4 rejected features. V1 stays FROZEN.

---

## P3.4-FEATURES — 2026-06-29 — bounded cross-lab-frontier feature attempt (Option B chosen; 4 candidates, cross-pipeline gate)

**Mandate:** crack the cross-LAB frontier V1 fails on the INDEPENDENT AIGen2026 surface
(GPT-Image/Gemini/Midjourney). The ACCEPT signal is cross-pipeline lift on AIGen2026
FAIL-GEN POOLED (gpt-image-1/1.5 + gemini-25-flash/3-pro + midjourneyv7, N=1522,
**base 0.418** anti-correlated), CI excl 0. Same-pipeline (OpenFake) lift is a
no-regression guard ONLY. Gate harness `scripts/xpipe_gate.py` (fit LR(22)/LR(22+cand) on
OF-TRAIN, predict on AIGen2026; the baseline reproduces frozen-V1 per-gen exactly ->
wiring validated). 4 candidates, one at a time, each with its own kill-criterion;
pre-committed STOP -> ship if all 4 fail.

### Candidate 1 — NOISE-PRINT residual (real-side sensor-noise whiteness) — REJECT (kill-test, gate not run)

White-box residual whiteness (acf1/2, spectral flatness, radial slope, flat-region
energy); `pipeline/features/noiseprint.py`. Kill-test:
- (A) signal-exists @ Q75: noise-print-only fresh-CV AUC = **0.575** AIGen fail-gens /
  0.594 OF — weak; the one discriminative feature (np_flat_energy) fires BACKWARDS (fakes
  7.7 > reals 3.9 flat-region residual = smoothness/JPEG-block artifact, NOT absent sensor
  noise).
- (B) Q-shift compression-coupling (DECISIVE): real -> fail-gen fake-prob gap = +0.018,
  but re-encoding reals at Q60 moves them **+0.033 toward fake** — LARGER than the whole
  gap. The feature reads COMPRESSION, not sensor noise -> kill-criterion triggered. The
  Q75-4:2:0 substrate destroys the physical noise the mechanism needs (same substrate-kill
  as CFA). **REJECT** before the gate.

### Candidate 2 — GENERALIZED UPSAMPLING-GRID PERIODICITY — REJECT (probe + JPEG confound, gate not run)

Grid-peak SNR scanned at NON-8-aligned periods {5,6,7,9,10,11,12,13,14} (JPEG's 8-grid
cannot peak there); `pipeline/features/grid_period.py`. Signal-exists: non-JPEG grid-only
fresh-CV AUC = **0.532** AIGen fail-gens / 0.511 OF — chance. JPEG confound check
(std-normalized real-fake separation): the LARGEST separator is **_gp_snr_p8 = 0.106 and
_gp_snr_p16 = 0.059** (the JPEG block periods), every non-JPEG decoder period <= 0.084 ->
what little periodicity exists is COMPRESSION, not generation. Q75-4:2:0 destroys faint
decoder grids AND imposes its own 8-grid; V1's 8/16 are the only recoverable grid signal.
**REJECT.**

### Candidate 3 — LOCAL PATCH SELF-CONSISTENCY — ACCEPT (first cross-pipeline-gate pass)

Spatial dispersion of per-tile spectral statistics (slope/HF-fraction/noise-floor across
8x8 tiles + adjacent-tile jumps); content-robust by construction.
`pipeline/features/self_consistency.py`.
- Probe signal-exists: self-consistency-only fresh-CV AUC **0.673** AIGen fail-gens /
  0.639 OF; CONTENT control PASS (within entropy-bin weighted 0.663 vs 0.673 — no
  collapse).
- **PRIMARY cross-pipeline gate (frozen-prediction, fit OF-TRAIN / predict AIGen FAIL-GEN
  POOLED N=1522):** base 0.418 -> **+SC6 0.445 (+0.028 [+0.022, +0.034])** -> **+SC5 0.452
  (+0.034)** after dropping the one rho=0.71 feature (sc_hffrac_adjjump; dropping it
  IMPROVES the lift). CI excl 0, P>0 = 1.00. EVERY fail-gen lifts CI-excl-0 (gpt-image-1
  .386 -> .424, gpt-image-1.5 .465 -> .487, gemini-2.5 .429 -> .445, gemini-3-pro
  .403 -> .430, midjourney .404 -> .440).
- No-regression (OFTEST, criterion 2): PASS — lifts broadly +0.03..+0.11 (miss-pooled
  .533 -> .585; seedream-v5 .514 -> .628; nano-banana-pro .653 -> .718). No same-pipeline
  regression.
- rho-independence: SC5 all |rho| <= 0.36 vs the 22 (key carrier sc_noisefloor_logdisp).
  Confound axes: content PASS (within-bin); compression PASS (real -> failgen gap
  +0.100 >> Q60 re-encode shift +0.026).
- **VERDICT: ACCEPT. Parsimonious set = 22 + SC5 = 27.** HONEST SCOPE: the lift is
  REAL/cross-pipeline/content-controlled but MODEST — it REDUCES the anti-correlation on
  the hardest cross-lab gens (gpt-image/gemini/midjourney) yet leaves them STILL <0.5
  (narrows the gap, does not achieve detection); helps more mid-tier (imagen4 .50 -> .52,
  flux-2-pro .53 -> .59).

### Candidate 4 — PHYSICS-GROUNDED CHANNEL CORRELATION — REJECT (tight-leash corpus-coupling kill)

G-guided cross-channel HF predictability (demosaicing physics, luma-preserved
escape-hatch); `pipeline/features/physics_channel.py`. PRIMARY kill = corpus-coupling
(real-vs-real across corpora): **news-vs-docci 0.745, laion-vs-docci 0.741**,
news-vs-laion 0.617 — the feature separates GENUINE photos from different corpora -> it
reads corpus chroma-history, not camera CFA -> would flag unseen-corpus reals as fake. The
mechanism is also inverted (fakes MORE G-predictable than reals). Same corpus-coupling
that sank color/lc_xchcorr, third confirmation; Q75-4:2:0 chroma subsampling makes any
cross-channel feature a corpus detector. **REJECT** (kill-criterion triggered before the
gate).

### P3.4-FEATURES NET CONCLUSION (2026-06-29): 4 tested, 1 ACCEPT (modest), cross-LAB white-box ceiling CONFIRMED

| Candidate | gate outcome | why |
|---|---|---|
| 1 noise-print residual | REJECT (kill-test) | substrate-killed (0.575 @Q75) + compression-dominated (Q60 shift +0.033 > gap +0.018) |
| 2 grid periodicity | REJECT (probe) | non-JPEG periods at chance (0.532); what exists is JPEG-8, not decoder |
| 3 self-consistency | **ACCEPT** | cross-pipeline lift +0.034 AIGen fail-gens (CI excl 0), content+compression-controlled, rho-indep |
| 4 physics channel | REJECT (kill-test) | corpus-coupling (real-vs-real cross-corpus 0.74) — tight-leash kill |

**Outcome:** exactly ONE pre-registered candidate (self-consistency) beat the
cross-pipeline gate. It adds a REAL, broad, content+compression-controlled,
mostly-independent lift (+0.03..+0.11 on BOTH surfaces, CI excl 0, no same-pipeline
regression) -> candidate **V2 = V1-22 + SC5 = 27 feats**. BUT it does NOT crack the
cross-LAB frontier: gpt-image/gemini/midjourney remain anti-correlated (<0.5) after the
lift — narrowed, not detected. The other 3 mechanisms died on the Q75-4:2:0 substrate
(noise/grid) or on corpus-coupling (channel) — **the white-box cross-LAB ceiling on this
substrate is now empirically confirmed by 4 mechanistic attempts, not assumed.**

**DISCIPLINE NOTE (banking V2):** the C3 lift is on the AIGen2026 DEV gate (used for
selection -> now soft-overfit). Per the standing rule, BANKING V2 as
verified-better-than-V1 cross-pipeline requires a LOCKED one-shot on a cross-pipeline
surface NOT used for selection (none currently reserved — AIGen spent for selection).
Until then C3 is an accepted CANDIDATE, not a locked FINAL.

**DISPOSITION — HOLD:** (A) bank V2 = 22+SC5 (modest validated improvement, broad lift,
no regression) and either lock a fresh cross-pipeline held-out to confirm or ship V2 with
the honest DEV-gate caveat; OR (B) ship V1 as-is (the SC lift is too modest to change the
headline — the hardest cross-lab gens stay <0.5). Either way the MEASURED scope stands:
covers FLUX-lineage + SD-lineage frontier incl FLUX.2; does NOT detect the cross-lab
frontier (GPT-Image/Gemini/Midjourney) on independent data — and that ceiling is now
mechanistically confirmed, not just observed. STOP feature work (pre-committed; 4 done).

---

## P3.4-LEGACY — 2026-06-30 — earlier prototype-era feature families through the cross-pipeline gate

**Mandate:** test the feature families from my earlier, confounded prototype pipeline
against the cross-LAB frontier gap (GPT-Image/Gemini/Midjourney — the AIGen2026 gens
V1/V2 fail on), ORDERED by what is genuinely untested on THIS clean pipeline. Iron rule:
prototype-era "it worked" means nothing — each is a fresh hypothesis at the gate.
Baseline = **V2 = 22 + SC5 = 27** (gate `scripts/xpipe_gate_v2.py`; V2 baseline FAIL-GEN
POOLED **0.452**, matches the SC5 number above -> wiring validated). Gate = cross-pipeline
lift on AIGen2026 FAIL-GEN POOLED (N=1522), frozen-prediction, CI excl 0; + no
same-pipeline regression (OFTEST); + rho-independence vs the 27; + content-matched-bin
confound audit; + own kill-criterion. One at a time, no batch-adding.

**SKIP (already cleanly rejected on THIS pipeline — re-running reproduces the kill;
stated, not re-run):** Benford-DCT (U3.2 Fam3, one-gen + redundant); CFA/channel-corr
(killed 3x, corpus-coupling on Q75-4:2:0, last P3.4 cand4); shot-noise/Poisson/
noise-signal (= noise-print, P3.4 cand1, Q75 destroys sensor noise -> reads compression);
phase-congruency (= phase-spectrum P3-cand09, no substrate signal); anisotropy/
radial-spectrum (covered by spectral-fractal reject P3-cand04); 45 polynomial terms
(explainability cap + poly rejected U3.2 Fam4).

**TEST (genuinely untested on the clean pipeline), in order:**
1. **Dark-channel prior** (f_dc_mean/var/p0) — real-side, physics-grounded; confound axis
   = content.
2. **Residual-domain GLCM** (f_rglcm_*) — texture on the noise residual; confound axis =
   content + compression.
3. **JPEG block-boundary** (f_block_*) — UP-FRONT KILL: prove the real-vs-AI block signal
   survives Q75 via real-vs-real Q-shift, else reject like noise-print.

### Candidate 1 — Dark-channel prior (f_dc_mean/var/p0) — REJECT (confound-shift: GPT-Image-only, AC-8 + no-regression fail; mechanism inverted)

**Cross-pipeline PRIMARY (AIGen FAIL-GEN POOLED N=1522):** base 0.452 -> cand 0.462,
**lift +0.010 [+0.004, +0.016] P>0=1.00** — the CI technically excludes 0, BUT the pooled
number is a confound RESHUFFLE, not a uniform lift:
- **AC-8 FAIL within the 5 fail-gens:** gpt-image-1 **+0.082** (.431 -> .512) &
  gpt-image-1.5 +0.027 LIFT, but **gemini-25-flash -0.046** (.454 -> .408) &
  **midjourney-v7 -0.018** REGRESS (CI excl 0 negative); gemini-3-pro null. The +0.010 is
  GPT-Image gain MINUS gemini/MJ loss -> confound shift, not detection.
- **No-regression guard FAIL (OFTEST):** MISS-pooled +0.046 is again GPT-Image-driven
  (gpt-image-1.5 +0.055, gpt-image-2 +0.047, ernie +0.04) while FLUX-lineage/diffusion
  REGRESS CI-excl-0: flux.2-klein -0.013, ideogram-2.0 -0.017, recraft-v3 -0.013,
  seedream-v5 -0.013. Broad same-pipeline regression.
- **AIGEN collateral:** hurts the gens V2 already covers — fast-sdxl -0.078, flux-2-max
  -0.055, seedream -0.038, flux-2-pro -0.033, z-image -0.032, ideogram -0.026.
- **Mechanism INVERTED (direction check):** the prior predicts AI -> higher dark-channel
  mean / lower near-zero fraction. DATA: fakes have LOWER f_dc_mean (0.177 vs 0.220) and
  HIGHER f_dc_p0 (0.347 vs 0.298) — AI fail-gens are DARKER than the AIGen news-photo
  reals. The feature is a brightness/corpus proxy, NOT the dark-channel-prior physics.
  (Content-bin control: lift trivial & flat +0.007/+0.011/+0.000 — not the point; the
  per-gen reshuffle + inversion are decisive.)
- rho-independence OK (max |rho| = 0.41 vs s_resid_skew/f2_cd_peak_k16) — but moot.

**Verdict: REJECT.** A GPT-Image-specific brightness tell that shifts the confound (lifts
OpenAI, regresses Gemini/MJ + FLUX-lineage), with the physics mechanism firing backwards.
Fails AC-8 and the no-regression guard. Accepted set stays V2=27. Implementation:
`pipeline/features/dark_channel.py`.

### Candidate 2 — Residual-domain GLCM (f_rglcm_contrast/homogeneity/energy/entropy/correlation) — REJECT (redundant + catastrophic no-regression)

**Cross-pipeline PRIMARY (AIGen FAIL-GEN POOLED N=1522):** base 0.452 -> cand 0.469,
**lift +0.017 [+0.011, +0.024] P>0=1.00** — CI excl 0, but dies on TWO independent gates
before content/compression matter:
- **rho-independence (gate 3) FAIL:** 3 of 5 features rho=0.83-0.85 with
  **iv_tv_curv_energy** (homogeneity 0.841, energy 0.849, entropy 0.830) + contrast
  rho=0.715 with f4_spectral_slope. Residual-GLCM homogeneity/energy/entropy = residual
  SMOOTHNESS, already captured by the TV-curvature-energy feature. Not a new axis — a
  noisier re-encoding of an existing one.
- **No-regression guard (gate 2) FAIL — catastrophic:** OFTEST MISS-pooled **-0.044**;
  nearly every same-pipeline gen collapses: recraft-v2 -0.187, seedream-v5 -0.167,
  recraft-v3 -0.140, lumina -0.116, ernie-turbo -0.101, illustrious -0.088, z-image
  -0.066. AIGEN collateral: fast-sdxl -0.178, flux-pro -0.068, reve -0.052. The 5
  redundant/noisy features destabilize the LR and WRECK V2 coverage.
- **AC-8 mixed within fail-gens:** midjourney +0.043 / gemini-3-pro +0.033 /
  gpt-image-1.5 +0.033 lift, but gpt-image-1 -0.014 regress, gemini-flash null ->
  reshuffle, same shape as cand-1.

**Verdict: REJECT.** The marginal +0.017 fail-gen lift is not independent (rho ~0.85 w/
iv_tv_curv_energy) and is bought by -0.04..-0.19 same-pipeline destruction. Fails
rho-independence + no-regression decisively (content/compression audit moot). Accepted
set stays V2=27. Implementation: `pipeline/features/residual_glcm.py`.

### Candidate 3 — JPEG block-boundary (f_block_h_excess/v_excess/ratio/p8_snr) — REJECT (kill-test; gate not run; substrate-killed)

**UP-FRONT KILL-TEST (noise-print pattern, before any full extraction;
`scripts/killtest_jpegblock.py`, 400 reals + 400 fail-gen @ Q75):**
- (A) signal-exists @Q75: block-only fresh-CV AUC = **0.605** — weak.
- (B) compression-coupling (DECISIVE): real -> fail-gen fake-prob gap = **+0.049**, but
  re-encoding the SAME reals at Q60 moves them **+0.045 toward fake** — 92% of the entire
  gap, from a real-vs-real Q-shift (Q90 d ~0.000). The feature reads the LAST compression
  (the Q75 substrate), NOT the reals' first-compression blocking history.
  **Kill-criterion triggered -> REJECT before the gate.**
- Direction: f_block_p8_snr is higher on reals (6.96 vs 6.08) — reals DO carry slightly
  stronger blocking — but it's dominated by the substrate's own uniform Q75 8-grid, so it
  is not recoverable.

**Verdict: REJECT.** Same substrate-kill as noise-print/CFA/shot-noise: the uniform
Q75-4:2:0 re-encode overwrites the first-compression trace the mechanism needs. Accepted
set stays V2=27. Implementation: `pipeline/features/jpeg_block.py`.

### P3.4-LEGACY NET CONCLUSION (2026-06-30): 3 tested, 0 accepted — cross-lab ceiling further confirmed

| Candidate | gate outcome | why |
|---|---|---|
| 1 dark-channel prior | REJECT | confound-SHIFT: GPT-Image-only (+0.082), regresses gemini/MJ (AC-8 fail) + FLUX-lineage same-pipe (no-regression fail); mechanism INVERTED (fakes darker -> brightness/corpus proxy) |
| 2 residual-GLCM | REJECT | rho=0.83-0.85 redundant w/ iv_tv_curv_energy (gate-3 fail) + catastrophic no-regression (OFTEST -0.044, gens -0.10..-0.19) |
| 3 JPEG block-boundary | REJECT (kill-test) | substrate-killed: Q60 real-vs-real shift +0.045 ~ gap +0.049 -> reads compression not first-compression history |

**Pattern:** none cleared the cross-pipeline gate. Cand-1/2 produced a positive POOLED
fail-gen lift (+0.010/+0.017, CI excl 0) that on inspection was a confound reshuffle
(GPT-Image-only) or a redundant re-encoding, both regressing the broader generator set —
the AC-8 + no-regression + rho-independence gates caught what the pooled headline hid.
Cand-3 died on the Q75-4:2:0 substrate, the 4th sensor/compression mechanism to do so
(noise-print, CFA, shot-noise, now block-boundary) -> **the Q75 substrate is now the
recurring wall for compression/sensor-history features.** Accepted set FINAL on this
substrate = **V2=27** (unchanged). Together with P3.4-FEATURES (4 tested, 1 modest accept
= SC5), the white-box cross-LAB frontier ceiling is confirmed across 7 mechanistic
attempts on this substrate.

**DISPOSITION — HOLD for review.** Per the standing plan, the BANKED NEXT TIER (5
structural/geometric candidates) is NOT actioned until I have reviewed these three and
confirmed a cross-lab gap still warrants reaching for them; otherwise accept the ceiling
and ship V2 with measured scope. The remaining non-feature lever is the SUBSTRATE
experiment (lossless / higher-Q to revive the 4 compression-killed mechanisms), NOT more
features on Q75. V1/V2 stay frozen.

---

## STRUCTURAL TIER (ACTIONED 2026-06-30) — structural/geometric real-side candidates

**STATUS at the time: IN PROGRESS.** I reviewed the three P3.4-LEGACY rejections,
confirmed a cross-lab gap remains, and actioned this tier. Baseline = V2=27
(`scripts/xpipe_gate_v2.py`). Same gate as everything: cross-pipeline lift on AIGen2026
FAIL-GEN POOLED (CI excl 0) + no same-pipeline regression + rho-independence vs the
then-current set + content-matched-bin audit + own kill-criterion. One at a time, no
batch-adding. Order below (1 defocus -> 5 chromatic-aberration). Results appended per
candidate as each landed.

### Structural Cand 1 — Defocus / depth-of-field consistency (f_dof_sharp_cv/autocorr/jump/compact) — REJECT (parsimony reveals gpt-image-only; uniformity propped by a redundant feature)

**Cross-pipeline PRIMARY (AIGen FAIL-GEN POOLED N=1522):** 4-feat **+0.016
[+0.013, +0.020]**, and superficially ALL 5 fail-gens lift (gpt-image-1 +0.030,
gpt-image-1.5 +0.022, gemini-3-pro +0.011, midjourney +0.011, gemini-flash +0.007) —
looked like the first uniform AC-8 pass. **BUT the parsimony test breaks it:**
- **rho-independence FAIL on the strongest feature:** `f_dof_sharp_cv` rho=**0.908** with
  s_noise_cov (a V2 feature) — it IS s_noise_cov in disguise (direction-correct: reals
  1.603 > fakes 1.447 sharpness CV). The other 3 (autocorr 0.259, jump 0.294, compact
  0.149) ARE independent.
- **Drop the redundant feature -> AC-8 collapses to gpt-image-only.** DOF3 =
  {autocorr, jump, compact}: pooled +0.012 [+0.008, +0.015] but per-target: gpt-image-1
  +0.029 / gpt-image-1.5 +0.017 (CI excl 0), **gemini-3-pro +0.006 (CI touches 0),
  midjourney +0.005 (null), gemini-flash +0.002 (null).** The "all-5-lift" appearance
  REQUIRED the s_noise_cov-clone; the genuinely-structural focus-map features carry only a
  gpt-image-specific signal.
- **Content audit CLEAN (its designated axis):** the lift holds across all 3
  f4_spectral_slope terciles (+0.023/+0.013/+0.016) — NOT content-coupled. Direction of
  the structural triple is tiny (autocorr -0.003, jump +0.010, compact +0.005) — weak
  mechanism. No catastrophic regression (DOF3 OFTEST +0.011).

**Verdict: REJECT into V2.** Fails the gate both ways: keep-4 -> rho-independence
(sharp_cv rho=0.908); drop-to-3 -> AC-8 (gpt-image-only on the fail targets, gemini/MJ
null). Cleaner than dark-channel (no regression, content-robust) but the independent
structural signal is NOT architecture-agnostic — it's a weak gpt-image-leaning tell that
leaves gpt-image still <0.5 and does not touch gemini/MJ. **Optional parked carve-out:**
the structural triple {f_dof_autocorr, f_dof_jump, f_dof_compact} as a weak,
content-robust, gpt-image-TARGETED lift (+0.029 gpt-image-1) — NOT banked (doesn't meet
SC5's uniform-lift bar). Accepted set stays V2=27. Implementation:
`pipeline/features/defocus_dof.py`.

### Structural Cand 2 — Scene-level natural-image statistics (MSCN/BRISQUE: nss_mscn_alpha/sigma/kurt, pair_corr_h/v) — REJECT (redundant + gemini-only + corpus-inverted mechanism)

**Cross-pipeline PRIMARY (AIGen FAIL-GEN POOLED N=1522):** +0.019 [+0.014, +0.025] CI
excl 0 — but fails on every other axis, and is the MIRROR IMAGE of defocus (helps a
different lab):
- **rho-independence FAIL:** 3 of 5 redundant with V2 inversion-TV — nss_mscn_sigma
  rho=0.891 (iv_tv_curv_energy), nss_mscn_kurt rho=0.801, nss_mscn_alpha rho=0.720
  (iv_tv_curv_kurt). MSCN-shape ~ curvature distribution, already in V2. Only
  pair_corr_h/v (0.39/0.47) independent.
- **AC-8 FAIL (mirror of defocus):** gpt-image-1 **-0.020 REGRESSES**, midjourney null;
  gemini-flash +0.022, gemini-3-pro +0.043, gpt-image-1.5 +0.046 lift -> GEMINI-targeted
  (defocus was gpt-image-targeted). Neither architecture-agnostic; each helps one lab.
- **No-regression FAIL:** OFTEST pooled ~0.000 hiding a violent reshuffle — seedream-v5
  **-0.145**, recraft-v2 -0.106, illustrious -0.068, lumina -0.069 collapse; flux.2-klein
  +0.054 lifts. AIGEN collateral: fast-sdxl -0.086, seedream -0.072, flux-pro -0.049.
- **Mechanism INVERTED by corpus (flagged risk realized):** the NSS thesis = real photos
  near-Gaussian MSCN (low kurtosis). DATA backwards — AIGen news-photo REALS are more
  kurtotic (0.632) than fakes (0.335); fakes look MORE textbook-Gaussian (alpha 1.89 vs
  1.67). The universal-NSS mechanism reads the news-photo corpus's own processing ->
  corpus-coupled. Content audit: lift concentrated in one tercile
  (+0.040/+0.004/+0.015).

**Verdict: REJECT.** Redundant with iv_tv, gemini-only (AC-8 fail), destabilizes OFTEST,
and the model-the-real mechanism fires backwards on news-photo reals (the corpus
confound). Accepted set stays V2=27. Implementation: `pipeline/features/scene_nss.py`.
**Note (meta):** defocus -> gpt-image-only; NSS -> gemini-only. Complementary single-lab
tells, neither architecture-agnostic — reinforces the cross-LAB ceiling.

### REAL-SIDE TEST (separate, decisive) — added 2026-06-30 after reframing the tier's purpose

**Purpose of this tier, reframed:** these are REAL-side features — their job is to
RECOGNIZE REAL IMAGES better (lower false-positive rate on reals) -> higher overall
accuracy. So a SEPARATE real-side test is now decisive for each candidate
(`scripts/realside_audit.py`): (1) real-vs-real cross-corpus separability across
LAION/NEWS/DOCCI must stay ~0.50 (>0.65 = corpus-coupled KILL — would flag unseen real
photos as AI); (2) adding the feature must LOWER reals' fake-prob / RAISE specificity
(TNR) consistently across corpora; (3) without dropping fake detection (TPR). On APPROVAL
-> production stress test (few hundred random real+fake, count correct). AUC lift is
necessary but NOT sufficient — it can rise by pushing fakes up while reals get WORSE at
threshold.

Retro real-side results on the two rejects (confirm the gate verdicts and validate the
lens):
- **defocus {autocorr, jump, compact}:** real-vs-real 0.59/0.64/0.62 (borderline, partial
  corpus signal); REAL-CATCHING FAILS — reals' fake-prob RISES (NEWS 0.410 -> 0.414, DOCCI
  0.306 -> 0.316), specificity DROPS (NEWS TNR 0.681 -> 0.654, DOCCI 0.838 -> 0.825). The
  +0.012 AUC was ranking-only (fakes pushed up more); bal-acc +0.007. **Catches reals
  WORSE -> reject confirmed on the real-side lens.**
- **scene-NSS (5):** real-vs-real LAION-vs-DOCCI **0.836**, NEWS-vs-DOCCI 0.775 ->
  **CORPUS-COUPLED KILL**; real-catching also negative (TNR drops on both corpora).
  Decisive real-side kill, matches the gate.

### Structural Cand 3 — Lighting / shadow coherence (f_light_R, tile_cstd, bimod) — REJECT (cleanest candidate, but NO discriminative signal)

**The inverse failure mode — passes every cleanliness test, carries no signal:**
- **rho-independence: BEST so far** — all 3 features rho <= 0.215 vs the 27
  (shading-coherence is a genuinely new axis).
- **Real-side cross-corpus: BEST so far** — real-vs-real 0.556/0.553/0.531
  (LAION/NEWS/DOCCI). Genuinely corpus-INVARIANT -> measures realness, not corpus.
  Exactly what a true real-side feature should look like.
- **BUT cross-pipeline FAIL-GEN POOLED = +0.001 [-0.000, +0.003] (NULL).** Per-fail-gen
  all +/-0.004 (gpt-image +0.004/+0.004, gemini -0.002/-0.003, midjourney +0.003). No
  separation of modern fakes from reals.
- **Real-catching: FLAT.** NEWS fake-prob 0.410 -> 0.409 (TNR 0.681 -> 0.679); DOCCI
  0.306 -> 0.302 (TNR 0.838 -> **0.843**, marginal+). Fake-integrity flat (TPR -0.005).
  Bal-acc 0.468 -> 0.464 (-0.004).

**Verdict: REJECT — clean mechanism, no signal.** Unlike the others (confound-shift/
redundant/corpus-coupled), lighting-coherence is corpus-invariant, independent, and
confound-free — it simply doesn't discriminate: modern generators get coarse
shading-coherence close enough to real that the white-box proxy can't separate them. (A
full 3D light-source-consistency estimator might; out of clean-white-box reach on a
single 512px image — the implementation-effort caveat realized.) Not approved -> no
stress test. Accepted set stays V2=27. Implementation:
`pipeline/features/lighting_coherence.py`.

### Structural Cand 4 — Perspective / line-geometry coherence (f_persp_orient_conc/hv_frac/tile_conc) — REJECT into V2 (strongest discriminator, but corpus-INCONSISTENT real-catching)

**Cross-pipeline PRIMARY: +0.030 [+0.021, +0.038] — BIGGEST lift of the tier,** driven by
midjourney **+0.100** (.436 -> .536, crosses 0.5), gpt-image-1 +0.042, gemini-3-pro
+0.024 (gemini-flash null, gpt-image-1.5 -0.011 -> AC-8 partial 3/5). rho-indep: 1 clean
(0.25), 2 moderate (hv_frac 0.62, tile_conc 0.51).
**Real-side test (DECISIVE, splits by corpus):** real-vs-real 0.526/**0.647**/0.636
(DOCCI partially separates). REAL-CATCHING INCONSISTENT: helps NEWS reals (fakeprob
0.410 -> 0.403, TNR 0.681 -> 0.684) but HURTS DOCCI reals (fakeprob 0.306 -> 0.319, TNR
0.838 -> **0.804**, -0.034). Bal-acc on NEWS-vs-fakes +0.012 (best of tier) — but measured
on the corpus it helps. **Verdict: REJECT into V2** — the strongest discriminator found,
but it shifts WHICH corpus's reals get caught (scene-type/corpus-coupled at 0.647),
failing the "catch reals consistently better" bar; partial redundancy compounds it.
Flagged as the standout near-miss. Implementation:
`pipeline/features/perspective_geom.py`.

### Structural Cand 5 — Lateral chromatic aberration — NOT RUN (the phase pivot to P3.5 superseded the Q75 tier)

### STRUCTURAL TIER — NET (2026-06-30): 4 tested on Q75, 0 banked; meta-pattern = single-lab tells, none architecture-agnostic

defocus -> gpt-image-only (hurts reals); NSS -> gemini-only + corpus-coupled (real-vs-real
0.84); lighting -> cleanest/most corpus-invariant but NO signal; perspective -> strongest
lift but corpus-inconsistent real-catching. **The real-side test (the reframing) proved
its worth:** AUC lift repeatedly turned out to be ranking-only (reals pushed toward fake
at threshold) or corpus-coupled. Accepted set stays **V2=27**. The Q75 substrate +
single-512 white-box ceiling for cross-LAB real-side features is confirmed across this
tier. -> motivated the P3.5 lossless two-stage pivot below.

---

## P3.5 — TWO-STAGE REAL-FIRST DETECTOR — P3.5.1 dataset + compression-history/production-validity audit (HOLD)

**Phase mandate:** a Stage-1 real-check on a LOSSLESS substrate gates into the frozen-V2
Stage-2 (Q75), one-bit boundary. Bounded phase, pre-committed stop. P3.5.1 = build+audit
the Stage-1 lossless dataset, run the compression-history (primary) +
production-validity audits FIRST, escalate BEFORE feature discovery, no training until
approved.

**FINDING 1 — DATA GAP (recon):** the lossless data the plan assumes largely does NOT
exist locally:
- Lossless reals: only **~1,000 RAISE PNGs** on disk (B-Free native-RAISE, 500 test + 500
  audit, already 1024 center-crops), NOT the assumed ~7,100 native ~4288 TIFFs (would need
  download; RAISE source reachability unverified — Zenodo was unreachable from my network
  earlier, HTTP 403).
- Lossless fakes: the ONLY native-PNG fakes on disk are B-Free **FLUX + SD3.5** — the
  lineage V2 ALREADY catches. **No native-lossless GPT-Image/Gemini/Midjourney** (the
  cross-lab frontier P3.5 targets) — those exist only as Q75 JPEG in AIGen2026. The
  cross-pipeline gate's independent surface is Q75 JPEG.
- All existing surface raw_paths (aigen2026, of_test_docci, of512_*) are 512 Q75 4:2:0
  JPEG (the substrate), not pre-compression originals.

**FINDING 2 — COMPRESSION-HISTORY AUDIT (data-in-hand,
`scripts/p351_comphist_audit.py`, 600 RAISE reals, lossless L vs same-image Q75-JPEG J,
paired; per-family real-vs-real L-vs-J AUC = compression-confound size):**

| family | L-vs-J AUC | read |
|---|---|---|
| dark_channel | 0.501 | compression-ROBUST |
| lighting | 0.500 | compression-ROBUST |
| perspective | 0.552 | compression-ROBUST |
| defocus | 0.602 | borderline |
| noiseprint | 0.677 | compression-CONFOUNDED |
| scene_nss | 0.804 | compression-CONFOUNDED |

**THE CATCH-22 (decisive):** the compression-ROBUST features (lighting/perspective/
dark-channel, L-vs-J 0.50-0.55) are de-confounded + production-valid on JPEG reals — but
gain NOTHING from lossless (compression-invariant => lossless ~ Q75) and were ALREADY
tested weak/corpus-coupled on Q75 in the structural tier. The compression-SENSITIVE
features that MOTIVATE the lossless substrate (noiseprint 0.68, scene-NSS 0.80; and by
construction shot-noise/CFA/CA) READ COMPRESSION not realness => they fail the
compression-history de-confound AND fail production-validity (JPEG production reals look
'not real'; Stage-1's confident-real gate would rarely fire on normal JPEG photos, making
it inert in production). The features lossless helps are exactly the ones it can't
de-confound; the clean ones don't need it.

**DISPOSITION — HOLD (no feature discovery, no training, per the gate).** On the audit
alone P3.5 hits its pre-committed stop: the compression-history confound can only be
'controlled' by excluding the very sensor/physics features that justify the lossless
substrate, AND the cross-lab lossless fake data does not exist locally. Options weighed:
(A) ship V2 + product (the pre-committed stop honored); (B) narrow P3.5 to ONLY
compression-robust structural features — but those showed no usable cross-lab signal on
Q75, so low EV; (C) invest in data acquisition (download ~7,100 RAISE TIFFs + re-pull
native-PNG cross-lab frontier fakes) to test the sensor/physics features on a
matched-lossless both-sides set — highest cost, still bounded by the production-validity
problem (JPEG reals). V2 stays frozen.

### P3.5 TWO-PHASE Q75 DEMO (directed follow-up) — real-confirmer -> V2 cascade — NO GAIN + leakage risk

**Architecture:** Phase-1 real-confirmer (13 real-modeling feats:
dark-channel + defocus + lighting + perspective, LR, P(real) >= tau -> STOP=REAL) -> else
route to frozen V2 (27). One-bit boundary. Train Phase-1 on OpenFake, evaluate the SYSTEM
on AIGen2026 (cross-pipeline, balanced 200 real + 200 fake incl cross-lab). tau=0.799
calibrated for <= 2% train-fake leakage (5-fold CV, no in-sample tau).
`scripts/demo_twophase.py`.

| | acc | real-recall (TNR) | fake-recall (TPR) |
|---|---|---|---|
| V2-alone @0.5 | 0.568 | 0.680 | 0.455 |
| Two-phase | 0.570 | 0.685 | 0.455 |
| **Delta** | **+0.002** | +0.005 | +0.000 |

**Phase-1 confirmed REAL: 10/400 — 5 correct reals, 5 FAKES LEAKED (3 cross-lab) -> 2.5%
unrecoverable fake-leakage.** Phase-1 real-model **AUC on the independent test = 0.518
(~chance).**
**Determination:** the cascade adds NOTHING (+0.002 acc) while introducing a 2.5%
unrecoverable false-negative (fakes confidently mis-confirmed real never reach V2). Root
cause: the real-modeling features have NO cross-pipeline signal as a standalone
real-confirmer (AUC 0.518) — so no tau catches reals without equally catching fakes
(5-correct/5-leaked at tau=0.799 is exactly the coin-flip). The two-phase ARCHITECTURE is
sound; what is missing is high-precision real-confirmer features that transfer
cross-pipeline. Same white-box ceiling, now shown END-TO-END at the system level.

**SMOKE TEST (follow-up: integrate the Q75-killed features on lossless data and see;
`scripts/p351_smoketest.py`):** 500 RAISE lossless reals vs 500 B-Free FLUX/SD3.5
lossless fakes (content-paired, FOV-matched 512 crop), each family on the SAME crop
LOSSLESS vs Q75 (fresh-CV real-vs-fake AUC; diagnostic, not a gate):

| family | LOSSLESS | Q75 | gain(L-Q75) |
|---|---|---|---|
| sensor_absence | 0.832 | 0.842 | -0.010 |
| noiseprint | 0.768 | **0.900** | **-0.132** |
| physics_channel | 0.793 | 0.820 | -0.027 |
| color_imperf | 0.875 | 0.857 | +0.018 |

**PREMISE FALSIFIED.** High in-set AUC (0.77-0.88) but **Q75 ties/BEATS lossless** on
every camera-physics family (noiseprint +0.132 BETTER on Q75 — a noise feature improving
after JPEG reads compression-interaction, not native sensor noise). Genuine
camera-imperfection signal CANNOT survive/improve under Q75, so the high AUC is NOT
sensor physics — it is a RAISE-source vs BFree-generation CONFOUND (the same
corpus/source-coupling that failed these exact features on the AIGen2026 cross-lab gate
in P3.4-FEATURES/U3.2). Lossless gives NO benefit. The fakes are SD/FLUX-lineage (V2
already catches) — not the cross-lab frontier. **The lossless Stage-1 core premise does
not hold empirically; this strengthens the recommendation to ship V2 (option A).**

### (plan as banked — kept for reference)

**STATUS at banking: DOCUMENTED PLAN ONLY** (subsequently actioned, above). Recorded
before actioning.

**Selection principle:** STRUCTURAL/GEOMETRIC real-side features chosen to SURVIVE the
Q75-4:2:0 substrate that killed the sensor-physics features (shot-noise, CFA,
noise-print). Fine-grained noise/chroma dies on this substrate; COARSE structural/optical
properties of real scenes survive it. Each keys on physics no generator simulates
correctly -> architecture-agnostic by mechanism. Risk shifts from compression-coupling to
**CONTENT-coupling** -> every one needs the content-matched-bin audit on the
cross-pipeline surface (the gate that caught color): a structural feature can secretly
measure scene-type rather than real-vs-AI.

Test order if reached (ranked by mechanism strength + confidence):
1. **DEFOCUS / DEPTH-OF-FIELD CONSISTENCY** (first) — real photos have a physical focal
   plane: sharpness falls off with depth in an optically consistent gradient; AI has
   impossible blur (sharp bg behind blurred fg, multiple focal planes, blur not following
   depth). Structural, architecture-agnostic, untested. Confound = content (some scenes
   naturally flat-focus) -> content-matched bins mandatory.
2. **SCENE-LEVEL NATURAL-IMAGE STATISTICS** (most aligned w/ the cross-lab thesis) — real
   photos obey NSS at COARSE compression-surviving scales (power-law spectra, gradient
   distributions, cross-image structure). Global version of self-consistency (the one win
   = the local version); the direct "model the real, flag AI as OOD" (SPAI) thesis at a
   compression-surviving scale. Confound = content + corpus (NSS varies by corpus) ->
   audit hard.
3. **LIGHTING / SHADOW DIRECTION CONSISTENCY** — coherent illumination (shadows agree w/
   light sources); generators violate (conflicting shadow dirs, sourceless light).
   Structural, compression-robust. HIGH implementation effort (white-box light-direction
   estimation nontrivial) — FLAG if not cleanly white-box.
4. **PERSPECTIVE / VANISHING-POINT COHERENCE** (higher-effort; hold unless 1-3 leave a
   gap) — single-camera projective geometry (straight lines converge to consistent VPs);
   AI subtly inconsistent. White-box but involved (line detection + convergence stats).
   Confound = "has straight lines" = content-coupled -> audit.
5. **LATERAL CHROMATIC ABERRATION, radial-consistency version** (higher-effort, lowest
   confidence) — real lenses produce radial color fringing increasing from center.
   RADIAL-STRUCTURAL framing (not pixel-level) may survive Q75 where fine CA didn't.
   KILL-CRITERION: prove the radial-CA signal survives Q75-4:2:0 (real-vs-real Q-shift)
   first — if compression-dependent like other chroma features, reject immediately.

**Gate (all five, same as everything):** cross-pipeline lift on AIGen2026 fail-gens (CI
excl 0); no same-pipeline regression; rho-independence vs the then-current set;
content-matched-bin confound audit on the cross-pipeline surface; own kill-criterion. One
at a time, frozen-prediction, no batch-adding.

**PRE-COMMITTED STOP (updated):** if the current three AND these five all fail the
cross-pipeline gate, the white-box cross-lab ceiling is confirmed across ~12 mechanistic
attempts -> ship V2 with honest measured scope, and the remaining lever is the
**SUBSTRATE experiment** (lossless / higher-Q to revive the compression-killed features),
NOT more features on Q75.

---

## P3.6 / V3 — FINAL DE-CONFOUNDED PLAN (primary analysis cross-checked by a from-scratch adversarial re-derivation) — 2026-06-30

**Context:** I challenged my own premature "cross-lab ceiling" claim. I did the
literature research and then ran a second, fully independent adversarial re-derivation of
the question (deliberately done without reference to the project code). Both analyses
converged: the ceiling is FIELD-WIDE (Feb-2026 benchmark: SOTA deep nets 35-42% on
frontier; MJv7 24%, Imagen4 19%, freq methods near-chance), the lever is REAL-MODELING /
DATA-COMPOSITION not more low-level features, and corpus-coupling is the documented
central confound. The adversarial pass corrected the primary analysis on two points
(conceded): (1) the anti-correlation is WEAK-but-consistent, not a strong inverted
signal; (2) an estimation reframe > new features.
**Key update:** the per-gen decomposition was ALREADY done (V2 baseline): gpt-image-1
0.43, gpt-image-1.5 0.50, gemini-flash 0.45, gemini-3-pro 0.44, mjv7 0.44 — CONSISTENT
0.43-0.50 => a SHARED exploitable axis (not one-gen noise). Mechanism: frontier gens are
OVER-SMOOTHED, sitting BELOW real's artifact floor -> V2's 'more-artifact => fake'
direction scores them 'more real than real'. A single linear direction can't serve both
regimes => a TWO-SIDED ONE-CLASS REAL-ENVELOPE is the mechanistically-correct fix.

**ORDERED EXPERIMENTS (bounded; each de-confounded; frozen-prediction + real-vs-real
gate + per-gen):**
- **Exp 1 (FREE, FIRST):** one-class real-envelope on V2's 27 feats (shrinkage
  Mahalanobis + robust-z), fit on TRAIN reals ONLY. Score = distance from the real
  manifold (two-sided). Gate (i) real-vs-real: fit on LAION reals, FPR on held-out
  NEWS+DOCCI reals must stay low (<= ~15-20%) else it's a corpus detector. Gate (ii)
  per-gen cross-lab AUC on AIGen. KILL: per-gen <= 0.55 OR cross-corpus real-FPR > 20%.
  P(useful) ~0.35, P(clears 0.5) ~0.55.
- **Exp 2 (hygiene):** rebuild the envelope on RAISE verified-camera reals (dirty-reals
  hypothesis); same real-vs-real gate. P alone ~0.15; de-risks Exp1/3.
- **Exp 3 (highest P, NO acquisition needed):** cross-corpus LEAVE-ONE-LAB-OUT diverse
  training — both of_test_docci AND aigen2026 already carry Q75 cross-lab fakes; train on
  one corpus's cross-lab gens, frozen-predict on the other corpus + a held-out lab.
  P ~0.5. Relaxes the one-lineage-extrapolation setup.
- **Wave 2 (only if needed):** PatchCraft rich-vs-poor texture-contrast + cross-crop
  consistency (new white-box feats, generator-agnostic by design) as new axes; same
  gates.
**PRE-COMMITTED STOP (evidence-based ceiling proof):** Exp1 envelope <= 0.55 per-gen at
acceptable FPR AND Exp3 LOLO held-out lab <= 0.6 => white-box frontier unreachable, ship
V2 scoped. Combined P(>= 1 of Exp1/Exp3 gives a genuine de-confounded lift) ~ 55-60%. V2
stays frozen as fallback throughout.
**Sources:** SPAI (CVPR25), SimLBR, PatchCraft (CVPR24), NPR (CVPR24), "AI detection in
the wild: what truly matters" (2507.10236), open-benchmark (2602.07814),
cropping-robustness (2511.14030).

### P3.6 Exp 1 — ONE-CLASS REAL-ENVELOPE — PARTIAL POSITIVE (first de-confounded cross-lab signal; Gemini is the gap)

**`scripts/exp1_oneclass.py`.** Fit a real-only density (robust-z diag + Ledoit-Wolf
Mahalanobis) on 3000 LAION train reals, V2's 27 feats; score = distance from the real
manifold (two-sided). Frozen, de-confounded.
- **GATE (i) real-vs-real FPR: PASS** (Maha: NEWS 4.3%, DOCCI 8.2%; robust-z 2.9%/3.2% —
  all << 20%). The envelope is NOT a corpus detector — different-corpus reals stay inside
  the real manifold. First real-side candidate to pass this cleanly (vs the structural
  tier, which all failed it).
- **GATE (ii) per-gen cross-lab AUC (Mahalanobis, envelope distance vs V2
  discriminative):** midjourney-v7 0.44 -> **0.65**, gpt-image-1 0.43 -> **0.56** (both
  clear 0.55); gpt-image-1.5 0.50 -> 0.45, gemini-flash 0.45 -> 0.41, gemini-3-pro
  0.44 -> **0.40** (Gemini WORSE — sits INSIDE the real manifold). POOLED 0.45 -> 0.49.
**Determination:** PARTIAL POSITIVE. The one-class reframe works for the OVER-SMOOTHED
labs (MJ +0.21, gpt-image-1 +0.13) and is DE-CONFOUNDED (corpus gate clean) — the first
genuine honest cross-lab lift in the project. But Gemini fakes lie within the real
distribution on V2 feats -> the envelope can't see them; pooled stays ~chance. Does NOT
trigger the kill (MJ 0.65 > 0.55). Mahalanobis > robust-z. **Next: Exp 3 (diverse LOLO
training) targets exactly the Gemini gap (learn its position vs rely on the
over-smoothing prior); Exp 2 (clean RAISE reals) may sharpen the envelope.**

### P3.6 Exp 3 — DIVERSE LOLO TRAINING — PARTIAL POSITIVE; + decisive Gemini probe

**`scripts/exp3_lolo.py`.** Train LR(27) on of_test_docci (2500 DOCCI reals + 2940 fakes
across 17 frontier gens), FROZEN-PREDICT on aigen2026 (news reals + cross-lab fakes).
Cross-corpus; Gemini = the truly held-out lab.

| gen | note | V2 | DIVERSE | Delta |
|---|---|---|---|---|
| gpt-image-1 | seen-lab x-corpus | 0.431 | **0.632** | +0.202 |
| gpt-image-1.5 | seen-lab x-corpus | 0.500 | 0.519 | +0.019 |
| gemini-flash | HELD-OUT LAB | 0.454 | 0.498 | +0.044 |
| gemini-3-pro | HELD-OUT LAB | 0.438 | 0.465 | +0.026 |
| midjourney-v7 | seen-lab x-corpus | 0.436 | 0.527 | +0.091 |
| POOLED | | ~0.45 | **0.528** | |

In-scope FLUX-lineage retained (0.716 -> 0.658). Diverse training FIXES the
anti-correlation and lifts SEEN labs cross-corpus (gpt-image-1 +0.20, mj +0.09); the
truly held-out lab (Gemini) barely moves.
**DECISIVE GEMINI PROBE (include Gemini in training, split, test the held-out Gemini
half):** AUC = **0.510** — STILL CHANCE even with direct supervision (gpt-image-1 0.655,
mj 0.544 same run). => **Gemini images are white-box-INSEPARABLE from real on V2's 27
feats even with supervision** — they occupy the real manifold and no linear combination
of the current features separates them. gpt-image/mj ARE separable (V2's frozen direction
just pointed wrong).

### P3.6 SYNTHESIS — V3 IS REAL for over-smoothed/seen labs; the Gemini-class is the precise residual wall

**The "no chance for V3 / total ceiling" claim was WRONG (pushing on it was right).**
De-confounded result:
- **V3 path = one-class envelope (Exp1, catches over-smoothed: MJ 0.65, gpt-image-1 0.56,
  NO cross-lab training) + diverse multi-lab discriminative training (Exp3, catches seen
  labs: gpt-image-1 0.63, mj 0.53).** Together they lift cross-lab from WORSE-than-chance
  (0.43-0.45, actively misleading) to genuinely ABOVE-chance (0.55-0.65) on
  gpt-image + midjourney, while RETAINING FLUX/SD lineage (0.66-0.78). All
  corpus-gate-clean (Exp1 real-vs-real FPR < 10%).
- **Precise residual wall = Gemini-class generators** whose images are
  white-box-inseparable from real on the current 27 feats EVEN WITH supervision (0.51).
  This is NOT "all cross-lab is hopeless" — it's one specific failure mode (images that
  sit inside the real manifold on every current axis).
**NEXT:** (Wave 2) test whether NEW white-box features — PatchCraft rich/poor
texture-contrast + cross-crop consistency — can open ANY axis that separates Gemini (the
only remaining wall); if yes -> V3 covers the frontier; if no after a bounded try -> V3
ships covering gpt-image/mj/FLUX-lineage with the Gemini-class documented as the honest
white-box-undetectable case. Exp 2 (clean RAISE reals) still pending as an
envelope-sharpener. V2 stays frozen as the in-scope fallback. The pre-estimated 55-60%
P(genuine de-confounded lift) — REALIZED.

## P3.6 Wave 2 — GEMINI-WALL RESEARCH (first half; from-scratch re-derivation still to do) — 2026-06-30

**Target:** the one remaining wall — Gemini-class images (gemini-2.5-flash, gemini-3-pro;
likely Nano-Banana/Imagen) are white-box-INSEPARABLE from real even WITH supervision
(0.51) on V2's 27 feats. Need an ORTHOGONAL white-box axis.

**PRIMARY LEAD — SynthID watermark detection (novel, explains the wall):**
- Field finding (Google's own SynthID Detector + published reverse-engineering, a
  123k-image study): Google's Gemini/Nano-Banana/Imagen embed **SynthID by default**; it
  lives in the FREQUENCY domain at a **fixed carrier frequency with CONSTANT PHASE**, is
  **robust to JPEG by design**, and its common signature is **statistically recoverable
  WITHOUT the key** by averaging complex spectra across images.
- **Why it explains the Gemini wall:** the 27 features measure spectral MAGNITUDE/SNR
  (periods 8/16) — NONE measure phase-coherence at an arbitrary carrier => a
  constant-phase watermark is fully ORTHOGONAL to them. Consistent with Gemini sitting
  inside the real manifold yet being machine-detectable to Google.
- **White-box, de-confoundable test:** estimate the SynthID carrier template from ~600
  training Gemini images (average the COMPLEX residual FFT — a constant-phase watermark
  adds coherently, random content cancels), then per-image feature = correlation/energy
  at that template (matched filter). Reals carry no watermark => it should separate
  gemini-fake from real AND trivially pass real-vs-real (reals have no SynthID). DECISIVE
  BAR: supervised gemini separability > 0.6 (vs 0.51). RISK: per-image SNR (watermark
  faint in one image); JPEG phase disruption (but SynthID is designed JPEG-robust). Must
  control vs JPEG-grid/screen-pattern false carriers.
- **Caveat:** SynthID is Google-specific => this is a Gemini/Google-family tell, not
  generation-general — but that's fine, Gemini is the specific wall and gpt-image/mj are
  already covered. Generalizes across Google models sharing SynthID. Also: only
  un-attacked (default) outputs carry it; a phase-shift attack removes ~91%.

**SECONDARY — PatchCraft rich/poor texture-contrast: assessed WEAKER for Gemini.** The
literature is explicit: as gens "produce distributions closer to natural images,
appearance-driven cues grow subtle/fragile under JPEG" — exactly the Gemini-class regime.
Still worth a cheap check as a generation-general (non-Gemini-specific) axis.

**Sources:** Google SynthID Detector (theregister
2025-11-20), PatchCraft/Rich-Poor-Texture (arXiv 2311.12397), RA-Det
robustness-asymmetry (2603.01544), Detecting-Generated-by-Real-Only (2311.00962).
**STATUS at the time:** Wave-2 plan held until I finished the from-scratch
re-derivation and squared the two analyses. Then execute: if SynthID (or another axis) cracks Gemini
supervised > 0.6 de-confounded -> V3 covers the frontier; else build V3 covering
gpt-image/mj/FLUX-lineage with the Gemini-class documented as white-box-undetectable.

## P3.6 Wave 2 — FINAL PLAN (SynthID lead squared with the from-scratch re-derivation) — 2026-06-30

**Squaring the two analyses:** the from-scratch pass's thesis = Gemini matches natural stats by
construction => INTRINSIC features strong enough to read the residual are strong enough
to read CORPUS => they die on real-vs-real (that pass rated Gemini ~55-70%
white-box-undetectable; best intrinsic bet = operator-response manifold-proximity ~40%).
It was done before the SynthID measurement existed, so it could not price in that
finding, which is EXTRINSIC (an added watermark, absent
from all reals) => it sidesteps the corpus-coupling executioner by construction. The two
leads are maximally complementary: added-signal vs intrinsic-dynamics.
**DECIDED — two decisive probes, supervised-bar-first (>0.6 on gemini), de-confound
immediately:**
- **Probe 1 SynthID matched-filter (extrinsic):** estimate the constant-phase carrier
  template from gemini TRAIN (complex-FFT-of-residual mean, minus the real mean to cancel
  shared JPEG), per-image projection + carrier energy. Cleanest de-confound (reals carry
  no watermark in any corpus). Control vs JPEG-8/16 false carriers.
  Google-family-specific (fine — gemini is the wall).
- **Probe 2 operator-response manifold-proximity (intrinsic, from the independent
  analysis):** denoise O in {bilateral, NLM}, r1=||I-O(I)||/||grad I||,
  r2=||O(I)-O(O(I))||/||I-O(I)|| contraction ratio, per-tile summaries; generalizes the
  TV-curvature feature. De-confound via ratios -> real-vs-real < 0.65.
**PRE-COMMITTED:** either cracks (supervised > 0.6 AND de-confound passes) -> V3 WITH
gemini; both miss -> ABSTAIN path (gemini sits in the one-class real-envelope -> V3
reports low-confidence/abstain), ship V3 covering gpt-image/mj/FLUX-lineage, gemini
documented white-box-undetectable (publishable negative). DEPRIORITIZED (both analyses
agree): PatchCraft, CA/vignetting (cropped), CFA (Q75-dead), color marginals (corpus).
**Sources added by the independent pass:** CHROMA(2606.08864), DCCT(2601.22778),
AEROBLADE(2401.17879), RIGID(2405.20112), multiLID(2307.02347), B-Free(2412.17671),
GPT-ImgEval(2504.02782).

### P3.6 Wave 2 Probe 1 — SynthID matched-filter — CRACKS GEMINI (de-confounded), modest

**`scripts/wave2_probe1_synthid.py` + `wave2_probe1b_deconfound.py`.** Estimate the
constant-phase carrier template from gemini TRAIN (mean complex-FFT of the high-pass
residual, minus the real mean to cancel shared JPEG); per-image feature = matched-filter
projection + peak energy. Frozen (template + LR on TRAIN, eval disjoint TEST).
- **DECISIVE: supervised gemini-real-vs-fake AUC = 0.648** (V2's 27 feats gave 0.51; bar
  0.60). FIRST feature to separate Gemini.
- **Specificity:** the same gemini-template classifier on FLUX-vs-real = 0.508 (chance)
  -> Gemini/Google-specific, NOT generic JPEG/generation.
- **Real-vs-real de-confound PASS:** LAION reals flagged-as-gemini 6.3% (news-vs-LAION
  AUC 0.459), DOCCI 15.0% (AUC 0.516) — does NOT separate real corpora => not
  corpus-coupling. (Reals carry no watermark by construction.)
- **Grid-excluded PASS:** zero out ALL 8/16/4-px (JPEG/VAE) template frequencies -> the
  supervised AUC SURVIVES at 0.609 => a genuine OFF-GRID carrier, not compression. (Top
  off-grid template peak ~73px period.)
**Determination:** a GENUINE DE-CONFOUNDED CRACK of the Gemini wall, supervised.
Consistent with SynthID — a Gemini-specific, off-grid, corpus-invariant freq-domain
watermark, ORTHOGONAL to the magnitude/SNR features (why Gemini was invisible). The
SynthID hypothesis (absent from the from-scratch pass, which predated it) beat its 55-70% "undetectable"
estimate. **CAVEATS (honest):** (1) 0.648 is MODEST (chance -> above-chance, not solved);
(2) it's a WATERMARK detector => Google-family-specific, removable by a phase-shift
attack, won't catch a future unwatermarked Google model; (3) Probe 2 (operator-response,
intrinsic) still worth running as attack-robust watermark-independent insurance. **Per
the pre-committed plan (a probe cracked it) -> BUILD V3 WITH Gemini:** V3 = V2-27
(FLUX/SD lineage) + one-class envelope & diverse training (gpt-image/mj) + SynthID feat
(Gemini).

### P3.6 Wave 2 Probe 2 + SYNTHESIS — operator-response (intrinsic, ATTACK-ROBUST) — SUCCESS (robustness, modest AUC)

**`scripts/wave2_probe2_opresp.py`, `wave2_synth_v3.py`.** Operator-response
manifold-proximity (pure-scipy denoisers bilateral/median/wiener;
r1=||I-O(I)||/||grad I||, r2=2nd-step contraction, per-tile mean/IQR — ratios only,
de-corpus by construction). Squaring the two: the primary BASELINE hit supervised gemini
AUC 0.648 (beat the independent PSD-bound ~0.51-0.55 pessimism again); the independent
pass's decisive addition = the watermark-STRIP attack test + combination + cross-corpus
de-confound.
**THREE HONEST NUMBERS (frozen, disjoint):** SynthID-only 0.648 clean / 0.620 stripped;
**opresp-only 0.648 clean / 0.652 STRIPPED (no drop -> genuinely watermark-INDEPENDENT,
attack-robust)**; COMBINED 0.651 / 0.644. corr(SynthID, opresp) = -0.008 (independent
values, but combining adds only +0.003 -> overlapping ERRORS, they catch the same gemini
images). De-confound PASS: opresp real-vs-real news-vs-LAION 0.462, news-vs-DOCCI 0.518
(< 0.65).
**Determination:** operator-response is a GENUINE, de-confounded, ATTACK-ROBUST intrinsic
Gemini signal — it removes SynthID's removability weakness (the "watermark-in-costume"
worst-case did NOT occur: stripped opresp = 0.652). The adversarial pre-estimate P ~0.27
was beaten. **HONEST LIMITS:** AUC modest ~0.65 (above-chance, not solved; Gemini still
the hardest class vs FLUX-lineage 0.70-0.78); combining barely helps (+0.003). Wave 2's
improvement is ROBUSTNESS, not headline AUC. **Untested higher-AUC lever (the
provably-not-PSD axes):** NL-recurrence-gain r6, cross-operator-disagreement r5,
white-box LID — only these are provably-not-PSD-functionals; could push past 0.65 (needs
pure-numpy NLM).

### P3.6 V3 — COMPOSITION (de-confounded frontier coverage, honest scope)

V3 = V2-27 (FLUX/SD lineage 0.66-0.78) + one-class real-envelope & diverse multi-lab
training (gpt-image/mj 0.55-0.65) + operator-response intrinsic + SynthID (Gemini ~0.65,
attack-robust via opresp). ALL passing the real-vs-real cross-corpus gate. First coherent
ABOVE-CHANCE coverage of the whole cross-lab frontier (GPT-Image/Gemini/Midjourney),
de-confounded — vs V2 which was anti-correlated (~0.45) on all three. Honest: modest AUCs
(0.55-0.65 frontier), Gemini hardest; a future unwatermarked Google model would lean on
the intrinsic opresp axis (~0.65) only.

### P3.6 Wave 2 — NON-PSD lever (cross-checked, fixes adopted) — SMALL de-confounded win via r5; cross-scale FAILED

**`scripts/wave2_nonpsd_v2.py`.** The independent critique was adopted: FIXED h (not
sigma_hat-tied, which nulled the signal — confirmed: the sigma_hat baseline gave
0.626 < 0.65), CROSS-SCALE recurrence, gradient-stratification, MARGINAL-over-baseline
eval, 3-corpus de-confound, pre-committed kill (+0.02 & real-vs-real < 0.65).
**Result (marginal over the reduced r1/r2 head 0.620):**
- r6c_mid CROSS-SCALE recurrence (the #1 bet): **+0.000 — FAILED** (Gemini matches real
  cross-scale).
- r6c_slope -0.020, r6s_mid (same-scale) -0.007 — no help.
- **r5medg_mid (cross-operator disagreement median-vs-gaussian, mid-gradient; the #2
  bet): +0.025** — CLEARS the kill. De-confound PASS: 3-corpus real-vs-real
  0.52/0.56/0.55 (all < 0.65). Gradient-stratification saved r5 from its flagged
  noise/camera confound.
**Determination:** a SMALL but genuine de-confounded lift — Gemini ~0.648 -> ~0.665 via
r5. The headline cross-scale idea failed honestly; the #2 (operator-disagreement) won,
right at the predicted ~0.30-0.35 edge.
**OPEN before banking (honest, not inflated):** (1) confirm r5 lifts the FULL head
(12-feat opresp + SynthID), not just the reduced 6-feat baseline; (2) r5 is flagged
noise-attack-vulnerable — confirm the lift survives +noise (it survives watermark-strip,
being intrinsic, but +noise is a separate threat).

## P3.6 Wave 2 — broader-model improvement (empirics squared with a from-scratch improvement analysis) — 2026-06-30

**(A) r5 FINALIZED:** on the FULL head (opresp12 + synthid2), r5 clean marginal +0.003
(negligible — the tile feats absorb it; the +0.025 was reduced-baseline-only). +noise
marginal +0.030 but +noise already RAISES the AUC (0.64 -> 0.72, the attack backfires).
**r5 DROPPED.** Gemini head = opresp + SynthID ~0.648, attack-robust.
**De-confound re-check CORRECTED (my first test had a train=test bug -> a bogus 0.84):**
the frozen Gemini detector on other-corpus reals = news-vs-LAION 0.475, news-vs-DOCCI
0.411 (does NOT flag other reals as fake) -> **Gemini head CONFIRMED de-confounded.**
**Census:** 35 generators, rich 2026 frontier; **NO Qwen / Sora / Wan / Kolors / Hunyuan**
(acquisition gap).
**SynthID -> Google generalization (test):** imagen4 0.569 (V2 0.53, partial +0.04),
nano-banana 0.524 (= gemini-3 dup, N=60 DOCCI). Google docs confirm the SAME carrier
across Imagen/Gemini/Nano-Banana, so the weak transfer is a carrier EPOCH/VERSION
mismatch -> needs a per-model carrier re-lock (the planned fallback).
**FULL V3 SCOREBOARD (best-of V2/diverse/envelope, aigen2026 news reals,
cross-pipeline):** pooled V2 0.590 -> best 0.641; **13/19 models improve** — midjourney
+0.213 (env), gpt-image-1 +0.202 (div), flux-2-max +0.115, flux-2-pro +0.075, sd3.5
+0.071, imagen4 +0.066, ideogram +0.036, reve +0.028, hidream +0.020, seedream
+0.020 (env).
**CRITICAL CAVEATS (honest, not inflated):**
1. The DIVERSE detector is CORPUS-COUPLED (single/double-corpus reals -> news-vs-DOCCI
   0.126 -> 0.194, fails the strict <0.65/|.15|). The failure direction makes the AIGen
   test CONSERVATIVE (news reals scored fake-ish), not inflated — but it is NOT
   corpus-neutral.
2. DIVERSE REGRESSES strong models (seedream 0.75 -> 0.50, flux_dev 0.78 -> 0.73) — the
   "LR-capacity blurs the boundary" risk. So "best-of" needs to KNOW the generator (not
   deployable); a single diverse model trades strong-model loss for weak-model gain.
**Codename dedup:** halfmoon = reve, nano-banana-pro = gemini-3-pro.
GPT-Image + aurora = autoregressive (token-grid axis, V2 blind).
flux-2/z-image/seedream = different VAE (grid refit).
**FINAL ROADMAP to a de-confounded, non-regressing, deployable V3 (the real assembly
problem):** (B1) multi-corpus diverse training + regularization/per-family weighting to
STOP strong-model regression; (B2) per-model SynthID carrier re-lock -> lift
imagen4/nano-banana (Google family); (B3) per-architecture VAE-grid refit (flux-2,
z-image); (B4) an AR token-grid axis (gpt-image/aurora, NEW, hardest de-confound on Q75);
(B5) combine sub-detectors (max-of-scores / meta-LR) WITHOUT inflating real
false-positives — the core V3 deployment design. Each through the real-vs-real + frozen
cross-pipeline gate, with an from-scratch re-derivation before banking.

### P3.6 Wave 2 B5 — DEPLOYABLE FUSION — the best-of-0.64 does NOT survive into one de-confounded score

**`scripts/wave2_b5_fusion.py`.** Meta-LR over the LOGITS of frozen sub-detectors
[V2-LR, DIVERSE-LR, ENVELOPE], meta trained on aigen-half, tested on aigen-half (the only
surface where all sub-detectors are out-of-sample); real-vs-real (news/LAION/DOCCI)
checks meta corpus-coupling. Sub-detectors V2-feature-based (the SynthID/opresp Google
channels were NOT in this run — they need raw features on all gens).

| config | pooled | seedream | flux_dev | de-confound news-vs-DOCCI | @90%spec sensitivity |
|---|---|---|---|---|---|
| V2-alone | 0.593 | 0.768 | 0.768 | 0.356 | 0.174 |
| META[v2,div,env] | **0.608** | **0.579 (regress)** | 0.755 | **0.189 (corpus-coupled)** | 0.188 |
| META[v2,env] (clean) | 0.593 | 0.765 | 0.762 | 0.346 (clean) | 0.175 |
| MAX-of-scores | 0.592 | 0.726 | 0.767 | — | 0.169 |

**Determination:** the only fusion beating V2 (META + diverse, +0.015) is
CORPUS-COUPLED + REGRESSES seedream (0.77 -> 0.58) — it inherits the diverse detector's
flaws; the meta's V2-fallback did not protect seedream. The CLEAN fusion (META[v2,env])
TIES V2 (no gain). MAX is worst (confirms FP-inflation). At 90% specificity the best meta
adds only +0.014 mean-sensitivity. **The per-gen best-of 0.64 is an artifact of picking
the tool per-generator; it does NOT survive into a single de-confounded non-regressing
deployable score.** What fuses CLEANLY (de-confounded, no regression): V2 (SD/FLUX) +
envelope (midjourney) + SynthID/opresp (Google) — a modest honest beat on those families
only. **This hits the B5 kill criterion: a fused score does not honestly beat V2.**
**Implication for V3 deployment:** ship V2 as the core + the two de-confounded add-on
channels (envelope for over-smoothed/MJ; SynthID + opresp for the Google family) as
score-level OR-gates, with per-generator scope labels; do NOT deploy the corpus-coupled
diverse channel. (Fusion-design re-derivation still to run — square the two before final.)

### P3.6 Wave 2 B5 — FINAL VERDICT (empirics and the from-scratch fusion design point the same way) — 2026-06-30

**The from-scratch fusion design (adopted):** calibrate each
sub-score as a one-sided REAL-REFERENCE p-value (p_s = 1 - ECDF_real(score)) ->
Stouffer-z -> regularized meta-LR over [V2, ENVELOPE, SynthID, opresp], DIVERSE EXCLUDED.
Then (b) anti-regression, (c) specificity, (d) de-confound hold BY CONSTRUCTION (a linear
combo of corpus-invariant inputs is corpus-invariant; max-FP-inflation is a
multiple-comparison effect, fixed by the p-value framing). Only (a) "beats V2" is open.
My empirical B5 CONFIRMED the design's prediction: including DIVERSE -> corpus-coupled
(news-vs-DOCCI 0.19) + seedream regress 0.77 -> 0.58; excluding it -> clean but ties V2
(my run lacked the SynthID + opresp channels, which is what would add the Gemini/Google
lift).
**SWING FACTOR RESOLVED — NEGATIVE:** the diverse-decoupling gate (multi-corpus reals +
class_weight balanced + C=0.5) FAILS: news-vs-DOCCI 0.196 (still coupled), **seedream
0.506 (< the 0.72 gate, still regressed)**, despite gpt-image-1 0.645 / flux-2-pro 0.674
lifting. The gpt-image direction fundamentally anti-correlates with seedream;
corpus-coupling persists (single-pipeline fakes). => DIVERSE cannot be admitted cleanly.
The pre-estimate P(fused beats V2 by > 0.03 at equal spec, no strong regression) drops to
~0.40.
**FINAL B5 VERDICT:** the per-gen best-of 0.64 does NOT survive into one
de-confounded non-regressing score. The deployable, de-confounded V3 = **V2 core +
p-value-fused de-confounded add-on channels (ENVELOPE for over-smoothed -> midjourney
0.65; SynthID + opresp for the Google family -> Gemini 0.65, imagen4 ~0.57)** +
per-generator SCOPE LABELS. DIVERSE NOT deployed. This is a genuine de-confounded
improvement over V2 on MJ + Google specifically; the pooled gain is modest (~+0.02-0.03,
at the kill bar). Validating the EXACT fused pooled number needs the one build task:
extend SynthID + opresp raw features to all 19 gens x both corpora, then the decisive
frozen test (stack-train of_test_docci, stack-test aigen, arch-select on a 3rd split).
Honest fallback (endorsed): if fused <= V2 + 0.03 -> ship V2 + truthful scope labels
rather than a tied fused score that implies coverage it lacks.

### P3.6 — PER-FEATURE one-class architecture (an idea worth testing: don't mix features, combine at the end) — tested

**`scripts/perfeature_hc.py`.** Per-feature two-sided p-value vs a multi-corpus REAL
ECDF; aggregate by Higher Criticism (sparse-signal-optimal), Fisher,
count-of-exceedances. vs joint V2-LR + joint Mahalanobis.

| | V2-LR | Mahal | HC(27) | count(27) | HC(45) | count(45) |
|---|---|---|---|---|---|---|
| POOLED | 0.590 | 0.509 | 0.519 | 0.519 | 0.505 | 0.509 |
| FAIL-GEN | 0.452 | 0.493 | 0.487 | 0.506 | 0.490 | 0.502 |
| gemini | — | 0.407 | 0.392 | — | 0.392 | 0.423 |
| seedream (strong) | 0.752 | 0.772 | 0.713 | — | 0.667 | 0.625 |

De-confound (aggregate real-vs-real): HC news-vs-DOCCI 0.595(27)/0.603(45) — borderline
(corpus-coupled feats leak); count cleaner (0.543/0.560).
**Determination:** per-feature un-mixed ~ joint Mahalanobis (~0.52), does NOT beat V2-LR
(0.59), does NOT crack Gemini (0.39 — it deviates on NO feature). Expanding 27 -> 45
features did NOT help (the added structural feats carry noise/corpus, not sparse signal).
**The bottleneck is NOT feature-mixing (architecture) — it is FEATURE POVERTY for the
cross-lab frontier**: there are no clean white-box features any cross-lab gen (esp.
Gemini) deviates on; no aggregation surfaces signal that isn't in the features. The idea
is principled and has one real virtue (it NEVER regresses strong models — seedream 0.67,
no 0.50 collapse, since it fits no shared direction) -> a cleaner deployable architecture
than diverse training, but not a breakthrough.
**SOTA / similar projects (web):** closest analogue "Handcrafted Feature Fusion" (arXiv
2601.19262: DCT/LBP/GLCM/wavelet + GBMs) reports 0.98 but IN-DISTRIBUTION on CIFAKE, not
cross-lab/de-confounded. Field-wide benchmark (2602.07814, 16 methods/291 gens): most are
black-box; the cross-lab frontier is unsolved (deep SOTA 35-42%). The white-box +
cross-pipeline-de-confound rigor here is a distinctive (harder) path. (Independent
re-derivation of this architecture test pending.)

### P3.6 — PER-FEATURE architecture + SOTA — FINAL (empirics and the from-scratch re-derivation agree) — 2026-06-30

**I ran the experiment twice, the second time from scratch; identical conclusion.** Per-feature
one-class p-values aggregated by Higher-Criticism/Berk-Jones (sparse-optimal) vs joint
Mahalanobis: HC+ 0.526 / BJ 0.525 are the WORST aggregators; Mahalanobis 0.534;
Fisher/Stouffer/sum-z^2 0.531-0.533 — all below V2-LR 0.590. Whitening (innovated HC)
didn't change it. **Killer diagnostic (per-gen de-confounded single-feature max-AUC):**
NO generator is "real on 34 feats, spikes 1" — the sparse-signal premise HC needs has
ZERO instances; the signal is DENSE where present (a correlated spectral block -> the
Mahalanobis regime) and ABSENT where it matters (Gemini max feat 0.64, 0 feats > 0.65;
gpt-image-1.5 only 1 feat > 0.60). **Determination: feature-mixing is NOT the binding
constraint — FEATURE POVERTY for the cross-lab frontier is.** An architecture change
can't surface signal absent from every marginal. KILL triggered (per-feature-HC is not a
replacement).
**Virtues banked from the idea:** (1) it NEVER regresses strong models (no shared
direction) — cleaner than diverse-LR; (2) it surfaced a REAL de-confound upgrade: **a
pooled-multi-corpus ECDF/Mahalanobis null fixes a hidden corpus leak in the LAION-only
envelope** (news-vs-DOCCI 0.678 -> 0.578, LAION-vs-DOCCI 0.662 -> 0.538) at ~zero
detection cost (mj 0.649 -> 0.633). **ADOPT the pooled null for V3's envelope channel.**
(The exp1 FPR-gate missed this ranking-level coupling — important lesson: gate
real-vs-real on AUC, not just FPR.)
**SOTA / similar projects (verified):** this exact assembly (HC/BJ over per-feature
p-values of hand-crafted forensic cues, one-class) appears GENUINELY NOVEL — publishable
framing ("ECOD-style per-feature p-values + sparse-optimal HC/BJ for training-free image
forensics"). Precedents to cite: SubsetGAN (BJ/HC over per-node p-values, deep
activations, PRL2022), ECOD/COPOD (per-dim ECDF-tail p-values, one-class tabular), SPAI
(deep model-the-real spectral, ~91%/13gen), Chroma-Clues (handcrafted color + SVM
soft-vote), ZED, AEROBLADE. Cross-lab UNSOLVED field-wide: Chameleon (SOTA flags <2%
in-the-wild), NTIRE-2026 "not yet solved" (42 gens), in-the-wild collapse ~38% (Flux
21% / MJv7 24% / Imagen4 19%). Prior correction: HF-anomaly POLARITY FLIPS GAN(excess) ->
diffusion(deficit); NTIRE detection is 2026 not 2025.
**CONCLUSION — the de-confounded WHITE-BOX CEILING is reached.** Pushing on the idea was
right (it was worth testing) but it confirms the ceiling rather than breaking it. Honest
path: consolidate V3 = V2 core + the de-confounded add-on channels (envelope -> MJ,
SynthID/opresp -> Google) WITH the pooled-ECDF null fix + per-gen scope labels. Further
capability gains require leaving the white-box-LR constraint (deep features) or riding
the SynthID watermark while it lasts — neither changes the white-box conclusion.

### P3.6 — V3 CONSOLIDATED (pooled-null fix adopted) — 2026-06-30

**The pooled-multi-corpus null was adopted for the envelope channel** (fixes the
LAION-only corpus leak: news-vs-DOCCI 0.678 -> 0.578, LAION-vs-DOCCI 0.662 -> 0.538;
detection retained mj 0.649 -> 0.633). **Final de-confounded V3 scoreboard (aigen2026,
news reals, cross-pipeline):** POOLED V2 0.590 -> V3 0.631 (+0.041). Cross-lab lifts
(were anti-correlated): midjourney 0.436 -> 0.633 (envelope), gemini-flash 0.454 -> 0.648
+ gemini-3-pro 0.438 -> 0.648 (SynthID + opresp), gpt-image-1 0.431 -> 0.569 (envelope),
imagen4 0.530 -> 0.570 (SynthID). Strong models retained (no regression). **GEMINI
BANKED: 0.454 -> 0.648** (SynthID + operator-response, supervised, de-confounded,
attack-robust). Deployment caveat documented: 0.631 is per-channel-best;
generator-unknown fused ~0.60-0.62; cross-lab is above-chance not deployment-grade
(90%-spec sensitivity ~0.18).
**Consolidated report written** (architecture, methodology, results,
Gemini confirmation, SOTA comparison table, what-was-missed, novel contributions,
limitations, references) — dissertation reference.

### P3.6 — MISSED-FEATURE VERIFICATION (PatchCraft texture-patch) + dissertation comparison — 2026-06-30

**Verified the literature's #1 "feature we missed" (PatchCraft rich/poor texture-patch
contrast) empirically.** Fresh-CV looked strong (gpt-image-1 0.648, midjourney 0.643,
seedream 0.910 — matching PatchCraft's published ~0.9) BUT the full feature FAILS the
de-confound (news-vs-DOCCI 0.694, corpus-coupled). The per-feature de-confound isolated a
CLEAN subset (pc_tex_rich news-vs-DOCCI 0.488 + gpt-image 0.627; pc_ac_contrast 0.507 +
gpt-image 0.593) — the texture-RICH patches erase the corpus confound by construction (as
the literature predicted). BUT under the FROZEN one-class gate the de-confounded signal
shrinks: gpt-image 0.534, midjourney 0.566, gemini 0.475 — **below the existing envelope
(gpt-image 0.57, MJ 0.63); does NOT beat what exists.**
**This is the definitive demonstration of BOTH methodological contributions on a
SOTA-cited feature:** (1) the de-confound gate exposes corpus-coupling (0.69 -> clean
subset); (2) frozen-prediction exposes the fresh-CV -> frozen gap (0.62 -> 0.53). The
"missed" feature evaporates under honest evaluation — confirming the ceiling AND
validating the methodology.
**Dissertation-related-work survey written** (primary-source-verified survey of
SPAI/AEROBLADE/B-Free/NPR/PatchCraft/Chroma/CIFAKE-fusion/ECOD/SubsetGAN/AIGI-Holmes +
GenImage/Chameleon/NTIRE2026/291-gen benchmark; scorecard; top-3 missed levers with the
empirical test here; 4 novel contributions; 6 limitations). Key positioning:
**bottom-quartile raw capability, TOP-quartile (plausibly #1) evaluation rigor +
efficiency.** Verified corrections logged (SPAI title; 19-24% = avg-detector recall not
best; CIFAKE uses GBM not SVM; SynthID is a proprietary learned verifier -> frame the
matched-filter as a high-precision/near-zero-recall auxiliary, not load-bearing;
ECOD ~ COPOD).
**Other adoptable (untested, future work):** NPR local-pixel residual (the most
generalizing handcrafted signal), color features (none in the set), Berk-Jones
subset-scanning (principled per-feature upgrade), calibration (ECE/Brier) +
augmentation-robustness reporting.

### P3.6 — 4 REMAINING MISSED FEATURES tested (NPR / color / multi-scale / Berk-Jones) — 2026-06-30

**`scripts/missed4_test.py` + inline BJ.** Each through supervised fresh-CV ->
de-confound (real-vs-real news/LAION/DOCCI) -> FROZEN one-class envelope (multi-corpus
null).

| feature | supervised (fresh-CV) | de-confound news-vs-DOCCI | frozen one-class | verdict |
|---|---|---|---|---|
| NPR residual (down-up, Nyquist) | gpt-image 0.677, mj 0.640, seedream 0.926 | **0.662** borderline-FAIL | gpt-image 0.619, mj 0.585, gemini 0.391 | MARGINAL/borderline-coupled |
| Color across-colorspaces | gpt-image 0.767, **gemini 0.701**, mj 0.731 | **0.724** FAIL | gpt-image 0.632 (inflated) | REJECT (corpus; color rejected 3x now) |
| Multi-scale (512+256 slope/hf/noise) | gpt-image 0.806, mj 0.809, seedream 0.921 | **0.851** CATASTROPHIC | gpt-image 0.699, mj 0.708 (inflated) | REJECT (resolution-history corpus detector) |
| Berk-Jones subset-scan (27 feats) | — | 0.587 (clean) | FAIL-GEN 0.484 (gpt1 0.541, mj 0.632) | NO GAIN (~ HC 0.487, Mahal 0.493) |

**Determination:** NONE of the 4 cleanly beats the existing V3 channels under honest
evaluation. The pattern is decisive and is the thesis in one table — the BEST-looking
supervised/frozen numbers (color 0.77 gemini 0.70; multi-scale 0.81 frozen 0.70) are the
MOST corpus-coupled (de-confound 0.72/0.85); the de-confound gate rejects exactly the
impressive ones. NPR is the only borderline case (frozen gpt-image 0.619, marginally
above the envelope's 0.57, but de-confound DOCCI 0.662 just fails) — a weak
gpt-image-only contributor at best. Berk-Jones (the principled per-feature upgrade) ties
Mahalanobis (0.484) — confirms the signal is NOT sparse (BJ's regime); its only edge is a
cleaner de-confound (DOCCI 0.587 vs Mahal 0.678), consistent with adopting the
pooled-ECDF/p-value null. **CONCLUSION: the white-box de-confounded ceiling holds across
ALL adoptable literature features. The methodological contribution (the de-confound gate
rejecting corpus-coupled 'wins') is empirically demonstrated on 5 SOTA-derived features
(texture-patch, NPR, color, multi-scale, BJ).** A full independent re-check of these
verdicts was queued.

### P3.6 — AUDIT FOLLOW-UP (from-scratch double-check acted on) — 2026-06-30

**The from-scratch double-check surfaced TWO valid methodological flaws (acted on both):** (1) the
FROZEN ONE-CLASS gate is biased against DIRECTIONAL features — the 0.98 -> 0.14 lesson
licenses "frozen not CV", NOT "one-class not discriminative" (orthogonal axes,
conflated); the fingerprint is visible (NPR 0.68 -> 0.62, multiscale 0.81 -> 0.70).
(2) Berk-Jones/HC run on RAW correlated features is artifactually handicapped
(Mahalanobis IS whitened-L2); should whiten first. Also: "ceiling across ALL features" is
over-generalized (n=5, no ensemble test, JPEG-ghost substrate blind spot).
**FAIR RE-TESTS (per the audit):**
- **Whitened BJ/HC:** FAIL-GEN 0.481/0.482 ~ raw (0.484/0.487) ~ Mahalanobis 0.493 ->
  **no capability gain CONFIRMED** in the whitened basis; cleaner de-confound (DOCCI
  0.563 vs Mahal 0.678) — consistent with the pooled-ECDF null already adopted. The
  per-feature signal genuinely is NOT sparse.
- **NPR frozen DISCRIMINATIVE incremental over V2 (the fair gate):** FAIL-GEN
  0.452 -> 0.430 (lift -0.021, HURTS), gpt-image-1.5 -0.114, no-regression guard -0.088;
  **npr_var rho=0.780 with f2_cd_peak_k8 -> REDUNDANT with V2's VAE-grid**
  (mechanistically obvious: both read the upsampling artifact). **REJECT confirmed MORE
  clearly than the one-class read.** The audit's prediction ("verdicts may survive a fair
  re-test") was borne out.
- Multi-scale: the audit CONFIRMED the resolution-history-coupling diagnosis (the frozen
  0.70 is inflated by the SAME axis as the 0.851 de-confound) -> REJECT upheld. Color:
  de-confound fail -> REJECT.
**CEILING CLAIM DOWNGRADED (per the audit):** from "holds across ALL adoptable features"
-> "across the hand-crafted features TESTED, under these gates." Honest caveats: (a) no
ensemble/composite of candidates tested; (b) JPEG-ghost / double-quantization features
are a STRUCTURAL blind spot of the Q75 substrate (cannot test); (c) the unrun strongest
counter-experiment = frozen discriminative leave-one-lab-out on the UNION of all features
after resolution-matched normalization + whitening. **Net: the audit improved rigor; all
4 + texture-patch verdicts SURVIVE the fairer re-tests; the universal ceiling language is
appropriately softened.** The independent double-check worked exactly as intended — it
caught real flaws, and the fix confirmed the science.

---

## V4 — 2026-07-02 — product-focused consolidation + unrun levers (IN PROGRESS)

**Context:** a full re-analysis of the pipeline and an improvement pass toward a solid
Accuracy/AUC product (same white-box-LR, de-confounded mindset), ending in a user-facing
web UI.
**Diagnosis:** the binding gaps are PRODUCT gaps, not benchmark gaps — (P2) the endorsed
B5 fused score was never completed (SynthID + opresp features existed only for probe
subsets -> the deployable single score was never actually tested); (P3) no
calibration/abstention anywhere; (P4) roadmap items B2/B3/B4 never run; (P5) single-crop
inference wastes most pixels of large uploads + sub-512 uploads have no path; (P6)
provenance/container evidence unused (a legitimate product-only channel).
**Survey:** see `pre-registrations/similar_tools_2026-07-02.md` — no deployed tool
formalizes an INCONCLUSIVE bucket (differentiator); channels-as-separate-panels is the
honest precedent (Sightengine/FotoForensics); commercial detectors ~78% in-the-wild
(Deepfake-Eval-2024) -> these numbers are competitive when honestly scoped; c2pa-python
for provenance; Mahalanobis-as-abstention-driver separately validated.
**Launched/staged (one at a time, standard gates):**
- V4.1 `scripts/v4_extract_channels.py` — SynthID + opresp features for ALL of aigen2026
  + of_test_docci + 1500 LAION reals; templates locked (T_nano: docci nano-banana-pro,
  fully-frozen arm; T_gem: aigen gemini train-half, half-split arm, train hashes saved).
- V4.2/4.3 `scripts/v4_fusion.py` — the decisive B5 fused-score test (stack-train
  of_test_docci A/B split, frozen-test aigen2026) + isotonic/Platt calibration,
  ECE/Brier, 3-band verdict thresholds. Pre-committed: accept fused only if pooled >=
  V2 + 0.03, no strong-gen regression, real-vs-real clean; else fallback = V2 + channels
  as separate scoped panels.
- V4.4 `scripts/v4_b2_carrier_relock.py` (B2, imagen4 per-model template) - V4.6
  `scripts/v4_b4_arprobe.py` (B4, large non-multiple-of-8 periods; pre-registered
  expectation REJECT given P3.4-C2)
- V4.7 `scripts/v4_multicrop_pilot.py` — multi-crop score aggregation A/B on fresh
  native-res OpenFake data (AC-9-clean estimation-variance reduction; distinct from the
  rejected multi-SCALE).
- V4.8 `pipeline/provenance.py` + tests (6/6 pass) — a product-only C2PA/XMP/EXIF/
  PNG-chunk channel, asymmetric semantics (presence => AI-positive; absence => NEUTRAL,
  never "real").
- Product path `pipeline/predict.py` — bundle-driven single-image predict with
  per-channel panels.
**Results appended below as each arm completed.**

### V4 RESULTS (all numbers from the run logs `data/features/v4_*.log`, runs of 2026-07-02 09:00-10:15) — 2026-07-02

**V4.1 channel extraction — DONE (10:07).** SynthID matched-filter (T_nano frozen arm +
T_gem half-split arm) + operator-response features extracted for ALL surfaces:
`v4_channels_aigen2026.parquet` (10,660), `v4_channels_of_test_docci.parquet` (18,504),
`v4_channels_laion.parquet` (1,500). Templates: `v4_templates.npz`. This closes P2's
build gap — the fused score became testable for the first time.

**V4.2 learned meta-LR fusion — FAIL (pre-committed bar not met; 10:08,
`v4_fusion.log`).** Stack-trained on of_test_docci A/B, frozen-tested aigen2026. Pooled
per-gen mean 0.588 vs V2 0.590 (the bar was >= +0.03). The meta coefs revealed WHY:
opresp dominated (+1.20) and envelope got a NEGATIVE coefficient (-0.35), i.e. the stack
learned train-corpus quirks; frozen transfer flipped them -> bytedance_seedream
0.752 -> 0.240 (catastrophic), flux_dev 0.781 -> 0.747 (regression), even though
gpt-image-1 +0.214 / fast-sdxl +0.123. Fused real-vs-real news-vs-DOCCI 0.155 (wildly
non-flat = corpus-coupled). V4.3 calibration on this score: ECE 0.38, AI-band FPR on news
reals 39% — unusable. **The fallback triggered exactly as pre-committed.**

**V4.2b fixed-form fusion (no learned weights; 10:10, `v4_fusion2.log`).** Added a
train-side channel ADMISSION gate (real-vs-real in [0.35, 0.65]): **opresp EXCLUDED
(LAION-vs-DOCCI 0.301 — corpus-coupled; retroactively explains V4.2's failure)**; env
0.569 + sid 0.355 admitted. All fixed rules (max, one-sided-max, Stouffer) on admitted
channels: pooled 0.555-0.572 < V2-alone 0.590, strong-gen worst delta up to -0.073 -> the
clean + no-regression candidate set is EMPTY -> **PICK R1_v2: the V2 score is THE ranking
score.** Banded verdicts from the pooled real null (AI >= 95th pct, REAL <= 30th pct):
news-real FPR into the AI band 8.8% / LAION 7.5% / DOCCI 4.1%; per-gen mean AI 16% /
INCONCLUSIVE 72% / REALmiss 11%. Deployable bundle persisted:
`data/features/v4_bundle2_candidate.joblib` (drives `pipeline/predict.py`).

**V4.2c OR-gate verdict rule (10:11, `v4_orgate.log`).** V2 + q99 env/sid verdict gates:
mean AI-catch 0.165 -> 0.173 (+0.007) at a real-FPR cost of LAION 7.9% -> 9.5%, DOCCI
4.1% -> 6.3% (~+20% relative FPR for +4% relative catch). **DECISION: NOT adopted at
verdict level — a bad trade. The channels stay as display-only evidence panels in the
product report (reversal path: the thresholds are in the bundle; flip `gates_enabled` in
the predict path).**

**V4.4 B2 imagen4 carrier re-lock — BELOW BAR (10:14, `v4_b2.log`).** Per-model template,
disjoint-half AUC 0.593 vs the 0.62 bar (V2 0.530, gemini-template transfer 0.569 ->
+0.063 real improvement but not deployable). De-confound clean (news-vs-LAION 0.517,
news-vs-DOCCI 0.504); FLUX specificity 0.525 OK. 3 of 12 template peaks are
JPEG-8-harmonics (partial confound). NOT adopted; the carrier-epoch-mismatch hypothesis
is partially confirmed but the Q75 substrate caps the matched filter.

**V4.6 B4 AR token-grid probe — REJECT as pre-registered (10:13, `v4_b4.log`).** Large
non-multiple-of-8 periods {18..60}: fresh-CV 0.524 (bar 0.60; the P3.4-C2 small-period
scan was 0.532). No AR grid survives Q75.

**V4.7 multi-crop pilot — first run lost to an engineering failure; the extraction code
was rewritten to stream images one at a time and relaunched.** (Result below.)

**V4.5 B3 (VAE-grid refit flux-2/z-image): still PENDING** (low prior after P3.4-C2 + B4
rejections, but the mechanism differs — VAE upsampling vs AR tokens). V4.9 product spec:
PENDING.

**Net V4 science verdict:** the fused-single-score hypothesis (B5) is now EMPIRICALLY
CLOSED — both the learned and the fixed-form fusions fail frozen cross-corpus transfer;
the admission gate caught opresp as corpus-coupled. The deployable product = **the V2
ranking score + pooled-null banded verdicts (AI / INCONCLUSIVE / REAL) + per-channel
evidence panels + a provenance channel (product-only) + scope labels.** This is the
honest consolidation the pre-committed fallback always pointed to.

### V4 RESULTS (continued) — V4.7 multi-crop ADOPTED - V4.5 B3 REJECT - predict path rebuilt — 2026-07-02

**V4.7 multi-crop inference pilot — ADOPT, size-gated (`v4_multicrop.log`).** Rerun (with
the streaming implementation) on 298 fresh OpenFake native-res images (deduped vs all
of512 splits, 2 dropped for non-finite SC5 feats). Frozen V2 score, same images A/B:
center-crop 0.752 -> **median-5 0.816 (+0.064), mean-5 0.828** — the accept bar (+0.02)
decisively met. **BUT the size split matters: [724, 1024): center 0.614 -> median-5 0.549
(HURTS — crops overlap heavily); [1024, inf): 0.748 -> 0.835 (+0.087).** DECISION: adopt
**median-5 only for uploads with min side >= 1024**; center-crop below. Caveats logged
honestly: (a) the surface is same-corpus (OpenFake train -> held-out portions) — the
RELATIVE center-vs-multicrop gain is the measured quantity, the cross-corpus relative
gain is unmeasurable (aigen2026 is stored as 512 crops); (b) crop-score dispersion is
class-asymmetric (real std 1.075 vs fake 0.500) — noted as a possible future feature, NOT
used (would need full gates). The envelope channel under aggregation also sharpened
(0.322 -> 0.220, direction-consistent). Reversal path: the `MULTICROP_MIN_SIDE` const +
`multicrop=False` flag in `pipeline/predict.py`.

**V4.5 B3 VAE-grid probe — REJECT as pre-registered (`v4_b3.log`).**
flux-2/-max/-pro + z-image_turbo vs news reals, periods {18..60} non-multiple-of-8 (the
only untested window after P3.4-C2 {5..14} and B4): fresh-CV 0.502 (bar 0.60); per-gen
0.468-0.529; top separator vae_snr_p60 at 0.14 sigma (noise). **With B2 (below bar), B3,
B4 all closed, the V4 capability arms are exhausted — consistent with the standing
white-box frontier ceiling. All remaining V4 value is product work.**

**`pipeline/predict.py` REBUILT on the adopted design** (it was consuming the rejected
meta-LR bundle): rule R1_v2 from `v4_bundle2_candidate.joblib` — V2 z vs the pooled real
null is THE score; bands t_lo/t_hi from the bundle; strength language = real-percentile
(no cross-corpus probability claimed — the isotonic ECE 0.38 failure stands); panels =
v2/env/sid display-only (opresp DROPPED from extraction + panels: failed channel
admission, corpus-coupled); provenance as a separate panel; multi-crop median-5 gated at
min side >= 1024; sub-512 -> UNSUPPORTED (never upscale, provenance still reported).
Smoke-tested: flux_dev fake -> LIKELY AI-GENERATED (z 2.65, 99.6th pctile); news real ->
INCONCLUSIVE (z 1.29, 90.2th pctile, correct abstention). **Remaining V4 work: the V4.9
product spec + a full re-verification before promoting the bundle from `data/features/` to
`frozen_models/`, then the web UI.**

### V4.10 (B1) — MULTI-CORPUS FAMILY-BALANCED RETRAIN — REJECT (pre-registered shot; confirmed package P1) — 2026-07-02

**`scripts/v4_b1_retrain.py` + `data/features/v4_b1_retrain.log`; design pre-registered
in `pre-registrations/v4_push_proposal_2026-07-02.md` §1 (4 hard bars + a directional
pattern; only C swept, train-side).** LR(27) on the union of512_train + of_test_docci, 19
fake families uniform-weighted, LAION + DOCCI reals corpus-balanced, C=0.03 (train-side
sweep; insensitive 0.856-0.857).
**Ablations first (both clean):** A2 phash near-dup audit docci-gpt vs aigen-gpt: 0.0000
rate (min hamming p1=16 — no content leakage). A1 leave-gpt-image-out: the gpt-image-1
gain PERSISTS without any gpt training data (0.431 -> 0.573, +0.14) -> the
diverse-training lift is genuine cross-generator physics, NOT prompt/content overlap.
Scientifically valuable independent of the verdict.
**FROZEN SHOT:** pooled per-gen 0.590 -> **0.620** (+0.030, bar B1 met exactly);
gpt-image-1 +0.185, mj +0.099, flux-2-max +0.126, imagen4 +0.054, gemini +0.066/+0.044.
**BUT B2 FAIL:** seedream 0.752 -> 0.540 (-0.213; the same zero-sum-coefficient collapse
as the V4.2 meta), flux_dev -0.034. **B3 FAIL:** news-vs-D_hold 0.180, L_hold-vs-D_hold
0.183 — training on DOCCI reals taught a DOCCI-real corpus axis (the score is wildly
non-flat across real corpora). B4 passed (news FPR 9.2%). Directional pattern: gpt/mj
matched, gemini and flux missed. **VERDICT: REJECT — keep V2, no re-sweeps, per the
pre-registration.** The adversarial pre-estimate (25%, "the strong-gen bar is the likely
killer") was correct. **B1-B5 now ALL adjudicated: every attempt to move the single
deployable score off frozen V2 fails the no-regression or de-confound gates. The
single-score white-box-LR ceiling is closed on all roadmap axes; the banked insight (A1)
is that ~half the gpt-image lift is transferable physics — usable only at the cost of
strong-gen regression, i.e. NOT deployable in one linear score.**

### V4 P2 PROBES — angular anisotropy REJECT - Benford-DCT REJECT (pooled), Gemini-family LEAD BANKED — 2026-07-02

**`scripts/v4_p2_probes.py` + `data/features/v4_p2.log`; package §2, pre-registered
expectation REJECT for both; a per-feature real-vs-real pre-screen (news/LAION/DOCCI,
1000 each) ran BEFORE any classifier saw fakes.** Pre-screen: 17/18 features admitted
(only ben_chi2_all corpus-coupled at 0.665) — both families are de-confound-clean,
unusual vs the 22 prior deaths.
**P2a angular anisotropy (9 feats):** frontier-vs-news fresh-CV 0.587 < 0.60 bar ->
REJECT.
**P2b Benford-DCT (8 feats):** 0.599 < 0.60 -> REJECT at the pooled pre-registered bar.
BUT per-gen: **gemini-3-pro 0.662, gemini-25-flash 0.631, gpt-image-1.5 0.632**
(gpt-image-1 0.489, mj 0.583) — the first de-confound-prescreened feature family to show
>0.6 fresh-CV on the Gemini class, which was white-box-INSEPARABLE (0.510) on V2's 27
features even with supervision. Fresh-CV inflates (the 0.62 -> 0.53 lesson); NOT adopted.
**LEAD BANKED with one pre-registered frozen shot added to the P4 nano-banana readouts
(d): train LR on the Benford-admitted feats [aigen gemini fakes vs news reals],
frozen-test [virgin nano-banana fakes vs LAION/DOCCI holds]. Registered expectation:
attenuates below 0.60; escalates to full gates only if it survives.** Honest note:
readout (d) is motivated by a per-gen look at a rejected probe — the escalation path (new
virgin surface, frozen cross-corpus, different real corpora train/test) is chosen
precisely so that this selection cannot inflate it.

### V4 P3a — MCD ROBUST ENVELOPE A/B — REJECT (keep Ledoit-Wolf) — 2026-07-02

**`scripts/v4_p3a_mcd.py` + `data/features/v4_p3a.log`; package §3, the envelope's
existing role only (fusion stays closed). Kill = any unseen-corpus real-FPR increase;
adopt = kill-clear AND env-gen mean dAUC >= +0.02.** MCD sf=0.75/0.9 both KILL-CLEAR
(unseen FPR actually improves: news 3.4% -> 1.8%, DOCCI 6.4% -> 2.6%) but dAUC only
+0.013/+0.007 < bar, AND the "gain" is a redistribution that sacrifices midjourney
0.651 -> 0.568 — the one generator the envelope channel is scoped to catch (V3 banked
mj-via-envelope). REJECT: LW stays. Banked note: if the product ever needs a lower-FPR
abstention driver at the cost of MJ catch, MCD sf=0.75 is the measured alternative (the
reversal path is documented).

### V4 P4 — VIRGIN SURFACE (bitmind/nano-banana, 500 imgs @1024^2, Gemini-2.5-Flash) — 4 pre-registered readouts — 2026-07-02

**`scripts/v4_p4_nanobanana.py` + `data/features/v4_p4.log`.**
**(a) GEMINI WALL REPLICATED on virgin data:** frozen V2 nano-vs-news AUC **0.411**
(aigen refs 0.454/0.438) — the white-box-inseparable claim now holds on two independent
corpora; a confirmatory null banked as ceiling evidence. **(b) MULTI-CROP VALIDATED
CROSS-CORPUS:** center 0.567 -> median-5 **0.661 (+0.094)** vs 300 fresh native >= 1024
OpenFake reals — V4.7's same-corpus caveat CLOSED; the >= 1024 median-5 adoption
confirmed. (Note: the absolute level is real-corpus-sensitive — 0.411 vs news but
0.567/0.661 vs OpenFake native reals; the registered readout is the DELTA.) **(c)
SYNTHID RECALL ~ ZERO on genuine Google outputs:** @q95 5.0% (= the null FPR), @q99 0.2%
— the T_nano template (locked on only 60 docci nano-banana-pro imgs) has NO detection
power on this dump (carrier-epoch mismatch, re-encode stripping, or template noise).
Motivates the approved option-3 re-lock on 300+ fresh imgs (next arm). Product note: the
SynthID panel copy must say "checked, not detected" — never imply absence of a watermark.
**(d) BENFORD-GEMINI LEAD DEAD AS REGISTERED:** frozen cross-corpus 0.521 (fresh-CV was
0.63-0.66); real-vs-real 0.657 borderline. The 25th feature family killed by the gates;
the fresh-CV -> frozen collapse (0.66 -> 0.52) repeats the canonical lesson. **P2/P4
capability conclusion: the white-box ceiling on the Gemini class survives its strongest
challenge to date on virgin data.**

### V4 COVERAGE SWEEP — 10 newest-gen models scope-labeled (frozen V2, no gate spending) — 2026-07-02

**`scripts/v4_sweep_newgens.py` + `data/features/v4_sweep.log`; source
Qwen/Qwen-Image-Bench (official, 300/model, native 1024, center-512/Q75 substrate),
reference = aigen news reals.**

| model | AUC | label | | model | AUC | label |
|---|---|---|---|---|---|---|
| FLUX.2_max | **0.810** | covered | | nano-banana-2.0 | 0.558 | partial |
| Seedream-5.0 | 0.623 | partial | | Imagen-4.0-Ultra | 0.537 | wall |
| kling_v2_1 | 0.617 | partial | | gpt-image-2 | 0.488 | wall |
| HunyuanImage-3.0 | 0.584 | partial | | Qwen-Image-2.0-pro | 0.486 | wall |
| | | | | GLM-Image | 0.456 | wall |
| | | | | GPT-Image-1.5 | **0.420** | wall |

The pattern holds exactly: open/diffusion-lineage covered (FLUX.2 0.81), closed frontier
labs (OpenAI/Google/Zhipu/Alibaba newest) at the wall -> abstention is the correct
product behavior; the band rates confirm it (walls: 6-14% AI-flag, 61-80% INCONCLUSIVE).
**Honest replication note:** FLUX.2-max measures 0.810 here vs 0.549 on aigen2026 — same
frozen model, same substrate; the delta is surface construction (bench = API-direct 1024
PNG originals vs aigen's harvested/post-processed copies) — evidence that
POST-PROCESSING/provenance of the fake matters as much as the generator; the product copy
must say "detects generator artifacts as delivered by the API; social-media laundering
degrades detection". GPT-Image-1.5 0.420 replicates its aigen anti-correlation (wall
confirmed twice). New product scope labels banked for the web UI (v4_sweep_*.npy features
kept for future re-scoring).

### V4 SID-2 RE-LOCK + CONFIRMATION — **FIRST FULLY-GATED WIN ON THE WALL: Google family 0.656-0.702, ADOPT pending the from-scratch re-verification** — 2026-07-02

**Arm 1 (`v4_sid_relock2.py` + `v4_sid2.log`, pre-registered):** T_nano2 locked on 300
genuine bitmind/nano-banana crops (disjoint from P4's 500; saved `data/raw/nano_sid/`,
400 crops). The registered verdict = KEEP OLD (nano-holdout recall 0.150 vs 0.060,
+0.09 < the +0.10 bar) — but the secondary readouts exploded: aigen Google gens
0.398/0.344/0.386 (old) -> **0.702/0.656/0.668** (new), real-vs-real LAION 0.493 / DOCCI
0.503 (the old was DOCCI-coupled 0.151), FLUX spec 0.520.
**Arm 2 confirmation (`v4_sid_confirm.py` + `v4_sidc.log`, pre-registered mechanism +
adoption bars): ALL 6 BARS PASS.** M1 mechanism: **carrier-like** — the off-grid-only
template (6/12 peaks, all JPEG harmonics zeroed) retains gemini-flash 0.621 /
gemini-3-pro 0.677 / imagen4 0.627 -> NOT a smoothness/harmonic-amplitude artifact. M2
attack: Q60 re-encode AUC 0.714 (survives). Protocol: template from bitmind nano (train
corpus) + news-A reals; LR train news-A vs nano; ALL evaluation frozen on news-B vs aigen
gens (disjoint reals, disjoint fake corpus, cross-MODEL transfer nano ->
gemini-flash/3-pro/imagen4). Templates saved `v4_templates2.npz`.
**Context: gemini was 'white-box-inseparable' on V2 feats (0.510 supervised probe) and
the old 60-img template was worthless on genuine outputs (P4c recall 5%). The fix was
purely DATA: 300 genuine carrier samples instead of 60 proxy ones. imagen4 0.668 also
beats B2's failed re-lock (0.593) and its 0.62 bar.** ADOPT as a SCOPED Google-family
channel (panel copy: "spectral pattern consistent with Google's SynthID watermark family;
removable by attack") — the bundle swap held until the from-scratch re-verification (my
standing rule before adoption).

### V4 SID-2 RE-VERIFICATION — ADOPTED WITH CONDITIONS (from-scratch re-derivation completed) — 2026-07-02

**A from-scratch re-verification pass over the code + logs + data (`v4_sid_relock2/confirm.py`).**
Leakage: CLEAN (rA/rB disjoint & deterministic; avg-hash near-dup scan nano-vs-aigen =
noise — the FLUX control had more collisions than the Google gens). Circularity: CLEAN.
Effect size: REAL (~10+ DeLong SEs; not sampling luck). Mechanism: the harmonic formula
verified; chroma-subsampling harmonics excluded; one off-grid pair sits 1 bin from a
64-harmonic (residual ambiguity) — hence condition 1. **Additional checks from the audit
(banked as official readouts):** (a) generic-AI-ness sweep —
gpt-image-1/mj-7/seedream-4.5/sd3.5/ideogram/z-image with T_nano2 = **0.46-0.55 all
chance -> genuinely Google-scoped**; (b) a TRUE third-corpus replication on of_test_docci
nano-banana-pro (60 imgs, never touched): **0.578 vs news / 0.537 vs DOCCI — real but
attenuated; honest expected range "0.55-0.70 depending on surface"**; (c) fusion-style
admission spot-check 0.508 (inside [0.35, 0.65]). Conceded: arm-2's "6 bars"
partly re-measured arm-1 numbers (same seed/samples) — only the ablation + Q60 were new
evidence.
**CONDITIONS (all adopted):** (1) deploy the OFF-GRID-ONLY 6-peak template (`Tn_offgrid`,
wins gemini-3-pro 0.677, FLUX spec 0.480); (2) attenuation disclosed (this entry + panel
copy); (3) display-only panel — NOT in the ranking/verdict score (matches the V4.2c
decision); (4) refit scS2/clfS2 on DOCCI-protocol reals + rebuild the sid ecdf on the
LAION+DOCCI null (the bundle2b job); (5) panel copy = "spectral pattern consistent with a
Google-family generation signature" (not asserted SynthID identity).

### V4 OPTION-1 NULL BROADENING — extension NOT adopted; null VALIDATED on 2 new corpora; BDD out-of-scope class discovered — 2026-07-02

**`scripts/v4_null_broaden.py` + `data/features/v4_null.log`.** New verified-camera
pre-genAI corpora: DIV2K-HR (800, pro camera 2K) + BDD100K (1500 dashcam 720p). **BDD
EXCLUDED by the pre-registered rule** (V2-score real-vs-real 0.83-0.91 vs every other
corpus; LOCO hold-bdd FPR 46%): dashcam/high-noise-compressed camera frames read as
artifact-heavy — **a documented out-of-scope REAL class, added to the product limitations
copy.** DIV2K CLEAN (0.41-0.58) and covered by the EXISTING null (LOCO FPR 3.4% at
nominal 5%) — the LAION+DOCCI null generalizes to an unseen pro-camera corpus. The
3-corpus extension changed nothing (news FPR 0.076 -> 0.081) -> **the null was NOT
extended (no benefit); the arm's value is the validation + the scope discovery.**
Artifacts kept: `v4_null_div2k.parquet`, `v4_null_bdd.parquet`, `v4_null_extended.npz`.

### V4 BUNDLE2B ADOPTED INTO THE PREDICT PATH (all audit conditions implemented) — 2026-07-02

**`scripts/v4_adopt_bundle2b.py` + `data/features/v4_adopt.log` ->
`v4_bundle2b_candidate.joblib`.** The sid channel rebuilt per the conditions: OFF-GRID
6-peak template; clfS2 refit DOCCI-protocol (DOCCI-fit 2000 vs nano-300); ecdf null
rebuilt on LAION+DOCCI holds; admission 0.450 (clean). **Deployed-scorer validation:
gemini-25-flash 0.707 / gemini-3-pro 0.646 / imagen4 0.658** — the wall-class effect
fully survives production wiring. The v2 verdict channel/bands UNCHANGED (the null
extension rejected above; sid stays display-only per V4.2c + condition 3).
`pipeline/predict.py` updated: bundle2b + off-grid template + softened panel copy
("Google-family spectral signature", measured range 0.55-0.70 disclosed) + limitations
copy (laundering degradation, dashcam-class reals). Smoke-tested 3 cases (news real
INCONCLUSIVE 56.6pct; flux_dev 80.9pct; gemini-3-pro shows sid panel +0.88 while verdict
LIKELY REAL — one of that gen's known 13% REALmiss; noted). **Product option flagged for
V4.9 (NOT adopted): a sid-veto rule (elevated sid z downgrades LIKELY REAL ->
INCONCLUSIVE only — asymmetric, no new false-AI verdicts); needs its own measured trade
before adoption.**

### V4.9-pre — SID-VETO RULE — REJECT per pre-registration — 2026-07-02

**`scripts/v4_veto_measure.py` + `data/features/v4_veto.log`.** Rule: LIKELY REAL
requires z_v2 <= t_lo AND z_sid < T (asymmetric; can only downgrade REAL ->
INCONCLUSIVE). Bars: Google REALmiss -33% rel AND real coverage loss <= 3pt AND
non-Google shift <= 2pt. **All three thresholds FAIL** (q80: -25% rel but 6.2pt DOCCI
coverage loss + 5.5pt non-Google leak; q95: only -8% rel). Mechanism: a 0.65-0.70 channel
AUC is panel-grade, not gate-grade — the sid z distributions overlap too much per-image;
consistent with the V4.2c OR-gate rejection. **The display-only architecture is
re-validated empirically; the verdict layer stays V2-only. The veto question is CLOSED.**

## 2026-07-03 — WEB UI: four-band verdict (LEANING AI added at q80)

After a FLUX-2 upload read INCONCLUSIVE, I considered lowering the AI verdict to >= q80.
The hard q80 cut was REJECTED by measurement: by construction 20% of null reals would be
stamped AI (vs 5% at the frozen q95), falsifying the published <9% AI-verdict FPR.
Compromise adopted: the verdict layer gains a LEANING AI-GENERATED band for s in
[q80, q95) (z 0.841-1.641 on the pooled null). t_hi/t_lo frozen and unchanged; t_mid
derived at predict-time from the stored null (np.quantile(real_null_scores, 0.80)). Spot
check, 40 aigen2026 held-out AI images: 5 AI / 11 LEANING / 17 INC / 7 REAL —
actionable-verdict recall 12.5% -> 40% on that sample. Public repo `pipeline/predict.py`
+ site updated; the <9% claim still refers to the hard AI band only.

## 2026-07-03 — DEEP TILING: REJECT (pre-registered, `pre-registrations/deep_tiling_prereg_2026-07-03.md`)

Question: tile the whole image ("deep analysis") instead of center/median-5?
Data: 3000 held-out native >= 1024 images (2x nativeRAISE real splits, 4 AI arms:
flux/sd35 x audit/test), 9 stride-256 crops each, v2 z per crop, 27k crops.
bfree_audit/real excluded PRE-SCORING (mostly <1024; geometry-label confound).
Split-matched pairs, median aggregator.
Mean AUC over 4 pairs: center-1 0.692 | median-5 (DEPLOYED) 0.704 | grid-4 0.710 |
dense-9 0.707. Deltas vs deployed: grid-4 +0.006, dense-9 +0.003 (95% CI
[+0.000, +0.006]) — both far below the pre-registered +0.02 adoption bar. Exploratory
aggregators (mean 0.715 / max 0.713 / q75 0.711) modestly above median but post-hoc, not
adoptable without a fresh pre-registration, and mean/max shift the null -> a full band
recalibration for <= +0.011. VERDICT: full-image tiling adds ~2-9x compute for <= +0.01
AUC; the deployed median-5 (>= 1024) + center (<512..1024) stands.
Note for the writeup: multi-crop's validated +0.09 came from stabilizing the ESTIMATE
(median of 5), not from coverage — the artifacts are global; more tiles mostly re-measure
the same evidence.

## 2026-07-03 — WHOLE-IMAGE RESIZE ARM: smoke test (500 imgs), REJECT as replacement, FUTURE WORK

Prereg addendum in the deep-tiling pre-registration. The whole image Lanczos-downscaled
into the 512 substrate vs the deployed crop protocol, same images. Mean AUC 0.721 vs
median-5 0.719 — but GENERATOR-SPLIT: FLUX 0.790/0.801 (crop 0.754/0.722, +0.04..+0.08)
while SD3.5 0.650/0.645 (crop 0.674/0.726, -0.02..-0.08). The stated prediction (uniform
artifact destruction) was WRONG for FLUX: its decoder signature survives/aliases through
the 2x downscale; SD3.5's does not. Fails the adoption rule (loses 2 pairs by > 0.01).
FUTURE WORK note: the resized view = a complementary generator-specific channel; any
dual-view fusion needs (a) a second independent >= 1024 real corpus for the real-vs-real
gate on the resize protocol (RAISE-only here — the gate is untestable), (b) a full
prereg, (c) a frozen fusion transfer test (all V4 fusions failed transfer). Decision: not
chasing this now; parked.
