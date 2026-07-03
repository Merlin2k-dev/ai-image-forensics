# V4 final-push proposal — 2026-07-02
**Status: recorded BEFORE execution — nothing below had been run at the time of writing.**
This plan was drawn up from my own analysis combined with a broad survey of
the external literature; the full survey is summarized below (§5). The experiment log's
"V4 RESULTS" entries record all prior arms.

## §1 P1 — V4.10 (roadmap B1, never run): multi-corpus family-balanced retrain — PRIMARY
Single LR(27 frozen features), trained on the UNION of the two train-side corpora:
of512_train (3000 flux/sd35 fakes + 3000 OpenFake reals) + of_test_docci (per-family-balanced
17-gen fakes + 4000 DOCCI reals). Weighting fixed a priori: UNIFORM per fake family (not swept).
Only C is swept, train-side only (of_test_docci A/B). V2 stays frozen; a pass creates a V5
candidate subject to a from-scratch re-verification before adoption.
**Pre-registered ablations FIRST (kill the content-leakage hypothesis):**
  A1 leave-family-out: retrain the union WITHOUT gpt-image fakes; if the aigen gpt-image gain
     persists, the gain = cross-generator physics. A2 phash near-dup audit
     of_test_docci-vs-aigen gpt-image (>1% near-dups = leakage, discount).
**ONE frozen shot at aigen2026 with FOUR pre-committed bars (all must pass, 0.61 = REJECT, no
re-sweeps):** (1) pooled per-gen mean >= 0.62; (2) worst strong-gen delta >= -0.03
(seedream/flux_dev/flux-pro/hidream); (3) score real-vs-real news/LAION/DOCCI pairwise in
[0.35,0.65]; (4) news-real FPR into the AI band (recomputed pooled-null thresholds) <= 10%.
**Directional pre-registration (harder to pass by luck):** gpt-image up strongly (+0.10..0.20),
mj up weakly, gemini flat (+/-0.03), flux-lineage small down (>= -0.03).
Adversarial pre-estimate: P(pass all bars) ~25%; most likely failure = the strong-gen bar
(zero-sum linear coefficient capacity). aigen2026 re-spend acknowledged: partially burned;
defensible only as ONE pre-specified shot with worst-case + directional bars.
Cost: minutes CPU (all features already extracted).

## §2 P2 — two cheap capability probes (pre-registered expectation REJECT)
Per-feature real-vs-real pre-screen BEFORE any classifier; hours each; standard gates if signal.
  P2a angular spectral anisotropy (axial-vs-diagonal energy, angular kurtosis at fixed radius) --
      upsampling artifacts are axial; the current features are radially averaged. Survival ~20%;
      likely killer = resize-history axial imprint in reals (multi-scale's confound class).
  P2b generalized-Benford DCT digit stats on the final Q75 JPEG (arXiv 2004.07682) -- the uniform
      substrate removes the cross-QF fragility. Survival ~12%; likely killer = content-density
      coupling + the Bond-U negative result on modern gens.
SKIPPED by agreement of both analyses: SRM/SPAM-lite (18% x 1-2 days = poor EV; the failure axis
duplicates the noise-print/opresp deaths), phase-spectrum, bispectrum, CA/CFA (substrate-killed),
wavelet QMF.

## §3 P3 — product/method arms (no capability risk)
  P3a MCD/EllipticEnvelope A/B vs Ledoit-Wolf Mahalanobis in the envelope's EXISTING role only
      (display/abstention; fusion stays closed). KILL: any unseen-corpus real-FPR increase.
  P3b Conformal bands (Decoupled Conformal Optimisation 2605.18354 + e-value selective risk
      2603.24704) -- V4.9 product phase, replaces/backs the percentile bands with finite-sample
      risk control.

## §4 P4 — new independent eval surface: bitmind/nano-banana (~5GB streamed subset)
HF: bitmind/nano-banana (MIT, 9457 imgs @1024^2, 100% Gemini-2.5-Flash-Image). Fake-only ->
paired against existing news/DOCCI reals. THREE pre-registered readouts:
  (a) Gemini-wall replication on virgin data (expect ~0.50 -- a confirmatory null strengthens the
      ceiling claim), (b) multi-crop relative gain at native 1024 cross-corpus (fixes V4.7's
      same-corpus caveat), (c) SynthID matched-filter recall on genuine Google outputs (first
      real-carrier recall measurement).
NTIRE vintage: already the 2026 edition, surface SPENT (P3.3) -- no download. ash12321 micro-sets:
last priority, non-evidential without a provenance spot-check; cut first.

## §5 External survey conclusions (key cites inline)
- NO published white-box pipeline meets this rigor bar. Closest peer: Uhlenbrock IH&MMSec'24
  (chroma residual co-occurrence + RF, true cross-gen SD->6 unseen, 91%/98% reported) -- never ran
  a real-vs-real gate; lives on Cb/Cr statistics the 4:2:0 Q75 substrate flattens; no frontier
  gens tested. dl.acm.org/doi/10.1145/3658664.3659652
- Corroborating negatives: Benford-on-GAI ~40% FN (Bond U.); CNN OOD collapse 99.8% -> 56-66%
  (PMC12711871) -- the frontier cliff is architecture-independent.
- Steganalysis niche: classical SRM/SPAM rich models never run on AI-image detection (open gap),
  but the same primitives fingerprint ISP/platform re-encodes (SNRCN2) -> expected de-confound
  death; deliberately skipped.
- "White-box"-branded 2025/26 papers (CoDA, CHROMA, GADNet, LAID) all hide learned backbones --
  excluded on inspection.
- Scorecard position after the sweep: capability bottom-quartile on frontier (a field-wide wall),
  evaluation rigor plausibly #1. The 0.59 frontier ceiling stands unchallenged externally.

## §6 Sequencing (2-3 days)
Day 1: P1 ablations -> single frozen shot -> log entry. Day 1-2: P2a/P2b probes. Day 2: P4 pull +
3 readouts. Day 3: P3a A/B + P3b spec.
