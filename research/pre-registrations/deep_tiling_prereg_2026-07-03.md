# Pre-registration — deep tiling vs deployed multi-crop (2026-07-03)

Question: the analyzer reads a center crop (median-of-5 at >=1024). Would
exhaustively tiling the whole image and aggregating per-tile scores ("deep analysis")
give better verdicts?

## Data (all held-out, never trained on, all >=1024 native)
- REAL: bfree_audit_nativeRAISE/real (500) - bfree_test_nativeRAISE/real (500)
  (bfree_audit/real was found to be mostly <1024 px and is EXCLUDED before scoring --
  mixing crop geometries across classes would confound geometry with the label.
  Both real sets are RAISE splits: one real corpus, so conclusions are scoped to
  RAISE-like reals at 1024; the arm-vs-arm comparison uses identical images per arm
  and is unaffected.)
- AI:   bfree_audit/flux (500) - bfree_audit/sd35 (500) - bfree_test/flux (500) - bfree_test/sd35 (500)
- Pairs are split-matched: audit reals vs audit AI arms, test reals vs test AI arms -- 4 pairs.

## Protocol
Per image: 9 crops of 512px at stride 256 (anchors 0/256/512 in each axis). This superset
contains every arm:
- **center-1**   -- crop at (256,256) only (deployed behaviour <1024)
- **median-5**   -- center + 4 corners, median of v2 z (DEPLOYED, frozen)
- **grid-4**     -- 4 non-overlapping corner tiles, median (full coverage, no overlap)
- **dense-9**    -- all 9 crops, median (full coverage + overlap = the "deep analysis" arm)
Aggregator for the primary comparison is the median (matches deployed logic).
Mean / max / q75 recorded as exploratory only.

## Metric and decision rule (fixed before any scores are seen)
AUC(real vs AI) per split-matched (real x AI-arm) pair -- 4 pairs -- using the v2 channel z,
identical images across arms. ADOPT a deep arm for the product only if it beats
median-5 by >= +0.02 mean AUC across the 4 pairs AND does not lose by > 0.01 on any
single pair. Otherwise the deployed median-5 stays and the result is documented.
If adopted, the real-photo null and verdict bands must be recomputed under the deep
protocol before it touches the verdict layer.

## Notes
aigen2026 is 512px-only and cannot support tiling arms; excluded. At 1024 the corner
crops tile the image exactly, so grid-4 == "cover everything without overlap" and
dense-9 == "cover everything with 2x overlap"; larger native images were not available
on-disk for a sparser-coverage regime.

## Addendum (2026-07-03, pre-registered before scoring) -- whole-image resize arm
Follow-up question: instead of cropping, scan the ENTIRE image. Arm: aspect-preserving
downscale of the full image to 512px (Lanczos; corpora are 1024x1024 so this is exactly
the whole image at half scale), then the standard q75 substrate + v2 z. Same 4 split-matched
pairs, same decision rule vs deployed median-5 (adopt at >= +0.02 mean, no pair worse by
> 0.01). Physics prediction, stated in advance: resampling launders pixel-scale generator
artifacts, so AUC should DROP relative to cropping. The prediction is not a result; measuring.
