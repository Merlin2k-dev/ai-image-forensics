# Research record

This folder is the experiment log behind the detector. I kept it the way I kept it while
working: one entry per experiment or model-affecting decision, dated, with the numbers as
they came out of the runs.

Every change that could touch the model went through a measurement first, and most ideas
did not survive it. Feature families, fusion schemes, retrains, and inference tricks were
each tested against pre-registered acceptance bars (frozen-prediction evaluation,
cross-corpus real-vs-real gates, per-generator breakdowns, no-regression guards), and the
majority were rejected. The rejections are recorded with the same care as the wins,
because the negative results are most of what I learned.

- `experiment-log.md` — the chronological log. Entries preserve their original dates and
  unedited numbers (AUC values, confidence intervals, thresholds, sample sizes,
  per-generator results) and the verdict as recorded at the time: ADOPTED or REJECTED,
  and why.
- `pre-registrations/` — documents written *before* running key experiments:
  pre-registered protocols with fixed decision rules, data-source surveys, and
  verification notes. Nothing has been re-run, re-rounded, or retrofitted in hindsight.
