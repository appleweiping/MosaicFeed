# Evaluation guide

MosaicFeed reports complementary metrics because no single scalar captures a feed's behavior.

## Policy comparisons and uncertainty

`mosaicfeed benchmark` evaluates the configured MosaicFeed policy, the same relevance score without
slate constraints, a popularity baseline, and a recency baseline on identical temporal holdouts. It
retains per-user observations, reports deterministic percentile-bootstrap confidence intervals, and
uses the same user resample for every policy when computing paired deltas. Catalog coverage is
recomputed from each draw's slates and union of eligible catalogs, and exposure Gini is recomputed from
that draw's exposures; neither is averaged as if it were a user-level metric. Logged IPS CTR remains a
point diagnostic because the report does not retain the
per-user propensity log required for a paired interval. The report also records a SHA-256 fingerprint
of every ranking-relevant input field.

The interval only describes resampling uncertainty in the observed replay population. It does not
correct exposure bias, missing-not-at-random feedback, catalog censoring, or policy-dependent user
behavior. Runtime is included as a diagnostic and is not comparable across machines without an
external benchmark protocol.

## MIND conversion boundary

The local MIND adapter uses only clicked impressions as positive events. It does not treat unclicked
impressions as dislikes, and it records but does not assign invented timestamps to history entries.
Because MIND lacks publication timestamps and publisher identities, callers must declare one catalog
availability time and category is explicitly labeled as a source proxy. These limitations must remain
visible when interpreting freshness, source-diversity, and popularity-baseline results.

| Metric | Range | Interpretation |
|---|---:|---|
| NDCG@k | 0–1 | Rewards recovering the held-out positive item near the top |
| Hit rate@k | 0–1 | Fraction of evaluated users whose holdout appears in the top `k` |
| MRR@k | 0–1 | Mean reciprocal position of the first relevant item |
| Intra-list diversity | 0–1 | Mean pairwise Jaccard topic distance within each slate |
| Source diversity | 0–1 | Distinct sources divided by returned items |
| Catalog coverage@k | 0–1 | Fraction of temporally eligible catalog items exposed in top-`k` slates |
| Exposure Gini | 0–1 | Concentration among items that received at least one exposure |
| Logged IPS CTR | 0–1 | Self-normalized inverse-propensity click/like diagnostic |

## Read the numbers together

MMR can trade a small amount of NDCG for greater intra-list diversity. A strict source cap can increase source diversity while shortening slates. High catalog coverage can coexist with concentrated exposure, so coverage and Gini should be inspected together.

## Limits of leave-last-out

The final positive event reflects the logging policy that exposed it. Unobserved items are not verified negatives, propensities may be wrong, and user preferences can change. Use this evaluator to catch regressions and compare controlled configurations—not as a substitute for online validation.

Self-normalized IPS uses events at or before the evaluation `as_of` that have an explicit propensity.
Missing propensities are skipped. The implementation rescales inverse weights before summing to avoid
floating-point overflow, but very small propensities still create high statistical variance; production
analysis should clip weights deliberately and report uncertainty.
