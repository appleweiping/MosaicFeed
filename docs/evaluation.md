# Evaluation guide

MosaicFeed reports complementary metrics because no single scalar captures a feed's behavior.

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
