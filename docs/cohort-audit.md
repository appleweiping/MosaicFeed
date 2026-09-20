# Declared-cohort feed audit

`mosaicfeed audit-cohorts` measures group-level differences on the same temporal
leave-last-out replay used by `evaluate`. The input is a strict UTF-8 JSON
object mapping **every user active at `--as-of`** to one caller-declared cohort.
It includes users later skipped for having no positive holdout. Duplicate keys,
missing or extra users, invalid labels, overlarge inputs, and groups with fewer
than `--minimum-group-size` *evaluable* users cause the command to fail. No
demographic attributes are inferred.

```bash
mosaicfeed audit-cohorts \
  --articles examples/articles.json --events examples/events.json \
  --cohorts examples/cohorts.json --as-of 2026-08-30T12:00:00Z \
  --k 3 --minimum-group-size 1 --bootstrap-samples 100 \
  --output cohort-audit.json
```

The committed data and size-one threshold above are **synthetic smoke inputs**.
For a real audit, declare cohorts before inspecting outcomes, set a meaningful
minimum group size, and handle the labels and source mapping as sensitive data.
The CLI reads at most 16 MiB each for article and event sources, 4 MiB for the
cohort mapping, and 64 KiB for configuration. The audit accepts at most 100,000
article or event rows, 10,000 active users, 64 declared cohorts, and five
million user-times-row evaluation work units; excessive inputs fail before
evaluation or report output. These are safety limits, not performance claims.
The output does not directly list user IDs; its mapping fingerprint is an
integrity aid, **not** an anonymization guarantee. It records content fingerprints, the temporal
cutoff, per-cohort active/evaluable/skipped counts, seeded user-bootstrap
percentile intervals, and the
maximum-minus-minimum cohort mean for NDCG, hit rate, reciprocal rank,
intra-list diversity, and source diversity. A larger gap indicates unequal
observed outcomes, not necessarily unfair treatment; the metric direction
matters. A shared seed gives deterministic, label-specific draws.
With only one evaluable cohort, gaps are `null`, not falsely zero.

Topic calibration is the smoothed reader-to-slate KL divergence. Each reader
profile uses only interactions **strictly earlier** than their held-out positive
event; tied and later events cannot inform it. Readers without positive prior
topic history or an exposed slate have no defined calibration observation, not
a perfect zero-error score. A cohort receives a calibration interval only if
at least the declared minimum number of its users have such observations.
The calibration gap is `null` unless every declared cohort meets that minimum.
The number of eligible calibration observations is reported even when the
interval is withheld.

This is an observational diagnostic, not a causal fairness test, a protected-
class inference procedure, an online experiment, or evidence of official MIND
benchmark parity. Selection bias, unequal cohort sizes, unavailable labels,
and imperfect exposure logs remain limitations. The audit does not claim
statistical significance merely because two percentile intervals differ.
