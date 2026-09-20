# Declared-cohort policy frontier

`compare-cohort-policies` compares 2-8 caller-declared `FeedConfig` policies on
the same frozen articles, events, user-to-cohort mapping, `as_of`, explicit `k`,
and leave-last-positive-out user holdouts. It builds on the single-policy
[`audit-cohorts`](cohort-audit.md) calculation. The comparison is observational:
it does not infer protected traits, establish causal fairness, or certify an
online deployment decision.

```bash
mosaicfeed compare-cohort-policies \
  --articles examples/articles.json --events examples/events.json \
  --cohorts examples/cohorts.json \
  --plan examples/cohort-policy-plan.json \
  --output cohort-policy-frontier.json
```

The committed inputs are synthetic. A real evaluation should predeclare
cohorts, policies, constraints, `as_of`, and a meaningful minimum evaluable
group size before viewing outcomes. Cohort labels and the mapping may be
sensitive. The report excludes user IDs but hashes are integrity evidence,
**not** anonymization.

The plan is strict UTF-8 JSON with `schema_version: 1`; duplicate fields,
unknown fields, non-finite numbers, duplicate policy names, and malformed
configs fail. Required fields are `as_of`, `k`, and `policies` (each with a
short lowercase `name` and `FeedConfig` object). Optional fields are
`minimum_group_size`, `bootstrap_samples`, `confidence`, `seed`, and
`hard_constraints`. Bounds are explicit point-estimate gates:
`minimum_ndcg`, `minimum_worst_cohort_ndcg`, `maximum_ndcg_gap`,
`maximum_topic_calibration`, `maximum_topic_calibration_gap`. Omitted bounds
do not filter. A bound requiring missing evidence fails, never passes as zero.

The five Pareto objectives are overall user-weighted NDCG and worst-cohort
mean NDCG (maximize), plus the max-minus-min cohort NDCG gap, user-weighted
topic-calibration error, and max-minus-min cohort calibration gap (minimize).
The cohort audit still supplies per-group counts and deterministic bootstrap
intervals; **Pareto dominance uses point estimates only**, not confidence
intervals or significance tests. Every objective must be observed and every
hard bound must pass to enter the dominance comparison. Consequently a
single-cohort run has `null` gaps and an empty frontier; missing calibration
in even one cohort also excludes that policy, rather than making it look
perfect. `frontier_eligible` means comparable, not necessarily on the
frontier. Equal five-dimensional vectors are represented by the
lexicographically smallest policy name, with other equal policies listing it
in `dominated_by`. All comparisons are exact floating-point comparisons;
there is no hidden tolerance or weighted-sum score.

`holdouts_sha256` seals the sorted selected `(user, article, holdout time)`
triples, without exposing those triples. `dataset_sha256`, `cohorts_sha256`,
and `plan_sha256` seal canonical semantic inputs. When invoked from the CLI,
`source_sha256` also contains hashes of the exact bounded raw article, event,
cohort, and plan bytes read once before parsing. The Python API accepts
optional `source_sha256` as **caller-declared** metadata and validates its
shape; it cannot authenticate those digests against already parsed objects.
Use the CLI if raw-file provenance is needed. Changing only source whitespace
changes the raw hash, not the semantic hash.

The CLI caps article/event inputs at 16 MiB each, cohorts at 4 MiB, plan at
64 KiB, report at 4 MiB, and evaluates at most eight policies. It checks a
total policy-multiplied budget of eight million user-times-input rows and
eight million bootstrap units before evaluation. Outputs require an existing
parent directory and are published with a same-directory fsynced temporary
file and no-replace hard link: an existing output, including a symlink, is
never overwritten. Failure cleans up staging. Do not interpret these limits
as throughput guarantees.

This remains an offline temporal replay, not official MIND benchmark parity.
It cannot correct selection bias, missing exposure labels, or self-selected
cohort membership. Repeatedly tuning policies to the same holdouts risks
overfitting, even when all hashes match.
