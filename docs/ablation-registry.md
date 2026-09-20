# Bounded local ablation registry

`mosaicfeed run-ablation` evaluates declared one-factor removals from a frozen
`FeedConfig` against the unchanged baseline. It uses the same point-in-time
leave-last-out users as `mosaicfeed benchmark` and the same user resample in
each bootstrap draw for every variant. It is an offline diagnostic; it does
not train a model, select a checkpoint, estimate a causal intervention, or
claim performance on official MIND-small.

```bash
mosaicfeed run-ablation \
  --articles examples/articles.json \
  --events examples/events.json \
  --plan examples/ablation-plan.json \
  --registry scratch/ablation-registry
```

The plan is strict UTF-8 JSON with schema version 1, a timezone-aware `as_of`,
an ordinary `FeedConfig` object, `k`, and one to seven distinct names from
`without_interest`, `without_freshness`, `without_quality`, `without_novelty`,
`without_popularity`, `without_exploration`, and
`without_slate_diversity`. `bootstrap_samples`, `confidence`, and `seed` are
optional; they default to 100, 0.95, and 17. Each named weight ablation sets
only that weight to zero; if this leaves no positive ranking weight, the run
fails instead of silently changing another parameter. The slate ablation
sets MMR relevance weight to one, calibration weight to zero, and relaxes the
source cap to at least the catalog size. It retains the declared rerank mode
but removes its diversity/calibration pressure.

The output is one `<experiment_id>.json` record under the registry directory.
The ID hashes a domain-separated triple of the exact article, event, and plan
byte-stream SHA-256 digests together with the declared runner protocol
`mosaicfeed-ablation-v1`. Its record contains the protocol, those three hashes, the
normalized dataset fingerprint, the complete plan and effective variant
configurations, ranking/diversity/exposure metrics, percentile confidence
intervals, and paired baseline-minus-ablation intervals. No raw user IDs or
event rows are copied into the record. A repeat with identical inputs has
the same ID and fails with “already registered”; records are never
overwritten. A temporary file is flushed and linked into place as a single
no-replace operation. On POSIX the containing directory is also fsynced.
This is an integrity/reproducibility aid, not cryptographic authentication or
a guarantee against storage-hardware failure. An error after the link but
before cleanup may leave a complete registered record; inspect the exact ID
path before retrying. Verify reported findings by
rerunning with the same source bytes and comparing records. The Python API
`verify_experiment_record` performs that complete byte-for-byte replay check,
including the content-addressed filename; a corrupted or changed record
returns false.
Any future semantic change to parsing, variant construction, holdouts,
metrics, or uncertainty computation must introduce a new runner-protocol
revision before publishing records, so identical inputs under different
algorithms never collide. The protocol is not a source-code commit hash;
archiving the exact release alongside the record remains the caller's job.

The runner captures each path only once as bounded immutable bytes before
parsing and hashing. Each article/event source is limited to 16 MiB, the plan
to 64 KiB, and the rendered record to 4 MiB. Article and event rows are each
limited to 50,000; active users to 5,000; `k` and baseline `size` to 100;
bootstrap draws to 2,000. Global evaluation work accounts for repeated
candidate selection and topic comparisons; bootstrap work accounts for
per-draw eligible-catalog scans, exposed slates, and exposure sorting. Both
estimates are checked before the first variant runs. Inputs may be JSON or JSONL, with
strict duplicate-field and non-finite-number rejection. The registry writer
is optional in the Python API: `run_ablation_experiment` returns a typed
`ExperimentRun`, and `write_experiment_record` publishes it.

With one evaluable user, bootstrap intervals necessarily collapse to a
single value. That is not evidence of narrow population uncertainty. Offline
leave-last-out also cannot remove exposure bias or confounding. Supplied
article quality and popularity fields may themselves have been computed from
events after the holdout; this runner cannot certify their point-in-time
feature provenance. The separate
logged-policy and declared-cohort audits provide different diagnostics; this
registry does not replace them.
When source events contain logged propensities, each variant's evaluation
payload also includes the pre-existing logged-policy summary over all events
up to `as_of`, not just leave-last-out ranking holdouts. Its fixed 1,000-draw
clustered interval is included in the evaluation work budget and is not
controlled by the plan's `bootstrap_samples`.
