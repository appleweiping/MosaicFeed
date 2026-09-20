# Bounded training experiments

`run-training-experiment` runs the existing pairwise and listwise impression
rankers on a caller-owned, MIND-shaped article/train/validation split. It fits
every declared candidate on **train only**, restores every checkpoint, scores
the same post-cutoff validation impressions, and selects the maximum declared
metric (`auc`, `mrr`, `ndcg@5`, or `ndcg@10`). Exact metric ties select the
lexicographically smallest candidate ID. The output is one immutable,
content-addressed JSON registry record containing every candidate checkpoint,
metric report, source SHA-256, canonical split fingerprints, selected ID, and
the versioned plan. The reader validates internal structure and *all*
checkpoint hashes/states before returning the selected model. Only replay with
the exact source bytes verifies metric provenance and selection outcomes by
rerunning training and comparing the complete canonical record byte-for-byte.
The reader checks split-hash shape and work-bound range, but cannot establish
that either matches the original rows without those source bytes. Record
digests detect accidental edits; they are not signatures or an authenticity
guarantee against someone able to rewrite the record.

Run the tiny **hand-written synthetic example** from the repository root:

```shell
mosaicfeed run-training-experiment \
  --articles examples/training_experiment_articles.json \
  --train examples/training_experiment_train.json \
  --validation examples/training_experiment_validation.json \
  --plan examples/training_experiment_plan.json \
  --registry ./training-experiment-registry
```

On PowerShell, put the command on one line or use PowerShell backticks instead
of shell backslashes. A second identical run refuses to overwrite its record.
Python callers may use `run_training_experiment(articles_bytes, train_bytes,
validation_bytes, plan_bytes)`, `write_training_experiment_record(directory,
record)`, `verify_training_experiment_record(path, ...)`, and
`read_selected_checkpoint(path)`.

The plan must contain schema version 1, a timezone-aware cutoff, a full or
partial `FeedConfig` mapping, a selection metric, and 2–8 uniquely named
candidate specifications. Each candidate specifies `id`, `objective`
(`pairwise` or `listwise`), `epochs`, `learning_rate`, `l2`, and `seed`. Both
train and validation files use the strict `write_mind_impressions` JSON shape.

Safety limits: 8 MiB per data source, 64 KiB plan, 2,000 articles, 200 train
impressions, 100 validation impressions, 16 candidates per impression, 20
million conservative aggregate work units, and 4 MiB output record. Every
train timestamp must be at/before the cutoff; every validation timestamp must
be after it; train IDs and validation IDs must be disjoint. Validation rows
must have both clicked and unclicked candidates for AUC. No validation labels
enter `.fit()`; only held-out IDs are passed to the model, while validation
labels are used only for metrics after scores are produced.
The caller must also ensure article `quality` and `popularity` values were
known at each impression's timestamp. This runner checks article publication
time but cannot infer when those derived features were computed; values
aggregated from later interactions would leak future information into training
or validation.

This is a local model-selection demonstration, **not** an official MIND-small
benchmark, general-purpose hyperparameter tuner, neural news encoder, or
unbiased estimate on an untouched test split. Reusing the validation metric to
claim final model quality would be selection bias. Official MIND data are not
bundled, fetched, or synthesized as if official; use only data you are
licensed to handle. Keep a separate, untouched test set for final assessment.
