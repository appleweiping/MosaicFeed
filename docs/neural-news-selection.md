# Local neural-news checkpoint selection

`run-neural-news-selection` runs the released [neural-news baseline](neural-news.md)
on the same immutable caller-owned article, train and validation bytes for a
baseline and one to five declared one-factor ablations. It fits each candidate
using training labels and only uses validation labels to measure and choose a
checkpoint. This is a local experiment protocol, not NRMS, an official MIND
benchmark, a causal comparison, or an estimate on an untouched test set.

The example plan is [`examples/neural_news_selection_plan.json`](../examples/neural_news_selection_plan.json).
Its strict versioned fields are: a timezone-aware cutoff, a valid baseline
`NeuralNewsConfig`, one to five distinct IDs and fields, a selection metric
(`auc`, `mrr`, `ndcg@5`, or `ndcg@10`), and bootstrap samples and seed. Each
ablation changes exactly one of `dimension`, `epochs`, `learning_rate`,
`max_vocabulary`, or `max_history`. The plan and each source are bounded;
the sum of all candidate work bounds is capped at 60 million coordinate-work
units before publication. All
candidate models, validation scores, metrics and work bounds are retained.
Equal selection metrics prefer the baseline, then the lexicographically first
ablation ID. Metrics are per-impression averages. The paired diagnostic is the
mean of each candidate's per-impression validation-metric difference from the
baseline, with a deterministic impression-resampling percentile interval.
These intervals are descriptive, not a multiplicity-adjusted significance
claim or uncertainty over users, time, retraining seeds, or new data.

```bash
mosaicfeed run-neural-news-selection \
  --articles examples/training_experiment_articles.json \
  --train examples/neural_news_train.json \
  --validation examples/neural_news_validation.json \
  --plan examples/neural_news_selection_plan.json \
  --registry neural-selection-registry
```

The command creates `<experiment_id>.json` in the registry without replacing
an existing file. The ID is derived from exact SHA-256 source hashes and the
protocol. The file contains an outer digest, all candidate checkpoints, and
their individual state digests. `read_selected_neural_checkpoint(path)` checks
the internal schema, plan, selected metric, state and work consistency before
returning a ranker. **An internal read does not authenticate its source data or
recalculate reported validation metrics.** For that, provide the exact four
source byte strings to `verify_neural_selection_record(path, articles, train,
validation, plan)`: it reruns every candidate and requires byte-for-byte
equality with the saved record. Keep those source snapshots if independent
replay matters. Neither hash nor replay proves ownership, dataset licensing,
or that the chosen validation result generalizes.

Changing validation labels may alter metrics and the chosen checkpoint but
must not change any fitted candidate state or raw score. Validation
impressions must be strictly after the declared cutoff, and training
impressions cannot cross it; see the neural-news contract for the exact
temporal and earlier-click-history semantics.
