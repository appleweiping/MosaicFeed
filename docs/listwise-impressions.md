# Impression-aware listwise ranking

`ListwiseImpressionRanker` is an opt-in local softmax ranker over the complete
displayed candidate list of each `MindImpression`. It does not invent negatives
from undisplayed news. For each mixed-label impression, clicked items share a
uniform target distribution. The model minimizes one cross-entropy over that
impression and performs one SGD update per impression per epoch. All-positive
and all-negative impressions are explicitly skipped during training; they are
still scored in rank output. Its scores are raw logits used only for
within-impression ranking, **not** calibrated click probabilities. Pointwise
and pairwise defaults and persisted model formats are unchanged.

After `import-mind`, prepare caller-owned disjoint training and post-cutoff
held-out `impressions.json` subsets in the exact imported record format. The
cutoff is inclusive for training and exclusive for held-out records. Run:

```sh
mosaicfeed train-listwise-model \
  --articles imported/articles.json \
  --impressions train-impressions.json \
  --held-out-impressions heldout-impressions.json \
  --partition local-train --cutoff 2019-11-13T00:00:00Z \
  --output listwise-model.json

mosaicfeed rank-listwise-model \
  --model listwise-model.json --articles imported/articles.json \
  --training-impressions train-impressions.json \
  --impressions heldout-impressions.json \
  --scores-output heldout-scores.json \
  --report-output rank-report.json

mosaicfeed evaluate-listwise-model \
  --model listwise-model.json --articles imported/articles.json \
  --training-impressions train-impressions.json \
  --impressions heldout-impressions.json \
  --scores-output evaluated-scores.json \
  --output evaluation-report.json
```

The rank command emits one MIND long-form score per displayed candidate in
source order. The evaluation command stages and reloads that score format,
computes MIND metrics on mixed-label held-out impressions, records skipped
counts, then publishes score and report together. Neither command uses held-out
clicks as history; changing the target labels cannot change their logits.

Training sorts `(occurred_at, impression_id)` and builds all feature vectors
at one timestamp from strictly earlier clicked history, then applies that
group's clicks to later profiles. Candidate order is canonicalized during
fitting; the output returns to original order. The model stores training
feature/history/catalog fingerprints, source-file hashes, versioned format and
state checksum. The checksum detects accidental corruption, not malicious
replacement and re-signing. Scoring requires an exact fitted catalog and
training-history match. It rejects overlapping or pre-cutoff target IDs.

Bounds match the pairwise data protocol: 100,000 articles, 10,000 impressions,
256 candidates per impression, 3 million feature cells, 100 epochs, 5 million
SGD updates, 5 million cumulative history scans, 10 million topic visits,
4 MiB model state, 64 MiB input source files, and 256 MiB score output. In
addition the listwise trainer caps comparable training candidates at one
million and rejects non-finite or out-of-range features/weights. This is a
local linear baseline, not an official MIND-small or neural-news result.
