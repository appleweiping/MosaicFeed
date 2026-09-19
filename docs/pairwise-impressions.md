# Impression-aware pairwise ranking

`PairwiseImpressionRanker` is a separate, opt-in objective. It consumes the
candidate-lossless `MindImpression` records produced by `import-mind` or
`write_mind_impressions`. A training pair consists of one clicked and one
displayed-but-unclicked candidate in the **same** impression. It does not
relabel unrelated events or undisplayed catalog items as negatives. Its score
is a raw linear logit for within-impression ordering, **not** a click
probability; the pointwise schema-1/2 model and default commands are unchanged.

## Local CLI workflow

Obtain MIND under its distributor's terms and run `import-mind` as documented
in the README. Make disjoint `train-impressions.json` and
`heldout-impressions.json` files in the exact JSON form written by that
command. Choose a cutoff such that all training records are at or before it
and all held-out records are strictly later. Splitting is caller-owned; the
trainer will not silently discard post-cutoff records.

```sh
mosaicfeed train-pairwise-model \
  --articles imported/articles.json \
  --impressions train-impressions.json \
  --held-out-impressions heldout-impressions.json \
  --partition local-train --cutoff 2019-11-13T00:00:00Z \
  --output pairwise-model.json

mosaicfeed rank-pairwise-model \
  --model pairwise-model.json --articles imported/articles.json \
  --training-impressions train-impressions.json \
  --impressions heldout-impressions.json \
  --scores-output heldout-scores.json \
  --report-output rank-report.json

mosaicfeed evaluate-pairwise-model \
  --model pairwise-model.json --articles imported/articles.json \
  --training-impressions train-impressions.json \
  --impressions heldout-impressions.json \
  --scores-output evaluated-scores.json \
  --output evaluation-report.json
```

The rank command writes one strict MIND long-form score row per candidate,
in source candidate order, including noncomparable all-positive/all-negative
impressions. The evaluation command writes those same rows, reloads them with
`load_mind_scores`, and then evaluates mixed-label impressions through
`evaluate_mind_impressions`. Its report gives skipped counts rather than
pretending AUC is defined for noncomparable impressions. Existing
`evaluate-mind --impressions ... --scores ...` also accepts a rank output if
all supplied impressions are mixed-label.

Reports and model state include the objective, raw-logit semantics, declared
partition and cutoff, SHA-256 hashes of source files, training example/history
fingerprints, and skipped counts. The model checks article and training file
hashes when ranking, and checks a declared held-out file hash if one was
given. Loading checks a versioned canonical-state SHA-256 checksum. A checksum
detects accidental corruption, not malicious replacement and re-signing.

## Temporal and resource contract

All records are sorted by `(occurred_at, impression_id)`. Every feature
vector at one timestamp is built from history strictly earlier than that
timestamp; only afterward are that group's clicked articles added to history.
Scoring held-out impressions uses verified training clicks only, never the
held-out labels or prior held-out clicks. Feature extraction uses the existing
seven non-text score components. In particular, this slice does **not** fit a
news-text vocabulary, neural encoder, or calibrated probability.

The trainer validates unique article and impression IDs, user IDs, candidate
membership and availability, split disjointness, and source provenance before
updating weights. It bounds catalog articles (100,000), impressions (10,000),
candidates per impression (256), feature cells (3 million), pairs (1 million),
epochs (100), SGD updates (5 million), and model state bytes (4 MiB), as well
as 5 million cumulative profile-history scans, 10 million candidate/history
topic visits, and numeric magnitudes. Input source files are capped at 64 MiB;
score output is separately capped at 256 MiB and may exceed the input cap.
No official MIND-small rows or benchmark scores are bundled or claimed here.
