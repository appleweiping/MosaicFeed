# MIND impression evaluation contract

MosaicFeed keeps two uses of MIND data separate:

- clicked candidates become dated `EventKind.CLICK` events for feed replay;
- the complete ordered candidate list and its binary labels become an immutable impression record for
  ranking evaluation.

That separation prevents an evaluation from silently treating the clicked subset as the candidate set.
`import-mind` reads local files only and writes `articles.json`, `events.json`, `impressions.json`, and
`metadata.json`. The metadata records SHA-256 digests of the exact `news.tsv` and `behaviors.tsv` bytes,
plus the normalized catalog availability instant and behavior-log UTC offset required to replay the conversion.
MosaicFeed never downloads or redistributes the MIND dataset.

## Score contract

The score file is a JSON list. Every record has exactly these fields:

```json
{"impression_id": "1", "article_id": "N1", "score": 0.25}
```

There must be exactly one finite score for every candidate in every imported impression. Missing or extra
impressions, missing or extra candidates, duplicate pairs, non-finite numbers, duplicate JSON keys, and
unknown fields are errors. Long form is deliberate: an object keyed by candidate could overwrite a
duplicate before the evaluator saw it.

## Metrics

Metrics are computed per impression and macro-averaged across impressions, matching the public MIND
evaluation protocol:

- AUC is the fraction of clicked–unclicked pairs whose clicked score is higher, with half credit for a tie.
- MRR is the mean reciprocal rank of all clicked candidates in the impression.
- nDCG@k uses binary gains and logarithmic discount, normalized by the ideal clicked-first ordering.

An impression must contain at least one clicked and one unclicked candidate. Without both classes its AUC
is undefined, so the complete evaluation is rejected instead of dropping that impression and changing the
evaluation population.

Scores are ordered descending. When scores tie, their source candidate order is retained. The public
NumPy-based reference evaluator does not specify a portable tie order, so parity claims should use unique
scores or acknowledge this deterministic clarification.

For labels `[1, 0, 1, 0]` and scores `[0.9, 0.8, 0.7, 0.6]`, the tests assert:

- AUC = `3 / 4`;
- MRR = `(1 + 1/3) / 2`;
- nDCG@2 = `1 / (1 + 1/log2(3))`;
- nDCG@4 = `(1 + 1/log2(4)) / (1 + 1/log2(3))`.

In addition to these golden cases, a fixed-seed test checks 200 untied random impressions against a
separately structured reference: rank-sum AUC plus ranked-label MRR and nDCG. That cross-check follows the
formulas in the commit-pinned public evaluator without making its unspecified tie ordering part of this
project's contract.

## Reproducibility boundary

The report fingerprints canonicalized impression labels and scores independently. It also records metric
schema version, cutoffs, averaging unit, candidate count, and tie rule. These fields make two reports
auditable; they do not prove that a model was trained without leakage. A study must still disclose its
training window, feature availability, hyperparameter selection, hardware, random seeds, and any filtered
impressions.

Primary references:

- [MIND dataset and terms](https://msnews.github.io/)
- [MIND paper](https://aclanthology.org/2020.acl-main.331/)
- [public MIND evaluator at `47fb985`](https://github.com/msnews/MIND/blob/47fb9852c97814d80ccf9c658d6b81f5c930b510/evaluate.py)
