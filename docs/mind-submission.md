# Bounded MIND-shaped truth/prediction text

`evaluate-mind-submission` evaluates a strict local subset of the public MIND
challenge's `ID [array]` text shape. Its formulas follow the frozen MIND
`evaluate.py` reference: macro impression AUC, MRR, binary nDCG@5 and nDCG@10,
using `1 / rank` as the candidate score. AUC counts clicked/unclicked pairs in
the right order; MRR averages reciprocal ranks over clicked candidates within
each impression. An independent, hand-calculated two-impression oracle checks
these values. The frozen script is referenced for semantics, not copied.

This is deliberately **not** full official evaluator compatibility: official
scoring tolerates missing prediction lines and ignores the prediction array of
masked truth rows. We require one exactly aligned prediction row per truth row.
For masked truth `[]`, prediction may be `[]` or a bounded 1..N permutation;
the hidden candidate count cannot be checked from truth alone. Nonmasked labels
must contain both classes and predictions must be 1..N rank permutations.
These restrictions avoid implicit padding, ambiguous ties, division-by-zero groups and unnoticed
candidate loss. This local tool has not been run against licensed MIND challenge
test data and makes no leaderboard or official score claim.

The format is UTF-8 without BOM, LF-terminated, precisely `ID [1,0]` for truth
or `ID [2,1]` for predictions, with no spaces inside the compact JSON array.
IDs use `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}` and are unique and ordered identically
in both files. Truth values are JSON integers 0 or 1; ranks are JSON integers
from 1 to the complete candidate count with no repeats. Masked truth rows are
`ID []`; the corresponding prediction is `[]` or a dense permutation, and the
row is consumed but skipped in macro metrics. At least one mixed-label
unmasked row is necessary for evaluation. Each input text file is limited to
8 MiB, 10,000 lines, 8 KiB per line, 1,000 candidates per line and 200,000
candidates total; the JSON export inputs are limited to 16 MiB each. Limit
failures do not hash or parse an oversized direct-API source.

```bash
mkdir -p scratch
mosaicfeed export-mind-predictions \
  --impressions examples/mind_impressions.json \
  --scores examples/mind_scores.json \
  --truth examples/mind_truth.txt \
  --output scratch/prediction.txt
mosaicfeed evaluate-mind-submission \
  --truth examples/mind_truth.txt \
  --prediction scratch/prediction.txt \
  --output scratch/submission-evaluation.json
```

`export-mind-predictions` consumes MosaicFeed's lossless `impressions.json`
schema and exact long-form `[{"impression_id":"...","article_id":"...",
"score":0.5}, ...]` rows. IDs and candidate pairs must be unique, score
coverage exact, scores finite; highest score receives rank 1 and tied scores
follow source candidate order. The optional `--truth` must match impression
IDs, order, counts and known labels; `[]` truth rows produce `[]` predictions.
Without truth, every impression is ranked. The `clicked` fields in the existing
interchange format are never used to compute ranks. A caller producing a real
hidden-test submission should **omit** `--truth` to emit ranks for every row;
the masked-`[]` path is for this tool's strict local evaluation only. The
caller remains responsible for point-in-time feature and training provenance;
this exporter cannot prove that score inputs were generated without label or
future information leakage.

CLI reads bounded source snapshots, checks path aliases including hardlinks,
rechecks source bytes before publication and creates the output only when the
destination does not already exist. Reports give exact raw-byte SHA-256 values
for the read inputs and output. Hashes are join/reproducibility aids, **not**
licenses, privacy protection, signatures, or proof of source authenticity.
Raw truth and candidate rows stay in caller-controlled local files, but hashes
may enable correlation with known data; review reports before sharing. Concurrent
external file mutation after the second snapshot check cannot be ruled out by
this CLI, and cross-filesystem/crash durability is platform-dependent.

The example input is hand-written synthetic data, not licensed MIND content.
Obtain any real MIND data through its distributor under its terms; the package
does not download or ship it.
