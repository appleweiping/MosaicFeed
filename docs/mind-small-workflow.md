# Offline MIND-small train/dev workflow

`scripts/run_mind_small.py` is a reproducible **local sampled baseline**, not a
MIND leaderboard submission or a neural-news model. The official MIND-small
release has **training and validation only**, not a separate small test set.
Each split contains `news.tsv`, `behaviors.tsv`, `entity_embedding.vec`, and
`relation_embedding.vec`. This workflow requires that four-file shape but
only reads the two TSVs; it does not use entity embeddings. [The official
description](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md)
defines the format. The [official download page](https://msnews.github.io/)
requires accepting Microsoft Research License Terms; the data is intended for
non-commercial research. Obtain it yourself from that page after reading the
terms. The tool has no network calls or packaged dataset, and its synthetic
test fixture is not Microsoft data.

## Run

Record SHA-256 digests of the **exact files** you obtained; the tool does not
provide or assume official fixed digests. For a ZIP, record both the archive
digest and the digests of its two contained TSVs. For an extracted directory,
record the two TSV digests; omit the archive option. Keep those files private
and outside the Git repository. On a Unix shell, for example:

```sh
sha256sum /private/MINDsmall_train.zip /private/MINDsmall_dev.zip
unzip -p /private/MINDsmall_train.zip news.tsv | sha256sum
unzip -p /private/MINDsmall_train.zip behaviors.tsv | sha256sum
unzip -p /private/MINDsmall_dev.zip news.tsv | sha256sum
unzip -p /private/MINDsmall_dev.zip behaviors.tsv | sha256sum
```

Substitute your observed lowercase hashes below:

```sh
python scripts/run_mind_small.py \
  --train /private/MINDsmall_train.zip \
  --validation /private/MINDsmall_dev.zip \
  --train-archive-sha256 "$TRAIN_ZIP_SHA" \
  --validation-archive-sha256 "$DEV_ZIP_SHA" \
  --train-news-sha256 "$TRAIN_NEWS_SHA" \
  --train-behaviors-sha256 "$TRAIN_BEHAVIORS_SHA" \
  --validation-news-sha256 "$DEV_NEWS_SHA" \
  --validation-behaviors-sha256 "$DEV_BEHAVIORS_SHA" \
  --catalog-published-at 2019-01-01T00:00:00Z \
  --behavior-utc-offset -8 \
  --max-train-impressions 1000 \
  --max-validation-impressions 1000 \
  --epochs 5 --seed 17 --output /private/mind-small-local-run
```

Run in the project environment (`uv run python scripts/run_mind_small.py`)
or from an installed package (`python -m mosaicfeed.mind_small_workflow`).
The output path must be new. It contains `model.json`, `scores.json`, and
`report.json`; it never contains the TSVs or original ZIPs. Do not publish the
outputs without checking whether derived IDs, scores, or other information
may be restricted by the dataset terms.

For extracted sources, point `--train` and `--validation` to their respective
directories and omit `--*-archive-sha256`. Both directories must contain all
four named files as regular files. ZIPs must contain those exact four names at
the archive root; unexpected or duplicate members, encrypted members, and
oversized inputs fail closed. The declared TSV digests are checked against
the bytes actually parsed. ZIP members are read from the exact immutable
archive byte snapshot whose digest was checked. Digest matching establishes
replay identity, **not** proof that
the files came from Microsoft's distributor.

## Selection and leakage contract

The runner parses both entire splits, requires every training behavior to be
strictly before every validation behavior, and checks the declared catalog
time is no later than the first training impression. It selects the earliest `(time, impression ID)` prefix
of eligible records in each split: 1,000 train and 1,000 validation impressions
by default, at most 2,000 each. The existing model accepts up to 256
candidates per selected impression, so larger raw slates are excluded in full
before this ordering and counted in the report; no candidate inside a selected
slate is dropped. It prefixes impression IDs with `train:` and `dev:` to avoid
cross-file ID collisions while preserving candidate order and labels. The
model fits only on the selected training labels/clicks and sees just the
validation impression IDs for overlap checks. Validation labels are used only
after scoring, for AUC/MRR/nDCG; their source hash is recorded in the report,
not in the model. An adversarial test changes validation labels and confirms
the model bytes and raw scores stay identical.

The baseline's seven existing non-text features use training-impression
clicks with a strict same-timestamp barrier. **MIND's separate undated user
history field is not used.** The model must receive a catalog covering both
train and dev candidates, so dev-only article metadata is treated as a
transductively available catalog during fitting. No dev click label enters
model fitting or profile state, but this catalog assumption is not proof of
real publication-time availability: MIND lacks publication timestamps, and
the caller-declared catalog time is a simplifying assumption. IDs present in
both splits must have identical metadata. These limits prevent a misleading
claim of a fully train-only feature protocol.

The baseline is bounded by 128 MiB per TSV, 512 MiB per ZIP, 250,000 raw
news rows and 250,000 raw impressions per split, 64 KiB per TSV row,
4,096 raw candidates per impression, 2 million candidate tokens and 20 million
history IDs per split, 100,000 selected train+dev candidates, 128 MiB score
output, the selected-prefix limits, and the underlying
pairwise model's article/candidate/pair/update/work ceilings. Bare CR-only
rows are rejected; LF and CRLF are accepted. A selected-sample limit may be
resolved by reducing the sample; a raw-input limit requires a conforming source
or a different importer. The output report records physical and parsed source
row counts, selected counts, source hashes and byte lengths,
catalog/timezone assumptions, split cutoff, training fingerprints, model and
score hashes, skipped noncomparable validation impressions, and macro metrics.
The model's checksum and source fingerprints make accidental changes
detectable; they are not cryptographic signatures. Output publication uses
exclusive directory creation and per-file no-replace hard links. Treat the
bundle as complete only when `report.json` exists (linked last) and its model
and score hashes match. An interrupted run may leave a partial directory;
the runner never overwrites it automatically.

All official-data metrics must remain unpublished/unclaimed until someone
actually runs this against user-obtained data and audits provenance, protocol,
and license. Even then, the earliest-prefix sample and linear baseline must
be labeled as such, not presented as a full MIND-small result.
