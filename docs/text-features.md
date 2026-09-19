# Opt-in MIND news text features

The default `PointwiseLogisticRanker` remains the v0.5 seven-feature model and
persists schema 1. `train-click-model --text-features` enables schema 2: an
inspectable TF-IDF news vocabulary plus an eighth `text_affinity` feature.
`rank-click-model` loads either schema automatically. This is a lexical local
baseline, **not** a neural news encoder or an official MIND-small result.

MIND's `news.tsv` has title, category, and subcategory fields. The existing
adapter places category/subcategory in `Article.topics`; the encoder uses
`title:<token>`, `category:<token>`, and `subcategory:<token>` features. It
case-folds Unicode word tokens, uses title term frequency `1 + log(count)`,
document frequency from the frozen training vocabulary, and
`idf = 1 + log((1 + N)/(1 + df))`. Each article vector is L2-normalized.
Unknown words are ignored; an entirely out-of-vocabulary article gets a zero
vector rather than an invented similarity. A missing title produces no title
tokens. MIND import internally uses `[missing MIND title]` only to satisfy the
legacy `Article` identity invariant; `articles.json` writes the original empty
title with `title_missing: true`, and loading recreates the marker. Missing
category retains the v0.5 topic behavior: the subcategory remains the only
topic when present, and `uncategorized` is used only when both fields are
empty. Explicit `mind_category` and `mind_subcategory` fields retain the two
source slots even when their values are identical, so both lexical namespaces
remain available without changing legacy topic scoring. `category_missing`
and `subcategory_missing` record empty source fields. The exact source bytes remain fingerprinted
and retained by `MindDataset`.

For a useful multi-document corpus, pass a separate
`--text-vocabulary-articles` JSON snapshot that you declare to be the
**training-news partition**. Its articles must exactly match entries in the
training catalog and all be available by the **first** eligible training event.
The model freezes vocabulary and IDF from only this snapshot; later event IDs,
held-out news, and scoring candidates never enlarge it. The persisted encoder
records `source_kind`, the first-event cutoff, and a digest of the exact news
content, so the caller can audit which declared snapshot produced the model.
The snapshot must be assembled without using future interaction labels or a
test partition; these external split assumptions cannot be inferred from a
generic article JSON file. If no snapshot is passed, text mode falls back to
the first eligible event's article only. This conservative fallback avoids
lookahead but has constant IDF because it has one document.
At each training event, its feature is
computed before that event updates the user's text history. At ranking time,
only the user's events visible at `as_of` and after their articles' publication
contribute to the signed history vector. Its weights are the configured event
signal times event weight; cosine with the candidate news vector is clamped to
`[-1, 1]`. Candidate news published after `as_of` is never ranked. New news
can project onto existing vocabulary without an interaction or retraining.

Text mode caps articles, events, title/category characters, title tokens,
vocabulary entries, and aggregate feature occurrences. Invalid or non-finite
state is rejected. Schema 2 embeds the complete text encoder and a SHA-256
checksum over the canonical model state; model files are capped at 16 MiB.
The checksum detects accidental corruption, not a malicious writer who can
recompute it. The old schema 1 has unchanged field names, semantics, and
serialized shape.

```shell
mosaicfeed train-click-model --articles articles.json --events events.json \
  --as-of 2019-11-16T00:00:00+00:00 --output click-text.json \
  --text-features --text-vocabulary-articles train-news.json \
  --text-max-vocabulary 4096
mosaicfeed rank-click-model --model click-text.json --articles articles.json \
  --events events.json --user U1 --as-of 2019-11-17T00:00:00+00:00
```

`--as-of` and the MIND catalog availability timestamp are caller-declared;
MIND does not supply per-article publication times. Treat the availability
assumption as part of the experiment protocol, not evidence of real publication
order. The v0.5 importer retains even impressions earlier than the declared
catalog time so the original timestamps remain auditable; fitting rejects
clicked training events that predate article availability, and ranking filters
news not yet visible at its `as_of`. Undated history remains unusable as timed
text feedback.
The official MIND-small frozen-data and neural-news baseline milestones
remain separate future work.
