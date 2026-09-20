# Bounded local neural-news baseline

`run-neural-news` trains a small, genuine parameterized news-title model from
caller-owned MIND-shaped impressions. It uses no external deep-learning package,
downloaded checkpoint, licensed MIND data, or outbound network call. The bundled
example is hand-written synthetic data; this is **not** NRMS, an official MIND
implementation, a leaderboard result, or a production recommender.

```shell
mosaicfeed run-neural-news \
  --articles examples/training_experiment_articles.json \
  --train examples/neural_news_train.json \
  --validation examples/neural_news_validation.json \
  --cutoff 2026-01-04T00:00:00Z --dimension 8 --epochs 4 \
  --output neural-news-local.json
```

For a title, case-fold Unicode alphanumeric tokens and average their trainable
embedding vectors, then apply coordinate-wise `tanh`. A missing or out-of-vocabulary
token has a fixed zero vector. The user representation is a trainable global bias
plus the mean of encoded titles clicked in **strictly earlier training
impressions**. The candidate's raw logit is its title vector's dot product with
that user representation. Binary cross-entropy on the complete displayed
training slate backpropagates into both title embeddings and the bias. Each
impression updates once; the other candidates in that same impression cannot
enter its history. Independent finite-difference tests check gradients both
without prior clicks and through a history-only title token.

The vocabulary is built only from distinct **training candidate** titles. A
validation-only word is OOV, never used to fit vocabulary or parameters. Training
impressions must be at or before the declared aware cutoff; all validation
impressions must be after it, with distinct IDs. The same-timestamp training
group is scored against the history snapshot that existed before that timestamp.
The standalone fit API also bounds and validates held-out impression IDs before
checking overlap, including type and length, so malformed IDs fail as validation
errors rather than leaking a raw hash/type error.
After training, validation scoring uses only frozen training clicks for each user:
it never appends validation clicks, even across later validation impressions.
Validation labels are consumed solely by the existing AUC/MRR/nDCG evaluator.
A test flips those labels and asserts identical model state and raw candidate
scores. Catalog articles must be published by each impression time. As MIND
does not provide genuine publication timestamps, a MIND import uses the caller's
declared catalog availability time; this is a transductive simplifying
assumption, **not** a guarantee of historical publication availability.

The JSON output contains the complete versioned model, every validation
candidate score in input order, metrics, exact source-byte SHA-256 hashes, a
model-state SHA-256, the cutoff, and a conservative training-work upper bound.
Hashes aid replay identity and accidental-corruption detection; they are not
cryptographic authentication of dataset origin or an untouched-test estimate.
The model reader validates schema, shapes, finite/bounded weights, vocabulary,
history, and work limits. Scores are raw logits, not calibrated probabilities.
The standalone `score_impressions()` method accepts a caller-provided catalog:
the saved model state does **not** prove that article titles match the training
or experiment catalog. Changing a known article's title can change its score
without changing the model state. The experiment's `source_sha256` binds the
exact catalog and train/dev files used by that one run; replay users must check
those hashes against their input bytes. New post-cutoff articles may be scored
when present in the supplied catalog and published by the impression time, but
their unseen title tokens map to the training-only OOV vector.

Safety limits are 8 MiB per source, 2,000 catalog articles, 200 train and 100
validation impressions, 16 candidates per slate, 512 title characters, 24
title tokens, 4,096 vocabulary slots, 32 dimensions, 10 epochs, 32 prior clicks
per user, 20 million conservative training work units, and 4 MiB output. These
are deliberate local-CPU limits, not a claim that this runner processes full
MIND-small. Oversized data must be reduced through an explicit, auditable
sampling protocol outside this command. There is no attention, multi-head
encoder, negative-sampling scheme, or pretrained embedding compatibility with
NRMS. The untouched official-data benchmark row stays open until licensed
data and an appropriate protocol are actually supplied and run.
