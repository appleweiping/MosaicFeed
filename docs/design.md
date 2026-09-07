# Design and invariants

MosaicFeed treats recommendation as a point-in-time computation. The caller supplies `as_of`; every stage must preserve that temporal boundary.

## Point-in-time contract

- Articles published after `as_of` are not candidates.
- Events after `as_of` and events for articles not yet published by `as_of` do not affect the user
  profile.
- An event for an article absent from the active catalog is ignored by profiling.
- Evaluation moves `as_of` backward to each user's holdout time and trains only on events with a
  strictly earlier timestamp. Events tied with the holdout cannot be causally ordered and are excluded.
- Ties use stable article identifiers, making equal-input runs byte-for-byte reproducible after JSON serialization.

## Profile semantics

Each event kind has a configurable signed strength. Its contribution is multiplied by exponential half-life decay and divided across the article's topics. The resulting topic vector is normalized by its largest absolute value, preserving negative feedback in `[-1, 1]`.

The seen set is separate from preference strength. An old interaction can have almost no profile weight while still preventing an item from being recommended when `exclude_seen` is enabled.

## Rank and rerank boundary

Pointwise scoring cannot enforce slate diversity. MosaicFeed therefore retains score evidence unchanged while MMR changes ordering at the slate layer. The displayed `score` is always the pointwise value; it is never mislabeled as the MMR selection objective.

Source limits are hard constraints. If three publishers are available, `max_per_source=1`, and the requested size is five, the result has at most three items.

## Determinism

Exploration uses the first 64 bits of SHA-256 over `user_id`, a zero byte, and `article_id`. It does not depend on Python's randomized `hash()` and does not consume shared random state. Synthetic generation uses a private seeded `random.Random` instance.

## Learned-model boundary

The pointwise logistic trainer orders eligible events by timestamp and original
input position. Each event is featurized before it is appended to that user's
history, so its label cannot explain its own profile features. Events after the
declared training cutoff are ignored; an event earlier than its article's
publication time is rejected. A private seeded generator shuffles only the
training-example indices and never changes process-global random state.

The label contract is observable feedback, not causal relevance: click/like is
one and view/hide is zero. The model therefore reports itself as pointwise. It
does not infer counterfactual outcomes for candidates absent from the log.

## Deliberate non-goals

- Online serving, streaming updates, and distributed indexes.
- Learned embeddings or large-language-model inference; the included learned
  model is a small, inspectable linear logistic ranker.
- Causal claims from observational interaction logs.
- Automatically defining quality, safety, or fairness policy.
- Relaxing constraints to fill every requested position.
