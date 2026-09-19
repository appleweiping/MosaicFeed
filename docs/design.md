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

Each event kind has a configurable signed strength. Its contribution is
multiplied by the event's positive finite weight and exponential half-life decay,
then divided across the article's topics. Legacy events default to weight one.
The resulting topic vector is normalized by its largest absolute value,
preserving negative feedback in `[-1, 1]`.

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

## Incremental-event boundary

The versioned interaction log is authoritative and append-only. One canonical
record contains a globally unique event ID; equivalent reuse is idempotent and a
different record with the same ID is a conflict. A batch is completely parsed,
validated, deduplicated, checked against per-user/global resource ceilings, and
planned before append. One `ProfileEventStore` lock orders its threads. Separate
processes are deliberately outside that guarantee and must not share writer
duties.

User replay has a total `(timestamp, event_id)` order. The watermark is the
maximum timestamp observed for that user minus the configured lateness. The
default policy rejects earlier events; the alternate policy explicitly accepts
and rebuilds affected users. Incremental accumulation and full chronological
replay implement the same weighted half-life equation. A point-in-time query
before the accumulator's newest timestamp replays only events at or before the
requested time.

A checkpoint is a cache of a complete log prefix, never the source of truth. It
binds canonical events and derived user state to the byte offset, prefix digest,
recursive event-chain digest, last-event digest, catalog/config digests, and
lateness policy. Restore independently rebuilds state and then replays later
complete log records. Only an incomplete final record is recoverable, and only
under an explicit truncation option; invalid complete records stop replay.

## Inference snapshot boundary

The HTTP service clones a validated fitted model and stores the complete catalog
and history as immutable tuples before it binds a socket. Requests can restrict
the candidate IDs that are scored, but profile construction still uses the full
catalog so interactions with non-candidate items retain their meaning. A running
process never trains, appends events, reloads files, or consults the wall clock.

Each request supplies an aware `as_of`. Profile events with
`occurred_at <= as_of` are visible, while later events and articles published
after `as_of` are not. Snapshot and request resource ceilings make the amount of
work finite; a bounded semaphore rejects excess connections before starting
another worker. Socket timeouts bound stalled bodies, and a wall-time check
prevents a completed-but-late ranking from being returned. Graceful server close
waits for existing non-daemon workers.

An event-store `history_snapshot()` captures one bounded canonical log-prefix
byte string and re-parses it to derive the frozen legacy events, exact applied
byte offset, and event-chain digest. Those public values cannot be supplied
independently of their retained source bytes. Creating such a snapshot does
not mutate a running HTTP service. Publishing it requires writing a history
export and deliberately restarting `serve-click-model`, preserving the service's
read-only request semantics.

## Deliberate non-goals

- Live mutation of a running model/history HTTP snapshot, multi-process event-log
  writers, distributed indexes, and Internet-facing production serving.
- Learned embeddings or large-language-model inference; the included learned
  model is a small, inspectable linear logistic ranker.
- Causal claims from observational interaction logs.
- Automatically defining quality, safety, or fairness policy.
- Relaxing constraints to fill every requested position.
