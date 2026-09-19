# Incremental interaction stream and profile checkpoints

MosaicFeed can maintain time-decayed user profiles from a bounded, local,
append-only event log. The log is the source of truth. A checkpoint is a
validated cache of one complete log prefix; deleting it and replaying the log
produces the same profiles.

This component uses only the Python standard library. It does not open a
listener, call a model provider, or update a running HTTP inference process.

## Version 1 event contract

Ingestion accepts newline-terminated UTF-8 JSONL with exactly one object per
line and exactly these fields:

| Field | Contract |
|---|---|
| `schema_version` | integer `1` (a boolean is not an integer here) |
| `event_id` | globally unique, non-empty printable string with no surrounding whitespace |
| `user_id` | non-empty printable string with no surrounding whitespace |
| `item_id` | ID in the active catalog |
| `timestamp` | ISO-8601 timestamp with an explicit timezone, not before item publication |
| `type` | `view`, `click`, `like`, or `hide` |
| `weight` | positive finite number, at most `100` by default |

Unknown and missing fields, duplicate JSON keys, invalid UTF-8, non-finite
numbers, blank lines, and a final line without a newline are rejected. Incoming
JSON may use ordinary whitespace and key order. Accepted records are normalized
to UTC and written in one canonical compact form.

`event_id` supplies idempotency. Re-ingesting byte-equivalent semantic content
for an existing ID succeeds as a duplicate and does not append another record.
Reusing the ID with any different canonical field is a conflict and rejects the
entire batch. Every record in a batch is validated and its resource impact is
calculated before any bytes are appended or in-memory state is changed.
The accepted block is flushed and fsynced before logical state advances. When
the log is first created, its parent directory is fsynced on POSIX. Windows,
the selected filesystem, and the storage device may provide weaker crash or
power-loss guarantees. A process or storage failure can still leave a
complete prefix of the batch and/or
an incomplete final record; reopen, recover the incomplete suffix explicitly if
present, and retry the same IDs. Complete records deduplicate and only missing
records append.

## Chronology, watermark, and profiles

Each user has a maximum observed event timestamp. With the default five-minute
lateness allowance, its watermark is:

```text
maximum observed timestamp - 300 seconds
```

Under the default `reject` policy, a new event strictly before that watermark is
rejected. An event exactly on it is accepted. An accepted event that sorts after
the user's current `(timestamp, event_id)` boundary updates the accumulator
incrementally. An event inside the lateness window but earlier in that total
order rebuilds only that user's state in chronological order. The explicit
`accept-rebuild` policy accepts arbitrarily late events and rebuilds affected
users; this makes ingestion potentially more expensive and should be selected
deliberately.

An event contributes `configured signal × weight`, divided across the item's
topics. Existing totals decay to the new event time before the contribution is
applied. A point-in-time read later decays the accumulator to `as_of`; a read
before the latest event replays only eligible earlier events. Both paths are
tested against the independent full `build_profile` implementation.

## CLI workflow

Append the committed example batch and atomically refresh a checkpoint:

```bash
mosaicfeed stream-ingest \
  --articles examples/articles.json \
  --log scratch/interactions.jsonl \
  --input examples/interaction-events.jsonl \
  --checkpoint scratch/profiles.checkpoint.json
```

Repeated execution reports three duplicates and leaves the log unchanged. Build
or refresh a checkpoint independently:

```bash
mosaicfeed stream-checkpoint \
  --articles examples/articles.json \
  --log scratch/interactions.jsonl \
  --checkpoint scratch/profiles.checkpoint.json \
  --output scratch/profiles-next.checkpoint.json
```

Replay every user, or repeat `--user` to select users:

```bash
mosaicfeed stream-replay \
  --articles examples/articles.json \
  --log scratch/interactions.jsonl \
  --checkpoint scratch/profiles.checkpoint.json \
  --as-of 2026-08-30T12:00:00Z \
  --output scratch/profiles.json \
  --events-output scratch/frozen-history.json
```

`--events-output` writes the legacy immutable history format consumed by
`train-click-model`, `rank-click-model`, and `serve-click-model`. The replay
result records the exact event count, byte offset, and recursive event-chain
digest used for that export. The in-memory snapshot re-parses one bounded
canonical log-prefix byte string and derives both events and metadata from
that same source; callers cannot attach independent provenance values.
Publishing new history to HTTP is explicit: create
the export and restart `serve-click-model`. A process that is already serving
keeps its original frozen model/catalog/history snapshot.

Only pass `--recover-torn-tail` after deciding to discard an incomplete final
record. Complete records are validated first; then the incomplete bytes are
truncated and fsynced. Without this option, complete records remain readable,
the number of trailing bytes is reported, and append/checkpoint operations are
refused.

## Python API

```python
from datetime import UTC, datetime

from mosaicfeed import ProfileEventStore, load_interaction_events
from mosaicfeed.io import load_articles

store = ProfileEventStore.open(
    "scratch/interactions.jsonl",
    load_articles("examples/articles.json"),
    checkpoint_path="scratch/profiles.checkpoint.json",
)
report = store.ingest(load_interaction_events("examples/interaction-events.jsonl"))
profile = store.profile("alex", as_of=datetime(2026, 8, 30, 12, tzinfo=UTC))
checkpoint = store.checkpoint("scratch/profiles-next.checkpoint.json")
snapshot = store.history_snapshot()
```

One `ProfileEventStore` serializes its threads with a re-entrant lock. Batch
append, profile snapshots, and checkpoint generation therefore observe a
consistent in-process prefix. The API does not coordinate separate store
instances or processes. A size or file-identity change made outside the
instance is detected and requires reopen/replay; use one writer per log.
Symlinks, non-regular files, and files with more than one hard link are refused.
Where available, opens also use `O_NOFOLLOW`, and descriptor identity is
checked against the path before and after mutable operations.

## Checkpoint and recovery contract

A checkpoint is compact strict JSON plus a final newline. Its versioned payload
contains:

- the canonical events in the applied prefix;
- derived per-user topic totals, seen IDs, counts, and update times;
- `last_applied_offset`, prefix SHA-256, last-event SHA-256, event count, and a
  recursive event-chain SHA-256;
- the profile-relevant catalog digest, complete `FeedConfig` digest, lateness
  allowance, and late-event policy.

The envelope checksum covers the canonical payload. Restore checks the checksum,
settings, configured limits, exact log-prefix bytes, event chain, last-event
digest, and independently rebuilt user state before replaying later complete
records. A mismatched catalog/config/policy is rejected instead of silently
reusing stale profiles.

Checkpoint replacement writes a same-directory temporary file, flushes and
fsyncs it, and uses atomic `os.replace`. On POSIX, the parent directory is
fsynced after publication. The old checkpoint therefore survives an ordinary
failure before replacement. Windows, filesystem, storage-controller, and
hardware guarantees can still vary, and the JSON checksum is not a signature
or message-authentication code: it catches accidental damage, not a local
attacker who can rewrite both payload and checksum. Protect the log and
checkpoint with operating-system file permissions and backups.

The only automatically recoverable log damage is an incomplete final record.
Corrupt, non-canonical, duplicate, blank, or semantically invalid complete
records stop replay. The tool never guesses past a corrupt complete line.

## Default resource ceilings

| Boundary | Default |
|---|---:|
| one canonical event | 16 KiB |
| one input batch file | 64 MiB / 10,000 events |
| authoritative log | 256 MiB / 1,000,000 events |
| serialized event state | 192 MiB |
| checkpoint | 256 MiB |
| users | 100,000 |
| events per user | 100,000 |
| catalog | 100,000 items / 64 MiB / 1,024 topics per item / 2,000,000 item-topic cells |
| configuration file | 64 KiB |
| live or materialized all-user topic cells | 2,000,000 |
| each identifier | 256 characters |
| event weight | 100 |

Every ceiling has a matching CLI option and a Python `EventStreamLimits` field.
Larger values increase memory, CPU, disk, and replay-time exposure.

The CLI also rejects collisions between input, catalog, config, authoritative
log, history export, report, and checkpoint paths before writing. The one
intentional exception is refreshing an existing checkpoint in place with
`stream-checkpoint --checkpoint PATH --output PATH`; replacement remains atomic.
