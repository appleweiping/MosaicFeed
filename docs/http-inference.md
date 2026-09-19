# HTTP inference protocol

`mosaicfeed serve-click-model` exposes one fitted
`PointwiseLogisticRanker` plus one catalog/history snapshot through a deliberately
small HTTP/1.1 JSON interface. The service uses only the Python standard library,
makes no outbound requests, never modifies the snapshot, and closes each response
connection.

## Start a snapshot

```bash
mosaicfeed serve-click-model \
  --model click-model.json \
  --articles articles.json \
  --events events.json \
  --host 127.0.0.1 \
  --port 8080
```

Startup is atomic from the API's perspective: each of the three files is read once
through a bounded immutable byte snapshot and the model must be fitted; the catalog must be non-empty
and unique; history cannot exceed its row limit, reference a missing article, or
predate publication. The service makes a private model copy and immutable
catalog/history tuples before accepting a connection. It does not watch files.
Restart it to publish another snapshot.

The defaults are:

| Boundary | Default |
|---|---:|
| request body | 65,536 bytes |
| response body | 2,097,152 bytes |
| `k` | 100 maximum |
| candidates per request | 10,000 maximum |
| catalog | 100,000 articles / 64 MiB |
| history | 1,000,000 events / 128 MiB |
| catalog topics | 1,024 per article / 2,000,000 item-topic cells |
| event weight | 100 maximum |
| model file | 4 MiB |
| concurrent workers | 16 |
| socket timeout and cooperative ranking deadline | 10 seconds |
| snapshot/request identifier or topic | 256 characters |

Every ceiling has a corresponding `serve-click-model` option. Values are
validated before binding. If the catalog exceeds the per-request candidate limit,
each request must provide a smaller `candidate_ids` subset.

## Authentication and binding

Loopback is the default and needs no credential. To protect metadata and ranking
on a shared machine, name an environment variable containing the token:

```bash
export MOSAICFEED_TOKEN='replace-with-a-long-random-value'
mosaicfeed serve-click-model ... --token-env MOSAICFEED_TOKEN
curl http://127.0.0.1:8080/metadata \
  -H "Authorization: Bearer $MOSAICFEED_TOKEN"
```

The token value has no CLI option, must contain 1-4,096 visible ASCII characters,
and is never printed. `/health` stays public;
`/metadata` and `/v1/rank` require exactly one matching `Authorization: Bearer`
header when authentication is configured. A non-loopback host requires both
`--allow-nonloopback` and authentication. Bearer HTTP is not encryption; read
the [security policy](../SECURITY.md) before any non-loopback use.

For the special `localhost` name, startup resolves the name once, requires every
returned address to be loopback, and binds the checked numeric address. Mixed or
non-loopback resolution is rejected instead of being re-resolved during bind.

## `GET /health`

This liveness endpoint does not inspect or disclose the snapshot:

```json
{"status":"ok"}
```

## `GET /metadata`

Metadata includes model format/schema/training-example digest, catalog/history
row counts, selected limits, and the time rule. It deliberately omits weights,
article text, paths, user IDs, and event contents.

## `POST /v1/rank`

The request must use `Content-Type: application/json`, a single decimal
`Content-Length`, identity content encoding, strict UTF-8 JSON, and exactly these
fields:

| Field | Required | Contract |
|---|---|---|
| `user_id` | yes | non-empty, no surrounding whitespace/non-printing characters, at most the configured limit (256 by default) |
| `as_of` | yes | ISO-8601 timestamp with an explicit timezone |
| `k` | no | integer from 1 through the configured maximum; default 10 |
| `candidate_ids` | no | unique known IDs within the configured maximum; an empty array is valid |

Unknown or missing fields, duplicate JSON keys, `NaN`/infinities, booleans used
as integers, duplicate candidates, invalid framing, and compressed/chunked bodies
are rejected. Candidate IDs absent from the snapshot produce a generic semantic
error so private identifiers are not echoed.

Example:

```json
{
  "user_id": "alex",
  "as_of": "2026-08-30T12:00:00-05:00",
  "k": 2,
  "candidate_ids": ["article-battery-reuse", "article-feed-diversity"]
}
```

Response timestamps are normalized to UTC and object keys have stable compact
serialization. The values below are reproduced by training with the README's
default command on the committed example catalog/history and then sending the
request above:

```json
{
  "as_of": "2026-08-30T17:00:00Z",
  "candidate_count": 2,
  "object": "mosaicfeed.click_ranking",
  "predictions": [
    {
      "article_id": "article-battery-reuse",
      "probability": 0.717513961155814,
      "rank": 1
    },
    {
      "article_id": "article-feed-diversity",
      "probability": 0.7149332757934674,
      "rank": 2
    }
  ],
  "requested_k": 2,
  "user_id": "alex"
}
```

The full catalog and history construct the user's profile even when
`candidate_ids` limits scoring. Events whose timestamp is equal to `as_of` are
included. Later events and later-published articles are excluded. Seen-item
filtering comes from the model's persisted `FeedConfig`. Probability ties use
article ID. Thus the same validated snapshot and semantic request yield the same
body regardless of concurrency or input candidate order.

## Errors and overload

Errors have one non-reflective shape:

```json
{"error":{"code":"invalid_request","message":"request has missing or unknown fields"}}
```

Relevant statuses include `400` (schema/framing), `401` (authentication), `404`,
`405`, `411`, `413`, `415`, `422` (catalog/ranking constraint), `408`/`504`
(timeout), `503` (concurrency cap), and `500` (redacted internal or response-size
failure). A busy response includes `Retry-After: 1`. Responses use `no-store`,
disable MIME sniffing, and never include a traceback or exception text. The
server intentionally disables request logging because paths, IDs, and credentials
can be sensitive.

`Ctrl-C` stops accepting work and closes the listening socket after existing
non-daemon workers finish. Stalled worker sockets are covered by the configured
request timeout. Ranking uses a monotonic cooperative deadline: bounded-interval
checks run while validating rows, constructing the user profile, traversing
article topics, and scoring candidates, with checks immediately before and after
the bounded final sort. A single Python or operating-system primitive is not
preempted, so this is not a strict hard-real-time guarantee; an overrun is returned
as `504` when the next check observes it.
