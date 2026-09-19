# Security policy

MosaicFeed's training, evaluation, import, and report workflows make no network
calls. The optional `serve-click-model` command accepts inbound HTTP requests but
makes no outbound calls. It binds to `127.0.0.1` by default.

Catalogs and event logs can contain sensitive data. Use synthetic or anonymized
fixtures in public reports and issues. Do not place secrets or personal data in
article text, user IDs, generated HTML, committed JSON, process arguments, or
shell history.

## Interaction-log boundary

- Keep one writer process per append-only interaction log. A
  `ProfileEventStore` serializes threads inside one instance and detects ordinary
  external size changes, but it is not a cross-process lock or distributed
  transaction system.
- Give logs, checkpoints, exported histories, and parent directories restrictive
  operating-system permissions. User/item IDs and topic preferences can be
  sensitive even when article text is public.
- Checkpoints carry an unkeyed SHA-256 checksum and bind to an exact log prefix,
  catalog/config digest, event chain, and independently rebuilt profile state.
  This detects accidental corruption and stale combinations. It does not
  authenticate an operator who can rewrite both data and checksum.
- An incomplete final log record is left untouched and blocks further writes by
  default. `--recover-torn-tail` is explicit authorization to discard only that
  incomplete suffix after complete records validate. Keep backups; corrupt
  complete records are refused rather than skipped.
- Input, configuration, log, checkpoint, catalog, user, event, per-article topic,
  aggregate catalog-topic, and live or materialized all-user topic state sizes
  have finite defaults.
  Increasing them can turn replay or late-event rebuilds into a local CPU, memory,
  or disk denial of service.
- Exporting event-store history does not update a running inference service.
  Review the snapshot metadata and restart the service deliberately. Do not use
  an event log as the HTTP service's writable backing store.

## Inference-server boundary

- Treat the loaded model, catalog, and complete history as one confidential,
  read-only snapshot. Restart the process to load an update.
- Keep the default loopback binding whenever possible. A non-loopback host is
  refused unless the operator supplies both `--allow-nonloopback` and
  `--token-env`; the named environment variable must hold a non-empty bearer
  token.
- HTTP bearer tokens are not transport encryption. For a controlled remote use,
  put the listener behind a correctly configured TLS reverse proxy, restrict the
  network path, rotate the token, and keep proxy access logs from recording
  request bodies or credentials. Do not expose this development server directly
  to the Internet.
- `/health` is intentionally unauthenticated and contains no snapshot details.
  `/metadata` and `/v1/rank` require the configured token. HTTP errors and request logs
  do not echo identifiers, payloads, tokens, paths, model exceptions, or stack
  traces.
- Default ceilings cover input/output bytes, requested `k`, candidates, loaded
  articles/events, snapshot file sizes, identifiers, article topics, event weight,
  worker concurrency, socket time, and a monotonic cooperative ranking deadline.
  Ranking checks that deadline at bounded intervals in its Python hot loops, but
  does not preempt a single Python or operating-system primitive and is not a hard
  real-time execution limit.
  Raising them increases CPU or memory denial-of-service exposure.
- The service does not provide TLS, authorization by user, rate limits across
  processes, audit storage, sandboxing, live snapshot mutation, or distributed
  isolation. Add those controls outside the process when the deployment requires
  them.

For a security-sensitive report, contact the repository owner privately through
the security advisory interface instead of opening a public issue. Supported
releases are the latest version on the default branch.
