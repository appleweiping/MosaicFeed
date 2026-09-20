# Changelog

All notable changes are recorded here. The format follows Keep a Changelog and versions follow semantic versioning.

## [Unreleased]

_No changes yet._

## [0.11.0] - 2026-09-19

### Added

- Offline, hash-checked MIND-small train/dev workflow for user-supplied ZIPs or
  extracted directories. It validates the four-file layout and temporal split,
  trains a bounded pairwise baseline, scores the held-out candidate sets, and
  publishes a deterministic model/scores/report bundle without dataset files.
  The sampled baseline is not a full official benchmark result.

## [0.10.0] - 2026-09-19

### Added

- Bounded local ablation experiment runner with seven one-factor variants,
  shared temporal holdouts, paired bootstrap comparisons, conservative work
  budgets, and a strict versioned, content-addressed, no-overwrite registry.
- Adversarial resource-bound and replay tests, package/CLI smoke, and docs
  identifying feature-provenance and cross-version migration limits. Model
  checkpoint selection, neural/MIND-small parity, and official scores remain
  open.

## [0.9.0] - 2026-09-19

### Added

- Added a strict, opt-in declared-cohort audit for temporal leave-last-out
  evaluation. It reports per-cohort active, evaluated, skipped, and
  calibration-eligible counts; seeded user-bootstrap intervals for utility,
  slate diversity, and pre-holdout topic calibration; and explicit
  maximum-minus-minimum gaps only when at least two groups have evidence.
- Added a bounded UTF-8 cohort mapping adapter, synthetic CLI example,
  hand-computed group arithmetic and same-time/future-event leakage tests,
  fail-closed group-size and work ceilings, CI smoke, and clean-package CLI
  checks. This observational diagnostic does not infer protected attributes
  or establish causal fairness or official MIND benchmark parity.

## [0.8.0] - 2026-09-19

### Added

- Added an opt-in `ListwiseImpressionRanker` with one stable softmax
  cross-entropy update per mixed-label displayed impression. Multiple clicked
  candidates share a uniform target; noncomparable impressions are reported
  explicitly. Existing pointwise and pairwise models retain their formats.
- Added bounded listwise train/rank/evaluate CLI workflows, a distinct
  checksummed model state, immutable training/catalog fingerprints, a
  candidate-lossless MIND score round trip, and the
  [listwise impression contract](docs/listwise-impressions.md).
- Added hand-computed gradient and same-timestamp SGD oracles, disjoint
  post-cutoff/atomic-output/resource tests, UTF-8 state-invariance regressions,
  and listwise wheel/sdist verification.

### Limitations

- Listwise scores are raw within-impression logits, not calibrated click
  probabilities. This release does not include a neural news encoder,
  official MIND-small benchmark result, or licensed MIND dataset rows.

## [0.7.0] - 2026-09-19

### Added

- Added a typed, opt-in `PairwiseImpressionRanker` that learns from clicked
  versus displayed-unclicked candidates within the same MIND impression.
  Same-timestamp impressions share strictly prior history; the pointwise
  schema-1/2 models and default workflows remain unchanged.
- Added bounded pairwise train/rank/evaluate CLI workflows with explicit
  training partition and cutoff, held-out validation, checksummed model state,
  source-file provenance, and candidate-lossless MIND logit score round trips.
- Added independent one-step and exhaustive tiny-impression SGD oracles,
  leakage and corruption regressions, resource/adversarial tests, and the
  [pairwise impression contract](docs/pairwise-impressions.md).

### Limitations

- Pairwise scores are raw ordering logits, not calibrated click probabilities.
  This release does not add pairwise text vocabulary fitting, a neural news
  model, or an official MIND-small benchmark result; no licensed dataset rows
  are bundled.

## [0.6.0] - 2026-09-19

### Added

- Added an opt-in, inspectable TF-IDF news-text encoder and `text_affinity`
  feature for click-model training, with a versioned, checksummed schema-2
  model; the seven-feature schema-1 default remains unchanged.
- Added a declared training-news vocabulary snapshot with first-event
  availability checks and persisted provenance, plus a conservative
  first-event-only fallback when no snapshot is supplied.
- Preserved MIND category and subcategory in separate text namespaces,
  including identical labels, without changing legacy topic-profile semantics.
- Added bounded text parsing, model validation, regression/property tests,
  end-to-end CLI examples, documentation, and cross-platform CI smoke tests.

## [0.5.0] - 2026-09-19

### Added

- Added a real standard-library HTTP inference service for frozen
  `PointwiseLogisticRanker` snapshots, with `GET /health`, authenticated snapshot
  metadata, and deterministic `POST /v1/rank` responses.
- Added strict JSON/framing checks and explicit ceilings for bodies, responses,
  requested ranks, candidates, snapshot rows/files, concurrent workers, and
  request time, plus graceful shutdown behavior.
- Added `serve-click-model`, loopback-only defaults, environment-only optional
  bearer credentials, guarded non-loopback binding, protocol/security docs, and
  real loopback integration tests.
- Added a strict version-1 append-only interaction schema, idempotent event-ID
  ingestion with conflict detection, per-user watermarks, chronological
  incremental profile accumulators, and explicit late-event rebuild policy.
- Added atomic checksummed profile checkpoints bound to an exact log prefix,
  event chain, catalog/config semantics, and independently rebuilt state, plus
  complete-tail replay and explicit torn-tail recovery.
- Added one-pass bounded file snapshots, a streaming catalog fingerprint, and
  explicit configuration, per-article-topic, aggregate catalog-topic, and live
  user-topic ceilings so resource checks cover parsed and derived state rather
  than only log bytes.
- Added `stream-ingest`, `stream-checkpoint`, and `stream-replay`, including a
  provenance-carrying frozen history export for model training and HTTP snapshot
  restart workflows. Tests cover manual profile arithmetic, randomized full
  rebuild equivalence, corruption, truncation, resource ceilings, and threaded
  exactly-once ingestion.
- Added a release-archive verifier with explicit path, file-count, uncompressed
  byte, metadata, and required-module allowlists for both sdist and wheel files.
- Added a branch-only coverage gate that independently checks covered arcs
  against all measured branches; the combined line/branch percentage is not
  treated as proof of the 90% branch requirement.
- Event-log mutation now rejects symbolic links and multiply-linked files and
  verifies path/descriptor identity around every append and torn-tail repair.
- MIND conversions and frozen history snapshots derive their records and
  fingerprints from retained, bounded immutable source bytes.

### Changed

- Learned ranking can restrict candidate IDs without discarding the complete
  catalog context used to construct the point-in-time user profile.
- Legacy `Event` values accept an optional positive finite `weight`; omitted
  weights remain exactly `1.0`, while incremental and offline profile builders
  apply the same weighted signal semantics. Benchmark dataset fingerprints now
  include this ranking-relevant field.
- Tuple-backed public results now snapshot tuple subclasses before validation,
  model loading rejects finite individual weights whose aggregate magnitude can
  overflow inference, and historical all-user profile reads share the global
  topic-cell ceiling used by live state.
- HTTP framing rejects pathological decimal `Content-Length` values as client
  errors, and release archives reject control characters, platform device names,
  portable path collisions, links, and special filesystem entries.
- HTTP ranking now carries a monotonic cooperative deadline through catalog and
  history validation, profile construction, topic traversal, and candidate
  scoring, returning `504` without completing oversized eligible histories once
  a bounded-interval check observes expiration.
- JSON, JSONL, HTML, and fitted-model destinations now use flushed atomic
  replacement with interruption-safe staging cleanup and old-target preservation;
  multi-output CLI commands stage all outputs and roll back earlier replacements
  if a later publication fails, including when a rename takes effect before
  reporting an exception. POSIX directory metadata is fsynced after renames;
  Windows, filesystem, and storage-hardware crash guarantees remain platform-specific.
- Mutating CLI workflows reject lexical, canonical, symlink, and hardlink input/output
  collisions before reading or writing their datasets.
- Public MIND and benchmark reports now validate their complete construction
  invariants and recursively freeze nested mappings.
- Portable release-path checks now include `CONIN$`, `CONOUT$`, and the Windows
  superscript-digit `COM¹`, `COM²`, `COM³`, `LPT¹`, `LPT²`, and `LPT³`
  device aliases.
- Bootstrap confidence controls now reject unrepresentable integers with a
  domain error, and averaging divides before summing to preserve finite means
  for large but valid observations.

## [0.4.0] - 2026-09-07

### Added

- Added `PointwiseLogisticRanker`, a genuine fitted click/like probability
  model with seven inspectable ranking features, seeded SGD, L2 regularization,
  strict event-time leakage prevention, and stable candidate ranking.
- Added versioned portable learned-model JSON with full feature/config/training
  provenance, an exact training-example SHA-256 digest, and strict tamper
  validation.
- Added `train-click-model` and `rank-click-model` CLI workflows plus
  deterministic, leakage, malformed-state, learning-behavior, and end-to-end
  tests.

## [0.3.0] - 2026-09-07

### Added

- Added a tag-gated release pipeline with locked builds, clean wheel and sdist
  installation checks, CycloneDX SBOM, SHA-256 manifest, and GitHub provenance.
- Candidate-lossless MIND impression interchange: `import-mind` retains ordered candidates and click
  labels, streams and hashes the exact source bytes in one pass, records the conversion time/offset needed
  for replay, and keeps the existing click-event conversion separate.
- `evaluate-mind` and the `mosaicfeed.mind` API compute macro impression AUC, MRR, and binary nDCG at
  explicit cutoffs using the public MIND evaluator formulas. Exact candidate-score coverage, duplicate
  rows, undefined AUC groups, non-finite scores, and schema drift are rejected rather than hidden.
- Versioned evaluation reports carry canonical label/score fingerprints, including normalized numeric
  representations and signed zero, and document deterministic source-order tie handling. Hand-calculated
  golden tests and a fixed-seed 200-case independent reference check cover the formulas; the CLI runs in CI
  on a synthetic MIND-format fixture, and no licensed dataset rows are vendored.
- Added a clustered-bootstrap confidence interval for logged-policy IPS, with whole-user resampling and
  pooled observations. A committed synthetic regression demonstrates why event-level resampling can be
  materially overconfident.
- Added Kish effective sample size, largest-weight share, and an explicit `concentrated` diagnostic. An
  interval is withheld with a reason when fewer than three users carry a propensity.
- Added `mosaicfeed.logged` APIs: `summarize_logged_policy`, `collect_logged_observations`,
  `self_normalized_estimate`, `effective_sample_size`, and `largest_weight_share`.
- Added calibrated slate construction following Steck (2018). `calibration_error` compares a reader's
  topic distribution with any resulting slate; smoothing keeps imperfect slates comparable and negative
  preference weights do not become proportions to serve.

### Changed

- Bootstrap draws aggregate precomputed per-user totals instead of revisiting every observation. This
  preserves the ratio-of-sums estimator to floating-point precision while making each draw proportional
  to the number of users after aggregation.
- MMR remains the default and its behavior is unchanged. A calibrated strategy without a profile is
  rejected instead of silently substituted, and the hard `max_per_source` cap applies to either strategy.

## [0.2.0] - 2026-08-31

### Added

- Added paired policy benchmarks against unconstrained relevance, popularity, and recency baselines.
- Added deterministic user-bootstrap confidence intervals, paired deltas, dataset fingerprints, and
  portable JSON/HTML benchmark reports.
- Added a strict, local-only MIND TSV adapter with explicit timestamp and metadata-loss declarations.

### Fixed

- Exclude same-timestamp events from leave-last-out training and future events from logged IPS.
- Compute coverage and exposure concentration from top-`k` slates and the holdout-eligible catalog.
- Reject duplicate JSON fields, non-finite JSON numbers, overflowing ranking weights, and non-finite
  IPS arithmetic.

## [0.1.0] - 2026-08-31

### Added

- Time-decayed signed topic profiling from views, clicks, likes, and hides.
- Explainable weighted ranking with deterministic exploration.
- MMR topic diversification and hard source caps.
- Temporal leave-last-out metrics and synthetic dataset generation.
- Strict JSON/JSONL I/O, CLI workflows, and portable HTML reports.
