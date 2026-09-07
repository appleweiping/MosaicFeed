# Changelog

All notable changes are recorded here. The format follows Keep a Changelog and versions follow semantic versioning.

## [Unreleased]

_No changes yet._

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
