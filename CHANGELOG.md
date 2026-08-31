# Changelog

All notable changes are recorded here. The format follows Keep a Changelog and versions follow semantic versioning.

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
