# Changelog

All notable changes are recorded here. The format follows Keep a Changelog and versions follow semantic versioning.

## [Unreleased]

### Added

- Calibrated slate construction, following Steck (2018). `rerank_strategy: "calibrated"`
  selects against the gap between the reader's topic distribution and the slate's, rather
  than against similarity between the slate's own items. `calibration_error` reports that
  gap for any slate, whichever reranker built it, so the two objectives can be compared on
  the same footing.
- Measured rather than argued: for a reader whose history is roughly four fifths one topic,
  over a five-topic corpus, MMR builds a slate that is 60% that topic and 10% each of four
  others -- three of which the reader has no history for at all -- scoring 0.362. The
  calibrated slate matches the reader's proportions at 0.000. MMR scores *higher* on
  intra-list diversity while doing it, which is the trade, and the README reports both
  numbers rather than only the favourable one.
- The divergence is smoothed, because a topic the slate misses entirely would otherwise be
  infinite and unable to rank two imperfect slates against each other. Negative topic
  weights take no share of the reader's distribution: they record what to push away, which
  is not a proportion to serve.

### Changed

- MMR remains the default and its behaviour is unchanged; all 178 existing tests pass
  untouched. Asking for the calibrated strategy without a profile is refused rather than
  falling back, because the two build different slates and a silent substitution would leave
  nothing in the output to show which one ran. The hard `max_per_source` cap binds under
  either strategy.

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
