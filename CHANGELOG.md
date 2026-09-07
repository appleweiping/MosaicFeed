# Changelog

All notable changes are recorded here. The format follows Keep a Changelog and versions follow semantic versioning.

## [Unreleased]

### Added

- A confidence interval for the logged-policy IPS estimate, which the README previously recorded
  as impossible because the report did not retain the per-user propensity log. It does now.
- The bootstrap resamples users and pools their observations, because one user contributes many
  correlated events. Measured on a synthetic population of forty users with twenty-five events
  each, resampling events instead reports an interval 2.7 times too narrow when users differ in
  how often they click, and one that is too wide when they differ only in logged propensity. The
  error has no fixed sign, so the wrong bootstrap is not conservative, merely wrong.
- Kish's effective sample size and the share of total weight on the single heaviest observation,
  since a self-normalized estimate can rest almost entirely on a few observations when the
  logging propensities were small. Below a tenth of the observations the summary marks itself
  concentrated.
- An interval is withheld with a stated reason, rather than reported as zero width, when fewer
  than three users carry a propensity. A bootstrap over one cluster resamples the same cluster
  every time.
- `mosaicfeed.logged` as a Python API: `summarize_logged_policy`, `collect_logged_observations`,
  `self_normalized_estimate`, `effective_sample_size`, and `largest_weight_share`.

### Changed

- Resampling walks per-user totals rather than every observation. The estimate is a ratio of two
  sums, and a sum over a pool of users is the sum of each user's own totals, so the results are
  identical to floating point while fifty thousand observations from two thousand users fell from
  thirty-five seconds to under two.

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
