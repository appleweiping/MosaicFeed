"""Reproducible policy comparisons with paired bootstrap uncertainty."""

from __future__ import annotations

import hashlib
import html
import json
import math
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import islice
from pathlib import Path
from types import MappingProxyType

from mosaicfeed.config import FeedConfig
from mosaicfeed.io import atomic_write_text
from mosaicfeed.logged import LoggedPolicySummary
from mosaicfeed.metrics import EvaluationReport, EvaluationSamples, evaluate_leave_last_out_samples
from mosaicfeed.models import Article, Event


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A point estimate and percentile-bootstrap interval."""

    mean: float
    lower: float
    upper: float
    confidence: float
    observations: int

    def __post_init__(self) -> None:
        for name in ("mean", "lower", "upper", "confidence"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        if self.lower > self.upper:
            raise ValueError("confidence interval lower must not exceed upper")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        if (
            isinstance(self.observations, bool)
            or not isinstance(self.observations, int)
            or self.observations < 0
        ):
            raise ValueError("observations must be a non-negative integer")
        if self.observations == 0 and any(
            value != 0.0 for value in (self.mean, self.lower, self.upper)
        ):
            raise ValueError("an empty confidence interval must contain only zeros")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "mean": self.mean,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "observations": self.observations,
        }


def _quantile(ordered: Sequence[float], probability: float) -> float:
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean(
    values: Sequence[float],
    *,
    samples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 17,
) -> ConfidenceInterval:
    """Estimate a deterministic percentile interval for a sample mean."""

    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    confidence_value = _finite_float(confidence, "confidence")
    if not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be a finite number in (0, 1)")
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("bootstrap values must be finite numbers")
        try:
            converted.append(float(value))
        except OverflowError as error:
            raise ValueError("bootstrap values must be finite numbers") from error
    observations = tuple(converted)
    if any(not math.isfinite(value) for value in observations):
        raise ValueError("bootstrap values must be finite numbers")
    if not observations:
        return ConfidenceInterval(0.0, 0.0, 0.0, confidence_value, 0)
    generator = random.Random(seed)
    count = len(observations)
    estimates = sorted(
        math.fsum(observations[generator.randrange(count)] / count for _ in range(count))
        for _ in range(samples)
    )
    tail = (1.0 - confidence_value) / 2.0
    return ConfidenceInterval(
        mean=math.fsum(value / count for value in observations),
        lower=_quantile(estimates, tail),
        upper=_quantile(estimates, 1.0 - tail),
        confidence=confidence_value,
        observations=count,
    )


_USER_MEAN_METRICS = (
    "ndcg",
    "hit_rate",
    "reciprocal_rank",
    "intra_list_diversity",
    "source_diversity",
)
_BOOTSTRAP_METRICS = (*_USER_MEAN_METRICS, "catalog_coverage", "exposure_gini")


def _lowercase_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.casefold()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_evaluation_report(report: EvaluationReport) -> None:
    for name in ("users_evaluated", "users_skipped"):
        value = getattr(report, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"evaluation {name} must be a non-negative integer")
    if isinstance(report.k, bool) or not isinstance(report.k, int) or report.k < 1:
        raise ValueError("evaluation k must be a positive integer")
    for name in (
        "ndcg",
        "hit_rate",
        "mean_reciprocal_rank",
        "intra_list_diversity",
        "source_diversity",
        "catalog_coverage",
        "exposure_gini",
        "logged_ips_ctr",
    ):
        value = _finite_float(getattr(report, name), f"evaluation {name}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"evaluation {name} must be a finite number in [0, 1]")
    summary = report.logged_policy
    if summary is not None:
        _validate_logged_policy(summary)
        if not math.isclose(report.logged_ips_ctr, summary.estimate, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("logged policy estimate must match logged_ips_ctr")


def _validate_logged_policy(summary: LoggedPolicySummary) -> None:
    if not isinstance(summary, LoggedPolicySummary):
        raise ValueError("logged_policy must be a LoggedPolicySummary")
    for name in ("users", "observations"):
        value = getattr(summary, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"logged policy {name} must be a non-negative integer")
    if summary.users > summary.observations or (summary.users == 0) != (summary.observations == 0):
        raise ValueError("logged policy users must not exceed observations")
    if isinstance(summary.resamples, bool) or not isinstance(summary.resamples, int):
        raise ValueError("logged policy resamples must be a positive integer")
    if summary.resamples < 1:
        raise ValueError("logged policy resamples must be a positive integer")
    estimate = _finite_float(summary.estimate, "logged policy estimate")
    effective = _finite_float(summary.effective_sample_size, "effective_sample_size")
    share = _finite_float(summary.largest_weight_share, "largest_weight_share")
    confidence = _finite_float(summary.confidence, "logged policy confidence")
    if not 0.0 <= estimate <= 1.0 or not 0.0 <= share <= 1.0:
        raise ValueError("logged policy rates must stay in [0, 1]")
    if not 0.0 <= effective <= summary.observations:
        raise ValueError("effective_sample_size must not exceed observations")
    if summary.observations == 0 and (effective != 0.0 or share != 0.0):
        raise ValueError("empty logged policy diagnostics must be zero")
    if summary.observations > 0 and (effective <= 0.0 or share <= 0.0):
        raise ValueError("non-empty logged policy diagnostics must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("logged policy confidence must be in (0, 1)")
    if (summary.lower is None) != (summary.upper is None):
        raise ValueError("logged policy interval bounds must both be present or absent")
    if summary.lower is not None and summary.upper is not None:
        lower = _finite_float(summary.lower, "logged policy lower")
        upper = _finite_float(summary.upper, "logged policy upper")
        if not 0.0 <= lower <= upper <= 1.0:
            raise ValueError("logged policy interval must stay in [0, 1]")
    if not isinstance(summary.reason, str):
        raise ValueError("logged policy reason must be a string")


def _gini(values: Sequence[int]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    total = sum(ordered)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2.0 * weighted) / (len(ordered) * total) - (len(ordered) + 1) / len(ordered)


def _resampled_metrics(samples: EvaluationSamples, indices: Sequence[int]) -> dict[str, float]:
    if not indices:
        return {name: 0.0 for name in _BOOTSTRAP_METRICS}
    result = {
        metric: math.fsum(float(getattr(samples.users[index], metric)) for index in indices)
        / len(indices)
        for metric in _USER_MEAN_METRICS
    }
    exposures = Counter(
        article_id for index in indices for article_id in samples.users[index].ranked_ids
    )
    eligible_catalog = {
        article_id for index in indices for article_id in samples.users[index].eligible_catalog_ids
    }
    result["catalog_coverage"] = len(exposures) / len(eligible_catalog) if eligible_catalog else 0.0
    result["exposure_gini"] = _gini(tuple(exposures.values()))
    return result


def _interval(
    point: float,
    draws: Sequence[float],
    *,
    confidence: float,
    observations: int,
) -> ConfidenceInterval:
    ordered = sorted(draws)
    tail = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        mean=point,
        lower=_quantile(ordered, tail),
        upper=_quantile(ordered, 1.0 - tail),
        confidence=confidence,
        observations=observations,
    )


def _point_metrics(report: EvaluationReport) -> dict[str, float]:
    return {
        "ndcg": report.ndcg,
        "hit_rate": report.hit_rate,
        "reciprocal_rank": report.mean_reciprocal_rank,
        "intra_list_diversity": report.intra_list_diversity,
        "source_diversity": report.source_diversity,
        "catalog_coverage": report.catalog_coverage,
        "exposure_gini": report.exposure_gini,
    }


@dataclass(frozen=True, slots=True)
class PolicyBenchmark:
    name: str
    report: EvaluationReport
    intervals: Mapping[str, ConfidenceInterval]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name != self.name.strip()
            or any(not character.isprintable() for character in self.name)
        ):
            raise ValueError("policy name must be a non-empty printable string")
        if not isinstance(self.report, EvaluationReport):
            raise ValueError("report must be an EvaluationReport")
        _validate_evaluation_report(self.report)
        if not isinstance(self.intervals, Mapping) or set(self.intervals) != set(
            _BOOTSTRAP_METRICS
        ):
            raise ValueError("intervals must contain every benchmark metric exactly once")
        points = _point_metrics(self.report)
        normalized: dict[str, ConfidenceInterval] = {}
        for metric in _BOOTSTRAP_METRICS:
            interval = self.intervals[metric]
            if not isinstance(interval, ConfidenceInterval):
                raise ValueError("intervals must contain ConfidenceInterval values")
            if interval.observations != self.report.users_evaluated:
                raise ValueError("interval observations must equal users_evaluated")
            if not 0.0 <= interval.lower <= interval.upper <= 1.0:
                raise ValueError("policy intervals must stay in [0, 1]")
            if not math.isclose(interval.mean, points[metric], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("interval means must match evaluation point estimates")
            normalized[metric] = interval
        object.__setattr__(self, "intervals", MappingProxyType(normalized))
        elapsed = _finite_float(self.elapsed_seconds, "elapsed_seconds")
        if elapsed < 0.0:
            raise ValueError("elapsed_seconds must be a finite non-negative number")
        object.__setattr__(self, "elapsed_seconds", elapsed)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "evaluation": self.report.to_dict(),
            "confidence_intervals": {
                key: value.to_dict() for key, value in sorted(self.intervals.items())
            },
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    dataset_sha256: str
    as_of: datetime
    k: int
    bootstrap_samples: int
    confidence: float
    seed: int
    policies: tuple[PolicyBenchmark, ...]
    paired_deltas_from_mosaic: Mapping[str, Mapping[str, ConfidenceInterval]]

    def __post_init__(self) -> None:
        _lowercase_digest(self.dataset_sha256, "dataset_sha256")
        if (
            not isinstance(self.as_of, datetime)
            or self.as_of.tzinfo is None
            or self.as_of.utcoffset() is None
        ):
            raise ValueError("as_of must be timezone-aware")
        for name in ("k", "bootstrap_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        confidence = _finite_float(self.confidence, "confidence")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be a finite number in (0, 1)")
        object.__setattr__(self, "confidence", confidence)
        try:
            policies = tuple(islice(self.policies, 1_001))
        except TypeError as error:
            raise ValueError("policies must contain PolicyBenchmark values") from error
        if (
            not policies
            or len(policies) > 1_000
            or any(not isinstance(policy, PolicyBenchmark) for policy in policies)
        ):
            raise ValueError("policies must be a non-empty bounded collection")
        names = [policy.name for policy in policies]
        if len(names) != len(set(names)) or names[0] != "mosaic":
            raise ValueError("policies must be unique and begin with mosaic")
        users = policies[0].report.users_evaluated
        for policy in policies:
            if policy.report.k != self.k or policy.report.users_evaluated != users:
                raise ValueError("policy reports must use the benchmark k and shared users")
            if any(
                interval.confidence != self.confidence for interval in policy.intervals.values()
            ):
                raise ValueError("policy interval confidence must match the benchmark")
        object.__setattr__(self, "policies", policies)
        if not isinstance(self.paired_deltas_from_mosaic, Mapping):
            raise ValueError("paired deltas must be a mapping")
        expected_names = set(names[1:])
        if set(self.paired_deltas_from_mosaic) != expected_names:
            raise ValueError("paired deltas must correspond to every non-mosaic policy")
        mosaic_points = _point_metrics(policies[0].report)
        policy_by_name = {policy.name: policy for policy in policies}
        paired: dict[str, Mapping[str, ConfidenceInterval]] = {}
        for name in names[1:]:
            values = self.paired_deltas_from_mosaic[name]
            if not isinstance(values, Mapping) or set(values) != set(_BOOTSTRAP_METRICS):
                raise ValueError("paired deltas must contain every benchmark metric")
            other_points = _point_metrics(policy_by_name[name].report)
            frozen: dict[str, ConfidenceInterval] = {}
            for metric in _BOOTSTRAP_METRICS:
                interval = values[metric]
                if not isinstance(interval, ConfidenceInterval):
                    raise ValueError("paired deltas must contain ConfidenceInterval values")
                expected = mosaic_points[metric] - other_points[metric]
                if (
                    interval.observations != users
                    or interval.confidence != self.confidence
                    or not -1.0 <= interval.lower <= interval.upper <= 1.0
                    or not math.isclose(interval.mean, expected, rel_tol=1e-12, abs_tol=1e-12)
                ):
                    raise ValueError("paired delta metadata or mean is inconsistent")
                frozen[metric] = interval
            paired[name] = MappingProxyType(frozen)
        object.__setattr__(self, "paired_deltas_from_mosaic", MappingProxyType(paired))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "dataset_sha256": self.dataset_sha256,
            "as_of": self.as_of.isoformat(),
            "k": self.k,
            "bootstrap_samples": self.bootstrap_samples,
            "confidence": self.confidence,
            "seed": self.seed,
            "policies": [policy.to_dict() for policy in self.policies],
            "paired_deltas_from_mosaic": {
                policy: {
                    metric: interval.to_dict() for metric, interval in sorted(intervals.items())
                }
                for policy, intervals in sorted(self.paired_deltas_from_mosaic.items())
            },
        }


def dataset_fingerprint(articles: Sequence[Article], events: Sequence[Event]) -> str:
    """Hash all ranking-relevant fields without retaining user data in a manifest."""

    payload = {
        "articles": [
            {
                "id": item.id,
                "title": item.title,
                "summary": item.summary,
                "topics": list(item.topics),
                "source": item.source,
                "published_at": item.published_at.isoformat(),
                "quality": item.quality,
                "popularity": item.popularity,
            }
            for item in sorted(articles, key=lambda value: value.id)
        ],
        "events": [
            {
                "user_id": item.user_id,
                "article_id": item.article_id,
                "kind": item.kind.value,
                "occurred_at": item.occurred_at.isoformat(),
                "propensity": item.propensity,
                "weight": item.weight,
            }
            for item in sorted(
                events,
                key=lambda value: (
                    value.occurred_at,
                    value.user_id,
                    value.article_id,
                    value.kind.value,
                    0.0 if value.propensity is None else value.propensity,
                    value.weight,
                ),
            )
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _policy_configs(
    config: FeedConfig, *, k: int, catalog_size: int
) -> tuple[tuple[str, FeedConfig], ...]:
    base = replace(config, size=max(config.size, k))
    unconstrained_source_cap = max(1, base.size, catalog_size)
    return (
        ("mosaic", base),
        (
            "relevance_without_slate_diversity",
            replace(base, mmr_lambda=1.0, max_per_source=unconstrained_source_cap),
        ),
        (
            "popularity",
            replace(
                base,
                interest_weight=0.0,
                freshness_weight=0.0,
                quality_weight=0.0,
                novelty_weight=0.0,
                popularity_weight=1.0,
                exploration_weight=0.0,
                mmr_lambda=1.0,
                max_per_source=unconstrained_source_cap,
            ),
        ),
        (
            "recency",
            replace(
                base,
                interest_weight=0.0,
                freshness_weight=1.0,
                quality_weight=0.0,
                novelty_weight=0.0,
                popularity_weight=0.0,
                exploration_weight=0.0,
                mmr_lambda=1.0,
                max_per_source=unconstrained_source_cap,
            ),
        ),
    )


def run_policy_benchmark(
    articles: Sequence[Article],
    events: Sequence[Event],
    *,
    as_of: datetime,
    config: FeedConfig,
    k: int | None = None,
    bootstrap_samples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 17,
) -> BenchmarkReport:
    """Evaluate MosaicFeed and three declared baselines on identical holdouts."""

    cutoff = config.size if k is None else k
    if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 1:
        raise ValueError("k must be a positive integer")
    # Validate statistical controls even when the dataset has no evaluable users.
    bootstrap_mean((), samples=bootstrap_samples, confidence=confidence, seed=seed)
    evaluations: list[tuple[str, EvaluationSamples, float]] = []
    for name, policy_config in _policy_configs(config, k=cutoff, catalog_size=len(articles)):
        started = time.perf_counter()
        samples = evaluate_leave_last_out_samples(
            articles,
            events,
            as_of=as_of,
            config=policy_config,
            k=cutoff,
        )
        evaluations.append((name, samples, time.perf_counter() - started))

    expected_holdouts = tuple(
        (item.user_id, item.holdout_article_id, item.holdout_at) for item in evaluations[0][1].users
    )
    if any(
        tuple((item.user_id, item.holdout_article_id, item.holdout_at) for item in sample.users)
        != expected_holdouts
        for _, sample, _ in evaluations
    ):
        raise RuntimeError("policy evaluations did not retain identical user holdouts")

    generator = random.Random(seed)
    user_count = len(expected_holdouts)
    draws: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in _BOOTSTRAP_METRICS} for name, _, _ in evaluations
    }
    for _ in range(bootstrap_samples):
        indices = tuple(generator.randrange(user_count) for _ in range(user_count))
        for name, samples, _ in evaluations:
            values = _resampled_metrics(samples, indices)
            for metric in _BOOTSTRAP_METRICS:
                draws[name][metric].append(values[metric])

    policies: list[PolicyBenchmark] = []
    for name, samples, elapsed in evaluations:
        aggregate = samples.report()
        points = _point_metrics(aggregate)
        intervals = {
            metric: _interval(
                points[metric],
                draws[name][metric],
                confidence=confidence,
                observations=user_count,
            )
            for metric in _BOOTSTRAP_METRICS
        }
        policies.append(PolicyBenchmark(name, aggregate, intervals, elapsed))

    paired: dict[str, dict[str, ConfidenceInterval]] = {}
    mosaic_points = _point_metrics(evaluations[0][1].report())
    for name, samples, _ in evaluations[1:]:
        points = _point_metrics(samples.report())
        paired[name] = {}
        for metric in _BOOTSTRAP_METRICS:
            deltas = [
                left - right
                for left, right in zip(draws["mosaic"][metric], draws[name][metric], strict=True)
            ]
            paired[name][metric] = _interval(
                mosaic_points[metric] - points[metric],
                deltas,
                confidence=confidence,
                observations=user_count,
            )

    return BenchmarkReport(
        dataset_sha256=dataset_fingerprint(articles, events),
        as_of=as_of,
        k=cutoff,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
        policies=tuple(policies),
        paired_deltas_from_mosaic=paired,
    )


def render_benchmark_html(report: BenchmarkReport) -> str:
    """Render a portable report without scripts or external assets."""

    def metric_cell(policy: PolicyBenchmark, metric: str, point: float) -> str:
        interval = policy.intervals[metric]
        label = f"{interval.confidence:.0%} CI"
        return (
            f"<td>{point:.4f}<small>{label} "
            f"[{interval.lower:.4f}, {interval.upper:.4f}]</small></td>"
        )

    rows = []
    for policy in report.policies:
        metrics = policy.report
        rows.append(
            "<tr>"
            f"<td>{html.escape(policy.name)}</td>"
            f"<td>{metrics.users_evaluated}</td>"
            f"{metric_cell(policy, 'ndcg', metrics.ndcg)}"
            f"{metric_cell(policy, 'hit_rate', metrics.hit_rate)}"
            f"{metric_cell(policy, 'reciprocal_rank', metrics.mean_reciprocal_rank)}"
            f"{metric_cell(policy, 'intra_list_diversity', metrics.intra_list_diversity)}"
            f"{metric_cell(policy, 'source_diversity', metrics.source_diversity)}"
            f"{metric_cell(policy, 'catalog_coverage', metrics.catalog_coverage)}"
            f"{metric_cell(policy, 'exposure_gini', metrics.exposure_gini)}"
            f"<td>{policy.elapsed_seconds:.4f}</td>"
            "</tr>"
        )
    return (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MosaicFeed policy benchmark</title><style>
body{font:15px system-ui,sans-serif;margin:2rem;color:#172033;background:#f6f8fc}
main{max-width:1120px;margin:auto;background:white;padding:2rem;border-radius:14px;
box-shadow:0 8px 30px #17203318}
table{border-collapse:collapse;width:100%}th,td{padding:.7rem;
border-bottom:1px solid #d9dfeb;text-align:right}
th:first-child,td:first-child{text-align:left}code{overflow-wrap:anywhere}
small{color:#536078;display:block;white-space:nowrap}
</style></head><body><main><h1>MosaicFeed policy benchmark</h1>
<p><small>Paired temporal holdouts and shared user-bootstrap draws.
Runtime is diagnostic, not a hardware-normalized score.</small></p>
<p>Dataset SHA-256: <code>"""
        + html.escape(report.dataset_sha256)
        + """</code></p>
<table><thead><tr><th>Policy</th><th>Users</th><th>NDCG</th><th>Hit</th>
<th>MRR</th><th>ILD</th><th>Source div.</th><th>Coverage</th><th>Exposure Gini</th>
<th>Seconds</th></tr></thead>
<tbody>"""
        + "".join(rows)
        + """</tbody></table></main></body></html>
"""
    )


def write_benchmark_html(path: str | Path, report: BenchmarkReport) -> None:
    atomic_write_text(path, render_benchmark_html(report))
