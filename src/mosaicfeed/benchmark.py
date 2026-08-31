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
from pathlib import Path

from mosaicfeed.config import FeedConfig
from mosaicfeed.metrics import EvaluationReport, EvaluationSamples, evaluate_leave_last_out_samples
from mosaicfeed.models import Article, Event


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A point estimate and percentile-bootstrap interval."""

    mean: float
    lower: float
    upper: float
    confidence: float
    observations: int

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
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a finite number in (0, 1)")
    confidence_value = float(confidence)
    if not math.isfinite(confidence_value) or not 0.0 < confidence_value < 1.0:
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
        math.fsum(observations[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    tail = (1.0 - confidence_value) / 2.0
    return ConfidenceInterval(
        mean=math.fsum(observations) / count,
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
        article_id
        for index in indices
        for article_id in samples.users[index].eligible_catalog_ids
    }
    result["catalog_coverage"] = (
        len(exposures) / len(eligible_catalog) if eligible_catalog else 0.0
    )
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
            }
            for item in sorted(
                events,
                key=lambda value: (
                    value.occurred_at,
                    value.user_id,
                    value.article_id,
                    value.kind.value,
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
        (item.user_id, item.holdout_article_id, item.holdout_at)
        for item in evaluations[0][1].users
    )
    if any(
        tuple(
            (item.user_id, item.holdout_article_id, item.holdout_at)
            for item in sample.users
        )
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
            f'{metric_cell(policy, "ndcg", metrics.ndcg)}'
            f'{metric_cell(policy, "hit_rate", metrics.hit_rate)}'
            f'{metric_cell(policy, "reciprocal_rank", metrics.mean_reciprocal_rank)}'
            f'{metric_cell(policy, "intra_list_diversity", metrics.intra_list_diversity)}'
            f'{metric_cell(policy, "source_diversity", metrics.source_diversity)}'
            f'{metric_cell(policy, "catalog_coverage", metrics.catalog_coverage)}'
            f'{metric_cell(policy, "exposure_gini", metrics.exposure_gini)}'
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
    Path(path).write_text(render_benchmark_html(report), encoding="utf-8", newline="\n")
