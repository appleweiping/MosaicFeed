"""Adversarial checks for published benchmark statistics and report metadata."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from mosaicfeed.benchmark import (
    BenchmarkReport,
    ConfidenceInterval,
    bootstrap_mean,
    run_policy_benchmark,
)
from mosaicfeed.config import FeedConfig
from mosaicfeed.logged import LoggedPolicySummary


def _empty_report() -> BenchmarkReport:
    return run_policy_benchmark(
        [], [], as_of=datetime(2026, 1, 1, tzinfo=UTC), config=FeedConfig(), bootstrap_samples=2
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"mean": True}, "finite number"),
        ({"mean": 10**1_000}, "finite number"),
        ({"lower": 2.0, "upper": 1.0}, "lower"),
        ({"confidence": 0.0}, "confidence"),
        ({"observations": True}, "observations"),
        ({"observations": -1}, "observations"),
        ({"observations": 0, "mean": 1.0}, "empty confidence"),
    ],
)
def test_confidence_interval_rejects_inconsistent_public_state(
    updates: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "mean": 0.0,
        "lower": 0.0,
        "upper": 0.0,
        "confidence": 0.95,
        "observations": 1,
    }
    values.update(updates)
    with pytest.raises(ValueError, match=message):
        ConfidenceInterval(**values)  # type: ignore[arg-type]


def test_bootstrap_extreme_inputs_have_finite_or_domain_error_results() -> None:
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_mean([1.0], confidence=10**1_000)
    with pytest.raises(ValueError, match="finite"):
        bootstrap_mean(["1"], samples=2)  # type: ignore[list-item]
    interval = bootstrap_mean([1e308, 1e308], samples=3, seed=2)
    assert interval.mean == interval.lower == interval.upper == 1e308
    assert bootstrap_mean([0.5], samples=2).lower == 0.5


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("users_evaluated", -1, "users_evaluated"),
        ("users_skipped", True, "users_skipped"),
        ("k", 0, "evaluation k"),
        ("ndcg", 1.1, "evaluation ndcg"),
        ("logged_ips_ctr", float("inf"), "finite number"),
    ],
)
def test_policy_rejects_malformed_evaluation_reports(
    field: str, value: object, message: str
) -> None:
    policy = _empty_report().policies[0]
    damaged = replace(policy.report, **{field: value})
    with pytest.raises(ValueError, match=message):
        replace(policy, report=damaged)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"name": " "}, "policy name"),
        ({"name": "bad\nname"}, "policy name"),
        ({"report": object()}, "EvaluationReport"),
        ({"intervals": []}, "every benchmark metric"),
        ({"intervals": {"ndcg": object()}}, "every benchmark metric"),
        ({"elapsed_seconds": -1.0}, "elapsed_seconds"),
    ],
)
def test_policy_rejects_malformed_envelope(updates: dict[str, object], message: str) -> None:
    policy = _empty_report().policies[0]
    with pytest.raises(ValueError, match=message):
        replace(policy, **updates)


def test_policy_rejects_wrong_interval_type_and_count() -> None:
    policy = _empty_report().policies[0]
    intervals: dict[str, object] = dict(policy.intervals)
    intervals["ndcg"] = object()
    with pytest.raises(ValueError, match="ConfidenceInterval"):
        replace(policy, intervals=intervals)
    intervals["ndcg"] = ConfidenceInterval(0.0, 0.0, 0.0, 0.95, 1)
    with pytest.raises(ValueError, match="observations"):
        replace(policy, intervals=intervals)


def _logged_summary() -> LoggedPolicySummary:
    return LoggedPolicySummary(
        estimate=0.0,
        users=0,
        observations=0,
        effective_sample_size=0.0,
        largest_weight_share=0.0,
        lower=None,
        upper=None,
        confidence=0.95,
        resamples=2,
        reason="no logged observations",
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("users", True, "logged policy users"),
        ("users", 1, "must not exceed observations"),
        ("resamples", 0, "resamples"),
        ("estimate", 1.2, "rates"),
        ("effective_sample_size", 1.0, "must not exceed observations"),
        ("largest_weight_share", 0.5, "must be zero"),
        ("confidence", 1.0, "confidence"),
        ("lower", 0.0, "both be present"),
        ("reason", 1, "reason"),
    ],
)
def test_policy_rejects_malformed_logged_diagnostics(
    field: str, value: object, message: str
) -> None:
    policy = _empty_report().policies[0]
    summary = replace(_logged_summary(), **{field: value})
    damaged = replace(policy.report, logged_policy=summary)
    with pytest.raises(ValueError, match=message):
        replace(policy, report=damaged)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"dataset_sha256": "0"}, "SHA-256"),
        ({"as_of": datetime(2026, 1, 1)}, "timezone-aware"),
        ({"k": 0}, "positive integer"),
        ({"bootstrap_samples": False}, "positive integer"),
        ({"seed": True}, "seed"),
        ({"confidence": 0.0}, "confidence"),
        ({"policies": ()}, "non-empty bounded"),
        ({"policies": None}, "PolicyBenchmark values"),
        ({"paired_deltas_from_mosaic": []}, "paired deltas must be a mapping"),
        ({"paired_deltas_from_mosaic": {}}, "every non-mosaic policy"),
    ],
)
def test_benchmark_rejects_malformed_public_envelope(
    updates: dict[str, object], message: str
) -> None:
    report = _empty_report()
    with pytest.raises(ValueError, match=message):
        replace(report, **updates)


def test_benchmark_rejects_duplicate_policies_and_split_mismatch() -> None:
    report = _empty_report()
    first = report.policies[0]
    with pytest.raises(ValueError, match="unique"):
        replace(report, policies=(first, first))
    with pytest.raises(ValueError, match="shared users"):
        replace(report, k=report.k + 1)
    with pytest.raises(ValueError, match="interval confidence"):
        replace(report, confidence=0.8)
    with pytest.raises(ValueError, match="begin with mosaic"):
        replace(report, policies=tuple(reversed(report.policies)))


def test_benchmark_rejects_malformed_paired_delta_mapping() -> None:
    report = _empty_report()
    paired = {name: dict(values) for name, values in report.paired_deltas_from_mosaic.items()}
    paired["popularity"] = {"ndcg": paired["popularity"]["ndcg"]}
    with pytest.raises(ValueError, match="every benchmark metric"):
        replace(report, paired_deltas_from_mosaic=paired)
    paired["popularity"] = dict(report.paired_deltas_from_mosaic["popularity"])
    paired["popularity"]["ndcg"] = object()
    with pytest.raises(ValueError, match="ConfidenceInterval"):
        replace(report, paired_deltas_from_mosaic=paired)
    paired["popularity"]["ndcg"] = ConfidenceInterval(0.0, 0.0, 0.0, 0.95, 1)
    with pytest.raises(ValueError, match="metadata or mean"):
        replace(report, paired_deltas_from_mosaic=paired)
