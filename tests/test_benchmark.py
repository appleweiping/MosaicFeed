from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import mosaicfeed.benchmark as benchmark_module
from mosaicfeed.benchmark import (
    bootstrap_mean,
    dataset_fingerprint,
    render_benchmark_html,
    run_policy_benchmark,
)
from mosaicfeed.cli import main
from mosaicfeed.config import FeedConfig
from mosaicfeed.metrics import EvaluationSamples, UserEvaluation
from mosaicfeed.models import Article, Event


def test_bootstrap_mean_is_deterministic_and_contains_point_estimate() -> None:
    first = bootstrap_mean([0.0, 0.5, 1.0], samples=200, seed=9)
    second = bootstrap_mean([0.0, 0.5, 1.0], samples=200, seed=9)

    assert first == second
    assert first.mean == pytest.approx(0.5)
    assert first.lower <= first.mean <= first.upper
    assert first.observations == 3
    assert bootstrap_mean([], samples=10).to_dict()["mean"] == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"samples": 0},
        {"samples": True},
        {"seed": True},
        {"confidence": 0.0},
        {"confidence": 1.0},
        {"confidence": float("nan")},
    ],
)
def test_bootstrap_rejects_invalid_controls(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        bootstrap_mean([1.0], **kwargs)  # type: ignore[arg-type]


def test_bootstrap_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        bootstrap_mean([float("inf")])
    with pytest.raises(ValueError, match="finite"):
        bootstrap_mean([True])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="finite"):
        bootstrap_mean([10**1_000])


def test_dataset_fingerprint_is_order_independent(
    articles: list[Article], events: list[Event]
) -> None:
    assert dataset_fingerprint(articles, events) == dataset_fingerprint(
        list(reversed(articles)), list(reversed(events))
    )


def _user_sample(
    user_id: str,
    ranked_ids: tuple[str, ...],
    eligible_catalog_ids: frozenset[str],
) -> UserEvaluation:
    return UserEvaluation(
        user_id=user_id,
        holdout_article_id=f"holdout-{user_id}",
        holdout_at=datetime(2026, 1, 1, tzinfo=UTC),
        ranked_ids=ranked_ids,
        eligible_catalog_ids=eligible_catalog_ids,
        ndcg=0.0,
        hit_rate=0.0,
        reciprocal_rank=0.0,
        intra_list_diversity=0.0,
        source_diversity=0.0,
    )


def test_resampling_recomputes_nonlinear_catalog_metrics() -> None:
    samples = EvaluationSamples(
        users=(
            _user_sample("u1", ("a", "b"), frozenset({"a", "b"})),
            _user_sample("u2", ("a", "c"), frozenset({"a", "b", "c"})),
        ),
        users_skipped=0,
        k=2,
        catalog_coverage=1.0,
        exposure_gini=1 / 6,
        logged_ips_ctr=0.0,
        eligible_catalog_size=3,
    )

    repeated_first_user = benchmark_module._resampled_metrics(samples, (0, 0))
    both_users = benchmark_module._resampled_metrics(samples, (0, 1))

    # The first draw's eligible catalog is {a, b}, not the full sample's {a, b, c}.
    assert repeated_first_user["catalog_coverage"] == 1.0
    assert repeated_first_user["exposure_gini"] == 0.0
    assert both_users["catalog_coverage"] == 1.0
    assert both_users["exposure_gini"] == pytest.approx(1 / 6)


def test_policy_benchmark_shares_each_user_resample_across_policies(
    monkeypatch: pytest.MonkeyPatch,
    articles: list[Article],
    events: list[Event],
    now: datetime,
) -> None:
    observed: list[tuple[int, ...]] = []
    original = benchmark_module._resampled_metrics

    def recording_resample(
        samples: EvaluationSamples, indices: tuple[int, ...]
    ) -> dict[str, float]:
        observed.append(indices)
        return original(samples, indices)

    monkeypatch.setattr(benchmark_module, "_resampled_metrics", recording_resample)
    run_policy_benchmark(
        articles,
        events,
        as_of=now,
        config=FeedConfig(size=3),
        bootstrap_samples=7,
        seed=4,
    )

    assert len(observed) == 7 * 4
    assert all(
        len(set(observed[start : start + 4])) == 1
        for start in range(0, len(observed), 4)
    )


def test_policy_benchmark_handles_no_evaluable_users(now: datetime) -> None:
    report = run_policy_benchmark(
        [],
        [],
        as_of=now,
        config=FeedConfig(),
        bootstrap_samples=5,
    )

    assert all(policy.report.users_evaluated == 0 for policy in report.policies)
    assert all(
        interval.mean == interval.lower == interval.upper == 0.0
        for policy in report.policies
        for interval in policy.intervals.values()
    )
    assert all(
        interval.mean == interval.lower == interval.upper == 0.0
        for metrics in report.paired_deltas_from_mosaic.values()
        for interval in metrics.values()
    )


def test_policy_benchmark_uses_identical_holdouts_and_reports_intervals(
    articles: list[Article], events: list[Event], now: datetime
) -> None:
    report = run_policy_benchmark(
        articles,
        events,
        as_of=now,
        config=FeedConfig(size=3, max_per_source=2),
        k=2,
        bootstrap_samples=50,
        seed=4,
    )

    assert [policy.name for policy in report.policies] == [
        "mosaic",
        "relevance_without_slate_diversity",
        "popularity",
        "recency",
    ]
    assert len({policy.report.users_evaluated for policy in report.policies}) == 1
    assert set(report.policies[0].intervals) == {
        "ndcg",
        "hit_rate",
        "reciprocal_rank",
        "intra_list_diversity",
        "source_diversity",
        "catalog_coverage",
        "exposure_gini",
    }
    assert set(report.paired_deltas_from_mosaic) == {
        "relevance_without_slate_diversity",
        "popularity",
        "recency",
    }
    assert report.to_dict()["schema_version"] == 1
    rendered = render_benchmark_html(report)
    assert "MosaicFeed policy benchmark" in rendered
    assert "95% CI" in rendered
    assert "Exposure Gini" in rendered
    assert report.dataset_sha256 in rendered


def test_policy_benchmark_rejects_bad_controls(
    articles: list[Article], events: list[Event], now: datetime
) -> None:
    with pytest.raises(ValueError, match="k"):
        run_policy_benchmark(articles, events, as_of=now, config=FeedConfig(), k=0)
    with pytest.raises(ValueError, match="samples"):
        run_policy_benchmark(
            articles,
            events,
            as_of=now,
            config=FeedConfig(),
            bootstrap_samples=0,
        )
    with pytest.raises(ValueError, match="confidence"):
        run_policy_benchmark(
            articles,
            events,
            as_of=now,
            config=FeedConfig(),
            confidence=1.0,
        )


def test_benchmark_cli_writes_json_and_html(tmp_path) -> None:
    examples = Path(__file__).parents[1] / "examples"
    output = tmp_path / "benchmark.json"
    html_output = tmp_path / "benchmark.html"
    code = main(
        [
            "benchmark",
            "--articles",
            str(examples / "articles.json"),
            "--events",
            str(examples / "events.json"),
            "--as-of",
            "2026-08-30T12:00:00Z",
            "--k",
            "3",
            "--bootstrap-samples",
            "20",
            "--output",
            str(output),
            "--html",
            str(html_output),
        ]
    )

    assert code == 0
    assert len(json.loads(output.read_text(encoding="utf-8"))["policies"]) == 4
    assert "<!doctype html>" in html_output.read_text(encoding="utf-8")
