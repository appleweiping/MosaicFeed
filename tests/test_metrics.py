from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mosaicfeed.config import FeedConfig
from mosaicfeed.metrics import (
    catalog_coverage,
    evaluate_leave_last_out,
    exposure_gini,
    hit_rate_at_k,
    intra_list_diversity,
    ndcg_at_k,
    reciprocal_rank,
    self_normalized_ips,
    source_diversity,
)
from mosaicfeed.models import (
    Article,
    Event,
    EventKind,
    Feed,
    Recommendation,
    ScoreBreakdown,
)


def recommendation(article_id: str, rank: int) -> Recommendation:
    breakdown = ScoreBreakdown(0, 0, 0, 0, 0, 0, 0.5, ())
    return Recommendation(article_id, rank, 0.5, breakdown)


@pytest.mark.parametrize("metric", [ndcg_at_k, reciprocal_rank, hit_rate_at_k])
def test_ranking_metrics_reject_non_positive_k(metric: object) -> None:
    with pytest.raises(ValueError, match="positive"):
        metric(["a"], {"a"}, 0)  # type: ignore[operator]


def test_ndcg_rewards_earlier_hits() -> None:
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == 1.0
    assert 0 < ndcg_at_k(["b", "a"], {"a"}, 2) < 1
    assert ndcg_at_k(["a"], set(), 1) == 0.0


def test_reciprocal_rank_and_hit_rate() -> None:
    ranking = ["x", "a", "b"]
    assert reciprocal_rank(ranking, {"a"}, 3) == 0.5
    assert reciprocal_rank(ranking, {"missing"}, 3) == 0.0
    assert hit_rate_at_k(ranking, {"a"}, 1) == 0.0
    assert hit_rate_at_k(ranking, {"a"}, 2) == 1.0


@pytest.mark.parametrize("metric", [ndcg_at_k, reciprocal_rank, hit_rate_at_k])
def test_ranking_metrics_reject_duplicates(metric: object) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        metric(["a", "a"], {"a"}, 2)  # type: ignore[operator]


def test_intra_list_diversity(articles: list[Article]) -> None:
    article_map = {article.id: article for article in articles}
    assert intra_list_diversity([], article_map) == 0.0
    assert intra_list_diversity(["a1"], article_map) == 0.0
    assert intra_list_diversity(["a1", "a2"], article_map) == 1.0
    assert 0 < intra_list_diversity(["a1", "a3"], article_map) < 1


def test_intra_list_diversity_rejects_unknown_articles(articles: list[Article]) -> None:
    article_map = {article.id: article for article in articles}
    with pytest.raises(ValueError, match="unknown"):
        intra_list_diversity(["missing", "a1"], article_map)
    with pytest.raises(ValueError, match="unknown"):
        intra_list_diversity(["a1", "missing"], article_map)
    with pytest.raises(ValueError, match="duplicate"):
        intra_list_diversity(["a1", "a1"], article_map)


def test_source_diversity(articles: list[Article]) -> None:
    article_map = {article.id: article for article in articles}
    assert source_diversity([], article_map) == 0.0
    assert source_diversity(["a1", "a2"], article_map) == 1.0
    assert source_diversity(["a1", "a3"], article_map) == 0.5
    with pytest.raises(ValueError, match="unknown"):
        source_diversity(["missing"], article_map)
    with pytest.raises(ValueError, match="duplicate"):
        source_diversity(["a1", "a1"], article_map)


def test_catalog_coverage_and_exposure_gini(now: datetime) -> None:
    feeds = [
        Feed("u1", now, (recommendation("a", 1), recommendation("b", 2))),
        Feed("u2", now, (recommendation("a", 1),)),
    ]
    assert catalog_coverage(feeds, 4) == 0.5
    assert catalog_coverage([], 0) == 0.0
    assert exposure_gini([]) == 0.0
    assert exposure_gini(feeds) == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match="catalog_size"):
        catalog_coverage(feeds, -1)
    with pytest.raises(ValueError, match="exceeds"):
        catalog_coverage(feeds, 1)


def test_self_normalized_ips() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    events = [
        Event("u", "a", EventKind.CLICK, now, 0.5),
        Event("u", "b", EventKind.VIEW, now, 0.25),
        Event("u", "c", EventKind.LIKE, now, None),
    ]
    assert self_normalized_ips(events) == pytest.approx(1 / 3)
    assert self_normalized_ips([events[2]]) == 0.0


def test_self_normalized_ips_is_stable_for_subnormal_propensities() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    events = [
        Event("u", "a", EventKind.CLICK, now, 5e-324),
        Event("u", "b", EventKind.VIEW, now, 1.0),
    ]
    assert self_normalized_ips(events) == pytest.approx(1.0)


def test_evaluate_leave_last_out_reports_all_metric_families(
    articles: list[Article], now: datetime
) -> None:
    events = [
        Event("u1", "a1", EventKind.CLICK, now - timedelta(hours=10), 0.5),
        Event("u1", "a2", EventKind.LIKE, now - timedelta(hours=2), 0.4),
        Event("u2", "a3", EventKind.VIEW, now - timedelta(hours=1), 0.6),
    ]
    report = evaluate_leave_last_out(
        articles,
        events,
        as_of=now,
        config=FeedConfig(size=3, exclude_seen=True),
        k=3,
    )
    assert report.users_evaluated == 1
    assert report.users_skipped == 1
    assert report.k == 3
    assert 0 <= report.ndcg <= 1
    assert 0 <= report.intra_list_diversity <= 1
    assert 0 <= report.exposure_gini <= 1
    assert set(report.to_dict()) >= {"hit_rate", "catalog_coverage", "logged_ips_ctr"}


def test_evaluate_handles_no_users(articles: list[Article], now: datetime) -> None:
    report = evaluate_leave_last_out(articles, [], as_of=now, config=FeedConfig(size=2))
    assert report.users_evaluated == 0
    assert report.ndcg == 0.0


def test_evaluate_rejects_duplicate_catalog(articles: list[Article], now: datetime) -> None:
    with pytest.raises(ValueError, match="unique"):
        evaluate_leave_last_out([articles[0], articles[0]], [], as_of=now, config=FeedConfig())


def test_evaluate_rejects_bad_k(articles: list[Article], now: datetime) -> None:
    with pytest.raises(ValueError, match="positive"):
        evaluate_leave_last_out(articles, [], as_of=now, config=FeedConfig(), k=0)


def test_evaluate_requires_aware_clock(articles: list[Article]) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate_leave_last_out(
            articles,
            [],
            as_of=datetime(2026, 1, 1),
            config=FeedConfig(),
        )


def test_evaluation_excludes_events_at_the_holdout_timestamp(
    articles: list[Article], now: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_training: list[Event] = []

    def capture_feed(
        user_id: str,
        available: list[Article],
        training: list[Event],
        *,
        as_of: datetime,
        config: FeedConfig,
    ) -> Feed:
        del available, config
        captured_training.extend(training)
        return Feed(user_id, as_of, ())

    monkeypatch.setattr("mosaicfeed.metrics.build_feed", capture_feed)
    events = [
        Event("u", "a1", EventKind.VIEW, now - timedelta(hours=1)),
        Event("u", "a2", EventKind.HIDE, now),
        Event("u", "a4", EventKind.CLICK, now),
    ]
    evaluate_leave_last_out(articles, events, as_of=now, config=FeedConfig())
    assert captured_training == [events[0]]


def test_evaluation_slate_metrics_use_top_k_and_holdout_eligible_catalog(
    articles: list[Article], now: datetime
) -> None:
    late = Article(
        "late",
        "Published after holdout",
        "",
        ("late",),
        "Late Source",
        now - timedelta(minutes=1),
    )
    holdout_time = now - timedelta(hours=2)
    report = evaluate_leave_last_out(
        [*articles, late],
        [Event("u", "a4", EventKind.CLICK, holdout_time)],
        as_of=now,
        config=FeedConfig(size=4, max_per_source=4),
        k=1,
    )
    assert report.catalog_coverage == pytest.approx(1 / 4)


def test_evaluation_ips_excludes_events_after_as_of(articles: list[Article], now: datetime) -> None:
    report = evaluate_leave_last_out(
        articles,
        [
            Event("u", "a1", EventKind.CLICK, now - timedelta(hours=1), 0.5),
            Event("u", "a2", EventKind.VIEW, now + timedelta(hours=1), 0.01),
        ],
        as_of=now,
        config=FeedConfig(),
    )
    assert report.logged_ips_ctr == 1.0
