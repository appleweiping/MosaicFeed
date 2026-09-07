from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from mosaicfeed.config import FeedConfig
from mosaicfeed.models import Article, Event, EventKind, UserProfile
from mosaicfeed.profile import build_profile, exponential_decay
from mosaicfeed.scoring import (
    deterministic_exploration,
    score_article,
    score_candidates,
    topic_interest,
)


def test_exponential_decay_hits_half_life() -> None:
    assert exponential_decay(0, 10) == 1.0
    assert exponential_decay(10, 10) == pytest.approx(0.5)


@pytest.mark.parametrize(("age", "half_life"), [(-1, 2), (1, 0), (1, -2)])
def test_exponential_decay_rejects_invalid_input(age: float, half_life: float) -> None:
    with pytest.raises(ValueError):
        exponential_decay(age, half_life)


def test_profile_uses_only_matching_past_known_events(
    articles: list[Article], now: datetime
) -> None:
    article_map = {article.id: article for article in articles}
    events = [
        Event("u", "a1", EventKind.LIKE, now - timedelta(days=1)),
        Event("other", "a2", EventKind.LIKE, now - timedelta(days=1)),
        Event("u", "missing", EventKind.LIKE, now - timedelta(days=1)),
        Event("u", "a2", EventKind.LIKE, now + timedelta(seconds=1)),
    ]
    profile = build_profile("u", events, article_map, as_of=now, config=FeedConfig())
    assert profile.event_count == 1
    assert profile.seen_article_ids == {"a1"}
    assert profile.topic_weights["energy"] == 1.0
    assert profile.topic_weights["climate"] == 1.0


def test_profile_ignores_events_for_items_not_yet_published(now: datetime) -> None:
    future = Article(
        "future",
        "Future item",
        "",
        ("future-topic",),
        "Source",
        now + timedelta(hours=1),
    )
    profile = build_profile(
        "u",
        [Event("u", future.id, EventKind.LIKE, now - timedelta(hours=1))],
        {future.id: future},
        as_of=now,
        config=FeedConfig(),
    )
    assert profile.event_count == 0
    assert profile.seen_article_ids == frozenset()
    assert dict(profile.topic_weights) == {}


def test_recent_event_has_more_influence(articles: list[Article], now: datetime) -> None:
    article_map = {article.id: article for article in articles}
    profile = build_profile(
        "u",
        [
            Event("u", "a1", EventKind.LIKE, now - timedelta(days=1)),
            Event("u", "a2", EventKind.LIKE, now - timedelta(days=60)),
        ],
        article_map,
        as_of=now,
        config=FeedConfig(profile_half_life_days=10),
    )
    assert profile.topic_weights["energy"] > profile.topic_weights["search"]


def test_hide_produces_negative_interests(articles: list[Article], now: datetime) -> None:
    profile = build_profile(
        "u",
        [Event("u", "a1", EventKind.HIDE, now - timedelta(hours=1))],
        {article.id: article for article in articles},
        as_of=now,
        config=FeedConfig(),
    )
    assert profile.topic_weights["energy"] == pytest.approx(-1.0)
    assert profile.topic_weights["climate"] == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("kind", "expected_sign"),
    [
        (EventKind.VIEW, 1),
        (EventKind.CLICK, 1),
        (EventKind.LIKE, 1),
        (EventKind.HIDE, -1),
    ],
)
def test_all_event_signals_are_supported(
    kind: EventKind, expected_sign: int, articles: list[Article], now: datetime
) -> None:
    profile = build_profile(
        "u",
        [Event("u", "a4", kind, now)],
        {article.id: article for article in articles},
        as_of=now,
        config=FeedConfig(),
    )
    assert profile.topic_weights["robotics"] * expected_sign > 0


def test_empty_profile_is_supported(articles: list[Article], now: datetime) -> None:
    profile = build_profile(
        "u", [], {article.id: article for article in articles}, as_of=now, config=FeedConfig()
    )
    assert profile.event_count == 0
    assert dict(profile.topic_weights) == {}


def test_profile_requires_aware_clock(articles: list[Article]) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        build_profile(
            "u",
            [],
            {article.id: article for article in articles},
            as_of=datetime(2026, 1, 1),
            config=FeedConfig(),
        )


def test_deterministic_exploration_is_stable_and_bounded() -> None:
    first = deterministic_exploration("u", "a")
    assert first == deterministic_exploration("u", "a")
    assert first != deterministic_exploration("u", "b")
    assert 0.0 <= first < 1.0


def test_topic_interest_maps_signed_average() -> None:
    profile = UserProfile("u", {"positive": 1.0, "negative": -1.0})
    assert topic_interest(profile, ["positive"]) == 1.0
    assert topic_interest(profile, ["negative"]) == 0.0
    assert topic_interest(profile, ["positive", "negative"]) == 0.5
    assert topic_interest(profile, []) == 0.0


def test_score_article_is_decomposed_and_explained(articles: list[Article], now: datetime) -> None:
    profile = UserProfile("u", {"energy": 1.0}, event_count=1)
    score = score_article(articles[0], profile, as_of=now, config=FeedConfig())
    assert score.interest > 0.5
    assert score.freshness > 0
    assert score.quality == 0.9
    assert score.novelty == 1.0
    assert 0 <= score.total <= 1
    assert any("positive interests" in reason for reason in score.reasons)
    assert "recently published" in score.reasons
    assert "high quality signal" in score.reasons


def test_score_article_explains_cold_start(articles: list[Article], now: datetime) -> None:
    score = score_article(articles[1], UserProfile("new"), as_of=now, config=FeedConfig())
    assert "cold-start profile: ranking uses item evidence" in score.reasons


def test_score_article_marks_seen_item_not_novel(articles: list[Article], now: datetime) -> None:
    score = score_article(
        articles[0],
        UserProfile("u", seen_article_ids=frozenset({"a1"})),
        as_of=now,
        config=FeedConfig(),
    )
    assert score.novelty == 0.0
    assert "not previously seen" not in score.reasons


def test_score_article_rejects_future_item(now: datetime) -> None:
    item = Article("future", "Future", "", ("ai",), "S", now + timedelta(seconds=1))
    with pytest.raises(ValueError, match="future"):
        score_article(item, UserProfile("u"), as_of=now, config=FeedConfig())


def test_scoring_requires_aware_clock(articles: list[Article]) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        score_article(
            articles[0],
            UserProfile("u"),
            as_of=datetime(2026, 1, 1),
            config=FeedConfig(),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        score_candidates([], UserProfile("u"), as_of=datetime(2026, 1, 1), config=FeedConfig())


def test_candidate_scoring_filters_seen_future_and_minimum(
    articles: list[Article], now: datetime
) -> None:
    future = Article("future", "Future", "", ("ai",), "S", now + timedelta(seconds=1))
    profile = UserProfile("u", seen_article_ids=frozenset({"a1"}))
    scores = score_candidates(
        [*articles, future],
        profile,
        as_of=now,
        config=FeedConfig(exclude_seen=True, minimum_score=0.4),
    )
    assert "a1" not in scores
    assert "future" not in scores


def test_candidate_scoring_can_include_seen(articles: list[Article], now: datetime) -> None:
    scores = score_candidates(
        articles,
        UserProfile("u", seen_article_ids=frozenset({"a1"})),
        as_of=now,
        config=FeedConfig(exclude_seen=False),
    )
    assert scores["a1"].novelty == 0.0


def test_candidate_scoring_rejects_duplicate_ids(articles: list[Article], now: datetime) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        score_candidates(
            [articles[0], articles[0]],
            UserProfile("u"),
            as_of=now,
            config=FeedConfig(),
        )


def test_candidate_scoring_rejects_duplicate_filtered_items(
    articles: list[Article], now: datetime
) -> None:
    future = Article("future", "Future", "", ("ai",), "S", now + timedelta(seconds=1))
    with pytest.raises(ValueError, match="duplicate"):
        score_candidates([future, future], UserProfile("u"), as_of=now, config=FeedConfig())


def test_custom_weights_control_total(articles: list[Article], now: datetime) -> None:
    config = FeedConfig(
        interest_weight=0,
        freshness_weight=0,
        quality_weight=1,
        novelty_weight=0,
        popularity_weight=0,
        exploration_weight=0,
    )
    score = score_article(articles[0], UserProfile("u"), as_of=now, config=config)
    assert score.total == 0.9
