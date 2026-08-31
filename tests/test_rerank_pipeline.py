from __future__ import annotations

from datetime import datetime

import pytest

from mosaicfeed.config import FeedConfig
from mosaicfeed.models import Article, Event, ScoreBreakdown
from mosaicfeed.pipeline import build_feed
from mosaicfeed.rerank import rerank, topic_similarity


def score(value: float) -> ScoreBreakdown:
    return ScoreBreakdown(value, value, value, value, value, value, value, ())


def test_topic_similarity_uses_jaccard(articles: list[Article]) -> None:
    assert topic_similarity(articles[0], articles[0]) == 1.0
    assert topic_similarity(articles[0], articles[2]) == pytest.approx(1 / 3)
    assert topic_similarity(articles[0], articles[1]) == 0.0


def test_rerank_orders_by_relevance_when_lambda_one(articles: list[Article]) -> None:
    scores = {"a1": score(0.4), "a2": score(0.9), "a3": score(0.6)}
    chosen = rerank(articles, scores, config=FeedConfig(size=3, mmr_lambda=1.0))
    assert chosen == ["a2", "a3", "a1"]


def test_rerank_favors_topic_novelty(articles: list[Article]) -> None:
    scores = {"a1": score(0.9), "a2": score(0.79), "a3": score(0.8)}
    chosen = rerank(articles, scores, config=FeedConfig(size=3, mmr_lambda=0.5))
    assert chosen[0] == "a1"
    assert chosen[1] == "a2"


def test_rerank_enforces_source_cap(articles: list[Article]) -> None:
    scores = {article.id: score(1.0 - index / 10) for index, article in enumerate(articles)}
    chosen = rerank(articles, scores, config=FeedConfig(size=4, max_per_source=1))
    sources = [
        next(article.source for article in articles if article.id == item) for item in chosen
    ]
    assert len(sources) == len(set(sources))
    assert len(chosen) == 3


def test_rerank_can_return_empty(articles: list[Article]) -> None:
    assert rerank(articles, {}, config=FeedConfig()) == []


def test_rerank_is_deterministic_on_ties(articles: list[Article]) -> None:
    scores = {article.id: score(0.5) for article in articles}
    config = FeedConfig(size=4, max_per_source=4)
    assert rerank(articles, scores, config=config) == rerank(articles, scores, config=config)


def test_rerank_rejects_duplicate_articles(articles: list[Article]) -> None:
    with pytest.raises(ValueError, match="unique"):
        rerank([articles[0], articles[0]], {"a1": score(0.5)}, config=FeedConfig())


def test_rerank_rejects_unknown_score(articles: list[Article]) -> None:
    with pytest.raises(ValueError, match="unknown"):
        rerank(articles, {"missing": score(0.5)}, config=FeedConfig())


def test_build_feed_returns_contiguous_ranked_items(
    articles: list[Article], events: list[Event], now: datetime
) -> None:
    feed = build_feed("u1", articles, events, as_of=now, config=FeedConfig(size=2))
    assert feed.user_id == "u1"
    assert [item.rank for item in feed.recommendations] == [1, 2]
    assert all(item.article_id not in {"a1", "a3"} for item in feed.recommendations)


def test_build_feed_default_config_and_cold_start(
    articles: list[Article], now: datetime
) -> None:
    feed = build_feed("new", articles, [], as_of=now)
    assert len(feed.recommendations) == len(articles)
    assert all(item.breakdown.reasons for item in feed.recommendations)


def test_build_feed_is_repeatable(
    articles: list[Article], events: list[Event], now: datetime
) -> None:
    first = build_feed("u1", articles, events, as_of=now)
    second = build_feed("u1", reversed(articles), events, as_of=now)
    assert first == second


def test_build_feed_rejects_duplicate_articles(articles: list[Article], now: datetime) -> None:
    with pytest.raises(ValueError, match="unique"):
        build_feed("u", [articles[0], articles[0]], [], as_of=now)


def test_minimum_score_can_empty_feed(articles: list[Article], now: datetime) -> None:
    feed = build_feed(
        "u",
        articles,
        [],
        as_of=now,
        config=FeedConfig(minimum_score=1.0),
    )
    assert feed.recommendations == ()
