"""Explainable article scoring before slate-level reranking."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime

from mosaicfeed.config import FeedConfig
from mosaicfeed.models import Article, ScoreBreakdown, UserProfile
from mosaicfeed.profile import exponential_decay


def _validate_clock(as_of: datetime) -> None:
    if not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")


def deterministic_exploration(user_id: str, article_id: str) -> float:
    """Stable pseudo-random value in [0, 1), independent of process hashing."""

    digest = hashlib.sha256(f"{user_id}\0{article_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def topic_interest(profile: UserProfile, topics: Iterable[str]) -> float:
    values = [profile.topic_weights.get(topic.casefold(), 0.0) for topic in topics]
    if not values:
        return 0.0
    raw = sum(values) / len(values)
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))


def score_article(
    article: Article,
    profile: UserProfile,
    *,
    as_of: datetime,
    config: FeedConfig,
) -> ScoreBreakdown:
    _validate_clock(as_of)
    if article.published_at > as_of:
        raise ValueError("cannot score an article published in the future")
    age_hours = (as_of - article.published_at).total_seconds() / 3_600.0
    interest = topic_interest(profile, article.topics)
    freshness = exponential_decay(age_hours, config.article_half_life_hours)
    novelty = 0.0 if article.id in profile.seen_article_ids else 1.0
    exploration = deterministic_exploration(profile.user_id, article.id)
    components = {
        "interest": interest,
        "freshness": freshness,
        "quality": article.quality,
        "novelty": novelty,
        "popularity": article.popularity,
        "exploration": exploration,
    }
    weights = config.ranking_weights()
    weight_total = sum(weights.values())
    total = sum(components[name] * weight for name, weight in weights.items()) / weight_total

    reasons: list[str] = []
    matched = sorted(
        topic for topic in article.topics if profile.topic_weights.get(topic, 0.0) > 0.0
    )
    if matched:
        reasons.append(f"positive interests: {', '.join(matched)}")
    elif profile.event_count == 0:
        reasons.append("cold-start profile: ranking uses item evidence")
    if freshness >= 0.75:
        reasons.append("recently published")
    if article.quality >= 0.75:
        reasons.append("high quality signal")
    if novelty == 1.0:
        reasons.append("not previously seen")
    return ScoreBreakdown(
        interest=interest,
        freshness=freshness,
        quality=article.quality,
        novelty=novelty,
        popularity=article.popularity,
        exploration=exploration,
        total=max(0.0, min(1.0, total)),
        reasons=tuple(reasons),
    )


def score_candidates(
    articles: Iterable[Article],
    profile: UserProfile,
    *,
    as_of: datetime,
    config: FeedConfig,
) -> dict[str, ScoreBreakdown]:
    _validate_clock(as_of)
    result: dict[str, ScoreBreakdown] = {}
    seen_ids: set[str] = set()
    for article in articles:
        if article.id in seen_ids:
            raise ValueError(f"duplicate article id: {article.id}")
        seen_ids.add(article.id)
        if article.published_at > as_of:
            continue
        if config.exclude_seen and article.id in profile.seen_article_ids:
            continue
        score = score_article(article, profile, as_of=as_of, config=config)
        if score.total >= config.minimum_score:
            result[article.id] = score
    return result
