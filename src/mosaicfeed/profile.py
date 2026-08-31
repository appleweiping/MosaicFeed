"""Time-aware preference profiling from implicit feedback."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime

from mosaicfeed.config import FeedConfig
from mosaicfeed.models import Article, Event, EventKind, UserProfile


def _event_signal(kind: EventKind, config: FeedConfig) -> float:
    return {
        EventKind.VIEW: config.view_signal,
        EventKind.CLICK: config.click_signal,
        EventKind.LIKE: config.like_signal,
        EventKind.HIDE: config.hide_signal,
    }[kind]


def exponential_decay(age: float, half_life: float) -> float:
    """Return a half-life decay factor for a non-negative age."""

    if age < 0.0:
        raise ValueError("age must not be negative")
    if half_life <= 0.0:
        raise ValueError("half_life must be positive")
    return math.exp(-math.log(2.0) * age / half_life)


def build_profile(
    user_id: str,
    events: Iterable[Event],
    articles: Mapping[str, Article],
    *,
    as_of: datetime,
    config: FeedConfig,
) -> UserProfile:
    """Build a normalized profile without using future or unknown events."""

    if not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    totals: defaultdict[str, float] = defaultdict(float)
    seen: set[str] = set()
    event_count = 0
    for event in events:
        if event.user_id != user_id or event.occurred_at > as_of:
            continue
        article = articles.get(event.article_id)
        if article is None or article.published_at > as_of:
            continue
        age_days = (as_of - event.occurred_at).total_seconds() / 86_400.0
        strength = _event_signal(event.kind, config) * exponential_decay(
            age_days, config.profile_half_life_days
        )
        per_topic = strength / len(article.topics)
        for topic in article.topics:
            totals[topic] += per_topic
        seen.add(article.id)
        event_count += 1

    scale = max((abs(value) for value in totals.values()), default=1.0)
    normalized = {topic: value / scale for topic, value in totals.items() if value != 0.0}
    return UserProfile(
        user_id=user_id,
        topic_weights=normalized,
        seen_article_ids=frozenset(seen),
        event_count=event_count,
    )
