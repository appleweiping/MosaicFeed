"""Time-aware preference profiling from implicit feedback."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime

from mosaicfeed.config import FeedConfig
from mosaicfeed.models import Article, Event, EventKind, UserProfile

DEADLINE_CHECK_INTERVAL = 64


def _check_deadline(
    deadline_check: Callable[[], None] | None,
    index: int,
) -> None:
    if deadline_check is not None and index % DEADLINE_CHECK_INTERVAL == 0:
        deadline_check()


def event_signal(kind: EventKind, config: FeedConfig) -> float:
    """Return the configured signed base signal for an interaction kind."""

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
    deadline_check: Callable[[], None] | None = None,
) -> UserProfile:
    """Build a normalized profile without using future or unknown events.

    When supplied, ``deadline_check`` is called at bounded intervals throughout
    event, topic, and normalization loops so serving code can cancel work
    cooperatively.
    """

    if not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    totals: defaultdict[str, float] = defaultdict(float)
    seen: set[str] = set()
    event_count = 0
    for event_index, event in enumerate(events):
        _check_deadline(deadline_check, event_index)
        if event.user_id != user_id or event.occurred_at > as_of:
            continue
        article = articles.get(event.article_id)
        if article is None or article.published_at > as_of:
            continue
        age_days = (as_of - event.occurred_at).total_seconds() / 86_400.0
        strength = (
            event_signal(event.kind, config)
            * event.weight
            * exponential_decay(age_days, config.profile_half_life_days)
        )
        per_topic = strength / len(article.topics)
        for topic_index, topic in enumerate(article.topics):
            _check_deadline(deadline_check, topic_index)
            totals[topic] += per_topic
        seen.add(article.id)
        event_count += 1

    scale: float | None = None
    for value_index, value in enumerate(totals.values()):
        _check_deadline(deadline_check, value_index)
        magnitude = abs(value)
        scale = magnitude if scale is None else max(scale, magnitude)
    active_scale = 1.0 if scale is None else scale
    normalized: dict[str, float] = {}
    for topic_index, (topic, value) in enumerate(totals.items()):
        _check_deadline(deadline_check, topic_index)
        if value != 0.0:
            normalized[topic] = value / active_scale
    if deadline_check is not None:
        deadline_check()
    return UserProfile(
        user_id=user_id,
        topic_weights=normalized,
        seen_article_ids=frozenset(seen),
        event_count=event_count,
    )
