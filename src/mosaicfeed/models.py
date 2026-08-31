"""Immutable domain models and their invariants."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType


def _require_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _require_aware(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _unit_interval(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be between 0 and 1")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{field_name} must be between 0 and 1") from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return number


def _normalized_labels(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)) or not all(isinstance(item, str) for item in values):
        raise ValueError("topics must be a sequence of strings")
    labels = tuple(dict.fromkeys(item.strip().casefold() for item in values if item.strip()))
    if not labels:
        raise ValueError("topics must contain at least one non-empty value")
    return labels


@dataclass(frozen=True, slots=True)
class Article:
    """A candidate item available to the feed ranker."""

    id: str
    title: str
    summary: str
    topics: tuple[str, ...]
    source: str
    published_at: datetime
    quality: float = 0.5
    popularity: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_identifier(self.id, "article id"))
        object.__setattr__(self, "title", _require_identifier(self.title, "article title"))
        object.__setattr__(self, "source", _require_identifier(self.source, "article source"))
        if not isinstance(self.summary, str):
            raise ValueError("article summary must be a string")
        object.__setattr__(self, "summary", self.summary.strip())
        object.__setattr__(self, "topics", _normalized_labels(self.topics))
        object.__setattr__(self, "published_at", _require_aware(self.published_at, "published_at"))
        object.__setattr__(self, "quality", _unit_interval(self.quality, "quality"))
        object.__setattr__(self, "popularity", _unit_interval(self.popularity, "popularity"))


class EventKind(StrEnum):
    """Supported implicit-feedback signals."""

    VIEW = "view"
    CLICK = "click"
    LIKE = "like"
    HIDE = "hide"


@dataclass(frozen=True, slots=True)
class Event:
    """A timestamped user interaction with an article."""

    user_id: str
    article_id: str
    kind: EventKind
    occurred_at: datetime
    propensity: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_id", _require_identifier(self.user_id, "user id"))
        object.__setattr__(self, "article_id", _require_identifier(self.article_id, "article id"))
        if not isinstance(self.kind, EventKind):
            raise ValueError("kind must be an EventKind")
        object.__setattr__(self, "occurred_at", _require_aware(self.occurred_at, "occurred_at"))
        if self.propensity is not None:
            propensity = _unit_interval(self.propensity, "propensity")
            if propensity == 0.0:
                raise ValueError("propensity must be in (0, 1]")
            object.__setattr__(self, "propensity", propensity)


@dataclass(frozen=True, slots=True)
class UserProfile:
    """A point-in-time preference profile derived only from prior events."""

    user_id: str
    topic_weights: Mapping[str, float] = field(default_factory=dict)
    seen_article_ids: frozenset[str] = frozenset()
    event_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_id", _require_identifier(self.user_id, "user id"))
        if (
            isinstance(self.event_count, bool)
            or not isinstance(self.event_count, int)
            or self.event_count < 0
        ):
            raise ValueError("event_count must be a non-negative integer")
        cleaned: dict[str, float] = {}
        for key, value in self.topic_weights.items():
            if not isinstance(key, str):
                raise ValueError("topic weight keys must be strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("topic weights must be finite numbers")
            if not math.isfinite(value):
                raise ValueError("topic weights must be finite numbers")
            if key.strip() and value != 0.0:
                cleaned[key.strip().casefold()] = float(value)
        if any(not -1.0 <= value <= 1.0 for value in cleaned.values()):
            raise ValueError("topic weights must be between -1 and 1")
        object.__setattr__(self, "topic_weights", MappingProxyType(dict(sorted(cleaned.items()))))
        normalized_seen = frozenset(
            _require_identifier(article_id, "seen article id")
            for article_id in self.seen_article_ids
        )
        object.__setattr__(self, "seen_article_ids", normalized_seen)


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Inspectable components for one article's pre-reranking score."""

    interest: float
    freshness: float
    quality: float
    novelty: float
    popularity: float
    exploration: float
    total: float
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "interest",
            "freshness",
            "quality",
            "novelty",
            "popularity",
            "exploration",
            "total",
        ):
            object.__setattr__(
                self,
                field_name,
                _unit_interval(getattr(self, field_name), field_name),
            )
        if not isinstance(self.reasons, tuple) or not all(
            isinstance(reason, str) and reason.strip() for reason in self.reasons
        ):
            raise ValueError("reasons must be a tuple of non-empty strings")


@dataclass(frozen=True, slots=True)
class Recommendation:
    article_id: str
    rank: int
    score: float
    breakdown: ScoreBreakdown

    def __post_init__(self) -> None:
        object.__setattr__(self, "article_id", _require_identifier(self.article_id, "article id"))
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("rank must be positive")
        object.__setattr__(self, "score", _unit_interval(self.score, "score"))
        if not math.isclose(self.score, self.breakdown.total, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("score must match breakdown total")


@dataclass(frozen=True, slots=True)
class Feed:
    user_id: str
    generated_at: datetime
    recommendations: tuple[Recommendation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_id", _require_identifier(self.user_id, "user id"))
        object.__setattr__(self, "generated_at", _require_aware(self.generated_at, "generated_at"))
        if not isinstance(self.recommendations, tuple) or not all(
            isinstance(item, Recommendation) for item in self.recommendations
        ):
            raise ValueError("recommendations must be a tuple of Recommendation objects")
        expected = list(range(1, len(self.recommendations) + 1))
        actual = [item.rank for item in self.recommendations]
        if actual != expected:
            raise ValueError("recommendation ranks must be contiguous and start at 1")
        ids = [item.article_id for item in self.recommendations]
        if len(ids) != len(set(ids)):
            raise ValueError("recommendations must not contain duplicate articles")
