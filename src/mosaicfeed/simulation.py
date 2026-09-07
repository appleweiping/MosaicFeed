"""Reproducible synthetic data for demos and smoke benchmarks."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from mosaicfeed.models import Article, Event, EventKind

TOPICS = (
    "ai",
    "climate",
    "economics",
    "health",
    "robotics",
    "science",
    "security",
    "software",
)
SOURCES = ("atlas", "beacon", "circuit", "dispatch", "element")


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    articles: tuple[Article, ...]
    events: tuple[Event, ...]
    user_ids: tuple[str, ...]
    as_of: datetime


def generate_synthetic(
    *,
    seed: int = 17,
    users: int = 20,
    articles: int = 100,
    events_per_user: int = 15,
    as_of: datetime | None = None,
) -> SyntheticDataset:
    if users < 1 or articles < 1 or events_per_user < 0:
        raise ValueError(
            "users and articles must be positive; events_per_user must not be negative"
        )
    clock = as_of or datetime(2026, 1, 15, 12, tzinfo=UTC)
    if clock.tzinfo is None or clock.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    rng = random.Random(seed)
    catalog: list[Article] = []
    for index in range(articles):
        first = TOPICS[index % len(TOPICS)]
        second = TOPICS[(index * 3 + 1) % len(TOPICS)]
        topics = (first,) if first == second else (first, second)
        catalog.append(
            Article(
                id=f"article-{index:04d}",
                title=f"Field note {index}: {first.title()} and {second.title()}",
                summary=f"A synthetic briefing covering {first} and {second}.",
                topics=topics,
                source=SOURCES[index % len(SOURCES)],
                published_at=clock - timedelta(days=45) + timedelta(hours=(index * 7) % (24 * 20)),
                quality=0.35 + 0.6 * rng.random(),
                popularity=0.15 + 0.8 * rng.random(),
            )
        )

    user_ids = tuple(f"user-{index:03d}" for index in range(users))
    log: list[Event] = []
    for user_index, user_id in enumerate(user_ids):
        preferred = {TOPICS[user_index % len(TOPICS)], TOPICS[(user_index + 3) % len(TOPICS)]}
        ordered = list(catalog)
        rng.shuffle(ordered)
        selected = ordered[: min(events_per_user, len(ordered))]
        for event_index, article in enumerate(selected):
            affinity = len(preferred & set(article.topics)) / len(set(article.topics))
            probability = min(0.92, 0.08 + 0.62 * affinity + 0.20 * article.quality)
            draw = rng.random()
            if draw < probability * 0.20:
                kind = EventKind.LIKE
            elif draw < probability:
                kind = EventKind.CLICK
            else:
                kind = EventKind.VIEW
            event_offset = (19 * 86_400) * (event_index + 1) / (len(selected) + 1)
            log.append(
                Event(
                    user_id=user_id,
                    article_id=article.id,
                    kind=kind,
                    occurred_at=clock
                    - timedelta(days=20, minutes=user_index)
                    + timedelta(seconds=event_offset),
                    propensity=max(0.05, min(1.0, probability)),
                )
            )
    return SyntheticDataset(tuple(catalog), tuple(log), user_ids, clock)


def expected_random_hit_rate(catalog_size: int, k: int) -> float:
    if catalog_size < 1 or k < 0:
        raise ValueError("catalog_size must be positive and k must not be negative")
    return min(k, catalog_size) / catalog_size


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials or z <= 0.0:
        raise ValueError("invalid binomial interval inputs")
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (rate + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials**2))
    margin /= denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)
