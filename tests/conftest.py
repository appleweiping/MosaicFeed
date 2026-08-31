from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mosaicfeed.models import Article, Event, EventKind


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 8, 30, 12, tzinfo=UTC)


@pytest.fixture
def articles(now: datetime) -> list[Article]:
    return [
        Article(
            id="a1",
            title="Grid flexibility",
            summary="Flexible demand and storage",
            topics=("energy", "climate"),
            source="Atlas",
            published_at=now - timedelta(hours=12),
            quality=0.9,
            popularity=0.7,
        ),
        Article(
            id="a2",
            title="Search diversity",
            summary="Exposure and source concentration",
            topics=("search", "responsible ai"),
            source="Beacon",
            published_at=now - timedelta(hours=24),
            quality=0.8,
            popularity=0.6,
        ),
        Article(
            id="a3",
            title="Climate maps",
            summary="Urban heat data",
            topics=("climate", "data"),
            source="Atlas",
            published_at=now - timedelta(hours=36),
            quality=0.75,
            popularity=0.5,
        ),
        Article(
            id="a4",
            title="Robot learning",
            summary="Reproducible control",
            topics=("robotics",),
            source="Circuit",
            published_at=now - timedelta(hours=48),
            quality=0.7,
            popularity=0.4,
        ),
    ]


@pytest.fixture
def events(now: datetime) -> list[Event]:
    return [
        Event("u1", "a1", EventKind.CLICK, now - timedelta(days=4), 0.5),
        Event("u1", "a3", EventKind.LIKE, now - timedelta(days=2), 0.4),
        Event("u2", "a4", EventKind.VIEW, now - timedelta(days=1), 0.8),
    ]
