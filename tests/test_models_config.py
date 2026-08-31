from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from mosaicfeed.config import FeedConfig
from mosaicfeed.models import (
    Article,
    Event,
    EventKind,
    Feed,
    Recommendation,
    ScoreBreakdown,
    UserProfile,
)


def article(**overrides: object) -> Article:
    values: dict[str, object] = {
        "id": " item ",
        "title": " Title ",
        "summary": " Summary ",
        "topics": ("AI", "ai", " Data "),
        "source": " Source ",
        "published_at": datetime(2026, 1, 1, tzinfo=UTC),
        "quality": 0.5,
        "popularity": 0.2,
    }
    values.update(overrides)
    return Article(**values)  # type: ignore[arg-type]


def breakdown(total: float = 0.5) -> ScoreBreakdown:
    return ScoreBreakdown(0.5, 0.5, 0.5, 1.0, 0.2, 0.1, total, ("reason",))


def test_article_normalizes_values() -> None:
    item = article()
    assert item.id == "item"
    assert item.title == "Title"
    assert item.summary == "Summary"
    assert item.source == "Source"
    assert item.topics == ("ai", "data")


@pytest.mark.parametrize("field", ["id", "title", "source"])
def test_article_rejects_blank_identity_fields(field: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        article(**{field: "  "})


def test_article_rejects_empty_topics() -> None:
    with pytest.raises(ValueError, match="topics"):
        article(topics=("", " "))


@pytest.mark.parametrize("field", ["quality", "popularity"])
@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_article_rejects_out_of_range_signals(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        article(**{field: value})


def test_article_requires_aware_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        article(published_at=datetime(2026, 1, 1))


def test_article_rejects_wrong_text_and_topic_types() -> None:
    with pytest.raises(ValueError, match="string"):
        article(id=7)
    with pytest.raises(ValueError, match="summary"):
        article(summary=7)
    with pytest.raises(ValueError, match="sequence of strings"):
        article(topics=("ai", 7))


def test_event_normalizes_and_validates() -> None:
    event = Event(" user ", " item ", EventKind.CLICK, datetime(2026, 1, 1, tzinfo=UTC), 0.2)
    assert (event.user_id, event.article_id) == ("user", "item")


@pytest.mark.parametrize("propensity", [0.0, -0.1, 1.1])
def test_event_rejects_bad_propensity(propensity: float) -> None:
    with pytest.raises(ValueError, match="propensity"):
        Event("u", "a", EventKind.VIEW, datetime(2026, 1, 1, tzinfo=UTC), propensity)


def test_event_requires_aware_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Event("u", "a", EventKind.VIEW, datetime(2026, 1, 1))


def test_event_requires_enum_kind_and_finite_propensity() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="EventKind"):
        Event("u", "a", "click", now)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="propensity"):
        Event("u", "a", EventKind.CLICK, now, math.nan)


def test_profile_freezes_and_normalizes_mapping() -> None:
    profile = UserProfile(" user ", {" AI ": 0.5, "unused": 0.0}, frozenset({"a"}), 2)
    assert profile.user_id == "user"
    assert dict(profile.topic_weights) == {"ai": 0.5}
    with pytest.raises(TypeError):
        profile.topic_weights["ai"] = 0.7  # type: ignore[index]


def test_profile_rejects_bad_counts_and_weights() -> None:
    with pytest.raises(ValueError, match="event_count"):
        UserProfile("u", event_count=-1)
    with pytest.raises(ValueError, match="event_count"):
        UserProfile("u", event_count=True)
    with pytest.raises(ValueError, match="topic weights"):
        UserProfile("u", {"ai": 1.1})
    with pytest.raises(ValueError, match="finite"):
        UserProfile("u", {"ai": math.inf})
    with pytest.raises(ValueError, match="keys"):
        UserProfile("u", {7: 0.5})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="seen article id"):
        UserProfile("u", seen_article_ids=frozenset({7}))  # type: ignore[arg-type]


def test_models_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        article().title = "changed"  # type: ignore[misc]


def test_recommendation_requires_positive_rank() -> None:
    with pytest.raises(ValueError, match="rank"):
        Recommendation("a", 0, 0.5, breakdown())


def test_score_breakdown_and_recommendation_are_consistent() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        breakdown(1.1)
    with pytest.raises(ValueError, match="reasons"):
        ScoreBreakdown(0, 0, 0, 0, 0, 0, 0, ("",))
    with pytest.raises(ValueError, match="must match"):
        Recommendation("a", 1, 0.4, breakdown(0.5))


def test_feed_accepts_contiguous_unique_recommendations() -> None:
    feed = Feed(
        "u",
        datetime(2026, 1, 1, tzinfo=UTC),
        (
            Recommendation("a", 1, 0.5, breakdown()),
            Recommendation("b", 2, 0.4, breakdown(0.4)),
        ),
    )
    assert len(feed.recommendations) == 2


def test_feed_rejects_rank_gaps_duplicates_and_naive_time() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="contiguous"):
        Feed("u", now, (Recommendation("a", 2, 0.5, breakdown()),))
    with pytest.raises(ValueError, match="duplicate"):
        Feed(
            "u",
            now,
            (
                Recommendation("a", 1, 0.5, breakdown()),
                Recommendation("a", 2, 0.4, breakdown(0.4)),
            ),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        Feed("u", datetime(2026, 1, 1), ())
    with pytest.raises(ValueError, match="tuple"):
        Feed("u", now, [])  # type: ignore[arg-type]


def test_default_config_is_valid_and_has_weights() -> None:
    config = FeedConfig()
    assert config.size == 10
    assert set(config.ranking_weights()) == {
        "interest",
        "freshness",
        "quality",
        "novelty",
        "popularity",
        "exploration",
    }


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"size": 0}, "size"),
        ({"max_per_source": 0}, "max_per_source"),
        ({"article_half_life_hours": 0}, "half-lives"),
        ({"profile_half_life_days": -1}, "half-lives"),
        ({"mmr_lambda": 1.1}, "mmr_lambda"),
        ({"minimum_score": -0.1}, "minimum_score"),
        ({"quality_weight": -1}, "weights"),
    ],
)
def test_config_rejects_invalid_values(override: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FeedConfig(**override)  # type: ignore[arg-type]


def test_config_rejects_all_zero_weights() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FeedConfig(
            interest_weight=0,
            freshness_weight=0,
            quality_weight=0,
            novelty_weight=0,
            popularity_weight=0,
            exploration_weight=0,
        )


@pytest.mark.parametrize(
    "override",
    [
        {"size": 1.5},
        {"size": True},
        {"max_per_source": 1.5},
        {"exclude_seen": "yes"},
        {"quality_weight": math.nan},
        {"article_half_life_hours": math.inf},
    ],
)
def test_config_rejects_wrong_types_and_non_finite_values(
    override: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        FeedConfig(**override)  # type: ignore[arg-type]


def test_config_mapping_and_json_are_strict(tmp_path: object) -> None:
    from pathlib import Path

    directory = Path(str(tmp_path))
    path = directory / "config.json"
    path.write_text(json.dumps({"size": 3, "max_per_source": 1}), encoding="utf-8")
    assert FeedConfig.from_json(path).size == 3
    assert FeedConfig.from_mapping({"size": 4}).size == 4
    assert FeedConfig(size=2).to_dict()["size"] == 2
    with pytest.raises(ValueError, match="unknown"):
        FeedConfig.from_mapping({"mystery": 1})
    with pytest.raises(ValueError, match="size"):
        FeedConfig.from_mapping({"size": "three"})
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        FeedConfig.from_json(path)
    path.write_text('{"size": 2, "size": 3}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON field"):
        FeedConfig.from_json(path)
    path.write_text('{"quality_weight": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON number"):
        FeedConfig.from_json(path)
    path.write_text('{"quality_weight": 1e400}', encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        FeedConfig.from_json(path)


def test_config_rejects_overflowing_weight_sum() -> None:
    with pytest.raises(ValueError, match="finite"):
        FeedConfig(
            interest_weight=1e308,
            freshness_weight=1e308,
            quality_weight=1e308,
            novelty_weight=1e308,
            popularity_weight=1e308,
            exploration_weight=1e308,
        )
