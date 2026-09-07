from __future__ import annotations

from datetime import datetime

import pytest

from mosaicfeed.simulation import expected_random_hit_rate, generate_synthetic, wilson_interval


def test_synthetic_dataset_is_repeatable_and_isolated() -> None:
    first = generate_synthetic(seed=9, users=3, articles=12, events_per_user=4)
    second = generate_synthetic(seed=9, users=3, articles=12, events_per_user=4)
    other = generate_synthetic(seed=10, users=3, articles=12, events_per_user=4)
    assert first == second
    assert first != other
    assert len(first.articles) == 12
    assert len(first.events) == 12
    assert len(first.user_ids) == 3
    article_map = {article.id: article for article in first.articles}
    assert all(
        article_map[event.article_id].published_at <= event.occurred_at for event in first.events
    )
    assert all(event.occurred_at <= first.as_of for event in first.events)


def test_synthetic_caps_events_at_catalog_size() -> None:
    dataset = generate_synthetic(users=2, articles=3, events_per_user=10)
    assert len(dataset.events) == 6


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("users", 0),
        ("articles", 0),
        ("events_per_user", -1),
    ],
)
def test_synthetic_rejects_invalid_sizes(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        if field == "users":
            generate_synthetic(users=value)
        elif field == "articles":
            generate_synthetic(articles=value)
        else:
            generate_synthetic(events_per_user=value)


def test_synthetic_requires_aware_clock() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        generate_synthetic(as_of=datetime(2026, 1, 1))


def test_expected_random_hit_rate() -> None:
    assert expected_random_hit_rate(100, 10) == 0.1
    assert expected_random_hit_rate(3, 10) == 1.0
    with pytest.raises(ValueError):
        expected_random_hit_rate(0, 1)
    with pytest.raises(ValueError):
        expected_random_hit_rate(2, -1)


def test_wilson_interval_bounds_observed_rate() -> None:
    low, high = wilson_interval(50, 100)
    assert 0 < low < 0.5 < high < 1
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0


@pytest.mark.parametrize(
    ("successes", "trials", "z"),
    [(-1, 10, 1.96), (11, 10, 1.96), (1, 0, 1.96), (1, 10, 0)],
)
def test_wilson_interval_rejects_invalid_inputs(successes: int, trials: int, z: float) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, trials, z)
