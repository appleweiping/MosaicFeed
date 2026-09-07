"""An interval for the logged-policy estimate, and when not to trust it."""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime

import pytest

from mosaicfeed.logged import (
    CONCENTRATION_THRESHOLD,
    MIN_USERS_FOR_INTERVAL,
    LoggedObservation,
    collect_logged_observations,
    effective_sample_size,
    largest_weight_share,
    self_normalized_estimate,
    summarize_logged_policy,
)
from mosaicfeed.metrics import self_normalized_ips
from mosaicfeed.models import Event, EventKind

ORIGIN = datetime(2026, 1, 1, tzinfo=UTC)


def make_event(user: str, article: str, kind: EventKind, *, propensity: float | None) -> Event:
    return Event(user, article, kind, ORIGIN, propensity)


def observation(user: str, reward: float, propensity: float) -> LoggedObservation:
    return LoggedObservation(user_id=user, reward=reward, propensity=propensity)


def population(
    *, users: int, per_user: int, click_rate: float, spread: float = 0.0, seed: int = 0
) -> list[LoggedObservation]:
    rng = random.Random(seed)
    found = []
    for index in range(users):
        rate = click_rate
        if spread:
            rate = min(0.99, max(0.01, rng.gauss(click_rate, spread)))
        propensity = rng.uniform(0.05, 0.9)
        for _ in range(per_user):
            found.append(observation(f"u{index}", 1.0 if rng.random() < rate else 0.0, propensity))
    return found


# ---------------------------------------------------------------------------
# The point estimate.
# ---------------------------------------------------------------------------


def test_equal_propensities_reduce_to_the_plain_rate() -> None:
    observations = [observation("a", 1.0, 0.5), observation("a", 0.0, 0.5)]
    assert self_normalized_estimate(observations) == pytest.approx(0.5)


def test_a_rarely_shown_click_counts_for_more() -> None:
    """The whole point of inverse-propensity weighting."""

    rare = [observation("a", 1.0, 0.1), observation("a", 0.0, 0.9)]
    common = [observation("a", 1.0, 0.9), observation("a", 0.0, 0.1)]
    assert self_normalized_estimate(rare) > 0.5
    assert self_normalized_estimate(common) < 0.5


def test_rescaling_every_propensity_leaves_the_ratio_alone() -> None:
    # The estimator is a ratio, so a common factor cancels.
    base = [observation("a", 1.0, 0.2), observation("a", 0.0, 0.6)]
    scaled = [observation("a", 1.0, 0.1), observation("a", 0.0, 0.3)]
    assert self_normalized_estimate(base) == pytest.approx(self_normalized_estimate(scaled))


def test_no_observations_estimate_nothing() -> None:
    assert self_normalized_estimate([]) == 0.0


def test_an_event_without_a_propensity_is_not_a_zero_propensity_event() -> None:
    events = [
        make_event("u1", "a1", EventKind.CLICK, propensity=0.5),
        make_event("u1", "a2", EventKind.VIEW, propensity=None),
    ]
    collected = collect_logged_observations(events)
    assert len(collected) == 1
    assert collected[0].propensity == 0.5


def test_the_public_metric_still_agrees_with_the_estimator() -> None:
    events = [
        make_event("u1", "a1", EventKind.CLICK, propensity=0.2),
        make_event("u1", "a2", EventKind.VIEW, propensity=0.8),
        make_event("u2", "a3", EventKind.LIKE, propensity=0.4),
    ]
    assert self_normalized_ips(events) == pytest.approx(
        self_normalized_estimate(collect_logged_observations(events))
    )


@pytest.mark.parametrize("kind", [EventKind.CLICK, EventKind.LIKE])
def test_clicks_and_likes_are_rewards(kind: EventKind) -> None:
    collected = collect_logged_observations([make_event("u1", "a1", kind, propensity=0.5)])
    assert collected[0].reward == 1.0


def test_a_view_is_not_a_reward() -> None:
    collected = collect_logged_observations(
        [make_event("u1", "a1", EventKind.VIEW, propensity=0.5)]
    )
    assert collected[0].reward == 0.0


# ---------------------------------------------------------------------------
# Effective sample size.
# ---------------------------------------------------------------------------


def test_equal_weights_give_back_the_observation_count() -> None:
    observations = [observation("a", 0.0, 0.5) for _ in range(20)]
    assert effective_sample_size(observations) == pytest.approx(20.0)


def test_skewed_propensities_shrink_the_effective_count() -> None:
    """A self-normalized estimate can rest on a handful of observations."""

    balanced = [observation("a", 0.0, 0.5) for _ in range(100)]
    skewed = [observation("a", 0.0, 0.001)] + [observation("a", 0.0, 0.9) for _ in range(99)]
    assert effective_sample_size(skewed) < effective_sample_size(balanced)
    assert largest_weight_share(skewed) > largest_weight_share(balanced)


def test_one_observation_carrying_everything_is_reported_as_such() -> None:
    observations = [observation("a", 1.0, 1e-6)] + [observation("a", 0.0, 0.9) for _ in range(500)]
    summary = summarize_logged_policy(observations, resamples=50)
    assert summary.largest_weight_share > 0.99
    assert summary.effective_sample_size < 2.0
    assert summary.concentrated


def test_concentration_is_measured_against_the_observation_count() -> None:
    summary = summarize_logged_policy(
        population(users=20, per_user=10, click_rate=0.3), resamples=50
    )
    assert summary.effective_sample_size >= CONCENTRATION_THRESHOLD * summary.observations
    assert not summary.concentrated


def test_an_empty_set_has_no_effective_sample() -> None:
    assert effective_sample_size([]) == 0.0
    assert largest_weight_share([]) == 0.0


# ---------------------------------------------------------------------------
# The interval.
# ---------------------------------------------------------------------------


def test_the_interval_contains_the_estimate() -> None:
    summary = summarize_logged_policy(population(users=30, per_user=10, click_rate=0.3))
    assert summary.has_interval
    assert summary.lower is not None and summary.upper is not None
    assert summary.lower <= summary.estimate <= summary.upper


def test_the_same_seed_gives_the_same_interval() -> None:
    observations = population(users=15, per_user=8, click_rate=0.4)
    first = summarize_logged_policy(observations, seed=5)
    second = summarize_logged_policy(observations, seed=5)
    assert (first.lower, first.upper) == (second.lower, second.upper)


def test_a_different_seed_moves_the_interval_a_little() -> None:
    observations = population(users=15, per_user=8, click_rate=0.4)
    first = summarize_logged_policy(observations, seed=1)
    second = summarize_logged_policy(observations, seed=2)
    assert first.lower != second.lower
    assert first.lower == pytest.approx(second.lower, abs=0.15)


def test_more_users_narrow_the_interval() -> None:
    small = summarize_logged_policy(population(users=8, per_user=10, click_rate=0.3, seed=1))
    large = summarize_logged_policy(population(users=200, per_user=10, click_rate=0.3, seed=1))
    assert large.width is not None and small.width is not None
    assert large.width < small.width


def test_a_wider_confidence_gives_a_wider_interval() -> None:
    observations = population(users=40, per_user=10, click_rate=0.3)
    narrow = summarize_logged_policy(observations, confidence=0.5)
    wide = summarize_logged_policy(observations, confidence=0.99)
    assert wide.width is not None and narrow.width is not None
    assert wide.width > narrow.width


def test_users_are_resampled_whole_not_their_events() -> None:
    """The reason the user has to be retained at all.

    A user who clicks on everything and one who clicks on nothing must move in
    and out of a resample together. Resampling events would mix them and report
    a narrower interval than the data supports.
    """

    observations = []
    for index in range(20):
        clicks = 1.0 if index % 2 == 0 else 0.0
        observations += [observation(f"u{index}", clicks, 0.5) for _ in range(25)]

    clustered = summarize_logged_policy(observations, seed=3)

    rng = random.Random(3)
    draws = []
    for _ in range(1000):
        pool = [observations[rng.randrange(len(observations))] for _ in range(len(observations))]
        draws.append(self_normalized_estimate(pool))
    draws.sort()
    naive_width = draws[974] - draws[24]

    assert clustered.width is not None
    assert clustered.width > naive_width * 2.0


# ---------------------------------------------------------------------------
# When there is not enough to bound.
# ---------------------------------------------------------------------------


def test_too_few_users_earn_no_interval_and_a_reason() -> None:
    """One cluster resampled is the same cluster every time, and reads as certainty."""

    observations = [observation("only", 1.0, 0.5), observation("only", 0.0, 0.5)]
    summary = summarize_logged_policy(observations)
    assert not summary.has_interval
    assert summary.estimate == pytest.approx(0.5)
    assert f"below the {MIN_USERS_FOR_INTERVAL}" in summary.reason


def test_no_propensities_earn_no_interval_and_a_reason() -> None:
    summary = summarize_logged_policy([])
    assert not summary.has_interval
    assert summary.estimate == 0.0
    assert "no logged event carried a propensity" in summary.reason


def test_a_bounded_summary_names_the_method_it_used() -> None:
    summary = summarize_logged_policy(population(users=10, per_user=5, click_rate=0.3))
    assert "clustered bootstrap" in summary.reason


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 1.5])
def test_an_impossible_confidence_is_refused(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        summarize_logged_policy(
            population(users=5, per_user=4, click_rate=0.3), confidence=confidence
        )


@pytest.mark.parametrize("resamples", [0, -1, 2.5, True])
def test_a_bad_resample_count_is_refused(resamples: object) -> None:
    with pytest.raises(ValueError, match="resamples"):
        summarize_logged_policy(
            population(users=5, per_user=4, click_rate=0.3),
            resamples=resamples,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_a_bad_seed_is_refused(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        summarize_logged_policy(
            population(users=5, per_user=4, click_rate=0.3),
            seed=seed,  # type: ignore[arg-type]
        )


def test_a_summary_serializes(tmp_path) -> None:
    payload = summarize_logged_policy(population(users=10, per_user=5, click_rate=0.3)).to_dict()
    assert payload["users"] == 10
    assert payload["lower"] <= payload["estimate"] <= payload["upper"]
    assert all(math.isfinite(payload[key]) for key in ("estimate", "lower", "upper"))
    json.dumps(payload, allow_nan=False)


def test_an_unbounded_summary_serializes_its_nulls() -> None:
    payload = summarize_logged_policy([observation("only", 1.0, 0.5)]).to_dict()
    assert payload["lower"] is None
    assert payload["upper"] is None
    assert payload["width"] is None
    json.dumps(payload, allow_nan=False)
