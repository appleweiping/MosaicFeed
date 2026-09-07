"""An interval for the logged-policy estimate, and when not to trust it.

The report carried `logged_ips_ctr` as a bare number. Every other metric gained
a bootstrap interval; this one could not, because the estimate was computed
from a flat list of events and the user each one belonged to was discarded on
the way in. Without that, there is no sampling unit to resample.

The sampling unit is the user, not the event. One user contributes many
correlated events, and resampling events would treat a hundred impressions from
one heavy user as a hundred independent observations.

The committed synthetic regression demonstrates the practical failure: when
users differ in how often they click, resampling events reports a materially
narrower interval than resampling whole users. A bare number invites caution
and a confident interval based on a false independence assumption does not, so
the event-level bootstrap is not offered.

The estimator is self-normalized, which makes it a ratio of two random sums
rather than a mean. A bootstrap resample therefore has to recompute the whole
ratio over the pooled observations of the resampled users. Averaging per-user
ratios would answer a different question and would break entirely for a user
whose weights sum to zero.

The effective sample size is reported alongside, because a self-normalized
estimate can rest almost entirely on a handful of observations when the logging
propensities were small. Kish's formula gives the number of equally weighted
observations carrying the same information. When it collapses to a small
fraction of the observations, the interval is honest about being wide and the
estimate is still resting on very little.

Nothing here makes the result causal. The interval is sampling uncertainty for
this logged population under its own logging policy. It does not correct for a
propensity model that was wrong, for actions the logging policy never took, or
for anything that changed between logging and now.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from mosaicfeed.models import Event, EventKind

#: Users below which no interval is attempted. A bootstrap over one cluster
#: resamples the same cluster every time and reports zero width, which would
#: read as certainty rather than as having one user.
MIN_USERS_FOR_INTERVAL = 3

#: Default resample count. Enough for a stable percentile interval at the two
#: and a half percent tails without making an evaluation noticeably slower.
DEFAULT_RESAMPLES = 1000

#: Default two-sided coverage.
DEFAULT_CONFIDENCE = 0.95

#: Effective sample size below this share of the observations is reported as a
#: concentration warning. At a tenth, nine tenths of the data is contributing
#: almost nothing and the estimate rests on the remainder.
CONCENTRATION_THRESHOLD = 0.1

REWARD_KINDS = frozenset({EventKind.CLICK, EventKind.LIKE})


@dataclass(frozen=True, slots=True)
class LoggedObservation:
    """One logged action, kept with the user it belongs to."""

    user_id: str
    reward: float
    propensity: float


@dataclass(frozen=True, slots=True)
class LoggedPolicySummary:
    """The logged-policy estimate with its uncertainty and its caveats."""

    estimate: float
    users: int
    observations: int
    effective_sample_size: float
    largest_weight_share: float
    lower: float | None
    upper: float | None
    confidence: float
    resamples: int
    reason: str

    @property
    def has_interval(self) -> bool:
        return self.lower is not None and self.upper is not None

    @property
    def concentrated(self) -> bool:
        """Whether the estimate rests on a small share of the observations."""

        if not self.observations:
            return False
        return self.effective_sample_size < CONCENTRATION_THRESHOLD * self.observations

    @property
    def width(self) -> float | None:
        if self.lower is None or self.upper is None:
            return None
        return self.upper - self.lower

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "users": self.users,
            "observations": self.observations,
            "effective_sample_size": self.effective_sample_size,
            "largest_weight_share": self.largest_weight_share,
            "concentrated": self.concentrated,
            "lower": self.lower,
            "upper": self.upper,
            "width": self.width,
            "confidence": self.confidence,
            "resamples": self.resamples,
            "reason": self.reason,
        }


def collect_logged_observations(events: Iterable[Event]) -> tuple[LoggedObservation, ...]:
    """Keep every event that carries a propensity, with its user.

    An event without a propensity is not a zero-propensity event; it is an
    event the logging policy did not record one for, and including it would
    invent a weight.
    """

    return tuple(
        LoggedObservation(
            user_id=event.user_id,
            reward=1.0 if event.kind in REWARD_KINDS else 0.0,
            propensity=event.propensity,
        )
        for event in events
        if event.propensity is not None
    )


def _weights(observations: Sequence[LoggedObservation]) -> list[float]:
    """Inverse-propensity weights, rescaled by the smallest propensity.

    The estimator is a ratio, so a common factor cancels. Rescaling keeps every
    weight at or below one, which stops a very small propensity from producing
    a sum large enough to lose precision.
    """

    smallest = min(item.propensity for item in observations)
    return [smallest / item.propensity for item in observations]


def self_normalized_estimate(observations: Sequence[LoggedObservation]) -> float:
    """Self-normalized inverse-propensity estimate of the reward rate."""

    if not observations:
        return 0.0
    weights = _weights(observations)
    total = math.fsum(weights)
    if total <= 0.0:
        return 0.0
    return (
        math.fsum(item.reward * weight for item, weight in zip(observations, weights, strict=True))
        / total
    )


def effective_sample_size(observations: Sequence[LoggedObservation]) -> float:
    """Kish's effective sample size: equally weighted observations of equal information.

    Equal weights give back the observation count. Weights concentrated on a
    few observations give a much smaller number, which is the honest count of
    what the estimate actually rests on.
    """

    if not observations:
        return 0.0
    weights = _weights(observations)
    total = math.fsum(weights)
    squared = math.fsum(weight * weight for weight in weights)
    if squared <= 0.0:
        return 0.0
    return (total * total) / squared


def largest_weight_share(observations: Sequence[LoggedObservation]) -> float:
    """Share of the total weight carried by the single heaviest observation."""

    if not observations:
        return 0.0
    weights = _weights(observations)
    total = math.fsum(weights)
    return max(weights) / total if total > 0.0 else 0.0


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile of an already sorted sequence."""

    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_logged_policy(
    observations: Sequence[LoggedObservation],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> LoggedPolicySummary:
    """Estimate the logged reward rate and bound it by a clustered bootstrap.

    Users are resampled with replacement and their observations pooled, so a
    user contributes all of their events or none of them. Resampling events
    instead would treat a hundred impressions from one heavy user as a hundred
    independent observations and report an interval several times too narrow.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    materialized = tuple(observations)
    estimate = self_normalized_estimate(materialized)
    by_user: dict[str, list[LoggedObservation]] = {}
    for item in materialized:
        by_user.setdefault(item.user_id, []).append(item)
    users = sorted(by_user)

    summary = LoggedPolicySummary(
        estimate=estimate,
        users=len(users),
        observations=len(materialized),
        effective_sample_size=effective_sample_size(materialized),
        largest_weight_share=largest_weight_share(materialized),
        lower=None,
        upper=None,
        confidence=confidence,
        resamples=resamples,
        reason="",
    )
    if not materialized:
        return _with_reason(summary, "no logged event carried a propensity")
    if len(users) < MIN_USERS_FOR_INTERVAL:
        return _with_reason(
            summary,
            f"{len(users)} user(s) is below the {MIN_USERS_FOR_INTERVAL} a clustered "
            "bootstrap needs; resampling one cluster reports zero width",
        )

    # The estimate is a ratio of two sums over the pooled observations, and a
    # sum over a pool of users is the sum of each user's own totals. Computing
    # those once makes each resample proportional to the number of users rather
    # than the number of observations.
    #
    # The weights carry a common factor of the smallest propensity, which
    # cancels in the ratio, so per-user totals computed against the global
    # smallest are exactly the totals a resample would have produced itself.
    smallest = min(item.propensity for item in materialized)
    totals: dict[str, tuple[float, float]] = {}
    for user in users:
        weights = [smallest / item.propensity for item in by_user[user]]
        totals[user] = (
            math.fsum(
                item.reward * weight for item, weight in zip(by_user[user], weights, strict=True)
            ),
            math.fsum(weights),
        )
    ordered_totals = [totals[user] for user in users]

    rng = random.Random(seed)
    draws: list[float] = []
    count = len(users)
    for _ in range(resamples):
        rewarded = 0.0
        weight = 0.0
        for _ in range(count):
            pair = ordered_totals[rng.randrange(count)]
            rewarded += pair[0]
            weight += pair[1]
        # A resample can miss every rewarding user, which is a real draw and
        # belongs in the interval rather than being discarded. A resample whose
        # weights sum to zero cannot form a ratio, and is reported as zero for
        # the same reason the point estimate is.
        draws.append(rewarded / weight if weight > 0.0 else 0.0)
    draws.sort()
    tail = (1.0 - confidence) / 2.0
    return _with_bounds(
        summary,
        _percentile(draws, tail),
        _percentile(draws, 1.0 - tail),
        "clustered bootstrap over users",
    )


def _with_reason(summary: LoggedPolicySummary, reason: str) -> LoggedPolicySummary:
    return LoggedPolicySummary(
        estimate=summary.estimate,
        users=summary.users,
        observations=summary.observations,
        effective_sample_size=summary.effective_sample_size,
        largest_weight_share=summary.largest_weight_share,
        lower=None,
        upper=None,
        confidence=summary.confidence,
        resamples=summary.resamples,
        reason=reason,
    )


def _with_bounds(
    summary: LoggedPolicySummary, lower: float, upper: float, reason: str
) -> LoggedPolicySummary:
    return LoggedPolicySummary(
        estimate=summary.estimate,
        users=summary.users,
        observations=summary.observations,
        effective_sample_size=summary.effective_sample_size,
        largest_weight_share=summary.largest_weight_share,
        lower=lower,
        upper=upper,
        confidence=summary.confidence,
        resamples=summary.resamples,
        reason=reason,
    )
