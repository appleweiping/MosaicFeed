"""Keep a slate's topic proportions faithful to the reader's own history.

Maximal marginal relevance rewards a slate whose items are unlike *each other*.
That is not the same as a slate that looks like the reader. A reader who spends
four fifths of their attention on one topic is served badly by a feed that
spreads evenly across five, and MMR will happily build that feed: every item is
maximally unlike the last, and the reader's actual proportions appear nowhere in
the objective.

Calibration, following Steck, *Calibrated Recommendations* (RecSys 2018),
measures the gap between the reader's topic distribution and the slate's, and
selects to close it. The two goals are genuinely different and can pull in
opposite directions, so this is offered alongside MMR rather than replacing it:
:func:`calibration_error` reports the gap for any slate, whichever reranker
built it, so the trade can be measured rather than argued.

The divergence is one-sided by design. It penalizes a topic the reader cares
about that the slate under-serves, which is the failure worth catching; a topic
the slate carries that the reader has no history for contributes only through
the mass it takes from the rest.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import fsum, isfinite, log

from mosaicfeed.models import Article, UserProfile

# Steck's smoothing weight. A topic the slate misses entirely would otherwise
# make the divergence infinite, which says only "something is missing" and
# cannot rank two imperfect slates against each other. Mixing a little of the
# reader's own distribution into the slate's keeps the quantity finite and
# still monotone in how badly a topic is under-served.
SMOOTHING = 0.01


def _normalized(weights: Mapping[str, float]) -> dict[str, float]:
    """Turn non-negative weights into a distribution, or return nothing."""

    positive = {
        topic: float(weight)
        for topic, weight in weights.items()
        if isfinite(float(weight)) and float(weight) > 0.0
    }
    total = fsum(positive.values())
    if total <= 0.0:
        return {}
    return {topic: weight / total for topic, weight in positive.items()}


def reader_distribution(profile: UserProfile) -> dict[str, float]:
    """The reader's topic distribution, from the positive part of the profile.

    Negative weights record topics the reader pushed away. They are evidence
    about what to avoid, not about what proportion to serve, so they take no
    share of the distribution rather than a negative one.
    """

    return _normalized(profile.topic_weights)


def slate_distribution(
    ranked_ids: Sequence[str], articles: Mapping[str, Article]
) -> dict[str, float]:
    """The topic distribution of a slate.

    An article divides one unit of attention equally among its own topics, so a
    single-topic article is a full vote for that topic and a five-topic article
    is a fifth of a vote for each. ``Article`` requires at least one topic, so
    there is no voteless case to handle.
    """

    mass: dict[str, float] = {}
    for article_id in ranked_ids:
        article = articles.get(article_id)
        if article is None:
            raise ValueError(f"slate references unknown article: {article_id}")
        share = 1.0 / len(article.topics)
        for topic in article.topics:
            mass[topic] = mass.get(topic, 0.0) + share
    return _normalized(mass)


def calibration_error(
    profile: UserProfile,
    ranked_ids: Sequence[str],
    articles: Mapping[str, Article],
    *,
    smoothing: float = SMOOTHING,
) -> float:
    """Return the smoothed KL divergence from reader to slate, in nats.

    Zero means the slate's proportions match the reader's. Larger values mean a
    topic the reader spends attention on is under-represented. A reader with no
    topic history, or an empty slate, has nothing to be calibrated against and
    scores zero rather than an arbitrary number.
    """

    if not 0.0 < smoothing < 1.0:
        raise ValueError("smoothing must be between 0 and 1, exclusive")
    reader = reader_distribution(profile)
    if not reader:
        return 0.0
    slate = slate_distribution(ranked_ids, articles)
    if not slate:
        return 0.0
    divergence = 0.0
    for topic, share in reader.items():
        blended = (1.0 - smoothing) * slate.get(topic, 0.0) + smoothing * share
        divergence += share * log(share / blended)
    # Floating point can leave a divergence a hair below zero when the two
    # distributions coincide; the quantity is non-negative by construction.
    return max(divergence, 0.0)


def _incremental_error(
    reader: Mapping[str, float], mass: Mapping[str, float], smoothing: float
) -> float:
    """Calibration error for a slate already reduced to topic mass.

    Only reached with the mass of a slate that has just gained an article, and
    every article carries topic mass, so there is no empty case to guard.
    """

    slate = _normalized(mass)
    divergence = 0.0
    for topic, share in reader.items():
        blended = (1.0 - smoothing) * slate.get(topic, 0.0) + smoothing * share
        divergence += share * log(share / blended)
    return max(divergence, 0.0)


def topic_mass(article: Article) -> dict[str, float]:
    """One unit of attention divided among an article's topics."""

    share = 1.0 / len(article.topics)
    return {topic: share for topic in article.topics}


def accumulate(mass: dict[str, float], addition: Iterable[tuple[str, float]]) -> dict[str, float]:
    """Add one article's topic mass to a running total."""

    combined = dict(mass)
    for topic, share in addition:
        combined[topic] = combined.get(topic, 0.0) + share
    return combined


def marginal_objective(
    reader: Mapping[str, float],
    mass: Mapping[str, float],
    article: Article,
    relevance: float,
    *,
    weight: float,
    smoothing: float = SMOOTHING,
) -> float:
    """Score adding one article to a partial slate.

    Steck's objective trades the total relevance of the slate against its
    divergence from the reader. Because relevance is additive and the divergence
    is not, the greedy step is evaluated on the slate that would result rather
    than on the article alone.
    """

    extended = accumulate(dict(mass), topic_mass(article).items())
    return (1.0 - weight) * relevance - weight * _incremental_error(reader, extended, smoothing)


__all__ = [
    "SMOOTHING",
    "accumulate",
    "calibration_error",
    "marginal_objective",
    "reader_distribution",
    "slate_distribution",
    "topic_mass",
]
