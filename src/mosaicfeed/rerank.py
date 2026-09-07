"""Slate construction with topical novelty and source caps."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from mosaicfeed.calibration import accumulate, marginal_objective, reader_distribution, topic_mass
from mosaicfeed.config import FeedConfig
from mosaicfeed.models import Article, ScoreBreakdown, UserProfile


def topic_similarity(left: Article, right: Article) -> float:
    left_topics = set(left.topics)
    right_topics = set(right.topics)
    union = left_topics | right_topics
    return len(left_topics & right_topics) / len(union) if union else 0.0


def rerank(
    articles: Sequence[Article],
    scores: Mapping[str, ScoreBreakdown],
    *,
    config: FeedConfig,
    profile: UserProfile | None = None,
) -> list[str]:
    """Greedily assemble a slate under a hard source cap.

    The default objective rewards relevance and topical novelty. The calibrated
    objective instead rewards a slate whose topic proportions match the reader's,
    which needs the profile; asking for it without one is refused rather than
    quietly falling back, because the two strategies build different slates and a
    silent substitution would be invisible in the output.
    """

    if config.rerank_strategy == "calibrated" and profile is None:
        raise ValueError("the calibrated rerank strategy requires a user profile")
    by_id = {article.id: article for article in articles}
    if len(by_id) != len(articles):
        raise ValueError("articles must have unique ids")
    missing = set(scores) - set(by_id)
    if missing:
        raise ValueError(f"scores reference unknown articles: {', '.join(sorted(missing))}")

    remaining = set(scores)
    selected: list[str] = []
    source_counts: Counter[str] = Counter()
    reader = reader_distribution(profile) if profile is not None else {}
    slate_mass: dict[str, float] = {}
    while remaining and len(selected) < config.size:
        eligible = [
            article_id
            for article_id in remaining
            if source_counts[by_id[article_id].source] < config.max_per_source
        ]
        if not eligible:
            break

        # Scored in place rather than through a closure: the calibrated
        # objective depends on the slate built so far, and a closure over that
        # running value is the kind of late binding worth not writing.
        ranked: list[tuple[float, float, str]] = []
        for article_id in eligible:
            relevance = scores[article_id].total
            if config.rerank_strategy == "calibrated":
                value = marginal_objective(
                    reader,
                    slate_mass,
                    by_id[article_id],
                    relevance,
                    weight=config.calibration_weight,
                )
            else:
                redundancy = max(
                    (
                        topic_similarity(by_id[article_id], by_id[chosen_id])
                        for chosen_id in selected
                    ),
                    default=0.0,
                )
                value = config.mmr_lambda * relevance - (1.0 - config.mmr_lambda) * redundancy
            ranked.append((value, relevance, article_id))

        chosen = max(ranked)[2]
        selected.append(chosen)
        source_counts[by_id[chosen].source] += 1
        slate_mass = accumulate(slate_mass, topic_mass(by_id[chosen]).items())
        remaining.remove(chosen)
    return selected
