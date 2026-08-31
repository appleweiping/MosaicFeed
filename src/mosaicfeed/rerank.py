"""Slate construction with topical novelty and source caps."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from mosaicfeed.config import FeedConfig
from mosaicfeed.models import Article, ScoreBreakdown


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
) -> list[str]:
    """Greedily maximize relevance and novelty under a hard source cap."""

    by_id = {article.id: article for article in articles}
    if len(by_id) != len(articles):
        raise ValueError("articles must have unique ids")
    missing = set(scores) - set(by_id)
    if missing:
        raise ValueError(f"scores reference unknown articles: {', '.join(sorted(missing))}")

    remaining = set(scores)
    selected: list[str] = []
    source_counts: Counter[str] = Counter()
    while remaining and len(selected) < config.size:
        eligible = [
            article_id
            for article_id in remaining
            if source_counts[by_id[article_id].source] < config.max_per_source
        ]
        if not eligible:
            break

        def objective(article_id: str) -> tuple[float, float, str]:
            redundancy = max(
                (
                    topic_similarity(by_id[article_id], by_id[chosen_id])
                    for chosen_id in selected
                ),
                default=0.0,
            )
            relevance = scores[article_id].total
            mmr = config.mmr_lambda * relevance - (1.0 - config.mmr_lambda) * redundancy
            return mmr, relevance, article_id

        chosen = max(eligible, key=objective)
        selected.append(chosen)
        source_counts[by_id[chosen].source] += 1
        remaining.remove(chosen)
    return selected
