"""Offline ranking and slate diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from mosaicfeed.config import FeedConfig
from mosaicfeed.models import Article, Event, EventKind, Feed
from mosaicfeed.pipeline import build_feed
from mosaicfeed.rerank import topic_similarity


def _validate_ranking(ranked_ids: Sequence[str], k: int | None = None) -> None:
    if k is not None and k < 1:
        raise ValueError("k must be positive")
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("ranking must not contain duplicate article ids")


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    _validate_ranking(ranked_ids, k)
    if not relevant_ids:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, article_id in enumerate(ranked_ids[:k], start=1)
        if article_id in relevant_ids
    )
    ideal_hits = min(k, len(relevant_ids))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    _validate_ranking(ranked_ids, k)
    for rank, article_id in enumerate(ranked_ids[:k], start=1):
        if article_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    _validate_ranking(ranked_ids, k)
    return float(bool(set(ranked_ids[:k]) & relevant_ids))


def intra_list_diversity(ranked_ids: Sequence[str], articles: Mapping[str, Article]) -> float:
    _validate_ranking(ranked_ids)
    if len(ranked_ids) < 2:
        return 0.0
    distances: list[float] = []
    for index, left_id in enumerate(ranked_ids):
        if left_id not in articles:
            raise ValueError(f"unknown article in ranking: {left_id}")
        for right_id in ranked_ids[index + 1 :]:
            if right_id not in articles:
                raise ValueError(f"unknown article in ranking: {right_id}")
            distances.append(1.0 - topic_similarity(articles[left_id], articles[right_id]))
    return sum(distances) / len(distances)


def source_diversity(ranked_ids: Sequence[str], articles: Mapping[str, Article]) -> float:
    _validate_ranking(ranked_ids)
    if not ranked_ids:
        return 0.0
    try:
        return len({articles[article_id].source for article_id in ranked_ids}) / len(ranked_ids)
    except KeyError as error:
        raise ValueError(f"unknown article in ranking: {error.args[0]}") from error


def catalog_coverage(feeds: Iterable[Feed], catalog_size: int) -> float:
    if catalog_size < 0:
        raise ValueError("catalog_size must not be negative")
    exposed = {
        recommendation.article_id
        for feed in feeds
        for recommendation in feed.recommendations
    }
    if len(exposed) > catalog_size:
        raise ValueError("exposed article count exceeds catalog_size")
    return len(exposed) / catalog_size if catalog_size else 0.0


def exposure_gini(feeds: Iterable[Feed]) -> float:
    counts = Counter(
        recommendation.article_id
        for feed in feeds
        for recommendation in feed.recommendations
    )
    values = sorted(counts.values())
    if not values:
        return 0.0
    total = sum(values)
    weighted = sum((index + 1) * value for index, value in enumerate(values))
    return (2.0 * weighted) / (len(values) * total) - (len(values) + 1) / len(values)


def self_normalized_ips(events: Iterable[Event]) -> float:
    """Estimate click rate from logged events with known propensities."""

    observations = [
        (float(event.kind in {EventKind.CLICK, EventKind.LIKE}), event.propensity)
        for event in events
        if event.propensity is not None
    ]
    if not observations:
        return 0.0
    minimum_propensity = min(propensity for _, propensity in observations)
    normalized = [
        (reward, minimum_propensity / propensity) for reward, propensity in observations
    ]
    weighted_rewards = math.fsum(reward * weight for reward, weight in normalized)
    total_weight = math.fsum(weight for _, weight in normalized)
    return weighted_rewards / total_weight


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    users_evaluated: int
    users_skipped: int
    k: int
    ndcg: float
    hit_rate: float
    mean_reciprocal_rank: float
    intra_list_diversity: float
    source_diversity: float
    catalog_coverage: float
    exposure_gini: float
    logged_ips_ctr: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "users_evaluated": self.users_evaluated,
            "users_skipped": self.users_skipped,
            "k": self.k,
            "ndcg": self.ndcg,
            "hit_rate": self.hit_rate,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "intra_list_diversity": self.intra_list_diversity,
            "source_diversity": self.source_diversity,
            "catalog_coverage": self.catalog_coverage,
            "exposure_gini": self.exposure_gini,
            "logged_ips_ctr": self.logged_ips_ctr,
        }


def evaluate_leave_last_out(
    articles: Sequence[Article],
    events: Sequence[Event],
    *,
    as_of: datetime,
    config: FeedConfig,
    k: int | None = None,
) -> EvaluationReport:
    """Evaluate each user on their last positive event at or before ``as_of``."""

    if not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    cutoff = config.size if k is None else k
    if cutoff < 1:
        raise ValueError("k must be positive")
    article_map = {article.id: article for article in articles}
    if len(article_map) != len(articles):
        raise ValueError("articles must have unique ids")
    user_ids = sorted({event.user_id for event in events if event.occurred_at <= as_of})
    feeds: list[Feed] = []
    ndcg_values: list[float] = []
    hits: list[float] = []
    reciprocal_values: list[float] = []
    ild_values: list[float] = []
    source_values: list[float] = []
    eligible_catalog_ids: set[str] = set()
    skipped = 0

    for user_id in user_ids:
        positives = [
            event
            for event in events
            if event.user_id == user_id
            and event.occurred_at <= as_of
            and event.kind in {EventKind.CLICK, EventKind.LIKE}
            and event.article_id in article_map
        ]
        if not positives:
            skipped += 1
            continue
        holdout = max(positives, key=lambda event: (event.occurred_at, event.article_id))
        training = [
            event
            for event in events
            if event.user_id == user_id
            and event.occurred_at < holdout.occurred_at
        ]
        available = [article for article in articles if article.published_at <= holdout.occurred_at]
        eligible_catalog_ids.update(article.id for article in available)
        feed = build_feed(
            user_id,
            available,
            training,
            as_of=holdout.occurred_at,
            config=config,
        )
        ranked = [recommendation.article_id for recommendation in feed.recommendations]
        relevant = {holdout.article_id}
        feeds.append(Feed(feed.user_id, feed.generated_at, feed.recommendations[:cutoff]))
        ndcg_values.append(ndcg_at_k(ranked, relevant, cutoff))
        hits.append(hit_rate_at_k(ranked, relevant, cutoff))
        reciprocal_values.append(reciprocal_rank(ranked, relevant, cutoff))
        ild_values.append(intra_list_diversity(ranked[:cutoff], article_map))
        source_values.append(source_diversity(ranked[:cutoff], article_map))

    evaluated = len(feeds)

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return EvaluationReport(
        users_evaluated=evaluated,
        users_skipped=skipped,
        k=cutoff,
        ndcg=mean(ndcg_values),
        hit_rate=mean(hits),
        mean_reciprocal_rank=mean(reciprocal_values),
        intra_list_diversity=mean(ild_values),
        source_diversity=mean(source_values),
        catalog_coverage=catalog_coverage(feeds, len(eligible_catalog_ids)),
        exposure_gini=exposure_gini(feeds),
        logged_ips_ctr=self_normalized_ips(
            event for event in events if event.occurred_at <= as_of
        ),
    )
