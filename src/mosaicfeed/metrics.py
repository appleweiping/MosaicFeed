"""Offline ranking and slate diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from mosaicfeed.config import FeedConfig
from mosaicfeed.logged import (
    LoggedObservation,
    LoggedPolicySummary,
    collect_logged_observations,
    self_normalized_estimate,
    summarize_logged_policy,
)
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
        recommendation.article_id for feed in feeds for recommendation in feed.recommendations
    }
    if len(exposed) > catalog_size:
        raise ValueError("exposed article count exceeds catalog_size")
    return len(exposed) / catalog_size if catalog_size else 0.0


def exposure_gini(feeds: Iterable[Feed]) -> float:
    counts = Counter(
        recommendation.article_id for feed in feeds for recommendation in feed.recommendations
    )
    values = sorted(counts.values())
    if not values:
        return 0.0
    total = sum(values)
    weighted = sum((index + 1) * value for index, value in enumerate(values))
    return (2.0 * weighted) / (len(values) * total) - (len(values) + 1) / len(values)


def self_normalized_ips(events: Iterable[Event]) -> float:
    """Estimate click rate from logged events with known propensities.

    Kept as the point estimate the report has always carried. The uncertainty
    around it lives in `mosaicfeed.logged`, which needs the user each event
    belongs to and so cannot be answered from a flat event list.
    """

    return self_normalized_estimate(collect_logged_observations(events))


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
    logged_policy: LoggedPolicySummary | None = None

    def to_dict(self) -> dict[str, Any]:
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
            "logged_policy": self.logged_policy.to_dict() if self.logged_policy else None,
        }


@dataclass(frozen=True, slots=True)
class UserEvaluation:
    """Point-in-time result for one held-out positive interaction."""

    user_id: str
    holdout_article_id: str
    holdout_at: datetime
    ranked_ids: tuple[str, ...]
    eligible_catalog_ids: frozenset[str]
    ndcg: float
    hit_rate: float
    reciprocal_rank: float
    intra_list_diversity: float
    source_diversity: float


@dataclass(frozen=True, slots=True)
class EvaluationSamples:
    """Per-user observations plus slate-wide diagnostics."""

    users: tuple[UserEvaluation, ...]
    users_skipped: int
    k: int
    catalog_coverage: float
    exposure_gini: float
    logged_ips_ctr: float
    eligible_catalog_size: int
    logged_observations: tuple[LoggedObservation, ...] = ()

    def logged_policy(
        self, *, confidence: float = 0.95, resamples: int = 1000, seed: int = 0
    ) -> LoggedPolicySummary:
        """Bound the logged-policy estimate by a bootstrap over users.

        Retained here rather than computed during evaluation because the
        resample count and confidence are the caller's to choose, and because
        a thousand resamples of a large log is work a caller may not want.
        """

        return summarize_logged_policy(
            self.logged_observations,
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        )

    def report(self) -> EvaluationReport:
        """Aggregate the retained observations into the stable public report."""

        def mean(values: Iterable[float]) -> float:
            items = tuple(values)
            return math.fsum(items) / len(items) if items else 0.0

        return EvaluationReport(
            users_evaluated=len(self.users),
            users_skipped=self.users_skipped,
            k=self.k,
            ndcg=mean(item.ndcg for item in self.users),
            hit_rate=mean(item.hit_rate for item in self.users),
            mean_reciprocal_rank=mean(item.reciprocal_rank for item in self.users),
            intra_list_diversity=mean(item.intra_list_diversity for item in self.users),
            source_diversity=mean(item.source_diversity for item in self.users),
            catalog_coverage=self.catalog_coverage,
            exposure_gini=self.exposure_gini,
            logged_ips_ctr=self.logged_ips_ctr,
            logged_policy=self.logged_policy() if self.logged_observations else None,
        )


def evaluate_leave_last_out_samples(
    articles: Sequence[Article],
    events: Sequence[Event],
    *,
    as_of: datetime,
    config: FeedConfig,
    k: int | None = None,
) -> EvaluationSamples:
    """Retain per-user leave-last-out observations for statistical analysis."""

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
    user_results: list[UserEvaluation] = []
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
            if event.user_id == user_id and event.occurred_at < holdout.occurred_at
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
        user_results.append(
            UserEvaluation(
                user_id=user_id,
                holdout_article_id=holdout.article_id,
                holdout_at=holdout.occurred_at,
                ranked_ids=tuple(ranked[:cutoff]),
                eligible_catalog_ids=frozenset(article.id for article in available),
                ndcg=ndcg_at_k(ranked, relevant, cutoff),
                hit_rate=hit_rate_at_k(ranked, relevant, cutoff),
                reciprocal_rank=reciprocal_rank(ranked, relevant, cutoff),
                intra_list_diversity=intra_list_diversity(ranked[:cutoff], article_map),
                source_diversity=source_diversity(ranked[:cutoff], article_map),
            )
        )

    logged_events = [event for event in events if event.occurred_at <= as_of]
    return EvaluationSamples(
        users=tuple(user_results),
        users_skipped=skipped,
        k=cutoff,
        catalog_coverage=catalog_coverage(feeds, len(eligible_catalog_ids)),
        exposure_gini=exposure_gini(feeds),
        logged_ips_ctr=self_normalized_ips(logged_events),
        eligible_catalog_size=len(eligible_catalog_ids),
        logged_observations=collect_logged_observations(logged_events),
    )


def evaluate_leave_last_out(
    articles: Sequence[Article],
    events: Sequence[Event],
    *,
    as_of: datetime,
    config: FeedConfig,
    k: int | None = None,
) -> EvaluationReport:
    """Evaluate each user on their last positive event at or before ``as_of``."""

    return evaluate_leave_last_out_samples(
        articles,
        events,
        as_of=as_of,
        config=config,
        k=k,
    ).report()
