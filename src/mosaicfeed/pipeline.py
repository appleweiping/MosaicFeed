"""End-to-end profile, score, and slate construction."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from mosaicfeed.config import FeedConfig
from mosaicfeed.models import Article, Event, Feed, Recommendation
from mosaicfeed.profile import build_profile
from mosaicfeed.rerank import rerank
from mosaicfeed.scoring import score_candidates


def build_feed(
    user_id: str,
    articles: Iterable[Article],
    events: Iterable[Event],
    *,
    as_of: datetime,
    config: FeedConfig | None = None,
) -> Feed:
    """Build a deterministic point-in-time feed with no future leakage."""

    active_config = config or FeedConfig()
    article_list = list(articles)
    article_map = {article.id: article for article in article_list}
    if len(article_map) != len(article_list):
        raise ValueError("articles must have unique ids")
    profile = build_profile(
        user_id,
        events,
        article_map,
        as_of=as_of,
        config=active_config,
    )
    scores = score_candidates(article_list, profile, as_of=as_of, config=active_config)
    chosen_ids = rerank(article_list, scores, config=active_config)
    recommendations = tuple(
        Recommendation(
            article_id=article_id,
            rank=rank,
            score=scores[article_id].total,
            breakdown=scores[article_id],
        )
        for rank, article_id in enumerate(chosen_ids, start=1)
    )
    return Feed(user_id=user_id, generated_at=as_of, recommendations=recommendations)
