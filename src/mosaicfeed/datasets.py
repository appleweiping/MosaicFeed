"""Adapters for public research datasets without network access."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path

from mosaicfeed.models import Article, Event, EventKind


@dataclass(frozen=True, slots=True)
class MindDataset:
    """The loss-aware subset of MIND fields that MosaicFeed can represent."""

    articles: tuple[Article, ...]
    events: tuple[Event, ...]
    impressions: int
    ignored_history_items: int


def fixed_offset(hours: float) -> tzinfo:
    """Construct and validate the timezone used by timestamp-naive MIND logs."""

    if isinstance(hours, bool) or not isinstance(hours, (int, float)):
        raise ValueError("UTC offset must be a finite number between -14 and 14")
    try:
        value = float(hours)
    except OverflowError as error:
        raise ValueError("UTC offset must be a finite number between -14 and 14") from error
    if not math.isfinite(value) or not -14.0 <= value <= 14.0:
        raise ValueError("UTC offset must be a finite number between -14 and 14")
    seconds = value * 3_600.0
    if not seconds.is_integer():
        raise ValueError("UTC offset must resolve to whole seconds")
    return timezone(timedelta(seconds=int(seconds)))


def _mind_time(value: str, zone: tzinfo, *, line_number: int) -> datetime:
    try:
        parsed = datetime.strptime(value, "%m/%d/%Y %I:%M:%S %p")
    except ValueError as error:
        raise ValueError(f"invalid MIND behavior time on line {line_number}: {value}") from error
    return parsed.replace(tzinfo=zone)


def _read_tsv(path: str | Path) -> list[tuple[int, list[str]]]:
    source = Path(path)
    rows: list[tuple[int, list[str]]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if line:
            rows.append((line_number, line.split("\t")))
    return rows


def load_mind(
    news_path: str | Path,
    behaviors_path: str | Path,
    *,
    catalog_published_at: datetime,
    behavior_timezone: tzinfo,
) -> MindDataset:
    """Load MIND ``news.tsv`` and ``behaviors.tsv`` as an honest click replay.

    MIND does not expose per-article publication timestamps or publishers. The
    caller must therefore declare one catalog availability time, and the adapter
    uses the news category as a source proxy. Unclicked impressions and undated
    history entries are not converted into preference events.
    """

    if (
        not isinstance(catalog_published_at, datetime)
        or catalog_published_at.tzinfo is None
        or catalog_published_at.utcoffset() is None
    ):
        raise ValueError("catalog_published_at must be timezone-aware")
    if not isinstance(behavior_timezone, tzinfo):
        raise ValueError("behavior_timezone must be a tzinfo instance")
    try:
        zone_offset = behavior_timezone.utcoffset(None)
    except (OverflowError, ValueError) as error:
        raise ValueError("behavior_timezone must have a fixed UTC offset") from error
    if zone_offset is None:
        raise ValueError("behavior_timezone must have a fixed UTC offset")

    articles: list[Article] = []
    article_ids: set[str] = set()
    for line_number, fields in _read_tsv(news_path):
        if len(fields) != 8:
            raise ValueError(f"MIND news line {line_number} must contain 8 tab-separated fields")
        (
            article_id,
            category,
            subcategory,
            title,
            abstract,
            _url,
            _title_entities,
            _abstract_entities,
        ) = fields
        if article_id in article_ids:
            raise ValueError(f"duplicate MIND news id on line {line_number}: {article_id}")
        article_ids.add(article_id)
        topics = tuple(value for value in (category, subcategory) if value.strip())
        if not topics:
            topics = ("uncategorized",)
        articles.append(
            Article(
                id=article_id,
                title=title,
                summary=abstract,
                topics=topics,
                source=category or "uncategorized",
                published_at=catalog_published_at,
                quality=0.5,
                popularity=0.0,
            )
        )

    events: list[Event] = []
    impression_ids: set[str] = set()
    ignored_history_items = 0
    for line_number, fields in _read_tsv(behaviors_path):
        if len(fields) != 5:
            raise ValueError(
                f"MIND behavior line {line_number} must contain 5 tab-separated fields"
            )
        impression_id, user_id, raw_time, history, impressions = fields
        if not impression_id:
            raise ValueError(f"MIND impression id is empty on line {line_number}")
        if not user_id:
            raise ValueError(f"MIND user id is empty on line {line_number}")
        if impression_id in impression_ids:
            raise ValueError(f"duplicate MIND impression id on line {line_number}: {impression_id}")
        impression_ids.add(impression_id)
        occurred_at = _mind_time(raw_time, behavior_timezone, line_number=line_number)
        history_ids = history.split() if history.strip() else []
        unknown_history = sorted(set(history_ids) - article_ids)
        if unknown_history:
            raise ValueError(
                f"MIND history on line {line_number} references unknown news: {unknown_history}"
            )
        ignored_history_items += len(history_ids)
        impression_tokens = impressions.split()
        if not impression_tokens:
            raise ValueError(f"MIND impressions are empty on line {line_number}")
        seen_articles: set[str] = set()
        for impression in impression_tokens:
            article_id, separator, label = impression.rpartition("-")
            if not separator or not article_id or label not in {"0", "1"}:
                raise ValueError(
                    f"invalid MIND impression token on line {line_number}: {impression}"
                )
            if article_id not in article_ids:
                raise ValueError(
                    f"MIND impression on line {line_number} references unknown news: {article_id}"
                )
            if article_id in seen_articles:
                raise ValueError(
                    f"duplicate MIND impression article on line {line_number}: {article_id}"
                )
            seen_articles.add(article_id)
            if label == "1":
                events.append(Event(user_id, article_id, EventKind.CLICK, occurred_at))

    return MindDataset(
        articles=tuple(articles),
        events=tuple(events),
        impressions=len(impression_ids),
        ignored_history_items=ignored_history_items,
    )
