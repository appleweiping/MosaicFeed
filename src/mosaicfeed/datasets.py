"""Adapters for public research datasets without network access."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path

from mosaicfeed.models import MISSING_MIND_TITLE, Article, Event, EventKind

MAX_MIND_SOURCE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class MindCandidate:
    """One labeled candidate in a MIND impression, in source-file order."""

    article_id: str
    clicked: bool

    def __post_init__(self) -> None:
        if not isinstance(self.article_id, str) or not self.article_id.strip():
            raise ValueError("MIND candidate article_id must be a non-empty string")
        if not isinstance(self.clicked, bool):
            raise ValueError("MIND candidate clicked must be a boolean")


@dataclass(frozen=True, slots=True)
class MindImpression:
    """A complete MIND candidate set retained for official-style evaluation."""

    impression_id: str
    user_id: str
    occurred_at: datetime
    candidates: tuple[MindCandidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.impression_id, str) or not self.impression_id.strip():
            raise ValueError("MIND impression_id must be a non-empty string")
        if not isinstance(self.user_id, str) or not self.user_id.strip():
            raise ValueError("MIND impression user_id must be a non-empty string")
        if (
            not isinstance(self.occurred_at, datetime)
            or self.occurred_at.tzinfo is None
            or self.occurred_at.utcoffset() is None
        ):
            raise ValueError("MIND impression occurred_at must be timezone-aware")
        if not isinstance(self.candidates, tuple):
            raise ValueError("MIND impression candidates must be a non-empty tuple")
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("MIND impression candidates must be a non-empty tuple")
        if not all(isinstance(candidate, MindCandidate) for candidate in candidates):
            raise ValueError("MIND impression candidates must contain MindCandidate values")
        object.__setattr__(self, "candidates", candidates)
        article_ids = [candidate.article_id for candidate in candidates]
        if len(article_ids) != len(set(article_ids)):
            raise ValueError("MIND impression candidates must have unique article ids")


@dataclass(frozen=True, slots=True, init=False)
class MindDataset:
    """A loss-aware MIND projection derived from two immutable byte snapshots."""

    articles: tuple[Article, ...]
    events: tuple[Event, ...]
    impressions: int
    ignored_history_items: int
    impression_records: tuple[MindImpression, ...]
    news_sha256: str
    behaviors_sha256: str
    _news_bytes: bytes = field(repr=False)
    _behaviors_bytes: bytes = field(repr=False)
    _catalog_published_at: datetime = field(repr=False)
    _behavior_timezone: tzinfo = field(repr=False)

    def __init__(
        self,
        news_bytes: bytes,
        behaviors_bytes: bytes,
        *,
        catalog_published_at: datetime,
        behavior_timezone: tzinfo,
        max_source_bytes: int = MAX_MIND_SOURCE_BYTES,
    ) -> None:
        if type(news_bytes) is not bytes or type(behaviors_bytes) is not bytes:
            raise ValueError("MIND source snapshots must be bytes")
        _positive_source_limit(max_source_bytes)
        if len(news_bytes) > max_source_bytes or len(behaviors_bytes) > max_source_bytes:
            raise ValueError("MIND source exceeds max_source_bytes")
        captured_catalog_time, zone = _validate_mind_assumptions(
            catalog_published_at, behavior_timezone
        )
        articles, events, ignored, impressions = _parse_mind(
            news_bytes,
            behaviors_bytes,
            catalog_published_at=captured_catalog_time,
            behavior_timezone=zone,
        )
        object.__setattr__(self, "articles", articles)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "impressions", len(impressions))
        object.__setattr__(self, "ignored_history_items", ignored)
        object.__setattr__(self, "impression_records", impressions)
        object.__setattr__(self, "news_sha256", hashlib.sha256(news_bytes).hexdigest())
        object.__setattr__(self, "behaviors_sha256", hashlib.sha256(behaviors_bytes).hexdigest())
        object.__setattr__(self, "_news_bytes", news_bytes)
        object.__setattr__(self, "_behaviors_bytes", behaviors_bytes)
        object.__setattr__(self, "_catalog_published_at", captured_catalog_time)
        object.__setattr__(self, "_behavior_timezone", zone)

    def verify_provenance(self) -> bool:
        """Reparse retained bytes and compare every public derived value."""

        articles, events, ignored, impressions = _parse_mind(
            self._news_bytes,
            self._behaviors_bytes,
            catalog_published_at=self._catalog_published_at,
            behavior_timezone=self._behavior_timezone,
        )
        return (
            articles == self.articles
            and events == self.events
            and ignored == self.ignored_history_items
            and impressions == self.impression_records
            and len(impressions) == self.impressions
            and hashlib.sha256(self._news_bytes).hexdigest() == self.news_sha256
            and hashlib.sha256(self._behaviors_bytes).hexdigest() == self.behaviors_sha256
        )


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


def _positive_source_limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_source_bytes must be a positive integer")
    return value


def _read_source(path: str | Path, maximum: int) -> bytes:
    with Path(path).open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("MIND source exceeds max_source_bytes")
    return data


def _iter_tsv_bytes(data: bytes, source: str) -> Iterator[tuple[int, list[str]]]:
    """Parse one immutable source snapshot without consulting the path again."""

    for line_number, raw_line in enumerate(data.splitlines(keepends=True), start=1):
        try:
            line = raw_line.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError as error:
            raise ValueError(f"{source} is not valid UTF-8 on line {line_number}") from error
        if line:
            yield line_number, line.split("\t")


def _validate_mind_assumptions(
    catalog_published_at: datetime, behavior_timezone: tzinfo
) -> tuple[datetime, tzinfo]:
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
    try:
        fixed_zone = timezone(zone_offset)
    except ValueError as error:
        raise ValueError("behavior_timezone must have a fixed UTC offset") from error
    return catalog_published_at.astimezone(UTC), fixed_zone


def _parse_mind(
    news_bytes: bytes,
    behaviors_bytes: bytes,
    *,
    catalog_published_at: datetime,
    behavior_timezone: tzinfo,
) -> tuple[tuple[Article, ...], tuple[Event, ...], int, tuple[MindImpression, ...]]:
    """Derive the complete public projection from exactly two byte snapshots."""

    catalog_time, zone = _validate_mind_assumptions(catalog_published_at, behavior_timezone)
    articles: list[Article] = []
    article_ids: set[str] = set()
    for line_number, fields in _iter_tsv_bytes(news_bytes, "MIND news"):
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
                title=title if title.strip() else MISSING_MIND_TITLE,
                summary=abstract,
                topics=topics,
                source=category if category.strip() else "uncategorized",
                published_at=catalog_time,
                quality=0.5,
                popularity=0.0,
                title_missing=not bool(title.strip()),
                category_missing=not bool(category.strip()),
                subcategory_missing=not bool(subcategory.strip()),
                mind_category=category,
                mind_subcategory=subcategory,
            )
        )

    events: list[Event] = []
    impression_records: list[MindImpression] = []
    impression_ids: set[str] = set()
    ignored_history_items = 0
    for line_number, fields in _iter_tsv_bytes(behaviors_bytes, "MIND behaviors"):
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
        occurred_at = _mind_time(raw_time, zone, line_number=line_number)
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
        candidates: list[MindCandidate] = []
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
            clicked = label == "1"
            candidates.append(MindCandidate(article_id, clicked))
            if clicked:
                events.append(Event(user_id, article_id, EventKind.CLICK, occurred_at))
        impression_records.append(
            MindImpression(
                impression_id=impression_id,
                user_id=user_id,
                occurred_at=occurred_at,
                candidates=tuple(candidates),
            )
        )
    return tuple(articles), tuple(events), ignored_history_items, tuple(impression_records)


def load_mind(
    news_path: str | Path,
    behaviors_path: str | Path,
    *,
    catalog_published_at: datetime,
    behavior_timezone: tzinfo,
    max_source_bytes: int = MAX_MIND_SOURCE_BYTES,
) -> MindDataset:
    """Load MIND ``news.tsv`` and ``behaviors.tsv`` as an honest click replay.

    MIND does not expose per-article publication timestamps or publishers. The
    caller must therefore declare one catalog availability time, and the adapter
    uses the news category as a source proxy. Unclicked impressions and undated
    history entries are not converted into preference events.
    """

    maximum = _positive_source_limit(max_source_bytes)
    news_bytes = _read_source(news_path, maximum)
    behaviors_bytes = _read_source(behaviors_path, maximum)
    return MindDataset(
        news_bytes,
        behaviors_bytes,
        catalog_published_at=catalog_published_at,
        behavior_timezone=behavior_timezone,
        max_source_bytes=maximum,
    )
