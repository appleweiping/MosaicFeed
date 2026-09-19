"""Versioned append-only interaction ingestion with checkpointed profile replay."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import stat
import tempfile
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import islice, pairwise
from pathlib import Path
from typing import Literal, Self, cast

from mosaicfeed.config import FeedConfig
from mosaicfeed.io import (
    _fsync_parent_directory,
    load_articles_bytes,
    load_json_text,
    parse_datetime,
)
from mosaicfeed.models import Article, Event, EventKind, UserProfile
from mosaicfeed.profile import event_signal, exponential_decay

EVENT_SCHEMA_VERSION = 1
CHECKPOINT_FORMAT = "mosaicfeed.profile-event-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
EMPTY_EVENT_CHAIN_SHA256 = hashlib.sha256(b"mosaicfeed.event-chain.v1").hexdigest()
LateEventPolicy = Literal["reject", "accept-rebuild"]
LATE_EVENT_POLICIES = frozenset({"reject", "accept-rebuild"})
_UNSPECIFIED_LOG_IDENTITY = object()


class EventConflictError(ValueError):
    """An existing event id was reused with different canonical content."""


class LateEventError(ValueError):
    """A new event fell behind the user's configured watermark."""


class LogIntegrityError(ValueError):
    """An append-only log or checkpoint failed an integrity invariant."""


def _log_identity(path: Path, *, allow_missing: bool = False) -> tuple[int, int] | None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        if allow_missing:
            return None
        raise
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise LogIntegrityError("event log must be a regular file, not a symlink")
    if status.st_nlink != 1:
        raise LogIntegrityError("event log must have exactly one hard link")
    return status.st_dev, status.st_ino


def _verify_open_log(descriptor: int, path: Path, expected: tuple[int, int] | None) -> None:
    status = os.fstat(descriptor)
    identity = (status.st_dev, status.st_ino)
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise LogIntegrityError("event log descriptor is not a single-link regular file")
    current = _log_identity(path)
    if current != identity or (expected is not None and identity != expected):
        raise LogIntegrityError("event log changed identity while it was being opened")


def _open_verified_log(
    path: Path,
    flags: int,
    *,
    create_if_missing: bool = False,
    required_identity: tuple[int, int] | object | None = _UNSPECIFIED_LOG_IDENTITY,
) -> tuple[int, bool]:
    expected = _log_identity(path, allow_missing=create_if_missing)
    if required_identity is not _UNSPECIFIED_LOG_IDENTITY and expected != required_identity:
        raise LogIntegrityError("event log identity changed before it could be opened")
    created = expected is None
    open_flags = flags | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    if created:
        open_flags |= os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, open_flags, 0o600)
    except FileExistsError as error:
        raise LogIntegrityError(
            "event log appeared concurrently while it was being created"
        ) from error
    try:
        _verify_open_log(descriptor, path, expected)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, created


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if strictly_positive and number <= 0.0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


def _identifier(value: object, name: str, *, maximum: int | None = None) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace")
    if any(not character.isprintable() for character in value):
        raise ValueError(f"{name} must contain only printing characters")
    if maximum is not None and len(value) > maximum:
        raise ValueError(f"{name} exceeds max_identifier_chars")
    return value


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _strict_fields(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} has missing or unknown fields")


def _integrity_fields(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise LogIntegrityError(f"{name} has missing or unknown fields")


@dataclass(frozen=True, slots=True)
class InteractionEvent:
    """One canonical v1 interaction record carried by the append-only log."""

    event_id: str
    user_id: str
    item_id: str
    timestamp: datetime
    kind: EventKind
    weight: float = 1.0

    def __post_init__(self) -> None:
        _identifier(self.event_id, "event_id")
        _identifier(self.user_id, "user_id")
        _identifier(self.item_id, "item_id")
        if (
            not isinstance(self.timestamp, datetime)
            or self.timestamp.tzinfo is None
            or self.timestamp.utcoffset() is None
        ):
            raise ValueError("timestamp must be timezone-aware")
        if not isinstance(self.kind, EventKind):
            raise ValueError("type must be a supported EventKind")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(UTC))
        object.__setattr__(
            self,
            "weight",
            _finite_number(self.weight, "weight", strictly_positive=True),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Parse the exact external v1 schema without aliases or defaults."""

        if not isinstance(value, Mapping):
            raise ValueError("interaction event must be a JSON object")
        _strict_fields(
            value,
            {"schema_version", "event_id", "user_id", "item_id", "timestamp", "type", "weight"},
            "interaction event",
        )
        schema_version = value["schema_version"]
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != EVENT_SCHEMA_VERSION
        ):
            raise ValueError("unsupported interaction event schema_version")
        raw_kind = value["type"]
        if not isinstance(raw_kind, str):
            raise ValueError("type must be a string")
        try:
            kind = EventKind(raw_kind)
        except ValueError as error:
            raise ValueError(f"unsupported interaction type: {raw_kind}") from error
        return cls(
            event_id=_identifier(value["event_id"], "event_id"),
            user_id=_identifier(value["user_id"], "user_id"),
            item_id=_identifier(value["item_id"], "item_id"),
            timestamp=parse_datetime(value["timestamp"], "timestamp"),
            kind=kind,
            weight=_finite_number(value["weight"], "weight", strictly_positive=True),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "item_id": self.item_id,
            "schema_version": EVENT_SCHEMA_VERSION,
            "timestamp": _utc_text(self.timestamp),
            "type": self.kind.value,
            "user_id": self.user_id,
            "weight": self.weight,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_dict())

    def log_line(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @property
    def order_key(self) -> tuple[datetime, str]:
        """Provide a total, stable chronological order for deterministic replay."""

        return (self.timestamp, self.event_id)

    def to_event(self) -> Event:
        """Convert losslessly into the history model used by frozen rank snapshots."""

        return Event(
            user_id=self.user_id,
            article_id=self.item_id,
            kind=self.kind,
            occurred_at=self.timestamp,
            weight=self.weight,
        )


@dataclass(frozen=True, slots=True)
class EventStreamLimits:
    """Finite memory, file, and batch boundaries for one event store."""

    max_event_bytes: int = 16 * 1024
    max_input_bytes: int = 64 * 1024 * 1024
    max_batch_events: int = 10_000
    max_log_bytes: int = 256 * 1024 * 1024
    max_checkpoint_bytes: int = 256 * 1024 * 1024
    max_state_bytes: int = 192 * 1024 * 1024
    max_events: int = 1_000_000
    max_users: int = 100_000
    max_events_per_user: int = 100_000
    max_catalog_articles: int = 100_000
    max_catalog_bytes: int = 64 * 1024 * 1024
    max_config_bytes: int = 64 * 1024
    max_topics_per_article: int = 1_024
    max_catalog_topic_cells: int = 2_000_000
    max_profile_topic_cells: int = 2_000_000
    max_identifier_chars: int = 256
    max_weight: float = 100.0

    def __post_init__(self) -> None:
        for name in (
            "max_event_bytes",
            "max_input_bytes",
            "max_batch_events",
            "max_log_bytes",
            "max_checkpoint_bytes",
            "max_state_bytes",
            "max_events",
            "max_users",
            "max_events_per_user",
            "max_catalog_articles",
            "max_catalog_bytes",
            "max_config_bytes",
            "max_topics_per_article",
            "max_catalog_topic_cells",
            "max_profile_topic_cells",
            "max_identifier_chars",
        ):
            _positive_int(getattr(self, name), name)
        _finite_number(self.max_weight, "max_weight", strictly_positive=True)
        if self.max_event_bytes > self.max_input_bytes:
            raise ValueError("max_event_bytes must not exceed max_input_bytes")
        if self.max_state_bytes > self.max_checkpoint_bytes:
            raise ValueError("max_state_bytes must not exceed max_checkpoint_bytes")


@dataclass(frozen=True, slots=True)
class IngestReport:
    accepted: int
    duplicates: int
    incremental_users: int
    rebuilt_users: int
    event_count: int
    log_offset: int
    event_chain_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "event_chain_sha256": self.event_chain_sha256,
            "event_count": self.event_count,
            "incremental_users": self.incremental_users,
            "log_offset": self.log_offset,
            "rebuilt_users": self.rebuilt_users,
        }


@dataclass(frozen=True, slots=True)
class CheckpointReport:
    event_count: int
    last_applied_offset: int
    event_chain_sha256: str
    checkpoint_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "event_chain_sha256": self.event_chain_sha256,
            "event_count": self.event_count,
            "last_applied_offset": self.last_applied_offset,
        }


@dataclass(frozen=True, slots=True, init=False)
class EventHistorySnapshot:
    """Immutable legacy history derived from one bounded canonical log prefix."""

    events: tuple[Event, ...]
    event_count: int
    last_applied_offset: int
    event_chain_sha256: str
    _source_bytes: bytes = field(repr=False)
    _max_event_bytes: int = field(repr=False, compare=False)

    def __init__(
        self,
        log_prefix_bytes: bytes,
        *,
        max_events: int = 1_000_000,
        max_bytes: int = 256 * 1024 * 1024,
        max_event_bytes: int = 16 * 1024,
    ) -> None:
        maximum = _positive_int(max_events, "max_events")
        byte_limit = _positive_int(max_bytes, "max_bytes")
        line_limit = _positive_int(max_event_bytes, "max_event_bytes")
        if type(log_prefix_bytes) is not bytes:
            raise ValueError("log_prefix_bytes must be bytes")
        if len(log_prefix_bytes) > byte_limit:
            raise ValueError("history snapshot exceeds max_bytes")
        if log_prefix_bytes and not log_prefix_bytes.endswith(b"\n"):
            raise ValueError("history snapshot log prefix has a torn final line")
        values: list[InteractionEvent] = []
        seen: set[str] = set()
        for line in log_prefix_bytes.splitlines(keepends=True):
            if len(line) > line_limit:
                raise ValueError("history snapshot event exceeds max_event_bytes")
            if len(values) >= maximum:
                raise ValueError("history snapshot exceeds max_events")
            try:
                payload = load_json_text(line[:-1].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                raise ValueError("history snapshot contains invalid strict JSON") from error
            if not isinstance(payload, dict):
                raise ValueError("history snapshot event must be a JSON object")
            event = InteractionEvent.from_mapping(payload)
            if event.log_line() != line:
                raise ValueError("history snapshot event is not in canonical form")
            if event.event_id in seen:
                raise ValueError("history snapshot contains a duplicate event_id")
            seen.add(event.event_id)
            values.append(event)
        source_events = tuple(values)
        chain = EMPTY_EVENT_CHAIN_SHA256
        for event in source_events:
            chain = _next_event_chain(chain, event.sha256)
        object.__setattr__(self, "_source_bytes", log_prefix_bytes)
        object.__setattr__(self, "_max_event_bytes", line_limit)
        object.__setattr__(self, "events", tuple(event.to_event() for event in source_events))
        object.__setattr__(self, "event_count", len(source_events))
        object.__setattr__(self, "last_applied_offset", len(log_prefix_bytes))
        object.__setattr__(self, "event_chain_sha256", chain)

    def verify_provenance(self) -> bool:
        """Re-derive all public values from the retained canonical source."""

        derived = EventHistorySnapshot(
            self._source_bytes,
            max_events=max(1, self.event_count),
            max_bytes=max(1, self.last_applied_offset),
            max_event_bytes=self._max_event_bytes,
        )
        return (
            derived.events == self.events
            and derived.event_count == self.event_count
            and derived.last_applied_offset == self.last_applied_offset
            and derived.event_chain_sha256 == self.event_chain_sha256
        )

    def metadata(self) -> dict[str, object]:
        return {
            "event_chain_sha256": self.event_chain_sha256,
            "event_count": self.event_count,
            "last_applied_offset": self.last_applied_offset,
        }


@dataclass(slots=True)
class _UserAccumulator:
    updated_through: datetime | None = None
    last_order_key: tuple[datetime, str] | None = None
    totals: dict[str, float] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)
    event_count: int = 0

    def copy(self) -> _UserAccumulator:
        return _UserAccumulator(
            updated_through=self.updated_through,
            last_order_key=self.last_order_key,
            totals=dict(self.totals),
            seen=set(self.seen),
            event_count=self.event_count,
        )

    def apply(
        self,
        event: InteractionEvent,
        article: Article,
        config: FeedConfig,
        *,
        max_topics: int,
    ) -> None:
        if self.updated_through is not None:
            if event.timestamp < self.updated_through:
                raise ValueError("internal profile replay is not chronological")
            if event.timestamp > self.updated_through:
                elapsed_days = (event.timestamp - self.updated_through).total_seconds() / 86_400.0
                decay = exponential_decay(elapsed_days, config.profile_half_life_days)
                self.totals = {
                    topic: decayed
                    for topic, value in self.totals.items()
                    if (decayed := value * decay) != 0.0
                }
        strength = event_signal(event.kind, config) * event.weight / len(article.topics)
        if not math.isfinite(strength):
            raise ValueError("weighted event signal is not finite")
        for topic in article.topics:
            updated = self.totals.get(topic, 0.0) + strength
            if not math.isfinite(updated):
                raise ValueError("incremental topic total is not finite")
            if updated == 0.0:
                self.totals.pop(topic, None)
            else:
                if topic not in self.totals and len(self.totals) >= max_topics:
                    raise ValueError("profile state exceeds max_profile_topic_cells")
                self.totals[topic] = updated
        self.seen.add(article.id)
        self.event_count += 1
        self.updated_through = event.timestamp
        self.last_order_key = (event.timestamp, event.event_id)

    def profile(self, user_id: str, *, as_of: datetime, config: FeedConfig) -> UserProfile:
        totals = dict(self.totals)
        if self.updated_through is not None and as_of > self.updated_through:
            elapsed_days = (as_of - self.updated_through).total_seconds() / 86_400.0
            decay = exponential_decay(elapsed_days, config.profile_half_life_days)
            totals = {
                topic: decayed
                for topic, value in totals.items()
                if (decayed := value * decay) != 0.0
            }
        scale = max((abs(value) for value in totals.values()), default=1.0)
        normalized = {topic: value / scale for topic, value in totals.items() if value != 0.0}
        return UserProfile(
            user_id=user_id,
            topic_weights=normalized,
            seen_article_ids=frozenset(self.seen),
            event_count=self.event_count,
        )


@dataclass(frozen=True, slots=True)
class _PreparedBatch:
    accepted: tuple[InteractionEvent, ...]
    duplicates: int
    user_events: Mapping[str, tuple[InteractionEvent, ...]]
    user_states: Mapping[str, _UserAccumulator]
    user_max_times: Mapping[str, datetime]
    state_bytes: int
    profile_topic_cells: int
    incremental_users: int
    rebuilt_users: int


class ProfileEventStore:
    """Single-process writer and deterministic reader for one profile event log."""

    def __init__(
        self,
        log_path: str | Path,
        articles: Iterable[Article],
        *,
        config: FeedConfig | None = None,
        limits: EventStreamLimits | None = None,
        allowed_lateness_seconds: float = 300.0,
        late_event_policy: LateEventPolicy = "reject",
    ) -> None:
        self._log_path = Path(log_path)
        if config is not None and not isinstance(config, FeedConfig):
            raise ValueError("config must be a FeedConfig")
        if limits is not None and not isinstance(limits, EventStreamLimits):
            raise ValueError("limits must be EventStreamLimits")
        self._config = config or FeedConfig()
        self._limits = limits or EventStreamLimits()
        lateness = _finite_number(
            allowed_lateness_seconds,
            "allowed_lateness_seconds",
            minimum=0.0,
        )
        self._allowed_lateness_seconds = 0.0 if lateness == 0.0 else lateness
        if self._allowed_lateness_seconds > 315_576_000.0:
            raise ValueError("allowed_lateness_seconds must not exceed ten years")
        if not isinstance(late_event_policy, str) or late_event_policy not in LATE_EVENT_POLICIES:
            raise ValueError("late_event_policy must be reject or accept-rebuild")
        self._late_event_policy = late_event_policy
        article_values = tuple(islice(articles, self._limits.max_catalog_articles + 1))
        if not article_values or any(
            not isinstance(article, Article) for article in article_values
        ):
            raise ValueError("articles must contain at least one Article")
        if len(article_values) > self._limits.max_catalog_articles:
            raise ValueError("catalog exceeds max_catalog_articles")
        self._articles = {article.id: article for article in article_values}
        if len(self._articles) != len(article_values):
            raise ValueError("catalog article ids must be unique")
        catalog_topic_cells = 0
        for article in article_values:
            _identifier(
                article.id,
                "catalog article id",
                maximum=self._limits.max_identifier_chars,
            )
            if len(article.topics) > self._limits.max_topics_per_article:
                raise ValueError("catalog article exceeds max_topics_per_article")
            catalog_topic_cells += len(article.topics)
            if catalog_topic_cells > self._limits.max_catalog_topic_cells:
                raise ValueError("catalog exceeds max_catalog_topic_cells")
            for topic in article.topics:
                _identifier(
                    topic,
                    "catalog topic",
                    maximum=self._limits.max_identifier_chars,
                )
        self._catalog_sha256 = _catalog_profile_sha256(
            article_values,
            maximum_bytes=self._limits.max_catalog_bytes,
        )
        self._config_sha256 = hashlib.sha256(_canonical_json(self._config.to_dict())).hexdigest()
        self._lock = threading.RLock()
        self._events_in_log_order: list[InteractionEvent] = []
        self._event_digests: dict[str, str] = {}
        self._events_by_user: dict[str, tuple[InteractionEvent, ...]] = {}
        self._user_states: dict[str, _UserAccumulator] = {}
        self._user_max_times: dict[str, datetime] = {}
        self._state_bytes = 0
        self._profile_topic_cells = 0
        self._log_offset = 0
        self._event_chain_sha256 = EMPTY_EVENT_CHAIN_SHA256
        self._last_event_sha256: str | None = None
        self._torn_tail_bytes = 0
        self._recovered_torn_tail_bytes = 0
        self._opened_log_identity: tuple[int, int] | None = None

    @classmethod
    def open(
        cls,
        log_path: str | Path,
        articles: Iterable[Article],
        *,
        config: FeedConfig | None = None,
        limits: EventStreamLimits | None = None,
        allowed_lateness_seconds: float = 300.0,
        late_event_policy: LateEventPolicy = "reject",
        checkpoint_path: str | Path | None = None,
        recover_torn_tail: bool = False,
    ) -> Self:
        """Open a log, optionally restore a checkpoint, then replay its complete tail."""

        store = cls(
            log_path,
            articles,
            config=config,
            limits=limits,
            allowed_lateness_seconds=allowed_lateness_seconds,
            late_event_policy=late_event_policy,
        )
        if not isinstance(recover_torn_tail, bool):
            raise ValueError("recover_torn_tail must be a boolean")
        if checkpoint_path is not None:
            store._restore_checkpoint(Path(checkpoint_path))
        store._replay_tail(recover_torn_tail=recover_torn_tail)
        return store

    @property
    def event_count(self) -> int:
        return len(self._events_in_log_order)

    @property
    def log_offset(self) -> int:
        return self._log_offset

    @property
    def event_chain_sha256(self) -> str:
        return self._event_chain_sha256

    @property
    def torn_tail_bytes(self) -> int:
        return self._torn_tail_bytes

    @property
    def recovered_torn_tail_bytes(self) -> int:
        """Return bytes explicitly discarded while opening this store."""

        return self._recovered_torn_tail_bytes

    @property
    def limits(self) -> EventStreamLimits:
        return self._limits

    def watermark(self, user_id: str) -> datetime | None:
        """Return the user's maximum event time minus allowed lateness."""

        normalized_user = _identifier(
            user_id,
            "user_id",
            maximum=self._limits.max_identifier_chars,
        )
        with self._lock:
            maximum = self._user_max_times.get(normalized_user)
            if maximum is None:
                return None
            return _watermark(maximum, self._allowed_lateness_seconds)

    def events(self) -> tuple[InteractionEvent, ...]:
        with self._lock:
            return tuple(self._events_in_log_order)

    def history_events(self) -> tuple[Event, ...]:
        """Return a lossless immutable history for an explicitly rebuilt rank snapshot."""

        return self.history_snapshot().events

    def history_snapshot(self) -> EventHistorySnapshot:
        """Freeze history and provenance together for a new ranker/server snapshot."""

        with self._lock:
            log_prefix = b"".join(event.log_line() for event in self._events_in_log_order)
            snapshot = EventHistorySnapshot(
                log_prefix,
                max_events=self._limits.max_events,
                max_bytes=self._limits.max_log_bytes,
                max_event_bytes=self._limits.max_event_bytes,
            )
            if (
                snapshot.last_applied_offset != self._log_offset
                or snapshot.event_chain_sha256 != self._event_chain_sha256
            ):
                raise LogIntegrityError("history snapshot does not match the applied log prefix")
            return snapshot

    def profile(self, user_id: str, *, as_of: datetime) -> UserProfile:
        """Read a profile at any point in time from the current consistent prefix."""

        normalized_user = _identifier(
            user_id,
            "user_id",
            maximum=self._limits.max_identifier_chars,
        )
        clock = _aware(as_of, "as_of")
        with self._lock:
            return self._profile_unlocked(normalized_user, clock)

    def profiles(self, *, as_of: datetime) -> tuple[UserProfile, ...]:
        """Read every profile from one in-process, lock-consistent event prefix."""

        clock = _aware(as_of, "as_of")
        with self._lock:
            profiles: list[UserProfile] = []
            topic_cells = 0
            for user_id in sorted(self._events_by_user):
                profile = self._profile_unlocked(user_id, clock)
                topic_cells += len(profile.topic_weights)
                if topic_cells > self._limits.max_profile_topic_cells:
                    raise ValueError("materialized profiles exceed max_profile_topic_cells")
                profiles.append(profile)
            return tuple(profiles)

    def ingest(self, events: Iterable[InteractionEvent]) -> IngestReport:
        """Atomically append one validated batch or make no logical state change."""

        values = tuple(islice(events, self._limits.max_batch_events + 1))
        if len(values) > self._limits.max_batch_events:
            raise ValueError("ingest batch exceeds max_batch_events")
        if any(not isinstance(event, InteractionEvent) for event in values):
            raise ValueError("events must contain InteractionEvent values")
        with self._lock:
            if self._torn_tail_bytes:
                raise LogIntegrityError("log has a torn tail; reopen with recover_torn_tail=True")
            prepared = self._prepare_batch(values, duplicate_mode="idempotent")
            block = b"".join(event.log_line() for event in prepared.accepted)
            if self._log_offset + len(block) > self._limits.max_log_bytes:
                raise ValueError("append would exceed max_log_bytes")
            self._assert_disk_offset()
            if block:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                descriptor, created = _open_verified_log(
                    self._log_path,
                    os.O_WRONLY | os.O_APPEND,
                    create_if_missing=True,
                    required_identity=self._opened_log_identity,
                )
                with os.fdopen(descriptor, "ab") as destination:
                    opened_identity = (
                        os.fstat(destination.fileno()).st_dev,
                        os.fstat(destination.fileno()).st_ino,
                    )
                    written = destination.write(block)
                    destination.flush()
                    os.fsync(destination.fileno())
                    _verify_open_log(destination.fileno(), self._log_path, None)
                if written != len(block):
                    raise OSError("event log append was incomplete")
                if created:
                    _fsync_parent_directory(self._log_path)
                current = self._log_path.lstat()
                if (
                    current.st_dev,
                    current.st_ino,
                ) != opened_identity or current.st_size != self._log_offset + len(block):
                    raise LogIntegrityError("event log changed concurrently during append")
                self._opened_log_identity = opened_identity
            self._commit_prepared(prepared)
            self._log_offset += len(block)
            return IngestReport(
                accepted=len(prepared.accepted),
                duplicates=prepared.duplicates,
                incremental_users=prepared.incremental_users,
                rebuilt_users=prepared.rebuilt_users,
                event_count=self.event_count,
                log_offset=self._log_offset,
                event_chain_sha256=self._event_chain_sha256,
            )

    def checkpoint(self, path: str | Path) -> CheckpointReport:
        """Atomically replace a checksummed checkpoint for the complete applied prefix."""

        destination = Path(path)
        if destination.resolve() == self._log_path.resolve():
            raise ValueError("checkpoint path must differ from the event log path")
        with self._lock:
            if self._torn_tail_bytes:
                raise LogIntegrityError("cannot checkpoint while the log has a torn tail")
            self._assert_disk_offset()
            prefix_sha256 = self._prefix_sha256()
            payload = self._checkpoint_payload(prefix_sha256)
            payload_bytes = _canonical_json(payload)
            checksum = hashlib.sha256(payload_bytes).hexdigest()
            envelope = {"checksum_sha256": checksum, "payload": payload}
            contents = _canonical_json(envelope) + b"\n"
            checkpoint_sha256 = hashlib.sha256(contents).hexdigest()
            if len(contents) > self._limits.max_checkpoint_bytes:
                raise ValueError("checkpoint exceeds max_checkpoint_bytes")
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    descriptor = -1
                    handle.write(contents)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
                _fsync_parent_directory(destination)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                if temporary.exists():
                    temporary.unlink()
            return CheckpointReport(
                event_count=self.event_count,
                last_applied_offset=self._log_offset,
                event_chain_sha256=self._event_chain_sha256,
                checkpoint_sha256=checkpoint_sha256,
            )

    def _prepare_batch(
        self,
        events: tuple[InteractionEvent, ...],
        *,
        duplicate_mode: Literal["idempotent", "reject"],
    ) -> _PreparedBatch:
        accepted: list[InteractionEvent] = []
        duplicates = 0
        batch_digests: dict[str, str] = {}
        user_max_times: dict[str, datetime] = {}
        user_additions: dict[str, list[InteractionEvent]] = {}
        state_bytes = self._state_bytes
        for event in events:
            self._validate_event(event)
            digest = event.sha256
            existing = batch_digests.get(event.event_id)
            if existing is None:
                existing = self._event_digests.get(event.event_id)
            if existing is not None:
                if existing != digest:
                    raise EventConflictError(
                        f"event_id conflicts with different content: {event.event_id}"
                    )
                if duplicate_mode == "reject":
                    raise LogIntegrityError(
                        f"event log contains duplicate event_id: {event.event_id}"
                    )
                duplicates += 1
                continue
            current_max = user_max_times.get(event.user_id)
            if current_max is None:
                current_max = self._user_max_times.get(event.user_id)
            if current_max is not None:
                watermark = _watermark(current_max, self._allowed_lateness_seconds)
                if event.timestamp < watermark and self._late_event_policy == "reject":
                    raise LateEventError(f"event {event.event_id} is older than the user watermark")
            batch_digests[event.event_id] = digest
            user_max_times[event.user_id] = (
                event.timestamp if current_max is None else max(current_max, event.timestamp)
            )
            pending_for_user = user_additions.setdefault(event.user_id, [])
            pending_for_user.append(event)
            existing_user_count = len(self._events_by_user.get(event.user_id, ()))
            if existing_user_count + len(pending_for_user) > self._limits.max_events_per_user:
                raise ValueError("user history exceeds max_events_per_user")
            line_bytes = len(event.log_line())
            state_bytes += line_bytes
            if state_bytes > self._limits.max_state_bytes:
                raise ValueError("in-memory event state exceeds max_state_bytes")
            accepted.append(event)
        if self.event_count + len(accepted) > self._limits.max_events:
            raise ValueError("event store exceeds max_events")
        new_user_count = sum(user_id not in self._events_by_user for user_id in user_additions)
        if len(self._events_by_user) + new_user_count > self._limits.max_users:
            raise ValueError("event store exceeds max_users")

        planned_events: dict[str, tuple[InteractionEvent, ...]] = {}
        planned_states: dict[str, _UserAccumulator] = {}
        profile_topic_cells = self._profile_topic_cells
        incremental_users = 0
        rebuilt_users = 0
        for user_id, addition_values in user_additions.items():
            additions = tuple(addition_values)
            existing_events = self._events_by_user.get(user_id, ())
            previous_state = self._user_states.get(user_id)
            previous_topic_cells = 0 if previous_state is None else len(previous_state.totals)
            other_topic_cells = profile_topic_cells - previous_topic_cells
            available_topic_cells = self._limits.max_profile_topic_cells - other_topic_cells
            if available_topic_cells < 0:
                raise ValueError("profile state exceeds max_profile_topic_cells")
            previous_key = None if previous_state is None else previous_state.last_order_key
            additions_are_after = previous_key is not None and all(
                event.order_key > previous_key for event in additions
            )
            additions_are_sorted = all(
                left.order_key < right.order_key for left, right in pairwise(additions)
            )
            if additions_are_after and additions_are_sorted:
                if previous_state is None:
                    raise RuntimeError("incremental plan is missing its previous user state")
                state = previous_state.copy()
                ordered = (*existing_events, *additions)
                for event in additions:
                    state.apply(
                        event,
                        self._articles[event.item_id],
                        self._config,
                        max_topics=available_topic_cells,
                    )
                incremental_users += 1
            else:
                ordered = tuple(
                    sorted((*existing_events, *additions), key=lambda item: item.order_key)
                )
                state = self._build_user_state(ordered, max_topics=available_topic_cells)
                rebuilt_users += 1
            profile_topic_cells = other_topic_cells + len(state.totals)
            if profile_topic_cells > self._limits.max_profile_topic_cells:
                raise ValueError("profile state exceeds max_profile_topic_cells")
            planned_events[user_id] = tuple(ordered)
            planned_states[user_id] = state
        return _PreparedBatch(
            accepted=tuple(accepted),
            duplicates=duplicates,
            user_events=planned_events,
            user_states=planned_states,
            user_max_times=user_max_times,
            state_bytes=state_bytes,
            profile_topic_cells=profile_topic_cells,
            incremental_users=incremental_users,
            rebuilt_users=rebuilt_users,
        )

    def _validate_event(self, event: InteractionEvent) -> None:
        _identifier(event.event_id, "event_id", maximum=self._limits.max_identifier_chars)
        _identifier(event.user_id, "user_id", maximum=self._limits.max_identifier_chars)
        _identifier(event.item_id, "item_id", maximum=self._limits.max_identifier_chars)
        if len(event.log_line()) > self._limits.max_event_bytes:
            raise ValueError("interaction event exceeds max_event_bytes")
        if event.weight > self._limits.max_weight:
            raise ValueError("interaction event weight exceeds max_weight")
        article = self._articles.get(event.item_id)
        if article is None:
            raise ValueError(f"interaction event references unknown item: {event.item_id}")
        if event.timestamp < article.published_at.astimezone(UTC):
            raise ValueError(f"interaction event for {event.item_id} predates publication")

    def _profile_unlocked(self, user_id: str, as_of: datetime) -> UserProfile:
        state = self._user_states.get(user_id)
        if state is None:
            return UserProfile(user_id)
        if state.updated_through is not None and as_of >= state.updated_through:
            return state.profile(user_id, as_of=as_of, config=self._config)
        eligible = tuple(
            event for event in self._events_by_user[user_id] if event.timestamp <= as_of
        )
        historical = self._build_user_state(
            eligible,
            max_topics=self._limits.max_profile_topic_cells,
        )
        return historical.profile(user_id, as_of=as_of, config=self._config)

    def _build_user_state(
        self,
        events: Iterable[InteractionEvent],
        *,
        max_topics: int,
    ) -> _UserAccumulator:
        state = _UserAccumulator()
        for event in events:
            state.apply(
                event,
                self._articles[event.item_id],
                self._config,
                max_topics=max_topics,
            )
        return state

    def _commit_prepared(self, prepared: _PreparedBatch) -> None:
        for event in prepared.accepted:
            digest = event.sha256
            self._events_in_log_order.append(event)
            self._event_digests[event.event_id] = digest
            self._event_chain_sha256 = _next_event_chain(self._event_chain_sha256, digest)
            self._last_event_sha256 = digest
        self._events_by_user.update(prepared.user_events)
        self._user_states.update(prepared.user_states)
        self._user_max_times.update(prepared.user_max_times)
        self._state_bytes = prepared.state_bytes
        self._profile_topic_cells = prepared.profile_topic_cells

    def _assert_disk_offset(self) -> None:
        identity = _log_identity(self._log_path, allow_missing=True)
        if identity != self._opened_log_identity:
            raise LogIntegrityError("event log identity changed; reopen and replay")
        actual = self._log_path.lstat().st_size if identity is not None else 0
        if actual != self._log_offset:
            raise LogIntegrityError(
                "event log size differs from the applied offset; reopen and replay"
            )

    def _prefix_sha256(self) -> str:
        if _log_identity(self._log_path, allow_missing=True) is None:
            return hashlib.sha256(b"").hexdigest()
        descriptor, _ = _open_verified_log(
            self._log_path,
            os.O_RDONLY,
            required_identity=self._opened_log_identity,
        )
        with os.fdopen(descriptor, "rb") as source:
            data = source.read(self._log_offset + 1)
            _verify_open_log(source.fileno(), self._log_path, None)
        if len(data) != self._log_offset:
            raise LogIntegrityError("event log changed while calculating its prefix digest")
        expected = b"".join(event.log_line() for event in self._events_in_log_order)
        if data != expected:
            raise LogIntegrityError("event log content differs from the applied event prefix")
        return hashlib.sha256(data).hexdigest()

    def _checkpoint_payload(self, prefix_sha256: str) -> dict[str, object]:
        return {
            "events": [event.to_dict() for event in self._events_in_log_order],
            "format": CHECKPOINT_FORMAT,
            "log": {
                "event_chain_sha256": self._event_chain_sha256,
                "event_count": self.event_count,
                "last_applied_offset": self._log_offset,
                "last_event_sha256": self._last_event_sha256,
                "prefix_sha256": prefix_sha256,
            },
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "settings": {
                "allowed_lateness_seconds": self._allowed_lateness_seconds,
                "catalog_profile_sha256": self._catalog_sha256,
                "config_sha256": self._config_sha256,
                "late_event_policy": self._late_event_policy,
            },
            "users": self._user_state_records(),
        }

    def _user_state_records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for user_id in sorted(self._user_states):
            state = self._user_states[user_id]
            records.append(
                {
                    "event_count": state.event_count,
                    "last_event_id": None
                    if state.last_order_key is None
                    else state.last_order_key[1],
                    "seen_item_ids": sorted(state.seen),
                    "topic_totals": dict(sorted(state.totals.items())),
                    "updated_through": (
                        None if state.updated_through is None else _utc_text(state.updated_through)
                    ),
                    "user_id": user_id,
                }
            )
        return records

    def _restore_checkpoint(self, path: Path) -> None:
        raw = _read_bounded_bytes(
            path,
            self._limits.max_checkpoint_bytes,
            "checkpoint",
            "max_checkpoint_bytes",
        )
        if not raw.endswith(b"\n"):
            raise LogIntegrityError("checkpoint is truncated or missing its final newline")
        try:
            decoded = raw.decode("utf-8")
            envelope = load_json_text(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise LogIntegrityError("checkpoint is not strict UTF-8 JSON") from error
        if not isinstance(envelope, dict):
            raise LogIntegrityError("checkpoint must be a JSON object")
        _integrity_fields(envelope, {"checksum_sha256", "payload"}, "checkpoint envelope")
        checksum = _digest_text(envelope["checksum_sha256"], "checkpoint checksum")
        payload = envelope["payload"]
        if not isinstance(payload, dict):
            raise LogIntegrityError("checkpoint payload must be a JSON object")
        if not _digest_equal(checksum, hashlib.sha256(_canonical_json(payload)).hexdigest()):
            raise LogIntegrityError("checkpoint checksum does not match its payload")
        self._validate_checkpoint_payload(cast(dict[str, object], payload))

    def _validate_checkpoint_payload(self, payload: dict[str, object]) -> None:
        _integrity_fields(
            payload,
            {"events", "format", "log", "schema_version", "settings", "users"},
            "checkpoint payload",
        )
        schema_version = payload["schema_version"]
        if (
            payload["format"] != CHECKPOINT_FORMAT
            or isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != CHECKPOINT_SCHEMA_VERSION
        ):
            raise LogIntegrityError("unsupported checkpoint format or schema version")
        settings = payload["settings"]
        if not isinstance(settings, dict):
            raise LogIntegrityError("checkpoint settings must be a JSON object")
        _integrity_fields(
            settings,
            {
                "allowed_lateness_seconds",
                "catalog_profile_sha256",
                "config_sha256",
                "late_event_policy",
            },
            "checkpoint settings",
        )
        try:
            checkpoint_lateness = _finite_number(
                settings["allowed_lateness_seconds"],
                "allowed_lateness_seconds",
                minimum=0.0,
            )
        except ValueError as error:
            raise LogIntegrityError("checkpoint lateness setting is invalid") from error
        checkpoint_policy = settings["late_event_policy"]
        if (
            checkpoint_lateness != self._allowed_lateness_seconds
            or _digest_text(settings["catalog_profile_sha256"], "catalog_profile_sha256")
            != self._catalog_sha256
            or _digest_text(settings["config_sha256"], "config_sha256") != self._config_sha256
            or not isinstance(checkpoint_policy, str)
            or checkpoint_policy != self._late_event_policy
        ):
            raise LogIntegrityError("checkpoint settings do not match this event store")
        raw_log = payload["log"]
        if not isinstance(raw_log, dict):
            raise LogIntegrityError("checkpoint log metadata must be a JSON object")
        _integrity_fields(
            raw_log,
            {
                "event_chain_sha256",
                "event_count",
                "last_applied_offset",
                "last_event_sha256",
                "prefix_sha256",
            },
            "checkpoint log metadata",
        )
        offset = _non_negative_int(raw_log["last_applied_offset"], "last_applied_offset")
        event_count = _non_negative_int(raw_log["event_count"], "event_count")
        if event_count > self._limits.max_events or offset > self._limits.max_log_bytes:
            raise ValueError("checkpoint log metadata exceeds configured limits")
        prefix_digest = _digest_text(raw_log["prefix_sha256"], "prefix_sha256")
        prefix = self._read_log_prefix(offset)
        if not _digest_equal(prefix_digest, hashlib.sha256(prefix).hexdigest()):
            raise LogIntegrityError("event log prefix does not match checkpoint")
        raw_events = payload["events"]
        if not isinstance(raw_events, list) or len(raw_events) != event_count:
            raise LogIntegrityError("checkpoint event list does not match event_count")
        events: list[InteractionEvent] = []
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                raise LogIntegrityError("checkpoint events must be JSON objects")
            try:
                event = InteractionEvent.from_mapping(raw_event)
            except ValueError as error:
                raise LogIntegrityError(
                    "checkpoint contains an invalid interaction event"
                ) from error
            events.append(event)
        if b"".join(event.log_line() for event in events) != prefix:
            raise LogIntegrityError("checkpoint events do not match the event log prefix")
        prepared = self._prepare_batch(tuple(events), duplicate_mode="reject")
        self._commit_prepared(prepared)
        if self.event_count != event_count:
            raise LogIntegrityError("checkpoint event count is inconsistent")
        expected_chain = _digest_text(raw_log["event_chain_sha256"], "event_chain_sha256")
        if not _digest_equal(expected_chain, self._event_chain_sha256):
            raise LogIntegrityError("checkpoint event chain is inconsistent")
        raw_last = raw_log["last_event_sha256"]
        last_digest = None if raw_last is None else _digest_text(raw_last, "last_event_sha256")
        if last_digest != self._last_event_sha256:
            raise LogIntegrityError("checkpoint last event digest is inconsistent")
        raw_users = payload["users"]
        expected_users = self._user_state_records()
        if not isinstance(raw_users, list) or _canonical_json(raw_users) != _canonical_json(
            expected_users
        ):
            raise LogIntegrityError("checkpoint derived user state is inconsistent")
        self._log_offset = offset

    def _read_log_prefix(self, offset: int) -> bytes:
        identity = _log_identity(self._log_path, allow_missing=True)
        if identity is None:
            if offset == 0:
                return b""
            raise LogIntegrityError("checkpoint offset exceeds the missing event log")
        if self._log_path.lstat().st_size > self._limits.max_log_bytes:
            raise ValueError("event log exceeds max_log_bytes")
        descriptor, _ = _open_verified_log(
            self._log_path,
            os.O_RDONLY,
            required_identity=identity,
        )
        with os.fdopen(descriptor, "rb") as source:
            prefix = source.read(offset)
            _verify_open_log(source.fileno(), self._log_path, identity)
        if len(prefix) != offset:
            raise LogIntegrityError("checkpoint offset exceeds the event log")
        return prefix

    def _replay_tail(self, *, recover_torn_tail: bool) -> None:
        identity = _log_identity(self._log_path, allow_missing=True)
        if identity is None:
            return
        size = self._log_path.lstat().st_size
        if size > self._limits.max_log_bytes:
            raise ValueError("event log exceeds max_log_bytes")
        if size < self._log_offset:
            raise LogIntegrityError("event log is shorter than the restored checkpoint offset")
        descriptor, _ = _open_verified_log(
            self._log_path,
            os.O_RDONLY,
            required_identity=identity,
        )
        with os.fdopen(descriptor, "rb") as source:
            opened_identity = (os.fstat(source.fileno()).st_dev, os.fstat(source.fileno()).st_ino)
            source.seek(self._log_offset)
            remaining_limit = self._limits.max_log_bytes - self._log_offset
            tail = source.read(remaining_limit + 1)
            _verify_open_log(source.fileno(), self._log_path, identity)
        if _log_identity(self._log_path) != opened_identity:
            raise LogIntegrityError("event log changed identity during replay")
        if self._log_offset + len(tail) > self._limits.max_log_bytes:
            raise ValueError("event log exceeds max_log_bytes")
        newline = tail.rfind(b"\n")
        complete_length = newline + 1
        complete = tail[:complete_length]
        torn = tail[complete_length:]
        events: list[InteractionEvent] = []
        tail_state_bytes = 0
        for line in complete.splitlines(keepends=True):
            if len(line) > self._limits.max_event_bytes:
                raise ValueError("event log line exceeds max_event_bytes")
            if line == b"\n":
                raise LogIntegrityError("event log contains a blank line")
            try:
                payload = load_json_text(line[:-1].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                raise LogIntegrityError("event log contains invalid strict JSON") from error
            if not isinstance(payload, dict):
                raise LogIntegrityError("event log line must be a JSON object")
            try:
                event = InteractionEvent.from_mapping(payload)
            except ValueError as error:
                raise LogIntegrityError("event log contains an invalid event") from error
            if line != event.log_line():
                raise LogIntegrityError("event log line is not in canonical form")
            events.append(event)
            if self.event_count + len(events) > self._limits.max_events:
                raise ValueError("event store exceeds max_events")
            tail_state_bytes += len(line)
            if self._state_bytes + tail_state_bytes > self._limits.max_state_bytes:
                raise ValueError("in-memory event state exceeds max_state_bytes")
        prepared = self._prepare_batch(tuple(events), duplicate_mode="reject")
        if torn and recover_torn_tail:
            descriptor, _ = _open_verified_log(
                self._log_path,
                os.O_RDWR,
                required_identity=identity,
            )
            with os.fdopen(descriptor, "r+b") as destination:
                destination.truncate(self._log_offset + complete_length)
                destination.flush()
                os.fsync(destination.fileno())
                _verify_open_log(destination.fileno(), self._log_path, identity)
            self._recovered_torn_tail_bytes = len(torn)
            torn = b""
        self._torn_tail_bytes = len(torn)
        self._commit_prepared(prepared)
        self._log_offset += len(complete)
        self._opened_log_identity = opened_identity


def _aware(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _read_bounded_bytes(path: Path, maximum: int, name: str, limit_name: str) -> bytes:
    with path.open("rb") as source:
        data = source.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f"{name} exceeds {limit_name}")
    return data


def _watermark(maximum: datetime, lateness_seconds: float) -> datetime:
    try:
        return maximum - timedelta(seconds=lateness_seconds)
    except OverflowError:
        return datetime.min.replace(tzinfo=UTC)


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LogIntegrityError(f"{name} must be a non-negative integer")
    return value


def _digest_text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.casefold()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LogIntegrityError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _digest_equal(left: str, right: str) -> bool:
    """Compare public integrity digests without accidental early-exit behavior."""

    return hmac.compare_digest(left, right)


def _next_event_chain(previous: str, event_digest: str) -> str:
    return hashlib.sha256(bytes.fromhex(previous) + bytes.fromhex(event_digest)).hexdigest()


def _catalog_profile_sha256(articles: Iterable[Article], *, maximum_bytes: int) -> str:
    """Hash canonical profile fields without materializing the complete catalog JSON."""

    digest = hashlib.sha256()
    total_bytes = 0

    def consume(chunk: bytes) -> None:
        nonlocal total_bytes
        total_bytes += len(chunk)
        if total_bytes > maximum_bytes:
            raise ValueError("catalog profile exceeds max_catalog_bytes")
        digest.update(chunk)

    consume(b"[")
    for index, article in enumerate(sorted(articles, key=lambda item: item.id)):
        if index:
            consume(b",")
        consume(
            _canonical_json(
                {
                    "id": article.id,
                    "published_at": _utc_text(article.published_at),
                    "topics": list(article.topics),
                }
            )
        )
    consume(b"]")
    return digest.hexdigest()


def load_interaction_events(
    path: str | Path,
    *,
    limits: EventStreamLimits | None = None,
) -> tuple[InteractionEvent, ...]:
    """Load a bounded newline-terminated v1 JSONL ingestion batch."""

    if limits is not None and not isinstance(limits, EventStreamLimits):
        raise ValueError("limits must be EventStreamLimits")
    active_limits = limits or EventStreamLimits()
    source = Path(path)
    data = _read_bounded_bytes(
        source,
        active_limits.max_input_bytes,
        "interaction input",
        "max_input_bytes",
    )
    if data and not data.endswith(b"\n"):
        raise ValueError("interaction input must end with a newline")
    events: list[InteractionEvent] = []
    for line in data.splitlines(keepends=True):
        if line == b"\n":
            raise ValueError("interaction input contains a blank line")
        if len(line) > active_limits.max_event_bytes:
            raise ValueError("interaction input event exceeds max_event_bytes")
        try:
            payload = load_json_text(line[:-1].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("interaction input contains invalid strict JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("interaction input lines must be JSON objects")
        events.append(InteractionEvent.from_mapping(payload))
        if len(events) > active_limits.max_batch_events:
            raise ValueError("interaction input exceeds max_batch_events")
    return tuple(events)


def load_event_store(
    log_path: str | Path,
    articles_path: str | Path,
    *,
    config: FeedConfig | None = None,
    limits: EventStreamLimits | None = None,
    allowed_lateness_seconds: float = 300.0,
    late_event_policy: LateEventPolicy = "reject",
    checkpoint_path: str | Path | None = None,
    recover_torn_tail: bool = False,
) -> ProfileEventStore:
    """Bound catalog loading before opening and replaying a profile event store."""

    if limits is not None and not isinstance(limits, EventStreamLimits):
        raise ValueError("limits must be EventStreamLimits")
    active_limits = limits or EventStreamLimits()
    catalog_path = Path(articles_path)
    catalog_bytes = _read_bounded_bytes(
        catalog_path,
        active_limits.max_catalog_bytes,
        "catalog",
        "max_catalog_bytes",
    )
    articles = load_articles_bytes(catalog_bytes, source_name=str(catalog_path))
    return ProfileEventStore.open(
        log_path,
        articles,
        config=config,
        limits=active_limits,
        allowed_lateness_seconds=allowed_lateness_seconds,
        late_event_policy=late_event_policy,
        checkpoint_path=checkpoint_path,
        recover_torn_tail=recover_torn_tail,
    )


def profile_to_dict(profile: UserProfile, *, as_of: datetime) -> dict[str, object]:
    return {
        "as_of": _utc_text(_aware(as_of, "as_of")),
        "event_count": profile.event_count,
        "seen_item_ids": sorted(profile.seen_article_ids),
        "topic_weights": dict(profile.topic_weights),
        "user_id": profile.user_id,
    }
