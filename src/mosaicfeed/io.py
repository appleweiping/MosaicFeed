"""Strict JSON adapters and stable output serialization."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, TypeVar, cast

from mosaicfeed.models import MISSING_MIND_TITLE, Article, Event, EventKind, Feed, ScoreBreakdown

_PathT = TypeVar("_PathT", str, Path)
MAX_JSON_NESTING = 256


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _validate_json_nesting(value: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > MAX_JSON_NESTING:
                raise ValueError("JSON nesting is too deep")
        elif character in "]}":
            depth -= 1


def load_json_text(value: str) -> object:
    """Parse strict JSON, rejecting non-finite numbers and duplicate object fields."""

    _validate_json_nesting(value)
    try:
        return json.loads(
            value,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except RecursionError as error:
        raise ValueError("JSON nesting is too deep") from error


def parse_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"invalid {field_name}: {value}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{field_name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite number")
    return number


def _records_from_text(value: str, source: Path) -> list[dict[str, Any]]:
    if source.suffix.casefold() == ".jsonl":
        lines = value.splitlines()
        values: object = [load_json_text(line) for line in lines if line.strip()]
    else:
        values = load_json_text(value)
    if not isinstance(values, list) or not all(isinstance(value, dict) for value in values):
        raise ValueError(f"{source} must contain a list of JSON objects")
    return cast(list[dict[str, Any]], values)


def _records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    return _records_from_text(source.read_text(encoding="utf-8"), source)


def _strict(record: dict[str, Any], allowed: set[str], required: set[str], kind: str) -> None:
    unknown = set(record) - allowed
    missing = required - set(record)
    if unknown:
        raise ValueError(f"unknown {kind} fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"missing {kind} fields: {', '.join(sorted(missing))}")


def _articles_from_records(records: Iterable[dict[str, Any]]) -> list[Article]:
    result: list[Article] = []
    allowed = {
        "id",
        "title",
        "summary",
        "topics",
        "source",
        "published_at",
        "quality",
        "popularity",
        "title_missing",
        "category_missing",
        "subcategory_missing",
        "mind_category",
        "mind_subcategory",
    }
    required = {"id", "title", "topics", "source", "published_at"}
    for record in records:
        _strict(record, allowed, required, "article")
        topics = record["topics"]
        if not isinstance(topics, list) or not all(isinstance(value, str) for value in topics):
            raise ValueError("article topics must be a list of strings")
        title_missing = record.get("title_missing", False)
        if not isinstance(title_missing, bool):
            raise ValueError("article title_missing must be a boolean")
        category_missing = record.get("category_missing", False)
        subcategory_missing = record.get("subcategory_missing", False)
        if not isinstance(category_missing, bool) or not isinstance(subcategory_missing, bool):
            raise ValueError("article category missing markers must be boolean")
        title = _text(record["title"], "article title")
        if title_missing:
            if title != "":
                raise ValueError("missing MIND title must export as an empty string")
            title = MISSING_MIND_TITLE
        result.append(
            Article(
                id=_text(record["id"], "article id"),
                title=title,
                summary=_text(record.get("summary", ""), "article summary"),
                topics=tuple(topics),
                source=_text(record["source"], "article source"),
                published_at=parse_datetime(record["published_at"], "published_at"),
                quality=_number(record.get("quality", 0.5), "quality"),
                popularity=_number(record.get("popularity", 0.0), "popularity"),
                title_missing=title_missing,
                category_missing=category_missing,
                subcategory_missing=subcategory_missing,
                mind_category=record.get("mind_category"),
                mind_subcategory=record.get("mind_subcategory"),
            )
        )
    ids = [article.id for article in result]
    if len(ids) != len(set(ids)):
        raise ValueError("articles contain duplicate ids")
    return result


def load_articles(path: str | Path) -> list[Article]:
    return _articles_from_records(_records(path))


def load_articles_bytes(data: bytes, *, source_name: str = "snapshot.json") -> list[Article]:
    """Parse one already-bounded immutable catalog byte snapshot."""

    if not isinstance(data, bytes):
        raise ValueError("article snapshot must be bytes")
    source = Path(source_name)
    return _articles_from_records(_records_from_text(data.decode("utf-8"), source))


def _events_from_records(records: Iterable[dict[str, Any]]) -> list[Event]:
    result: list[Event] = []
    allowed = {"user_id", "article_id", "kind", "occurred_at", "propensity", "weight"}
    required = {"user_id", "article_id", "kind", "occurred_at"}
    for record in records:
        _strict(record, allowed, required, "event")
        try:
            raw_kind = _text(record["kind"], "event kind")
            kind = EventKind(raw_kind)
        except ValueError as error:
            raise ValueError(f"unknown event kind: {record['kind']}") from error
        propensity = record.get("propensity")
        result.append(
            Event(
                user_id=_text(record["user_id"], "user id"),
                article_id=_text(record["article_id"], "article id"),
                kind=kind,
                occurred_at=parse_datetime(record["occurred_at"], "occurred_at"),
                propensity=None if propensity is None else _number(propensity, "propensity"),
                weight=_number(record.get("weight", 1.0), "event weight"),
            )
        )
    return result


def load_events(path: str | Path) -> list[Event]:
    return _events_from_records(_records(path))


def load_events_bytes(data: bytes, *, source_name: str = "snapshot.json") -> list[Event]:
    """Parse one already-bounded immutable history byte snapshot."""

    if not isinstance(data, bytes):
        raise ValueError("event snapshot must be bytes")
    source = Path(source_name)
    return _events_from_records(_records_from_text(data.decode("utf-8"), source))


def breakdown_to_dict(breakdown: ScoreBreakdown) -> dict[str, object]:
    return {
        "interest": breakdown.interest,
        "freshness": breakdown.freshness,
        "quality": breakdown.quality,
        "novelty": breakdown.novelty,
        "popularity": breakdown.popularity,
        "exploration": breakdown.exploration,
        "total": breakdown.total,
        "reasons": list(breakdown.reasons),
    }


def feed_to_dict(feed: Feed) -> dict[str, object]:
    return {
        "user_id": feed.user_id,
        "generated_at": feed.generated_at.isoformat(),
        "recommendations": [
            {
                "article_id": recommendation.article_id,
                "rank": recommendation.rank,
                "score": recommendation.score,
                "breakdown": breakdown_to_dict(recommendation.breakdown),
            }
            for recommendation in feed.recommendations
        ],
    }


def write_json(path: str | Path, value: object) -> None:
    atomic_write_text(path, json_text(value))


def json_text(value: object) -> str:
    """Serialize a value using the stable human-readable JSON format."""

    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_jsonl(path: str | Path, records: Iterable[dict[str, object]]) -> None:
    body = "".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records)
    atomic_write_text(path, body)


def atomic_write_text(path: str | Path, body: str) -> None:
    """Replace one text file without exposing a partial destination.

    File contents are flushed before publication.  On POSIX, the destination
    directory is also fsynced after the rename.  Windows and some filesystems
    provide weaker crash/power-loss guarantees; callers must not interpret
    this as storage-hardware durability.
    """

    atomic_write_texts({Path(path): body})


def _fsync_parent_directory(path: str | Path) -> None:
    """Persist directory metadata on POSIX when the platform supports it."""

    if os.name != "posix":
        return
    destination = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(destination.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _entry_status(path: Path) -> os.stat_result | None:
    """Return leaf metadata without following a symbolic link."""

    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _entry_matches(path: Path, expected: os.stat_result) -> bool:
    current = _entry_status(path)
    return current is not None and os.path.samestat(current, expected)


def _replace_observed(
    source: Path,
    destination: Path,
    source_status: os.stat_result,
) -> BaseException | None:
    """Replace a path and report an exception that arrived after the move.

    Some filesystems and fault injectors can report an error after the rename
    became visible.  The source entry's identity is therefore the commit record,
    rather than successful return from ``os.replace``.
    """

    failure: BaseException | None = None
    try:
        os.replace(source, destination)
    except BaseException as error:
        failure = error
    moved = _entry_matches(destination, source_status) and not _entry_matches(source, source_status)
    if moved:
        return failure
    if failure is not None:
        raise failure
    raise OSError(f"atomic replacement of {destination} did not publish the staged entry")


def atomic_write_texts(outputs: Mapping[_PathT, str]) -> None:
    """Publish several text files as one recoverable in-process transaction.

    Every body is staged and flushed before any destination changes. Existing
    files are moved to same-directory backups; if publication raises, restoration
    is attempted before the exception escapes. A persistent recovery failure is
    attached to that exception and leaves its backup for manual recovery. Each
    rename is atomic, but readers can observe intermediate states across
    directories and a process or power failure may leave recoverable files.
    """

    entries = tuple((Path(path), body) for path, body in outputs.items())
    resolved = tuple(os.path.normcase(str(path.resolve(strict=False))) for path, _ in entries)
    aliases = len(set(resolved)) != len(entries)
    for index, (left, _) in enumerate(entries):
        for right, _ in entries[index + 1 :]:
            with suppress(OSError):
                aliases = aliases or os.path.samefile(left, right)
    if aliases:
        raise ValueError("transaction outputs must not refer to the same destination")
    if any(not isinstance(body, str) for _, body in entries):
        raise TypeError("transaction bodies must be strings")
    if not entries:
        return

    staged: list[tuple[Path, Path, os.stat_result]] = []
    backups: dict[Path, tuple[Path, os.stat_result]] = {}
    installed: dict[Path, os.stat_result] = {}
    try:
        for destination, body in entries:
            descriptor, name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
            )
            temporary = Path(name)
            temporary_status = temporary.lstat()
            staged.append((destination, temporary, temporary_status))
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                    descriptor = -1
                    handle.write(body)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)

        for destination, _, _ in staged:
            original_status = _entry_status(destination)
            if original_status is not None:
                descriptor, name = tempfile.mkstemp(
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    suffix=".bak",
                )
                os.close(descriptor)
                backup = Path(name)
                placeholder_status = backup.lstat()
                try:
                    delayed_error = _replace_observed(
                        destination,
                        backup,
                        original_status,
                    )
                except BaseException:
                    if _entry_matches(backup, placeholder_status):
                        with suppress(OSError):
                            backup.unlink()
                    raise
                backups[destination] = (backup, original_status)
                if delayed_error is not None:
                    raise delayed_error
                _fsync_parent_directory(destination)

        for destination, temporary, temporary_status in staged:
            delayed_error = _replace_observed(temporary, destination, temporary_status)
            installed[destination] = temporary_status
            if delayed_error is not None:
                raise delayed_error
            _fsync_parent_directory(destination)
    except BaseException as error:
        rollback_errors: list[BaseException] = []
        for destination, _, _ in reversed(staged):
            try:
                saved = backups.get(destination)
                if saved is not None:
                    current = _entry_status(destination)
                    installed_status = installed.get(destination)
                    if current is not None and (
                        installed_status is None or not os.path.samestat(current, installed_status)
                    ):
                        raise OSError(
                            f"destination changed during transaction rollback: {destination}"
                        )
                    backup, original_status = saved
                    delayed_error = _replace_observed(backup, destination, original_status)
                    if delayed_error is not None and not _entry_matches(
                        destination, original_status
                    ):
                        raise delayed_error
                else:
                    installed_status = installed.get(destination)
                    if installed_status is not None:
                        if not _entry_matches(destination, installed_status):
                            raise OSError(
                                f"transaction destination changed during rollback: {destination}"
                            )
                        destination.unlink()
                _fsync_parent_directory(destination)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            error.add_note(
                "transaction rollback was incomplete; retained backups may require recovery: "
                + "; ".join(str(item) for item in rollback_errors)
            )
        raise
    else:
        for backup, original_status in backups.values():
            if _entry_matches(backup, original_status):
                with suppress(OSError):
                    backup.unlink()
                    _fsync_parent_directory(backup)
    finally:
        for _, temporary, temporary_status in staged:
            if _entry_matches(temporary, temporary_status):
                with suppress(OSError):
                    temporary.unlink()
