"""Strict JSON adapters and stable output serialization."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, cast

from mosaicfeed.models import Article, Event, EventKind, Feed, ScoreBreakdown


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_json_text(value: str) -> object:
    """Parse strict JSON, rejecting non-finite numbers and duplicate object fields."""

    return json.loads(
        value,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_json_object,
    )


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


def _records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.casefold() == ".jsonl":
        lines = source.read_text(encoding="utf-8").splitlines()
        values: object = [load_json_text(line) for line in lines if line.strip()]
    else:
        values = load_json_text(source.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, dict) for value in values):
        raise ValueError(f"{source} must contain a list of JSON objects")
    return cast(list[dict[str, Any]], values)


def _strict(record: dict[str, Any], allowed: set[str], required: set[str], kind: str) -> None:
    unknown = set(record) - allowed
    missing = required - set(record)
    if unknown:
        raise ValueError(f"unknown {kind} fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"missing {kind} fields: {', '.join(sorted(missing))}")


def load_articles(path: str | Path) -> list[Article]:
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
    }
    required = {"id", "title", "topics", "source", "published_at"}
    for record in _records(path):
        _strict(record, allowed, required, "article")
        topics = record["topics"]
        if not isinstance(topics, list) or not all(isinstance(value, str) for value in topics):
            raise ValueError("article topics must be a list of strings")
        result.append(
            Article(
                id=_text(record["id"], "article id"),
                title=_text(record["title"], "article title"),
                summary=_text(record.get("summary", ""), "article summary"),
                topics=tuple(topics),
                source=_text(record["source"], "article source"),
                published_at=parse_datetime(record["published_at"], "published_at"),
                quality=_number(record.get("quality", 0.5), "quality"),
                popularity=_number(record.get("popularity", 0.0), "popularity"),
            )
        )
    ids = [article.id for article in result]
    if len(ids) != len(set(ids)):
        raise ValueError("articles contain duplicate ids")
    return result


def load_events(path: str | Path) -> list[Event]:
    result: list[Event] = []
    allowed = {"user_id", "article_id", "kind", "occurred_at", "propensity"}
    required = {"user_id", "article_id", "kind", "occurred_at"}
    for record in _records(path):
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
            )
        )
    return result


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
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: str | Path, records: Iterable[dict[str, object]]) -> None:
    body = "".join(
        json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records
    )
    Path(path).write_text(body, encoding="utf-8", newline="\n")
