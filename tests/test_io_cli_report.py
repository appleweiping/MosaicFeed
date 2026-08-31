from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mosaicfeed.cli import _article_record, _event_record, main
from mosaicfeed.config import FeedConfig
from mosaicfeed.io import (
    breakdown_to_dict,
    feed_to_dict,
    load_articles,
    load_events,
    parse_datetime,
    write_json,
    write_jsonl,
)
from mosaicfeed.models import Article, Event, EventKind, Feed
from mosaicfeed.pipeline import build_feed
from mosaicfeed.report import render_feed_report


def write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def article_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "a",
        "title": "Title",
        "summary": "Summary",
        "topics": ["AI"],
        "source": "Source",
        "published_at": "2026-01-01T00:00:00Z",
        "quality": 0.8,
        "popularity": 0.4,
    }
    record.update(overrides)
    return record


def event_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "user_id": "u",
        "article_id": "a",
        "kind": "click",
        "occurred_at": "2026-01-02T00:00:00Z",
        "propensity": 0.5,
    }
    record.update(overrides)
    return record


def test_parse_datetime_accepts_z_and_rejects_bad_values() -> None:
    assert parse_datetime("2026-01-01T00:00:00Z", "time").tzinfo is not None
    with pytest.raises(ValueError, match="ISO-8601"):
        parse_datetime(4, "time")
    with pytest.raises(ValueError, match="invalid"):
        parse_datetime("not-a-date", "time")
    with pytest.raises(ValueError, match="timezone"):
        parse_datetime("2026-01-01T00:00:00", "time")


def test_load_articles_json_and_jsonl(tmp_path: Path) -> None:
    json_path = write(tmp_path / "articles.json", [article_record()])
    item = load_articles(json_path)[0]
    assert item.id == "a"
    jsonl_path = tmp_path / "articles.jsonl"
    write_jsonl(jsonl_path, [article_record(id="a"), article_record(id="b")])
    assert [value.id for value in load_articles(jsonl_path)] == ["a", "b"]


def test_load_articles_defaults_optional_fields(tmp_path: Path) -> None:
    record = article_record()
    for field in ("summary", "quality", "popularity"):
        del record[field]
    item = load_articles(write(tmp_path / "articles.json", [record]))[0]
    assert item.summary == ""
    assert item.quality == 0.5
    assert item.popularity == 0.0


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"bad": "shape"}, "list"),
        ([article_record(extra=1)], "unknown"),
        ([{"id": "a"}], "missing"),
        ([article_record(topics="ai")], "topics"),
        ([article_record(id=7)], "string"),
        ([article_record(quality="high")], "finite number"),
        ([article_record(id="a"), article_record(id="a")], "duplicate"),
    ],
)
def test_load_articles_rejects_invalid_data(tmp_path: Path, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_articles(write(tmp_path / "articles.json", value))


def test_json_input_rejects_duplicate_fields_and_non_finite_numbers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '[{"id":"a","id":"b","title":"T","topics":["ai"],'
        '"source":"S","published_at":"2026-01-01T00:00:00Z"}]',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON field"):
        load_articles(duplicate)

    non_finite = tmp_path / "non-finite.json"
    non_finite.write_text(
        '[{"id":"a","title":"T","topics":["ai"],"source":"S",'
        '"published_at":"2026-01-01T00:00:00Z","quality":Infinity}]',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite JSON number"):
        load_articles(non_finite)

    huge_integer = tmp_path / "huge-integer.json"
    huge_integer.write_text(
        '[{"id":"a","title":"T","topics":["ai"],"source":"S",'
        '"published_at":"2026-01-01T00:00:00Z","quality":'
        + "9" * 400
        + "}]",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite number"):
        load_articles(huge_integer)


def test_json_output_rejects_non_finite_numbers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        write_json(tmp_path / "invalid.json", {"score": float("nan")})
    with pytest.raises(ValueError, match="JSON compliant"):
        write_jsonl(tmp_path / "invalid.jsonl", [{"score": float("inf")}])


def test_load_events_and_optional_propensity(tmp_path: Path) -> None:
    first = event_record()
    second = event_record(article_id="b")
    del second["propensity"]
    loaded = load_events(write(tmp_path / "events.json", [first, second]))
    assert loaded[0].kind is EventKind.CLICK
    assert loaded[1].propensity is None


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"bad": "shape"}, "list"),
        ([event_record(extra=1)], "unknown"),
        ([{"user_id": "u"}], "missing"),
        ([event_record(kind="share")], "unknown event kind"),
        ([event_record(user_id=7)], "string"),
        ([event_record(propensity="likely")], "finite number"),
    ],
)
def test_load_events_rejects_invalid_data(tmp_path: Path, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_events(write(tmp_path / "events.json", value))


def test_serializers_are_stable(
    articles: list[Article], events: list[Event], now: datetime, tmp_path: Path
) -> None:
    feed = build_feed("u1", articles, events, as_of=now, config=FeedConfig(size=1))
    payload = feed_to_dict(feed)
    assert payload["user_id"] == "u1"
    recommendation = feed.recommendations[0]
    assert breakdown_to_dict(recommendation.breakdown)["total"] == recommendation.score
    output = tmp_path / "feed.json"
    write_json(output, payload)
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert b"\r\n" not in output.read_bytes()


def test_report_renders_feed_and_empty_state(
    articles: list[Article], events: list[Event], now: datetime, tmp_path: Path
) -> None:
    article_map = {article.id: article for article in articles}
    feed = build_feed("u1", articles, events, as_of=now, config=FeedConfig(size=2))
    output = tmp_path / "report.html"
    render_feed_report(feed, article_map, output=output)
    body = output.read_text(encoding="utf-8")
    assert "MosaicFeed" in body
    assert "ranking score before diversity reranking" in body
    assert "<script" not in body
    assert b"\r\n" not in output.read_bytes()
    assert all(line == line.rstrip() for line in body.splitlines())
    empty = tmp_path / "empty.html"
    render_feed_report(Feed("u", now, ()), article_map, output=empty)
    assert "No eligible candidates" in empty.read_text(encoding="utf-8")


def test_report_escapes_all_user_controlled_text(now: datetime, tmp_path: Path) -> None:
    marker = '<script>alert("unsafe")</script>'
    article = Article("a", marker, marker, (marker,), marker, now)
    feed = build_feed(marker, [article], [], as_of=now, config=FeedConfig(size=1))
    output = tmp_path / "escaped.html"
    render_feed_report(feed, {article.id: article}, output=output)
    body = output.read_text(encoding="utf-8")
    assert marker not in body
    assert "&lt;script&gt;" in body


def test_cli_validate_and_unknown_reference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    articles_path = write(tmp_path / "articles.json", [article_record()])
    events_path = write(tmp_path / "events.json", [event_record()])
    assert main(["validate", "--articles", str(articles_path), "--events", str(events_path)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    write(events_path, [event_record(article_id="missing")])
    assert main(["validate", "--articles", str(articles_path), "--events", str(events_path)]) == 2
    assert "unknown" in capsys.readouterr().err


def test_cli_reports_invalid_config_as_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    articles_path = write(tmp_path / "articles.json", [article_record()])
    events_path = write(tmp_path / "events.json", [])
    config_path = tmp_path / "config.json"
    config_path.write_text('{"quality_weight": NaN}', encoding="utf-8")
    result = main(
        [
            "recommend",
            "--articles",
            str(articles_path),
            "--events",
            str(events_path),
            "--user",
            "u",
            "--config",
            str(config_path),
        ]
    )
    assert result == 2
    assert "non-finite JSON number" in capsys.readouterr().err


def test_cli_recommend_stdout_and_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    articles_path = write(
        tmp_path / "articles.json",
        [article_record(id="a"), article_record(id="b", source="Other")],
    )
    events_path = write(tmp_path / "events.json", [])
    args = [
        "recommend",
        "--articles",
        str(articles_path),
        "--events",
        str(events_path),
        "--user",
        "u",
        "--as-of",
        "2026-01-03T00:00:00Z",
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["recommendations"]
    output = tmp_path / "feed.json"
    report = tmp_path / "feed.html"
    config = write(tmp_path / "config.json", {"size": 1})
    file_args = [
        *args,
        "--config",
        str(config),
        "--output",
        str(output),
        "--html",
        str(report),
    ]
    assert main(file_args) == 0
    assert len(json.loads(output.read_text(encoding="utf-8"))["recommendations"]) == 1
    assert report.exists()


def test_cli_evaluate_stdout_and_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    articles_path = write(
        tmp_path / "articles.json",
        [article_record(id="a"), article_record(id="b", source="Other")],
    )
    events_path = write(tmp_path / "events.json", [event_record()])
    args = [
        "evaluate",
        "--articles",
        str(articles_path),
        "--events",
        str(events_path),
        "--as-of",
        "2026-01-03T00:00:00Z",
        "--k",
        "1",
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["users_evaluated"] == 1
    output = tmp_path / "metrics.json"
    assert main([*args, "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["k"] == 1


def test_cli_simulate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    directory = tmp_path / "synthetic"
    assert main(
        [
            "simulate",
            "--directory",
            str(directory),
            "--seed",
            "3",
            "--users",
            "2",
            "--articles",
            "5",
            "--events-per-user",
            "2",
        ]
    ) == 0
    assert "wrote 5 articles and 4 events" in capsys.readouterr().out
    assert len(load_articles(directory / "articles.json")) == 5
    assert len(load_events(directory / "events.json")) == 4
    assert (directory / "metadata.json").exists()


def test_cli_record_helpers_reject_wrong_types() -> None:
    with pytest.raises(TypeError, match="Article"):
        _article_record(object())
    with pytest.raises(TypeError, match="Event"):
        _event_record(object())


def test_cli_record_helpers_include_expected_fields() -> None:
    item = Article("a", "T", "S", ("ai",), "Source", datetime(2026, 1, 1, tzinfo=UTC))
    event = Event("u", "a", EventKind.VIEW, datetime(2026, 1, 2, tzinfo=UTC))
    assert _article_record(item)["topics"] == ["ai"]
    assert "propensity" not in _event_record(event)
