from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mosaicfeed import __version__
from mosaicfeed import io as io_module
from mosaicfeed.cli import _article_record, _event_record, main
from mosaicfeed.config import FeedConfig
from mosaicfeed.io import (
    atomic_write_texts,
    breakdown_to_dict,
    feed_to_dict,
    load_articles,
    load_events,
    load_json_text,
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


def test_load_json_text_normalizes_excessive_nesting() -> None:
    deeply_nested = "[" * 2_000 + "]" * 2_000
    with pytest.raises(ValueError, match="nesting is too deep"):
        load_json_text(deeply_nested)
    assert isinstance(load_json_text("[" * 256 + "]" * 256), list)
    with pytest.raises(ValueError, match="nesting is too deep"):
        load_json_text("[" * 257 + "]" * 257)
    assert load_json_text(json.dumps({"literal": "[" * 2_000})) == {"literal": "[" * 2_000}
    assert load_json_text('{"literal":"[[[\\"still a string"}') == {"literal": '[[["still a string'}


def test_cli_version_uses_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["--version"])
    assert stopped.value.code == 0
    assert capsys.readouterr().out == f"mosaicfeed {__version__}\n"


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
        '"published_at":"2026-01-01T00:00:00Z","quality":' + "9" * 400 + "}]",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite number"):
        load_articles(huge_integer)


def test_json_output_rejects_non_finite_numbers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        write_json(tmp_path / "invalid.json", {"score": float("nan")})
    with pytest.raises(ValueError, match="JSON compliant"):
        write_jsonl(tmp_path / "invalid.jsonl", [{"score": float("inf")}])


@pytest.mark.parametrize(
    ("stage", "error_type"),
    (("fsync", KeyboardInterrupt), ("replace", SystemExit)),
)
def test_text_outputs_preserve_old_target_and_clean_staging_on_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    error_type: type[BaseException],
) -> None:
    destination = tmp_path / "result.json"
    destination.write_text("sentinel\n", encoding="utf-8")

    def interrupt(*_args: object) -> None:
        raise error_type()

    monkeypatch.setattr(io_module.os, stage, interrupt)
    with pytest.raises(error_type):
        write_json(destination, {"replacement": True})

    assert destination.read_text(encoding="utf-8") == "sentinel\n"
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


@pytest.mark.parametrize("error_type", (OSError, KeyboardInterrupt))
def test_multi_output_transaction_rolls_back_across_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    left = tmp_path / "left" / "result.json"
    right = tmp_path / "right" / "result.html"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_text("old-left", encoding="utf-8")
    right.write_text("old-right", encoding="utf-8")
    original_replace = io_module.os.replace
    calls = 0

    def fail_second_install(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise error_type("forced second output failure")
        original_replace(source, destination)

    monkeypatch.setattr(io_module.os, "replace", fail_second_install)
    with pytest.raises(error_type):
        atomic_write_texts({left: "new-left", right: "new-right"})

    assert left.read_text(encoding="utf-8") == "old-left"
    assert right.read_text(encoding="utf-8") == "old-right"
    assert not tuple(tmp_path.rglob("*.tmp"))
    assert not tuple(tmp_path.rglob("*.bak"))


@pytest.mark.parametrize("error_type", (OSError, KeyboardInterrupt, SystemExit))
@pytest.mark.parametrize("failure_call", (1, 2, 3, 4))
@pytest.mark.parametrize("after_effect", (False, True))
def test_multi_output_transaction_reconciles_every_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
    failure_call: int,
    after_effect: bool,
) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text("old-left", encoding="utf-8")
    right.write_text("old-right", encoding="utf-8")
    original_replace = io_module.os.replace
    calls = 0

    def fail_selected_replace(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            if after_effect:
                original_replace(source, destination)
            raise error_type("injected replace failure")
        original_replace(source, destination)

    monkeypatch.setattr(io_module.os, "replace", fail_selected_replace)
    with pytest.raises(error_type):
        atomic_write_texts({left: "new-left", right: "new-right"})

    assert left.read_text(encoding="utf-8") == "old-left"
    assert right.read_text(encoding="utf-8") == "old-right"
    assert not tuple(tmp_path.glob(".*.tmp"))
    assert not tuple(tmp_path.glob(".*.bak"))


@pytest.mark.parametrize("error_type", (OSError, KeyboardInterrupt, SystemExit))
@pytest.mark.parametrize("failure_call", (1, 2, 3))
@pytest.mark.parametrize("after_effect", (False, True))
def test_mixed_existing_and_new_outputs_reconcile_every_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
    failure_call: int,
    after_effect: bool,
) -> None:
    existing = tmp_path / "existing.json"
    created = tmp_path / "created.json"
    existing.write_text("old", encoding="utf-8")
    original_replace = io_module.os.replace
    calls = 0

    def fail_selected_replace(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            if after_effect:
                original_replace(source, destination)
            raise error_type("injected replace failure")
        original_replace(source, destination)

    monkeypatch.setattr(io_module.os, "replace", fail_selected_replace)
    with pytest.raises(error_type):
        atomic_write_texts({existing: "new-existing", created: "new-created"})

    assert existing.read_text(encoding="utf-8") == "old"
    assert not created.exists()
    assert not tuple(tmp_path.glob(".*.tmp"))
    assert not tuple(tmp_path.glob(".*.bak"))


def test_rollback_reconciles_an_interrupt_reported_after_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result.json"
    destination.write_text("old", encoding="utf-8")
    original_replace = io_module.os.replace
    calls = 0

    def fail_install_then_interrupt_restoration(source: object, target: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("installation failed")
        original_replace(source, target)
        if calls == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(io_module.os, "replace", fail_install_then_interrupt_restoration)
    with pytest.raises(OSError, match="installation failed"):
        atomic_write_texts({destination: "new"})

    assert destination.read_text(encoding="utf-8") == "old"
    assert not tuple(tmp_path.glob(".*.tmp"))
    assert not tuple(tmp_path.glob(".*.bak"))


def test_persistent_restore_failure_retains_recoverable_backup_and_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result.json"
    destination.write_text("old", encoding="utf-8")
    original_replace = io_module.os.replace
    calls = 0

    def fail_install_and_restore(source: object, target: object) -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise OSError(f"injected failure {calls}")
        original_replace(source, target)

    monkeypatch.setattr(io_module.os, "replace", fail_install_and_restore)
    with pytest.raises(OSError, match="injected failure 2") as stopped:
        atomic_write_texts({destination: "new"})

    assert not destination.exists()
    backups = tuple(tmp_path.glob(".*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "old"
    assert any("rollback was incomplete" in note for note in stopped.value.__notes__)
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_multi_output_transaction_rejects_filesystem_aliases(tmp_path: Path) -> None:
    original = tmp_path / "original.json"
    original.write_text("old", encoding="utf-8")
    hardlink = tmp_path / "hardlink.json"
    os.link(original, hardlink)

    with pytest.raises(ValueError, match="same destination"):
        atomic_write_texts({str(original): "first", str(hardlink): "second"})
    assert original.read_text(encoding="utf-8") == "old"

    symlink = tmp_path / "symlink.json"
    try:
        symlink.symlink_to(original)
    except OSError:
        return
    with pytest.raises(ValueError, match="same destination"):
        atomic_write_texts({str(original): "first", str(symlink): "second"})
    assert original.read_text(encoding="utf-8") == "old"


@pytest.mark.skipif(os.name != "nt", reason="Windows path aliases are case-insensitive")
def test_multi_output_transaction_rejects_case_aliases_before_creation(tmp_path: Path) -> None:
    lower = tmp_path / "result.json"
    upper = tmp_path / "RESULT.JSON"

    with pytest.raises(ValueError, match="same destination"):
        atomic_write_texts({str(lower): "first", str(upper): "second"})
    assert tuple(tmp_path.iterdir()) == ()


def test_directory_fsync_failure_rolls_back_an_installed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "fsync-result.json"
    destination.write_text("old", encoding="utf-8")
    calls = 0

    def fail_after_install(_path: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")

    monkeypatch.setattr(io_module, "_fsync_parent_directory", fail_after_install)
    with pytest.raises(OSError, match="directory fsync"):
        write_json(destination, {"new": True})
    assert destination.read_text(encoding="utf-8") == "old"


def test_recommend_cli_rolls_back_json_when_html_install_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, now: datetime
) -> None:
    articles_path = write(tmp_path / "articles.json", [article_record()])
    events_path = write(tmp_path / "events.json", [])
    output = tmp_path / "feed.json"
    html_output = tmp_path / "feed.html"
    output.write_text("old-json", encoding="utf-8")
    html_output.write_text("old-html", encoding="utf-8")
    original_replace = io_module.os.replace
    calls = 0

    def fail_second_install(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("forced second output failure")
        original_replace(source, destination)

    monkeypatch.setattr(io_module.os, "replace", fail_second_install)
    assert (
        main(
            [
                "recommend",
                "--articles",
                str(articles_path),
                "--events",
                str(events_path),
                "--user",
                "u",
                "--as-of",
                now.isoformat(),
                "--output",
                str(output),
                "--html",
                str(html_output),
            ]
        )
        == 2
    )
    assert output.read_text(encoding="utf-8") == "old-json"
    assert html_output.read_text(encoding="utf-8") == "old-html"


def test_load_events_and_optional_propensity(tmp_path: Path) -> None:
    first = event_record()
    second = event_record(article_id="b")
    del second["propensity"]
    third = event_record(article_id="c", weight=2.5)
    loaded = load_events(write(tmp_path / "events.json", [first, second, third]))
    assert loaded[0].kind is EventKind.CLICK
    assert loaded[1].propensity is None
    assert loaded[1].weight == 1.0
    assert loaded[2].weight == 2.5


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"bad": "shape"}, "list"),
        ([event_record(extra=1)], "unknown"),
        ([{"user_id": "u"}], "missing"),
        ([event_record(kind="share")], "unknown event kind"),
        ([event_record(user_id=7)], "string"),
        ([event_record(propensity="likely")], "finite number"),
        ([event_record(weight="heavy")], "finite number"),
        ([event_record(weight=0)], "event weight"),
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


def test_cli_rejects_hardlinked_input_output_before_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    articles_path = write(tmp_path / "articles.json", [article_record()])
    events_path = write(tmp_path / "events.json", [])
    output = tmp_path / "feed.json"
    os.link(events_path, output)
    before = events_path.read_bytes()

    assert (
        main(
            [
                "recommend",
                "--articles",
                str(articles_path),
                "--events",
                str(events_path),
                "--user",
                "u",
                "--as-of",
                "2026-01-02T00:00:00Z",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert "--events and --output must refer to different paths" in capsys.readouterr().err
    assert events_path.read_bytes() == before
    assert output.read_bytes() == before


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
    assert (
        main(
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
        )
        == 0
    )
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
    assert (
        _event_record(
            Event("u", "a", EventKind.VIEW, datetime(2026, 1, 2, tzinfo=UTC), weight=2.0)
        )["weight"]
        == 2.0
    )
