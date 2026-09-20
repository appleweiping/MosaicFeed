"""Independent arithmetic and temporal-boundary tests for cohort audits."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import mosaicfeed.cohorts as cohort_module
from mosaicfeed.benchmark import ConfidenceInterval
from mosaicfeed.cli import main
from mosaicfeed.cohorts import (
    MAX_COHORT_INPUT_BYTES,
    MAX_USERS,
    audit_cohorts,
    load_audit_sources,
    load_declared_cohorts,
)
from mosaicfeed.config import FeedConfig
from mosaicfeed.metrics import EvaluationSamples, UserEvaluation
from mosaicfeed.models import Article, Event, EventKind

NOW = datetime(2026, 9, 1, 12, tzinfo=UTC)


def _article(article_id: str, topic: str) -> Article:
    return Article(
        article_id,
        article_id,
        "summary",
        (topic,),
        "source",
        NOW - timedelta(days=2),
    )


def _user(user_id: str, value: float) -> UserEvaluation:
    return UserEvaluation(
        user_id=user_id,
        holdout_article_id="a",
        holdout_at=NOW,
        ranked_ids=("a",),
        eligible_catalog_ids=frozenset({"a"}),
        ndcg=value,
        hit_rate=value,
        reciprocal_rank=value,
        intra_list_diversity=0.0,
        source_diversity=value,
    )


def test_hand_computed_group_means_gaps_and_missing_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    users = (_user("u1", 1.0), _user("u2", 0.0), _user("u3", 1.0), _user("u4", 1.0))
    samples = EvaluationSamples(users, 0, 1, 1.0, 0.0, 0.0, 1)
    monkeypatch.setattr(cohort_module, "evaluate_leave_last_out_samples", lambda *a, **k: samples)
    monkeypatch.setattr(
        cohort_module,
        "_calibration_values",
        lambda *a, **k: {"u1": 0.1, "u2": 0.3, "u3": None, "u4": None},
    )
    events = [Event(user.user_id, "a", EventKind.CLICK, NOW) for user in users]
    mapping = {"u1": "A", "u2": "A", "u3": "B", "u4": "B"}
    report = audit_cohorts(
        [_article("a", "topic")],
        events,
        mapping,
        as_of=NOW,
        config=FeedConfig(),
        minimum_group_size=2,
        bootstrap_samples=25,
    )
    assert report.cohorts["A"].metrics["ndcg"].mean == 0.5
    assert report.cohorts["B"].metrics["ndcg"].mean == 1.0
    assert report.max_minus_min["ndcg"] == 0.5
    assert report.max_minus_min["intra_list_diversity"] == 0.0
    assert report.cohorts["A"].topic_calibration is not None
    assert report.cohorts["A"].topic_calibration.mean == pytest.approx(0.2)
    assert report.cohorts["A"].calibration_observations == 2
    assert report.cohorts["A"].active_users == 2
    assert report.cohorts["A"].users_skipped == 0
    assert report.cohorts["B"].topic_calibration is None
    assert report.max_minus_min["topic_calibration"] is None
    assert report.to_dict()["schema_version"] == 1
    assert "u1" not in json.dumps(report.to_dict())
    assert report.cohorts["A"].metrics["ndcg"].observations == 2
    reversed_report = audit_cohorts(
        [_article("a", "topic")],
        events,
        dict(reversed(list(mapping.items()))),
        as_of=NOW,
        config=FeedConfig(),
        minimum_group_size=2,
        bootstrap_samples=25,
    )
    assert report.to_dict() == reversed_report.to_dict()


def test_future_view_and_same_time_event_do_not_leak_into_calibration() -> None:
    articles = [_article("history", "a"), _article("holdout", "b")]
    old = Event("reader", "history", EventKind.CLICK, NOW - timedelta(hours=2))
    holdout = Event("reader", "holdout", EventKind.CLICK, NOW)
    same_time = Event("reader", "holdout", EventKind.VIEW, NOW)
    future = Event("reader", "holdout", EventKind.VIEW, NOW + timedelta(minutes=1))
    kwargs = dict(
        as_of=NOW + timedelta(hours=1),
        config=FeedConfig(size=1),
        minimum_group_size=1,
        bootstrap_samples=12,
    )
    baseline = audit_cohorts(articles, [old, holdout], {"reader": "group"}, **kwargs)
    altered = audit_cohorts(
        articles, [old, holdout, same_time, future], {"reader": "group"}, **kwargs
    )
    assert baseline.dataset_sha256 != altered.dataset_sha256
    assert baseline.cohorts["group"].to_dict() == altered.cohorts["group"].to_dict()
    assert baseline.cohorts["group"].topic_calibration is not None


def test_no_prior_positive_history_reports_missing_calibration() -> None:
    article = _article("holdout", "a")
    report = audit_cohorts(
        [article],
        [
            Event("reader", "holdout", EventKind.CLICK, NOW),
            Event("skipped", "holdout", EventKind.VIEW, NOW),
        ],
        {"reader": "group", "skipped": "group"},
        as_of=NOW,
        config=FeedConfig(size=1),
        minimum_group_size=1,
        bootstrap_samples=4,
    )
    assert report.cohorts["group"].topic_calibration is None
    assert report.cohorts["group"].calibration_observations == 0
    assert report.cohorts["group"].active_users == 2
    assert report.cohorts["group"].users_skipped == 1
    assert report.max_minus_min["ndcg"] is None
    assert report.max_minus_min["topic_calibration"] is None


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({}, "exactly"),
        ({"u": "x", "other": "x"}, "exactly"),
        ({"u": " x"}, "printable"),
        ({"u": "x\n"}, "printable"),
        ({"u": "x" * 65}, "printable"),
    ],
)
def test_declared_cohorts_are_exact_and_bounded(mapping: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        audit_cohorts(
            [_article("a", "a")],
            [Event("u", "a", EventKind.CLICK, NOW)],
            mapping,
            as_of=NOW,
            config=FeedConfig(),
            minimum_group_size=1,
            bootstrap_samples=2,
        )


def test_small_group_and_work_budget_fail_closed() -> None:
    events = [Event("u", "a", EventKind.CLICK, NOW)]
    with pytest.raises(ValueError, match="minimum_group_size"):
        audit_cohorts(
            [_article("a", "a")],
            events,
            {"u": "g"},
            as_of=NOW,
            config=FeedConfig(),
            minimum_group_size=2,
            bootstrap_samples=2,
        )
    with pytest.raises(ValueError, match="work limit"):
        audit_cohorts(
            [_article("a", "a")],
            events,
            {"u": "g"},
            as_of=NOW,
            config=FeedConfig(),
            minimum_group_size=1,
            bootstrap_samples=2_000_000,
        )


@pytest.mark.parametrize(
    "source",
    ['{"u":"a","u":"b"}', '{"u": 1}', '["a"]', '{"u":NaN}', "\ud800"],
)
def test_cohort_loader_rejects_malformed_inputs(tmp_path: Path, source: str) -> None:
    path = tmp_path / "cohorts.json"
    path.write_bytes(source.encode("utf-8", errors="surrogatepass"))
    with pytest.raises(ValueError, match=r"strict UTF-8 JSON|JSON object"):
        load_declared_cohorts(path)


def test_cli_cohort_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    article_path = tmp_path / "articles.json"
    event_path = tmp_path / "events.json"
    cohort_path = tmp_path / "cohorts.json"
    article_path.write_text(
        json.dumps(
            [
                {
                    "id": "a",
                    "title": "A",
                    "summary": "topic a",
                    "topics": ["a"],
                    "source": "source",
                    "published_at": (NOW - timedelta(days=1)).isoformat(),
                }
            ]
        ),
        encoding="utf-8",
    )
    event_path.write_text(
        json.dumps(
            [
                {
                    "user_id": "u",
                    "article_id": "a",
                    "kind": "click",
                    "occurred_at": NOW.isoformat(),
                }
            ]
        ),
        encoding="utf-8",
    )
    cohort_path.write_text('{"u":"synthetic"}', encoding="utf-8")
    assert (
        main(
            [
                "audit-cohorts",
                "--articles",
                str(article_path),
                "--events",
                str(event_path),
                "--cohorts",
                str(cohort_path),
                "--as-of",
                NOW.isoformat(),
                "--minimum-group-size",
                "1",
                "--bootstrap-samples",
                "4",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["cohorts"]["synthetic"]["users"] == 1
    assert payload["cohorts"]["synthetic"]["active_users"] == 1
    assert payload["max_minus_min"]["ndcg"] is None
    assert payload["cohorts"]["synthetic"]["topic_calibration"] is None


def test_public_report_constructors_reject_inconsistent_or_mutable_state() -> None:
    report = audit_cohorts(
        [_article("a", "a")],
        [Event("u", "a", EventKind.CLICK, NOW)],
        {"u": "group"},
        as_of=NOW,
        config=FeedConfig(),
        minimum_group_size=1,
        bootstrap_samples=2,
    )
    summary = report.cohorts["group"]
    with pytest.raises(TypeError):
        report.cohorts["evil"] = summary  # type: ignore[index]
    with pytest.raises(TypeError):
        summary.metrics["ndcg"] = summary.metrics["ndcg"]  # type: ignore[index]
    for changes in (
        {"active_users": 2},
        {"users": 0},
        {"calibration_observations": 2},
        {"metrics": {}},
        {"metrics": {name: ConfidenceInterval(0, 0, 0, 0.95, 2) for name in summary.metrics}},
        {"topic_calibration": ConfidenceInterval(0, 0, 0, 0.95, 1)},
        {"metrics": {**summary.metrics, "ndcg": ConfidenceInterval(-0.1, -0.1, 0.0, 0.95, 1)}},
        {"metrics": {**summary.metrics, "ndcg": ConfidenceInterval(0, 0, 0, 0.8, 1)}},
    ):
        with pytest.raises(ValueError):
            replace(summary, **changes)
    for changes in (
        {"cohorts": {}},
        {"users_skipped": 1},
        {"max_minus_min": {}},
        {"max_minus_min": {**report.max_minus_min, "ndcg": float("nan")}},
        {"max_minus_min": {**report.max_minus_min, "ndcg": 0.5}},
        {"dataset_sha256": "not-a-digest"},
        {"cohorts_sha256": "A" * 64},
        {"as_of": NOW.replace(tzinfo=None)},
        {"k": 0},
        {"minimum_group_size": True},
        {"bootstrap_samples": 0},
        {"confidence": float("nan")},
        {"seed": True},
        {"cohorts": {"bad\nlabel": summary}},
    ):
        with pytest.raises(ValueError):
            replace(report, **changes)


def test_public_report_rejects_zero_observation_calibration_interval() -> None:
    report = audit_cohorts(
        [_article("a", "a")],
        [Event("u", "a", EventKind.CLICK, NOW)],
        {"u": "group"},
        as_of=NOW,
        config=FeedConfig(),
        minimum_group_size=1,
        bootstrap_samples=2,
    )
    summary = report.cohorts["group"]
    with pytest.raises(ValueError, match="calibration interval"):
        replace(summary, topic_calibration=ConfidenceInterval(0, 0, 0, 0.95, 0))
    with pytest.raises(ValueError, match="calibration interval"):
        replace(
            summary,
            calibration_observations=1,
            topic_calibration=ConfidenceInterval(0, 0, 0, 0.8, 1),
        )


def test_source_and_config_size_limits_fail_without_writing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cohort_module, "MAX_AUDIT_SOURCE_BYTES", 10)
    article_path = tmp_path / "articles.json"
    article_path.write_bytes(b" " * 11)
    with pytest.raises(ValueError, match="input size limit"):
        load_audit_sources(article_path, tmp_path / "events.json")

    monkeypatch.setattr(cohort_module, "MAX_AUDIT_SOURCE_BYTES", 16 * 1024 * 1024)
    article_path.write_text("[]", encoding="utf-8")
    event_path = tmp_path / "events.json"
    event_path.write_text("[]", encoding="utf-8")
    cohort_path = tmp_path / "cohorts.json"
    cohort_path.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_bytes(b" " * (64 * 1024 + 1))
    output_path = tmp_path / "report.json"
    assert (
        main(
            [
                "audit-cohorts",
                "--articles",
                str(article_path),
                "--events",
                str(event_path),
                "--cohorts",
                str(cohort_path),
                "--config",
                str(config_path),
                "--as-of",
                NOW.isoformat(),
                "--output",
                str(output_path),
            ]
        )
        == 2
    )
    assert "max_config_bytes" in capsys.readouterr().err
    assert not output_path.exists()


def test_direct_api_row_and_work_limits_fail_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    article = _article("a", "a")
    event = Event("u", "a", EventKind.CLICK, NOW)
    monkeypatch.setattr(cohort_module, "MAX_AUDIT_ROWS", 2)
    with pytest.raises(ValueError, match="row limit"):
        audit_cohorts([article] * 3, [event], {"u": "g"}, as_of=NOW, config=FeedConfig())
    monkeypatch.setattr(cohort_module, "MAX_AUDIT_ROWS", 100_000)
    monkeypatch.setattr(cohort_module, "MAX_EVALUATION_WORK", 2)
    with pytest.raises(ValueError, match="evaluation work limit"):
        audit_cohorts([article], [event, event], {"u": "g"}, as_of=NOW, config=FeedConfig())


def test_invalid_controls_and_empty_population_fail_before_evaluation() -> None:
    article = _article("a", "a")
    events = [Event("u", "a", EventKind.CLICK, NOW)]
    base = dict(as_of=NOW, config=FeedConfig(), minimum_group_size=1, bootstrap_samples=2)
    for options in (
        {"minimum_group_size": True},
        {"bootstrap_samples": 0},
        {"seed": True},
        {"as_of": NOW.replace(tzinfo=None)},
    ):
        with pytest.raises(ValueError):
            audit_cohorts([article], events, {"u": "group"}, **{**base, **options})
    with pytest.raises(ValueError, match="no active users"):
        audit_cohorts([article], [], {}, **base)
    with pytest.raises(ValueError, match="mapping"):
        audit_cohorts([article], events, None, **base)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strings"):
        audit_cohorts([article], events, {"u": None}, **base)  # type: ignore[dict-item]


def test_cohort_count_and_input_size_limits(tmp_path: Path) -> None:
    article = _article("a", "a")
    many_events = [Event(f"u{i}", "a", EventKind.CLICK, NOW) for i in range(65)]
    many_labels = {event.user_id: f"group-{i}" for i, event in enumerate(many_events)}
    with pytest.raises(ValueError, match="too many"):
        audit_cohorts(
            [article],
            many_events,
            many_labels,
            as_of=NOW,
            config=FeedConfig(),
            minimum_group_size=1,
            bootstrap_samples=2,
        )
    too_many_users = [Event(f"u{i}", "a", EventKind.CLICK, NOW) for i in range(MAX_USERS + 1)]
    with pytest.raises(ValueError, match="user limit"):
        audit_cohorts(
            [article],
            too_many_users,
            {},
            as_of=NOW,
            config=FeedConfig(),
            minimum_group_size=1,
            bootstrap_samples=2,
        )
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (MAX_COHORT_INPUT_BYTES + 1))
    with pytest.raises(ValueError, match="size limit"):
        load_declared_cohorts(path)
