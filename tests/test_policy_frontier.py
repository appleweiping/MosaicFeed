"""Independent arithmetic, temporal, provenance, and publication tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import mosaicfeed.policy_frontier as frontier_module
from mosaicfeed.benchmark import ConfidenceInterval
from mosaicfeed.cli import main
from mosaicfeed.cohorts import COHORT_METRICS, CohortSummary, audit_cohorts
from mosaicfeed.config import FeedConfig
from mosaicfeed.metrics import evaluate_leave_last_out_samples
from mosaicfeed.models import Article, Event, EventKind
from mosaicfeed.policy_frontier import (
    FrontierConstraints,
    NamedPolicy,
    PolicyFrontierPlan,
    compare_cohort_policies,
    write_frontier_report,
)

NOW = datetime(2026, 9, 20, tzinfo=UTC)


def _article(article_id: str, topic: str = "topic") -> Article:
    return Article(article_id, article_id, "summary", (topic,), "source", NOW - timedelta(days=2))


def _source() -> tuple[list[Article], list[Event], dict[str, str]]:
    articles = [_article("a", "one"), _article("b", "two"), _article("c", "one")]
    events = [
        Event(user, "a", EventKind.CLICK, NOW - timedelta(days=1)) for user in ("u1", "u2", "u3")
    ] + [Event(user, "b", EventKind.LIKE, NOW) for user in ("u1", "u2", "u3")]
    return articles, events, {"u1": "A", "u2": "A", "u3": "B"}


def _plan(*names: str, constraints: FrontierConstraints | None = None) -> PolicyFrontierPlan:
    return PolicyFrontierPlan(
        NOW,
        2,
        tuple(
            NamedPolicy(name, FeedConfig(size=2, minimum_score=index / 10))
            for index, name in enumerate(names)
        ),
        minimum_group_size=1,
        bootstrap_samples=4,
        hard_constraints=constraints if constraints is not None else FrontierConstraints(),
    )


def _interval(mean: float, observations: int) -> ConfidenceInterval:
    return ConfidenceInterval(mean, mean, mean, 0.95, observations)


def _fake_audit(template: object, values: tuple[float, float, float, float]) -> object:
    from mosaicfeed.cohorts import CohortAudit

    assert isinstance(template, CohortAudit)
    a, b, ca, cb = values
    counts = {"A": 2, "B": 1}
    summaries = {
        label: CohortSummary(
            count,
            count,
            0,
            count,
            {
                metric: _interval(utility if metric == "ndcg" else 0.0, count)
                for metric in COHORT_METRICS
            },
            _interval(calibration, count),
        )
        for label, (count, utility, calibration) in {
            "A": (counts["A"], a, ca),
            "B": (counts["B"], b, cb),
        }.items()
    }
    gaps: dict[str, float | None] = {metric: 0.0 for metric in COHORT_METRICS}
    gaps["ndcg"] = abs(a - b)
    gaps["topic_calibration"] = abs(ca - cb)
    return replace(template, cohorts=summaries, max_minus_min=gaps)


def test_hand_computed_pareto_and_lexical_equal_vector_representative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    articles, events, cohorts = _source()
    template = audit_cohorts(
        articles,
        events,
        cohorts,
        as_of=NOW,
        config=FeedConfig(size=2),
        k=2,
        minimum_group_size=1,
        bootstrap_samples=4,
    )
    # Three users: alpha=(2*1+0)/3, beta=.7, gamma=(2*.9+.6)/3=.8.
    # beta dominates alpha; beta/gamma trade utility against worst-group utility.
    scenarios = {
        0.0: (1.0, 0.0, 0.2, 0.4),
        0.1: (0.7, 0.7, 0.15, 0.15),
        0.2: (0.9, 0.6, 0.1, 0.1),
        0.3: (0.9, 0.6, 0.1, 0.1),
    }
    monkeypatch.setattr(
        frontier_module,
        "audit_cohorts",
        lambda *args, **kwargs: _fake_audit(template, scenarios[kwargs["config"].minimum_score]),
    )
    report = compare_cohort_policies(
        articles, events, cohorts, _plan("alpha", "beta", "gamma", "zeta")
    )
    assert report.outcomes["alpha"].objectives["ndcg"] == pytest.approx(2 / 3)
    assert report.outcomes["alpha"].objectives["worst_cohort_ndcg"] == 0.0
    assert report.outcomes["alpha"].objectives["ndcg_gap"] == 1.0
    assert report.outcomes["alpha"].objectives["topic_calibration"] == pytest.approx(0.8 / 3)
    assert report.outcomes["alpha"].objectives["topic_calibration_gap"] == 0.2
    assert report.outcomes["alpha"].dominated_by == ("beta", "gamma", "zeta")
    assert report.outcomes["zeta"].dominated_by == ("gamma",)
    assert report.pareto_frontier == ("beta", "gamma")
    assert report.to_dict()["objective_directions"]["ndcg_gap"] == "minimize"
    with pytest.raises(ValueError, match="Pareto frontier"):
        replace(report, pareto_frontier=("alpha",))
    with pytest.raises(ValueError, match="objectives"):
        replace(
            report.outcomes["beta"], objectives={**report.outcomes["beta"].objectives, "ndcg": 0.0}
        )
    with pytest.raises(ValueError, match="digest"):
        replace(report, plan_sha256="0" * 64)
    with pytest.raises(ValueError, match="dominance"):
        replace(
            report,
            outcomes={**report.outcomes, "zeta": replace(report.outcomes["zeta"], dominated_by=())},
        )


def test_constraints_and_missing_calibration_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    articles, events, cohorts = _source()
    template = audit_cohorts(
        articles,
        events,
        cohorts,
        as_of=NOW,
        config=FeedConfig(size=2),
        k=2,
        minimum_group_size=1,
        bootstrap_samples=4,
    )
    complete = _fake_audit(template, (0.8, 0.6, 0.2, 0.1))
    assert isinstance(complete, type(template))
    missing = replace(
        complete,
        cohorts={
            **complete.cohorts,
            "B": replace(complete.cohorts["B"], calibration_observations=0, topic_calibration=None),
        },
        max_minus_min={**complete.max_minus_min, "topic_calibration": None},
    )
    monkeypatch.setattr(
        frontier_module,
        "audit_cohorts",
        lambda *args, **kwargs: complete if kwargs["config"].minimum_score == 0.0 else missing,
    )
    report = compare_cohort_policies(
        articles,
        events,
        cohorts,
        _plan(
            "complete",
            "missing",
            constraints=FrontierConstraints(minimum_ndcg=0.5, maximum_topic_calibration_gap=0.5),
        ),
    )
    assert report.pareto_frontier == ("complete",)
    assert report.outcomes["missing"].objectives["topic_calibration"] is None
    assert report.outcomes["missing"].missing_objectives == (
        "topic_calibration",
        "topic_calibration_gap",
    )
    assert report.outcomes["missing"].failed_constraints == ("maximum_topic_calibration_gap",)
    blocked = compare_cohort_policies(
        articles,
        events,
        cohorts,
        _plan("complete", "missing", constraints=FrontierConstraints(minimum_ndcg=0.9)),
    )
    assert blocked.pareto_frontier == ()
    with pytest.raises(ValueError, match="constraint flags"):
        replace(
            report,
            outcomes={
                **report.outcomes,
                "complete": replace(
                    report.outcomes["complete"], failed_constraints=("minimum_ndcg",)
                ),
            },
        )


def test_one_group_and_absent_calibration_have_no_frontier() -> None:
    articles = [_article("a")]
    events = [Event("u", "a", EventKind.CLICK, NOW)]
    report = compare_cohort_policies(articles, events, {"u": "only"}, _plan("first", "second"))
    assert report.pareto_frontier == ()
    for outcome in report.outcomes.values():
        assert outcome.objectives["ndcg_gap"] is None
        assert outcome.objectives["topic_calibration"] is None
        assert outcome.objectives["topic_calibration_gap"] is None
        assert outcome.to_dict()["frontier_eligible"] is False


def test_holdout_fingerprint_matches_evaluator_on_ties_future_and_unknown() -> None:
    articles = [_article("a"), _article("b"), _article("z")]
    prior = Event("u", "a", EventKind.CLICK, NOW - timedelta(days=1))
    baseline = [prior, Event("u", "b", EventKind.CLICK, NOW)]
    changed = [*baseline, Event("u", "z", EventKind.LIKE, NOW)]
    irrelevant = [
        *changed,
        Event("u", "z", EventKind.VIEW, NOW),
        Event("u", "a", EventKind.LIKE, NOW + timedelta(seconds=1)),
        Event("u", "zz", EventKind.LIKE, NOW),
    ]

    def evaluate(events: list[Event]) -> str:
        return (
            evaluate_leave_last_out_samples(
                articles, events, as_of=NOW, config=FeedConfig(size=2), k=2
            )
            .users[0]
            .holdout_article_id
        )

    assert evaluate(baseline) == "b"
    assert evaluate(changed) == "z"
    assert evaluate(irrelevant) == "z"
    assert frontier_module._holdout_digest(
        articles, baseline, NOW
    ) != frontier_module._holdout_digest(articles, changed, NOW)
    assert frontier_module._holdout_digest(
        articles, changed, NOW
    ) == frontier_module._holdout_digest(articles, irrelevant, NOW)
    expected = [["u", "z", NOW.isoformat()]]
    encoded = json.dumps(
        expected, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    assert (
        frontier_module._holdout_digest(articles, changed, NOW)
        == hashlib.sha256(encoded).hexdigest()
    )


def test_future_and_tied_nonpositive_events_cannot_change_policy_outcomes() -> None:
    articles, events, cohorts = _source()
    plan = _plan("first", "second")
    baseline = compare_cohort_policies(articles, events, cohorts, plan)
    changed = compare_cohort_policies(
        articles,
        [
            *events,
            Event("u1", "c", EventKind.VIEW, NOW),
            Event("u2", "c", EventKind.HIDE, NOW + timedelta(seconds=1)),
        ],
        cohorts,
        plan,
    )
    assert changed.dataset_sha256 != baseline.dataset_sha256
    assert changed.holdouts_sha256 == baseline.holdouts_sha256
    for name in ("first", "second"):
        assert changed.outcomes[name].objectives == baseline.outcomes[name].objectives
        for label in ("A", "B"):
            assert (
                changed.outcomes[name].audit.cohorts[label].to_dict()
                == baseline.outcomes[name].audit.cohorts[label].to_dict()
            )


@pytest.mark.parametrize(
    "invalid",
    [
        b'{"schema_version":1,"schema_version":1}',
        b'{"schema_version":1,"as_of":"2026-09-20T00:00:00Z","k":1,"policies":[],"extra":1}',
        b'{"schema_version":1,"as_of":"2026-09-20T00:00:00Z","k":1,"policies":[],"confidence":NaN}',
        b"\xff",
        b" " * (frontier_module.MAX_PLAN_BYTES + 1),
    ],
    ids=("duplicate-field", "unknown-field", "nonfinite", "invalid-utf8", "oversized"),
)
def test_plan_rejects_malformed_or_oversized_bytes(invalid: bytes) -> None:
    with pytest.raises(ValueError):
        PolicyFrontierPlan.from_bytes(invalid)


def test_duplicate_names_bad_bounds_and_work_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="unique"):
        _plan("same", "same")
    duplicate = _plan("first", "second").to_dict()
    policies = duplicate["policies"]
    assert isinstance(policies, list)
    policies[1]["name"] = "first"
    with pytest.raises(ValueError, match="unique"):
        PolicyFrontierPlan.from_bytes(json.dumps(duplicate).encode("utf-8"))
    for value in (float("nan"), True, -1.0, float("inf")):
        with pytest.raises(ValueError):
            FrontierConstraints(maximum_ndcg_gap=value)
    articles, events, cohorts = _source()
    monkeypatch.setattr(frontier_module, "MAX_TOTAL_EVALUATION_WORK", 10)
    with pytest.raises(ValueError, match="total evaluation work"):
        compare_cohort_policies(articles, events, cohorts, _plan("a", "b"))
    monkeypatch.setattr(frontier_module, "MAX_TOTAL_EVALUATION_WORK", 8_000_000)
    monkeypatch.setattr(frontier_module, "MAX_TOTAL_BOOTSTRAP_WORK", 1)
    with pytest.raises(ValueError, match="total bootstrap work"):
        compare_cohort_policies(articles, events, cohorts, _plan("a", "b"))


def test_public_plan_and_policy_constructors_reject_bad_controls() -> None:
    base = _plan("first", "second")
    for name in ("", "UPPER", " space", "x" * 65):
        with pytest.raises(ValueError, match="policy name"):
            NamedPolicy(name, FeedConfig())
    with pytest.raises(ValueError, match="policy config"):
        NamedPolicy("valid", FeedConfig(size=101))
    with pytest.raises(ValueError, match="policy config"):
        NamedPolicy("valid", None)  # type: ignore[arg-type]
    for change in (
        {"as_of": NOW.replace(tzinfo=None)},
        {"k": True},
        {"minimum_group_size": 0},
        {"bootstrap_samples": 2_001},
        {"policies": list(base.policies)},
        {"policies": base.policies[:1]},
        {"policies": (*base.policies, None)},
        {"seed": True},
        {"seed": 2**64},
        {"confidence": float("nan")},
        {"confidence": 2**1_024},
        {"hard_constraints": {}},
    ):
        with pytest.raises(ValueError):
            replace(base, **change)
    for bound in (
        {"minimum_ndcg": 1.1},
        {"minimum_worst_cohort_ndcg": 1.1},
        {"maximum_ndcg_gap": 1.1},
        {"maximum_topic_calibration": 2**1_024},
    ):
        with pytest.raises(ValueError):
            FrontierConstraints(**bound)


def test_plan_parser_rejects_nonobject_bad_policy_and_bounds() -> None:
    valid = _plan("first", "second").to_dict()
    for payload in (
        [],
        {**valid, "schema_version": 2},
        {**valid, "policies": {}},
        {**valid, "policies": [valid["policies"][0], {"name": "second", "config": 1}]},
        {**valid, "hard_constraints": {"unknown": 0.2}},
        {**valid, "hard_constraints": []},
    ):
        with pytest.raises(ValueError):
            PolicyFrontierPlan.from_bytes(json.dumps(payload).encode())
    assert PolicyFrontierPlan.from_bytes(json.dumps(valid).encode()).to_dict() == valid


def test_public_report_constructors_verify_internal_consistency() -> None:
    articles, events, cohorts = _source()
    report = compare_cohort_policies(articles, events, cohorts, _plan("first", "second"))
    outcome = report.outcomes["first"]
    with pytest.raises(ValueError, match="missing-evidence"):
        replace(outcome, missing_objectives=("ndcg",))
    with pytest.raises(ValueError, match="constraint flags"):
        replace(outcome, failed_constraints=("unknown",))
    with pytest.raises(ValueError, match="dominance"):
        replace(outcome, dominated_by=("second", "second"))
    with pytest.raises(ValueError, match="digests"):
        replace(report, holdouts_sha256="not-a-digest")
    with pytest.raises(ValueError, match="outcomes"):
        replace(report, outcomes={"first": outcome})
    with pytest.raises(ValueError, match="source digests"):
        replace(report, source_sha256={"articles": "bad"})
    with pytest.raises(ValueError, match="outcomes"):
        foreign_audit = replace(outcome.audit, dataset_sha256="f" * 64)
        foreign = replace(
            outcome,
            audit=foreign_audit,
            objectives=frontier_module._objectives(foreign_audit),
        )
        replace(report, outcomes={**report.outcomes, "first": foreign})
    with pytest.raises(ValueError, match="holdout counts"):
        summary = outcome.audit.cohorts["B"]
        updated_summary = replace(
            summary,
            active_users=2,
            users=2,
            metrics={key: _interval(value.mean, 2) for key, value in summary.metrics.items()},
            calibration_observations=2 if summary.topic_calibration is not None else 0,
            topic_calibration=(
                _interval(summary.topic_calibration.mean, 2)
                if summary.topic_calibration is not None
                else None
            ),
        )
        changed_audit = replace(
            outcome.audit,
            cohorts={**outcome.audit.cohorts, "B": updated_summary},
        )
        changed_outcome = replace(
            outcome,
            audit=changed_audit,
            objectives=frontier_module._objectives(changed_audit),
        )
        replace(report, outcomes={**report.outcomes, "first": changed_outcome})


def test_direct_api_rejects_bad_sources_and_mismatched_audits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    articles, events, cohorts = _source()
    plan = _plan("first", "second")
    with pytest.raises(ValueError, match="plan"):
        compare_cohort_policies(articles, events, cohorts, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cohorts"):
        compare_cohort_policies(articles, events, None, plan)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Article"):
        compare_cohort_policies([None], events, cohorts, plan)  # type: ignore[list-item]
    monkeypatch.setattr(frontier_module, "MAX_AUDIT_ROWS", 1)
    with pytest.raises(ValueError, match="row limit"):
        compare_cohort_policies(articles, events, cohorts, plan)
    monkeypatch.setattr(frontier_module, "MAX_AUDIT_ROWS", 100_000)
    monkeypatch.setattr(frontier_module, "MAX_USERS", 2)
    with pytest.raises(ValueError, match="declared user limit"):
        compare_cohort_policies(articles, events, cohorts, plan)
    monkeypatch.setattr(frontier_module, "MAX_USERS", 10_000)
    template = audit_cohorts(
        articles,
        events,
        cohorts,
        as_of=NOW,
        config=FeedConfig(size=2),
        k=2,
        minimum_group_size=1,
        bootstrap_samples=4,
    )
    calls = iter((template, replace(template, dataset_sha256="f" * 64)))
    monkeypatch.setattr(frontier_module, "audit_cohorts", lambda *args, **kwargs: next(calls))
    with pytest.raises(ValueError, match="same cohort holdouts"):
        compare_cohort_policies(articles, events, cohorts, plan)
    with pytest.raises(ValueError, match="complete evidence"):
        frontier_module._dominates(
            {key: None for key in frontier_module._OBJECTIVES},
            {key: None for key in frontier_module._OBJECTIVES},
        )


def test_cli_exact_raw_source_hashes_and_atomic_no_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from mosaicfeed.cli import _article_record, _event_record

    articles, events, cohorts = _source()
    source = {
        "articles": (json.dumps([_article_record(item) for item in articles]) + "\n").encode(),
        "events": json.dumps([_event_record(item) for item in events]).encode(),
        "cohorts": (json.dumps(cohorts, sort_keys=True) + "  \n").encode(),
        "plan": json.dumps(_plan("first", "second").to_dict()).encode(),
    }
    for name, data in source.items():
        (tmp_path / f"{name}.json").write_bytes(data)
    parent = tmp_path / "nested"
    parent.mkdir()
    output = parent / "frontier.json"
    args = [
        "compare-cohort-policies",
        "--articles",
        str(tmp_path / "articles.json"),
        "--events",
        str(tmp_path / "events.json"),
        "--cohorts",
        str(tmp_path / "cohorts.json"),
        "--plan",
        str(tmp_path / "plan.json"),
        "--output",
        str(output),
    ]
    assert main(args) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["source_sha256"] == {
        name: hashlib.sha256(data).hexdigest() for name, data in source.items()
    }
    assert result["plan_sha256"] != result["source_sha256"]["plan"]
    assert "u1" not in output.read_text(encoding="utf-8")
    original = output.read_bytes()
    assert main(args) == 2
    assert "File exists" in capsys.readouterr().err or output.read_bytes() == original
    assert output.read_bytes() == original
    assert not list(parent.glob(".frontier.json.*.tmp"))
    assert main([*args[:-1], str(tmp_path / "articles.json")]) == 2
    assert (tmp_path / "articles.json").read_bytes() == source["articles"]


def test_cli_malformed_source_fails_without_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    article_path = tmp_path / "articles.json"
    event_path = tmp_path / "events.json"
    cohort_path = tmp_path / "cohorts.json"
    plan_path = tmp_path / "plan.json"
    article_path.write_bytes(b"[{]")
    event_path.write_bytes(b"[]")
    cohort_path.write_bytes(b"{}")
    plan_path.write_bytes(json.dumps(_plan("a", "b").to_dict()).encode())
    output = tmp_path / "out.json"
    assert (
        main(
            [
                "compare-cohort-policies",
                "--articles",
                str(article_path),
                "--events",
                str(event_path),
                "--cohorts",
                str(cohort_path),
                "--plan",
                str(plan_path),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert "error:" in capsys.readouterr().err
    assert not output.exists()


def test_public_api_source_digest_is_caller_declared_not_recomputed() -> None:
    articles, events, cohorts = _source()
    declared = {key: "0" * 64 for key in ("articles", "events", "cohorts", "plan")}
    report = compare_cohort_policies(
        articles, events, cohorts, _plan("first", "second"), source_sha256=declared
    )
    assert report.source_sha256 == declared
    assert report.dataset_sha256 != declared["articles"]
    with pytest.raises(ValueError, match="source_sha256"):
        compare_cohort_policies(
            articles, events, cohorts, _plan("first", "second"), source_sha256={"articles": "bad"}
        )


def test_publish_link_failure_cleans_staging_and_never_replaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    articles, events, cohorts = _source()
    report = compare_cohort_policies(articles, events, cohorts, _plan("first", "second"))
    destination = tmp_path / "result.json"

    def fail_link(source: object, target: object) -> None:
        raise OSError("injected link failure")

    monkeypatch.setattr(frontier_module.os, "link", fail_link)
    with pytest.raises(OSError, match="injected"):
        write_frontier_report(destination, report)
    assert not destination.exists()
    assert not list(tmp_path.glob(".result.json.*.tmp"))


def test_report_size_limit_fails_before_creating_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    articles, events, cohorts = _source()
    report = compare_cohort_policies(articles, events, cohorts, _plan("first", "second"))
    monkeypatch.setattr(frontier_module, "MAX_REPORT_BYTES", 1)
    with pytest.raises(ValueError, match="output size limit"):
        write_frontier_report(tmp_path / "report.json", report)
    assert not list(tmp_path.iterdir())


def test_existing_symlink_is_never_replaced(tmp_path: Path) -> None:
    articles, events, cohorts = _source()
    report = compare_cohort_policies(articles, events, cohorts, _plan("first", "second"))
    target = tmp_path / "private.txt"
    target.write_text("original", encoding="utf-8")
    link = tmp_path / "report.json"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    with pytest.raises(FileExistsError):
        write_frontier_report(link, report)
    assert link.is_symlink()
    assert target.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".report.json.*.tmp"))
