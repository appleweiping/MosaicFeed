from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

import mosaicfeed.experiments as experiments
from mosaicfeed.cli import main
from mosaicfeed.experiments import (
    AblationPlan,
    read_experiment_source,
    run_ablation_experiment,
    verify_experiment_record,
    write_experiment_record,
)


def _sources() -> tuple[bytes, bytes, bytes]:
    articles = [
        {
            "id": "seen",
            "title": "Seen energy",
            "topics": ["energy"],
            "source": "one",
            "published_at": "2026-01-01T00:00:00Z",
            "quality": 0.5,
        },
        {
            "id": "holdout",
            "title": "Energy grid",
            "topics": ["energy"],
            "source": "two",
            "published_at": "2026-01-01T00:00:00Z",
            "quality": 0.2,
        },
        {
            "id": "other",
            "title": "Robotics",
            "topics": ["robotics"],
            "source": "three",
            "published_at": "2026-01-01T00:00:00Z",
            "quality": 1.0,
        },
    ]
    events = [
        {
            "user_id": "reader",
            "article_id": "seen",
            "kind": "click",
            "occurred_at": "2026-01-02T00:00:00Z",
        },
        {
            "user_id": "reader",
            "article_id": "holdout",
            "kind": "click",
            "occurred_at": "2026-01-03T00:00:00Z",
        },
    ]
    plan = {
        "schema_version": 1,
        "as_of": "2026-01-04T00:00:00Z",
        "config": {
            "size": 1,
            "interest_weight": 0.9,
            "quality_weight": 0.1,
            "freshness_weight": 0.0,
            "novelty_weight": 0.0,
            "popularity_weight": 0.0,
            "exploration_weight": 0.0,
        },
        "k": 1,
        "ablations": ["without_interest", "without_slate_diversity"],
        "bootstrap_samples": 9,
        "seed": 4,
    }
    return tuple(json.dumps(value).encode() for value in (articles, events, plan))  # type: ignore[return-value]


def _replace_plan(source: bytes, **changes: object) -> bytes:
    value = json.loads(source)
    value.update(changes)
    return json.dumps(value).encode()


def test_single_user_ablation_has_hand_checked_rank_and_paired_oracle() -> None:
    articles, events, plan = _sources()
    first = run_ablation_experiment(articles, events, plan)
    second = run_ablation_experiment(articles, events, plan)
    assert first.to_dict() == second.to_dict()
    payload = first.to_dict()
    assert payload["runner_protocol"] == "mosaicfeed-ablation-v1"
    outcomes = payload["outcomes"]
    assert isinstance(outcomes, dict)
    baseline = outcomes["baseline"]["evaluation"]
    without_interest = outcomes["without_interest"]["evaluation"]
    # One pre-holdout energy click makes the energy holdout rank first. Without
    # interest, the higher-quality robotics item wins. At k=1, NDCG is 1 vs 0.
    assert baseline["users_evaluated"] == 1
    assert baseline["ndcg"] == 1.0
    assert without_interest["ndcg"] == 0.0
    assert outcomes["without_slate_diversity"]["evaluation"] == baseline
    delta = payload["paired_deltas_from_baseline"]["without_interest"]["ndcg"]
    assert delta == {
        "mean": 1.0,
        "lower": 1.0,
        "upper": 1.0,
        "confidence": 0.95,
        "observations": 1,
    }
    zero_delta = payload["paired_deltas_from_baseline"]["without_slate_diversity"]["ndcg"]
    assert zero_delta["mean"] == zero_delta["lower"] == zero_delta["upper"] == 0.0
    assert payload["provenance"]["articles_sha256"] == hashlib.sha256(articles).hexdigest()
    assert payload["provenance"]["events_sha256"] == hashlib.sha256(events).hexdigest()
    assert payload["provenance"]["plan_sha256"] == hashlib.sha256(plan).hexdigest()
    assert set(outcomes) == {"baseline", "without_interest", "without_slate_diversity"}
    assert set(payload["paired_deltas_from_baseline"]) == {
        "without_interest",
        "without_slate_diversity",
    }
    with pytest.raises(TypeError):
        first.outcomes["baseline"]["evaluation"]["ndcg"] = 0.0  # type: ignore[index]


def test_exact_source_bytes_change_registry_identity_but_not_semantics() -> None:
    articles, events, plan = _sources()
    other = run_ablation_experiment(articles + b"\n", events, plan)
    original = run_ablation_experiment(articles, events, plan)
    assert other.experiment_id != original.experiment_id
    assert other.dataset_sha256 == original.dataset_sha256
    assert other.outcomes == original.outcomes


def test_runner_protocol_revision_separates_identical_input_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _sources()
    original = run_ablation_experiment(*sources)
    monkeypatch.setattr(experiments, "RUNNER_PROTOCOL", "mosaicfeed-ablation-v2")
    revised = run_ablation_experiment(*sources)
    assert revised.experiment_id != original.experiment_id
    assert revised.to_dict()["runner_protocol"] == "mosaicfeed-ablation-v2"


def test_all_variants_use_identical_multireader_bootstrap_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    articles, events_source, plan = _sources()
    events = json.loads(events_source)
    events.extend(
        [
            {
                "user_id": "another",
                "article_id": "seen",
                "kind": "click",
                "occurred_at": "2026-01-02T00:00:00Z",
            },
            {
                "user_id": "another",
                "article_id": "other",
                "kind": "click",
                "occurred_at": "2026-01-03T00:00:00Z",
            },
        ]
    )
    observed: list[tuple[int, ...]] = []
    original = experiments._resampled_metrics

    def recording(sample: object, indices: tuple[int, ...]) -> dict[str, float]:
        observed.append(indices)
        return original(sample, indices)  # type: ignore[arg-type]

    monkeypatch.setattr(experiments, "_resampled_metrics", recording)
    run_ablation_experiment(articles, json.dumps(events).encode(), plan)
    assert len(observed) == 9 * 3
    assert all(len(indices) == 2 for indices in observed)
    assert all(
        observed[index] == observed[index + 1] == observed[index + 2] for index in range(0, 27, 3)
    )
    assert len(set(observed)) > 1


def test_inconsistent_variant_holdouts_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = experiments.evaluate_leave_last_out_samples
    count = 0

    def inconsistent(*args: object, **kwargs: object) -> object:
        nonlocal count
        count += 1
        result = original(*args, **kwargs)  # type: ignore[arg-type]
        if count == 2:
            return replace(
                result,
                users=(replace(result.users[0], holdout_article_id="forged"),),
            )
        return result

    monkeypatch.setattr(experiments, "evaluate_leave_last_out_samples", inconsistent)
    with pytest.raises(RuntimeError, match="identical temporal holdouts"):
        run_ablation_experiment(*_sources())


def test_registry_record_is_no_replace_and_complete(tmp_path: Path) -> None:
    run = run_ablation_experiment(*_sources())
    registry = tmp_path / "registry"
    record = write_experiment_record(registry, run)
    body = record.read_bytes()
    assert record.name == f"{run.experiment_id}.json"
    assert json.loads(body) == run.to_dict()
    assert body.endswith(b"\n")
    assert list(registry.iterdir()) == [record]
    assert verify_experiment_record(record, *_sources())
    articles, events, plan = _sources()
    assert not verify_experiment_record(record, articles + b"\n", events, plan)
    if os.name == "posix":
        assert record.stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        write_experiment_record(registry, run)
    assert record.read_bytes() == body
    assert list(registry.iterdir()) == [record]


def test_registry_replay_detects_mutation_and_rejects_forged_identity(tmp_path: Path) -> None:
    sources = _sources()
    run = run_ablation_experiment(*sources)
    record = write_experiment_record(tmp_path / "registry", run)
    with pytest.raises(ValueError, match="ID does not match"):
        replace(run, experiment_id="0" * 64)
    for protocol in ("mosaicfeed-ablation-v0", "mosaicfeed-ablation-v\u0661", "unknown"):
        with pytest.raises(ValueError, match="runner_protocol"):
            replace(run, runner_protocol=protocol)
    record.write_bytes(record.read_bytes().replace(b'"schema_version":1', b'"schema_version":2'))
    assert not verify_experiment_record(record, *sources)
    wrong_registry = tmp_path / "not-a-directory"
    wrong_registry.write_text("occupied")
    with pytest.raises(ValueError, match="must be a directory"):
        write_experiment_record(wrong_registry, run)


def test_registry_link_failure_leaves_no_partial_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = run_ablation_experiment(*_sources())
    registry = tmp_path / "registry"

    def fail_link(source: object, destination: object) -> None:
        raise OSError("injected no-replace failure")

    monkeypatch.setattr(experiments.os, "link", fail_link)
    with pytest.raises(OSError, match="injected"):
        write_experiment_record(registry, run)
    assert list(registry.iterdir()) == []


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"as_of": "2026-01-04T00:00:00"},
        {"k": 0},
        {"k": True},
        {"k": 101},
        {"ablations": []},
        {"ablations": ["without_interest", "without_interest"]},
        {"ablations": ["not_an_ablation"]},
        {"ablations": "without_interest"},
        {"bootstrap_samples": 0},
        {"bootstrap_samples": 2001},
        {"confidence": 1.0},
        {"seed": True},
        {"seed": 2**63},
        {"config": {"size": 101}},
    ],
)
def test_invalid_plan_controls_fail_closed(change: dict[str, object]) -> None:
    articles, events, plan = _sources()
    with pytest.raises(ValueError):
        run_ablation_experiment(articles, events, _replace_plan(plan, **change))


def test_plan_requires_strict_schema_and_valid_json() -> None:
    _, _, plan = _sources()
    for source in (
        b"[]",
        b"\xff",
        b'{"schema_version":1,"schema_version":1}',
        b'{"schema_version":NaN}',
        plan.replace(b'"k": 1, ', b""),
        _replace_plan(plan, unknown=1),
    ):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            AblationPlan.from_bytes(source)


def test_empty_and_invalid_ablation_configuration() -> None:
    articles, events, plan = _sources()
    base = json.loads(plan)
    base["config"] = {
        "interest_weight": 1.0,
        "freshness_weight": 0.0,
        "quality_weight": 0.0,
        "novelty_weight": 0.0,
        "popularity_weight": 0.0,
        "exploration_weight": 0.0,
    }
    base["ablations"] = ["without_interest"]
    with pytest.raises(ValueError, match="invalid configuration"):
        run_ablation_experiment(articles, events, json.dumps(base).encode())
    empty = run_ablation_experiment(b"[]", b"[]", plan)
    assert empty.to_dict()["outcomes"]["baseline"]["evaluation"]["users_evaluated"] == 0


def test_without_slate_diversity_disables_both_reranker_modes() -> None:
    _, _, plan_source = _sources()
    base = AblationPlan.from_bytes(plan_source).config
    mmr = experiments._variant_config("without_slate_diversity", base, 3, 1)
    assert mmr.mmr_lambda == 1.0
    assert mmr.calibration_weight == 0.0
    assert mmr.max_per_source >= 3
    calibrated = experiments._variant_config(
        "without_slate_diversity",
        experiments.FeedConfig.from_mapping({**base.to_dict(), "rerank_strategy": "calibrated"}),
        3,
        1,
    )
    assert calibrated.rerank_strategy == "calibrated"
    assert calibrated.calibration_weight == 0.0


def test_source_and_work_limits_precede_registry_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    articles, events, plan = _sources()
    path = tmp_path / "source.json"
    path.write_bytes(b"abcd")
    with pytest.raises(ValueError, match="immutable byte snapshot"):
        read_experiment_source(path, maximum=3)
    with pytest.raises(ValueError, match="maximum"):
        read_experiment_source(path, maximum=0)
    with pytest.raises(ValueError, match="immutable byte snapshot"):
        run_ablation_experiment(bytearray(articles), events, plan)  # type: ignore[arg-type]
    monkeypatch.setattr(experiments, "MAX_EXPERIMENT_ROWS", 2)
    with pytest.raises(ValueError, match="row limit"):
        run_ablation_experiment(articles, events, plan)
    monkeypatch.setattr(experiments, "MAX_EXPERIMENT_ROWS", 50_000)
    monkeypatch.setattr(experiments, "MAX_EVALUATION_WORK", 1)
    with pytest.raises(ValueError, match="evaluation work"):
        run_ablation_experiment(articles, events, plan)
    monkeypatch.setattr(experiments, "MAX_EVALUATION_WORK", 5_000_000)
    monkeypatch.setattr(experiments, "MAX_BOOTSTRAP_WORK", 1)
    with pytest.raises(ValueError, match="bootstrap work"):
        run_ablation_experiment(articles, events, plan)
    assert not (tmp_path / "registry").exists()


def test_bootstrap_bound_counts_catalog_scan_slate_and_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    articles_source, events, plan = _sources()
    articles = json.loads(articles_source)
    for index in range(27):
        articles.append(
            {
                "id": f"extra-{index}",
                "title": f"Extra {index}",
                "topics": ["robotics"],
                "source": f"source-{index}",
                "published_at": "2026-01-01T00:00:00Z",
            }
        )
    catalog = json.dumps(articles).encode()
    # The old metrics-only estimate was 9 draws * 3 variants * 1 user *
    # 7 metrics = 189 units, even though every draw scans a 30-ID catalog.
    old_metrics_only = 9 * 3 * 1 * 7
    monkeypatch.setattr(experiments, "MAX_BOOTSTRAP_WORK", old_metrics_only + 1)
    with pytest.raises(ValueError, match="bootstrap work"):
        run_ablation_experiment(catalog, events, plan)
    users = 1
    slate_width = 1
    exposed_upper = min(30, users * slate_width)
    per_variant_draw = (
        users * (1 + 5 + slate_width + 30) + exposed_upper * (exposed_upper.bit_length() + 3) + 7
    )
    exact_upper = 9 * 3 * per_variant_draw
    monkeypatch.setattr(experiments, "MAX_BOOTSTRAP_WORK", exact_upper - 1)
    with pytest.raises(ValueError, match="bootstrap work"):
        run_ablation_experiment(catalog, events, plan)
    monkeypatch.setattr(experiments, "MAX_BOOTSTRAP_WORK", exact_upper)
    assert run_ablation_experiment(catalog, events, plan).outcomes["baseline"]


def test_evaluation_bound_counts_slates_and_topic_comparisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    articles_source, events, plan = _sources()
    articles = json.loads(articles_source)
    for index in range(27):
        articles.append(
            {
                "id": f"extra-{index}",
                "title": f"Extra {index}",
                "topics": ["robotics", "news"],
                "source": f"source-{index}",
                "published_at": "2026-01-01T00:00:00Z",
            }
        )
    catalog = json.dumps(articles).encode()
    old_users_catalog_events = 3 * 1 * (30 + 2)
    monkeypatch.setattr(experiments, "MAX_EVALUATION_WORK", old_users_catalog_events + 1)
    with pytest.raises(ValueError, match="evaluation work"):
        run_ablation_experiment(catalog, events, plan)
    topic_count = 3 + 27 * 2
    per_variant = 2 + 30 + 2 + 1 * (2 * (1 + topic_count) + 30 + 30 * topic_count * 1 * 3)
    exact_upper = 3 * per_variant
    monkeypatch.setattr(experiments, "MAX_EVALUATION_WORK", exact_upper - 1)
    with pytest.raises(ValueError, match="evaluation work"):
        run_ablation_experiment(catalog, events, plan)
    monkeypatch.setattr(experiments, "MAX_EVALUATION_WORK", exact_upper)
    assert run_ablation_experiment(catalog, events, plan).outcomes["baseline"]


def test_calibrated_reader_topic_scan_is_admitted_only_with_full_work_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, plan_source = _sources()
    plan = json.loads(plan_source)
    plan["config"]["rerank_strategy"] = "calibrated"
    articles = [
        {
            "id": f"a-{index}",
            "title": f"Article {index}",
            "topics": [f"topic-{index}"],
            "source": f"source-{index}",
            "published_at": "2026-01-01T00:00:00Z",
        }
        for index in range(40)
    ]
    events = [
        {
            "user_id": "reader",
            "article_id": f"a-{index}",
            "kind": "click",
            "occurred_at": f"2026-01-02T00:{index:02d}:00Z",
        }
        for index in range(20)
    ]
    events.append(
        {
            "user_id": "reader",
            "article_id": "a-20",
            "kind": "click",
            "occurred_at": "2026-01-03T00:00:00Z",
        }
    )
    sources = (
        json.dumps(articles).encode(),
        json.dumps(events).encode(),
        json.dumps(plan).encode(),
    )
    old_catalog_slate_estimate = 3 * (21 + 40 + 21 + (40 + 40) * 4)
    monkeypatch.setattr(experiments, "MAX_EVALUATION_WORK", old_catalog_slate_estimate + 1)
    with pytest.raises(ValueError, match="evaluation work"):
        run_ablation_experiment(*sources)
    topic_units = 40
    event_profile_work = 21 * (1 + topic_units)
    candidate_topic_work = 40 * topic_units * 1 * 3
    exact_upper = 3 * (21 + 40 + 21 + event_profile_work + 40 + candidate_topic_work)
    monkeypatch.setattr(experiments, "MAX_EVALUATION_WORK", exact_upper - 1)
    with pytest.raises(ValueError, match="evaluation work"):
        run_ablation_experiment(*sources)
    monkeypatch.setattr(experiments, "MAX_EVALUATION_WORK", exact_upper)
    assert run_ablation_experiment(*sources).outcomes["baseline"]


def test_logged_policy_fixed_bootstrap_is_in_preflight_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    articles, events_source, plan = _sources()
    events = json.loads(events_source)
    for event in events:
        event["propensity"] = 0.5
    for user_id in ("second", "third"):
        events.append(
            {
                "user_id": user_id,
                "article_id": "seen",
                "kind": "click",
                "occurred_at": "2026-01-02T00:00:00Z",
                "propensity": 0.5,
            }
        )
    sources = (articles, json.dumps(events).encode(), plan)
    topic_units = 3
    slate_steps = 1
    event_profile_work = 4 * (1 + topic_units)
    candidate_topic_work = 3 * topic_units * slate_steps * (slate_steps + 2)
    without_logged = 3 * (4 + 3 + 3 * (event_profile_work + 3 + candidate_topic_work))
    monkeypatch.setattr(experiments, "MAX_EVALUATION_WORK", without_logged + 1)
    with pytest.raises(ValueError, match="evaluation work"):
        run_ablation_experiment(*sources)
    logged_report_work = 4 + 1_000 * (3 + (1_000).bit_length())
    exact_upper = 3 * (
        4 + 3 + logged_report_work + 3 * (event_profile_work + 3 + candidate_topic_work)
    )
    monkeypatch.setattr(experiments, "MAX_EVALUATION_WORK", exact_upper - 1)
    with pytest.raises(ValueError, match="evaluation work"):
        run_ablation_experiment(*sources)
    monkeypatch.setattr(experiments, "MAX_EVALUATION_WORK", exact_upper)
    assert run_ablation_experiment(*sources).outcomes["baseline"]


def test_record_output_bound_is_checked_before_directory_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = run_ablation_experiment(*_sources())
    monkeypatch.setattr(experiments, "MAX_RECORD_BYTES", 1)
    registry = tmp_path / "registry"
    with pytest.raises(ValueError, match="byte limit"):
        write_experiment_record(registry, run)
    assert not registry.exists()


def test_cli_registers_one_reproducible_record_and_rejects_duplicate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    articles, events, plan = _sources()
    paths = [tmp_path / name for name in ("articles.json", "events.json", "plan.json")]
    for path, source in zip(paths, (articles, events, plan), strict=True):
        path.write_bytes(source)
    registry = tmp_path / "registry"
    args = [
        "run-ablation",
        "--articles",
        str(paths[0]),
        "--events",
        str(paths[1]),
        "--plan",
        str(paths[2]),
        "--registry",
        str(registry),
    ]
    assert main(args) == 0
    assert "registered experiment" in capsys.readouterr().out
    record = next(registry.glob("*.json"))
    assert json.loads(record.read_text())["experiment_id"] == record.stem
    assert main(args) == 2
    assert "already registered" in capsys.readouterr().err
    assert len(list(registry.iterdir())) == 1


def test_naive_public_plan_constructor_rejected() -> None:
    _, _, plan_source = _sources()
    parsed = AblationPlan.from_bytes(plan_source)
    with pytest.raises(ValueError, match="timezone-aware"):
        AblationPlan(
            datetime(2026, 1, 1),
            parsed.config,
            parsed.k,
            parsed.ablations,
        )
    assert parsed.as_of.tzinfo == UTC
