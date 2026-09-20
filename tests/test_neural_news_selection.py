"""Independent selection, paired-delta, replay, and adversarial tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import mosaicfeed.neural_news_selection as selection
from mosaicfeed.cli import main
from mosaicfeed.mind import impression_ndcg, load_mind_impressions_bytes
from mosaicfeed.neural_news_selection import (
    MAX_PLAN_BYTES,
    MAX_SOURCE_BYTES,
    NeuralAblation,
    NeuralSelectionPlan,
    read_selected_neural_checkpoint,
    run_neural_selection,
    verify_neural_selection_record,
    write_neural_selection_record,
)

ROOT = Path(__file__).resolve().parents[1]


def _sources() -> tuple[bytes, bytes, bytes, bytes]:
    examples = ROOT / "examples"
    return (
        (examples / "training_experiment_articles.json").read_bytes(),
        (examples / "neural_news_train.json").read_bytes(),
        (examples / "neural_news_validation.json").read_bytes(),
        (examples / "neural_news_selection_plan.json").read_bytes(),
    )


def _plan(**changes: object) -> bytes:
    value = json.loads(_sources()[3])
    value.update(changes)
    return json.dumps(value, sort_keys=True).encode()


def _redigest(record: dict[str, object]) -> dict[str, object]:
    value = copy.deepcopy(record)
    value.pop("record_sha256")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    value["record_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def test_one_factor_selection_and_independent_paired_ndcg_oracle() -> None:
    sources = _sources()
    first = run_neural_selection(*sources)
    assert first == run_neural_selection(*sources)
    assert first["selected_id"] == "small-embedding"
    assert first["work_units_upper_bound"] == 5760
    assert [row["id"] for row in first["outcomes"]] == [
        "baseline",
        "small-embedding",
        "one-epoch",
        "fast-step",
    ]
    assert first["source_sha256"]["train"] == hashlib.sha256(sources[1]).hexdigest()
    validation = load_mind_impressions_bytes(sources[2])
    per_model: dict[str, tuple[float, ...]] = {}
    for row in first["outcomes"]:
        predictions = {
            (entry["impression_id"], entry["article_id"]): entry["score"]
            for entry in row["experiment"]["scores"]
        }
        per_model[row["id"]] = tuple(
            impression_ndcg(
                [item.clicked for item in impression.candidates],
                [
                    predictions[impression.impression_id, item.article_id]
                    for item in impression.candidates
                ],
                5,
            )
            for impression in validation
        )
    oracle = sum(
        selected - baseline
        for selected, baseline in zip(
            per_model["small-embedding"], per_model["baseline"], strict=True
        )
    ) / len(validation)
    assert first["paired_deltas"]["small-embedding"]["mean"] == pytest.approx(oracle)
    assert first["paired_deltas"]["baseline"]["mean"] == 0


def test_validation_labels_change_audit_not_fitted_checkpoints() -> None:
    articles, train, validation, plan = _sources()
    original = run_neural_selection(articles, train, validation, plan)
    rows = json.loads(validation)
    for row in rows:
        for candidate in row["candidates"]:
            candidate["clicked"] = not candidate["clicked"]
    flipped = json.dumps(rows).encode()
    changed = run_neural_selection(articles, train, flipped, plan)
    assert [row["experiment"]["model"] for row in original["outcomes"]] == [
        row["experiment"]["model"] for row in changed["outcomes"]
    ]
    assert [row["experiment"]["scores"] for row in original["outcomes"]] == [
        row["experiment"]["scores"] for row in changed["outcomes"]
    ]
    assert original["source_sha256"]["validation"] != changed["source_sha256"]["validation"]
    assert (
        original["outcomes"][0]["experiment"]["metrics"]
        != changed["outcomes"][0]["experiment"]["metrics"]
    )


def test_registry_replay_selected_model_and_no_overwrite(tmp_path: Path) -> None:
    sources = _sources()
    record = run_neural_selection(*sources)
    target = write_neural_selection_record(tmp_path / "registry", record)
    assert verify_neural_selection_record(target, *sources)
    selected = read_selected_neural_checkpoint(target)
    assert selected.to_state() == next(
        row["experiment"]["model"]
        for row in record["outcomes"]
        if row["id"] == record["selected_id"]
    )
    with pytest.raises(FileExistsError):
        write_neural_selection_record(tmp_path / "registry", record)
    assert not verify_neural_selection_record(target, sources[0] + b" ", *sources[1:])
    target.write_text(
        target.read_text(encoding="utf-8").replace("small-embedding", "tampered"), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        read_selected_neural_checkpoint(target)


def test_non_ascii_title_uses_released_neural_checkpoint_digest(tmp_path: Path) -> None:
    articles, train, validation, plan = _sources()
    rows = json.loads(articles)
    rows[0]["title"] = "Énergie grid"
    changed_articles = json.dumps(rows, ensure_ascii=False).encode("utf-8")
    record = run_neural_selection(changed_articles, train, validation, plan)
    target = write_neural_selection_record(tmp_path / "registry", record)
    assert verify_neural_selection_record(target, changed_articles, train, validation, plan)
    assert read_selected_neural_checkpoint(target).to_state()["format"] == "mosaicfeed.neural-news"


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"ablations": []}, "bounded array"),
        ({"selection_metric": "precision"}, "metric"),
        ({"bootstrap_samples": 1}, "bootstrap_samples"),
        ({"cutoff": "2026-01-04T00:00:00"}, "timezone"),
        (
            {"ablations": [{"id": "same", "field": "dimension", "value": 4}]},
            "change exactly",
        ),
        (
            {"ablations": [{"id": "seed", "field": "seed", "value": 5}]},
            "unsupported",
        ),
        (
            {"ablations": [{"id": "bool", "field": "dimension", "value": True}]},
            "change exactly",
        ),
        (
            {"ablations": [{"id": "same", "field": "epochs", "value": 1}] * 2},
            "distinct",
        ),
    ],
)
def test_plan_fail_closed(change: dict[str, object], reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        NeuralSelectionPlan.from_bytes(_plan(**change))


def test_record_tamper_even_with_recomputed_outer_digest(tmp_path: Path) -> None:
    record = run_neural_selection(*_sources())
    changed = copy.deepcopy(record)
    changed["outcomes"][0]["experiment"]["model"]["config"]["dimension"] = 3
    with pytest.raises(ValueError, match="checkpoint"):
        write_neural_selection_record(tmp_path / "bad", _redigest(changed))
    changed = copy.deepcopy(record)
    changed["selected_id"] = "one-epoch"
    with pytest.raises(ValueError, match="result"):
        write_neural_selection_record(tmp_path / "bad", _redigest(changed))
    changed = copy.deepcopy(record)
    changed["outcomes"][0]["experiment"]["work_units_upper_bound"] += 1
    with pytest.raises(ValueError, match="work bound"):
        write_neural_selection_record(tmp_path / "bad", _redigest(changed))
    changed = copy.deepcopy(record)
    changed["paired_deltas"]["baseline"]["mean"] = 1
    with pytest.raises(ValueError, match="baseline paired"):
        write_neural_selection_record(tmp_path / "bad", _redigest(changed))
    changed = copy.deepcopy(record)
    changed["paired_deltas"]["baseline"]["mean"] = 10**1000
    with pytest.raises(ValueError, match="finite"):
        write_neural_selection_record(tmp_path / "bad", _redigest(changed))
    assert not (tmp_path / "bad").exists()


def test_source_and_work_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    sources = _sources()
    with pytest.raises(ValueError, match="bounded immutable bytes"):
        run_neural_selection(b" " * (MAX_SOURCE_BYTES + 1), *sources[1:])
    with pytest.raises(ValueError, match="bounded immutable bytes"):
        run_neural_selection(*sources[:3], b" " * (MAX_PLAN_BYTES + 1))
    monkeypatch.setattr("mosaicfeed.neural_news_selection.MAX_AGGREGATE_WORK", 1)
    with pytest.raises(ValueError, match="aggregate work"):
        run_neural_selection(*sources)


def test_registry_refuses_file_parent_and_cli_path_alias(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = run_neural_selection(*_sources())
    occupied = tmp_path / "occupied"
    occupied.write_text("unchanged", encoding="utf-8")
    with pytest.raises(ValueError, match="real directory"):
        write_neural_selection_record(occupied, record)
    assert occupied.read_text(encoding="utf-8") == "unchanged"
    root = ROOT / "examples"
    assert (
        main(
            [
                "run-neural-news-selection",
                "--articles",
                str(root / "training_experiment_articles.json"),
                "--train",
                str(root / "neural_news_train.json"),
                "--validation",
                str(root / "neural_news_validation.json"),
                "--plan",
                str(root / "neural_news_selection_plan.json"),
                "--registry",
                str(root / "neural_news_selection_plan.json"),
            ]
        )
        == 2
    )
    assert "must refer to different paths" in capsys.readouterr().err


def test_cli_selection_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = ROOT / "examples"
    registry = tmp_path / "registry"
    assert (
        main(
            [
                "run-neural-news-selection",
                "--articles",
                str(root / "training_experiment_articles.json"),
                "--train",
                str(root / "neural_news_train.json"),
                "--validation",
                str(root / "neural_news_validation.json"),
                "--plan",
                str(root / "neural_news_selection_plan.json"),
                "--registry",
                str(registry),
            ]
        )
        == 0
    )
    assert "registered neural-news selection" in capsys.readouterr().out
    assert len(list(registry.glob("*.json"))) == 1


@pytest.mark.parametrize(
    "raw,reason",
    [
        (b"\xff", "strict JSON"),
        (b"{", "strict JSON"),
        (b"[]", "fields"),
        (b"{}", "fields"),
        (b'{"schema_version": 1, "schema_version": 1}', "strict JSON"),
    ],
)
def test_plan_rejects_noncanonical_structure(raw: bytes, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        NeuralSelectionPlan.from_bytes(raw)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("schema_version", True, "schema"),
        ("schema_version", 2, "schema"),
        ("cutoff", "bad", "cutoff"),
        ("selection_metric", True, "metric"),
        ("bootstrap_samples", True, "bootstrap_samples"),
        ("bootstrap_samples", 501, "bootstrap_samples"),
        ("bootstrap_seed", True, "bootstrap_seed"),
        ("bootstrap_seed", -1, "bootstrap_seed"),
        ("bootstrap_seed", 2**32, "bootstrap_seed"),
        ("ablations", {}, "bounded array"),
        ("ablations", [{}], "fields"),
        ("ablations", ["dimension"], "fields"),
        ("ablations", [{"id": "baseline", "field": "dimension", "value": 3}], "id"),
        ("ablations", [{"id": "UPPER", "field": "dimension", "value": 3}], "id"),
        ("ablations", [{"id": "new", "field": "wrong", "value": 3}], "field"),
        ("ablations", [{"id": "new", "field": "epochs", "value": 3.0}], "change exactly"),
        (
            "ablations",
            [
                {"id": "first", "field": "epochs", "value": 2},
                {"id": "second", "field": "epochs", "value": 3},
            ],
            "distinct",
        ),
    ],
)
def test_plan_rejects_bad_fields(field: str, value: object, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        NeuralSelectionPlan.from_bytes(_plan(**{field: value}))


def test_plan_constructors_and_roundtrip() -> None:
    plan = NeuralSelectionPlan.from_bytes(_sources()[3])
    assert NeuralSelectionPlan.from_bytes(json.dumps(plan.to_dict()).encode()) == plan
    assert NeuralAblation("other", "dimension", 3).config(plan.baseline).dimension == 3
    with pytest.raises(ValueError, match="baseline"):
        NeuralSelectionPlan(plan.cutoff, object(), plan.ablations)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="distinct"):
        NeuralSelectionPlan(plan.cutoff, plan.baseline, (plan.ablations[0],) * 2)
    with pytest.raises(ValueError, match="timezone"):
        NeuralSelectionPlan(plan.cutoff.replace(tzinfo=None), plan.baseline, plan.ablations)
    with pytest.raises(ValueError, match="bounded immutable bytes"):
        NeuralSelectionPlan.from_bytes("not bytes")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, None, "0.4", float("nan"), -0.1, 1.1, 10**1000])
def test_unit_metric_rejects_non_unit_values(value: object) -> None:
    with pytest.raises(ValueError, match="unit-interval"):
        selection._unit(value)


@pytest.mark.parametrize("value", [True, None, "0.4", float("inf"), -1.1, 1.1, 10**1000])
def test_paired_delta_rejects_non_finite_or_unbounded_values(value: object) -> None:
    with pytest.raises(ValueError, match=r"finite|bounded"):
        selection._signed_unit(value)


@pytest.mark.parametrize(
    "report,name,reason",
    [
        (None, "auc", "report"),
        ({"ndcg": None}, "ndcg@5", "nDCG"),
        ({"auc": None}, "auc", "unit-interval"),
        ({"ndcg": {"5": 2}}, "ndcg@5", "unit-interval"),
    ],
)
def test_metric_reader_rejects_malformed_reports(report: object, name: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        selection._metric(report, name)


@pytest.mark.parametrize("metric", ["auc", "mrr", "ndcg@10"])
def test_other_metrics_replay_from_exact_sources(tmp_path: Path, metric: str) -> None:
    articles, train, validation, _ = _sources()
    plan = _plan(selection_metric=metric)
    record = run_neural_selection(articles, train, validation, plan)
    target = write_neural_selection_record(tmp_path / metric, record)
    assert verify_neural_selection_record(target, articles, train, validation, plan)
    for row in record["outcomes"]:
        assert 0 <= selection._metric(row["experiment"]["metrics"], metric) <= 1


def _set_nested(value: dict[str, object], path: str, replacement: object) -> None:
    node: object = value
    names = path.split(".")
    for name in names[:-1]:
        node = node[int(name)] if isinstance(node, list) else node[name]
    if isinstance(node, list):
        node[int(names[-1])] = replacement
    else:
        node[names[-1]] = replacement


@pytest.mark.parametrize(
    "path,replacement,reason",
    [
        ("protocol", "other", "protocol"),
        ("schema_version", True, "protocol"),
        ("scope", "official MIND", "protocol"),
        ("source_sha256.articles", "not-a-digest", "source digests"),
        ("source_sha256.plan", "0" * 64, "identity"),
        ("selection_metric", "auc", "differs from plan"),
        ("work_units_upper_bound", True, "work bound"),
        ("work_units_upper_bound", -1, "work bound"),
        ("outcomes", [], "outcome count"),
        ("outcomes.0.id", "not-in-plan", "ID"),
        ("outcomes.0.changed_field", "dimension", "field"),
        ("outcomes.0.experiment.format", "other", "experiment"),
        ("outcomes.0.experiment.cutoff", "2020-01-01T00:00:00+00:00", "experiment"),
        ("outcomes.0.experiment.source_sha256.train", "0" * 64, "sources mismatch"),
        ("outcomes.0.experiment.state_sha256", "0" * 64, "checkpoint digest"),
        ("outcomes.0.experiment.work_units_upper_bound", 0, "work bound"),
        ("outcomes.0.experiment.metrics", [], "report"),
        ("outcomes.0.experiment.metrics.ndcg", [], "nDCG"),
        ("outcomes.0.experiment.metrics.ndcg.5", 2, "unit-interval"),
        ("paired_deltas", {}, "result or paired"),
        ("paired_deltas.baseline", {}, "paired diagnostic"),
        ("paired_deltas.baseline.unit", "wrong", "paired diagnostic"),
        ("paired_deltas.baseline.lower_95", 0.1, "baseline paired"),
        ("paired_deltas.baseline.upper_95", "zero", "finite"),
        ("paired_deltas.baseline.mean", 1.1, "bounded"),
    ],
)
def test_registry_rejects_redigested_structural_tamper(
    tmp_path: Path, path: str, replacement: object, reason: str
) -> None:
    changed = copy.deepcopy(run_neural_selection(*_sources()))
    _set_nested(changed, path, replacement)
    with pytest.raises(ValueError, match=reason):
        write_neural_selection_record(tmp_path / "registry", _redigest(changed))
    assert not (tmp_path / "registry").exists()


def test_record_reader_rejects_invalid_bytes_name_and_size(tmp_path: Path) -> None:
    record = run_neural_selection(*_sources())
    target = write_neural_selection_record(tmp_path / "registry", record)
    other = tmp_path / "wrong.json"
    other.write_bytes(target.read_bytes())
    with pytest.raises(ValueError, match="filename"):
        read_selected_neural_checkpoint(other)
    other.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="strict JSON"):
        read_selected_neural_checkpoint(other)
    other.write_bytes(b"{")
    with pytest.raises(ValueError, match="strict JSON"):
        read_selected_neural_checkpoint(other)
    other.write_bytes(b"x" * (selection.MAX_RECORD_BYTES + 1))
    with pytest.raises(ValueError, match="byte limit"):
        read_selected_neural_checkpoint(other)
    assert not verify_neural_selection_record(other, *_sources())


def test_internal_reader_is_not_a_source_replay(tmp_path: Path) -> None:
    record = copy.deepcopy(run_neural_selection(*_sources()))
    # This metric is not used for checkpoint selection; a recomputed outer
    # digest cannot make the forged result an authentic source-backed run.
    record["outcomes"][0]["experiment"]["metrics"]["auc"] = 0.123456789
    target = write_neural_selection_record(tmp_path / "registry", _redigest(record))
    assert read_selected_neural_checkpoint(target).config.dimension == 2
    assert not verify_neural_selection_record(target, *_sources())


def test_selection_ties_prioritize_baseline_then_stable_ablation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = selection.run_neural_news_experiment

    def equal_scores(*args: object, **kwargs: object) -> dict[str, object]:
        result = original(*args, **kwargs)
        result["metrics"]["ndcg"]["5"] = 0.5
        return result

    monkeypatch.setattr(selection, "run_neural_news_experiment", equal_scores)
    assert run_neural_selection(*_sources())["selected_id"] == "baseline"

    def equal_variant_scores(*args: object, **kwargs: object) -> dict[str, object]:
        result = equal_scores(*args, **kwargs)
        if (
            kwargs["config"].dimension == 4
            and kwargs["config"].epochs == 2
            and kwargs["config"].learning_rate == 0.1
        ):
            result["metrics"]["ndcg"]["5"] = 0.0
        return result

    monkeypatch.setattr(selection, "run_neural_news_experiment", equal_variant_scores)
    # The three ablation outcomes tie; lexical ID order wins, independent of
    # the declared order in the example plan.
    assert run_neural_selection(*_sources())["selected_id"] == "fast-step"


def test_link_failure_cleans_stage_and_never_clobbers(tmp_path: Path) -> None:
    record = run_neural_selection(*_sources())
    root = tmp_path / "registry"
    with (
        patch.object(selection.os, "link", side_effect=OSError("simulated link failure")),
        pytest.raises(OSError, match="simulated link failure"),
    ):
        write_neural_selection_record(root, record)
    assert list(root.iterdir()) == []
    target = write_neural_selection_record(root, record)
    existing = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_neural_selection_record(root, record)
    assert target.read_bytes() == existing
    assert sorted(root.iterdir()) == [target]
