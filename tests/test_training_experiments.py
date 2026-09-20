from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

import pytest

from mosaicfeed.cli import main
from mosaicfeed.io import load_articles_bytes
from mosaicfeed.listwise import ListwiseImpressionRanker
from mosaicfeed.mind import evaluate_mind_impressions, load_mind_impressions_bytes
from mosaicfeed.pairwise import PairwiseImpressionRanker
from mosaicfeed.training_experiments import (
    MAX_PLAN_BYTES,
    MAX_SOURCE_BYTES,
    TrainingCandidate,
    TrainingExperimentPlan,
    read_selected_checkpoint,
    run_training_experiment,
    select_candidate,
    verify_training_experiment_record,
    write_training_experiment_record,
)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True).encode()


def _sources() -> tuple[bytes, bytes, bytes, bytes]:
    articles = [
        {
            "id": ident,
            "title": ident,
            "topics": [topic],
            "source": ident,
            "published_at": "2026-01-01T00:00:00Z",
            "quality": quality,
        }
        for ident, topic, quality in (
            ("a", "energy", 0.9),
            ("b", "sports", 0.1),
            ("c", "energy", 0.8),
            ("d", "sports", 0.2),
        )
    ]
    train = [
        {
            "impression_id": "t1",
            "user_id": "u",
            "occurred_at": "2026-01-02T00:00:00Z",
            "candidates": [
                {"article_id": "a", "clicked": True},
                {"article_id": "b", "clicked": False},
            ],
        },
        {
            "impression_id": "t2",
            "user_id": "v",
            "occurred_at": "2026-01-03T00:00:00Z",
            "candidates": [
                {"article_id": "c", "clicked": True},
                {"article_id": "d", "clicked": False},
            ],
        },
    ]
    validation = [
        {
            "impression_id": "v1",
            "user_id": "u",
            "occurred_at": "2026-01-05T00:00:00Z",
            "candidates": [
                {"article_id": "a", "clicked": True},
                {"article_id": "d", "clicked": False},
            ],
        },
        {
            "impression_id": "v2",
            "user_id": "v",
            "occurred_at": "2026-01-06T00:00:00Z",
            "candidates": [
                {"article_id": "b", "clicked": False},
                {"article_id": "c", "clicked": True},
            ],
        },
    ]
    plan = {
        "schema_version": 1,
        "cutoff": "2026-01-04T00:00:00Z",
        "feature_config": {},
        "selection_metric": "auc",
        "candidates": [
            {
                "id": "p",
                "objective": "pairwise",
                "epochs": 2,
                "learning_rate": 0.2,
                "l2": 0.0,
                "seed": 5,
            },
            {
                "id": "l",
                "objective": "listwise",
                "epochs": 2,
                "learning_rate": 0.2,
                "l2": 0.0,
                "seed": 5,
            },
        ],
    }
    return tuple(map(_json, (articles, train, validation, plan)))  # type: ignore[return-value]


def _change(sources: tuple[bytes, ...], index: int, change: object) -> tuple[bytes, ...]:
    changed = list(sources)
    changed[index] = _json(change)
    return tuple(changed)


def _resign(record: dict[str, object]) -> None:
    record["record_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in record.items() if key != "record_sha256"},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def test_training_replays_real_models_and_independent_metric_oracle(tmp_path: Path) -> None:
    sources = _sources()
    record = run_training_experiment(*sources)
    assert record == run_training_experiment(*sources)
    assert record["protocol"] == "mosaicfeed-training-experiment-v1"
    assert record["source_sha256"] == {
        name: hashlib.sha256(raw).hexdigest()
        for name, raw in zip(("articles", "train", "validation", "plan"), sources, strict=True)
    }
    articles = load_articles_bytes(sources[0])
    train = load_mind_impressions_bytes(sources[1])
    validation = load_mind_impressions_bytes(sources[2])
    outcomes = record["outcomes"]
    assert isinstance(outcomes, list)
    for outcome in outcomes:
        state = outcome["checkpoint"]
        if outcome["objective"] == "pairwise":
            model = PairwiseImpressionRanker.from_state(state)
        else:
            model = ListwiseImpressionRanker.from_state(state)
        scores = model.score_impressions(articles, validation, training_impressions=train)
        assert outcome["validation"] == evaluate_mind_impressions(validation, scores).to_dict()
        manual_auc = sum(
            float(
                scores[row.impression_id][next(c.article_id for c in row.candidates if c.clicked)]
                > scores[row.impression_id][
                    next(c.article_id for c in row.candidates if not c.clicked)
                ]
            )
            for row in validation
        ) / len(validation)
        assert outcome["validation"]["auc"] == pytest.approx(manual_auc)
        assert state["training"]["source_sha256"] == {
            "articles": hashlib.sha256(sources[0]).hexdigest(),
            "train": hashlib.sha256(sources[1]).hexdigest(),
        }
    chosen = min(outcomes, key=lambda row: (-row["validation"]["auc"], row["candidate_id"]))
    assert record["selected_candidate_id"] == chosen["candidate_id"]

    def hand_split_sha(source: bytes) -> str:
        rows = json.loads(source)
        for row in rows:
            row["occurred_at"] = row["occurred_at"].replace("Z", "+00:00")
        return hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    assert record["split_sha256"] == {
        "train": hand_split_sha(sources[1]),
        "validation": hand_split_sha(sources[2]),
    }
    path = write_training_experiment_record(tmp_path, record)
    assert verify_training_experiment_record(path, *sources)
    assert read_selected_checkpoint(path).to_state() == chosen["checkpoint"]
    with pytest.raises(FileExistsError):
        write_training_experiment_record(tmp_path, record)


def test_validation_labels_do_not_enter_fit_but_change_evaluation() -> None:
    articles, train, validation, plan = _sources()
    changed = json.loads(validation)
    for row in changed:
        for candidate in row["candidates"]:
            candidate["clicked"] = not candidate["clicked"]
    before = run_training_experiment(articles, train, validation, plan)
    after = run_training_experiment(articles, train, _json(changed), plan)
    for left, right in zip(before["outcomes"], after["outcomes"], strict=True):
        assert left["checkpoint"] == right["checkpoint"]
        assert left["checkpoint_sha256"] == right["checkpoint_sha256"]
        assert left["validation"]["impression_sha256"] != right["validation"]["impression_sha256"]
    assert before["source_sha256"]["validation"] != after["source_sha256"]["validation"]


def test_metric_directed_selection_and_exact_ties() -> None:
    def report(auc: float, mrr: float, ndcg: float) -> dict[str, object]:
        return {"auc": auc, "mrr": mrr, "ndcg": {"5": ndcg, "10": ndcg}}

    outcomes = [
        {"candidate_id": "z", "validation": report(0.8, 0.3, 0.4)},
        {"candidate_id": "a", "validation": report(0.8, 0.4, 0.5)},
        {"candidate_id": "b", "validation": report(0.7, 0.9, 0.9)},
    ]
    assert select_candidate(outcomes, "auc") == "a"
    assert select_candidate(outcomes, "mrr") == "b"
    assert select_candidate(outcomes, "ndcg@5") == "b"
    assert select_candidate(list(reversed(outcomes)), "auc") == "a"
    with pytest.raises(ValueError):
        select_candidate([{"candidate_id": "x", "validation": {}}], "auc")
    with pytest.raises(ValueError):
        select_candidate([{"candidate_id": "x", "validation": {"ndcg": {}}}], "ndcg@5")
    with pytest.raises(ValueError):
        select_candidate([{"candidate_id": "x", "validation": {"auc": 10**400}}], "auc")


@pytest.mark.parametrize("candidate_index", [0, 1])
def test_every_checkpoint_tamper_is_detected(tmp_path: Path, candidate_index: int) -> None:
    sources = _sources()
    record = run_training_experiment(*sources)
    path = write_training_experiment_record(tmp_path, record)
    changed = copy.deepcopy(record)
    changed["outcomes"][candidate_index]["checkpoint"]["weights"][0] += 0.01
    path.write_text(json.dumps(changed), encoding="utf-8")
    assert not verify_training_experiment_record(path, *sources)
    with pytest.raises(ValueError):
        read_selected_checkpoint(path)


def test_temporal_overlap_unmixed_and_limits_rejected_before_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _sources()
    original = json.loads(sources[2])
    for field, value in (("occurred_at", "2026-01-04T00:00:00Z"), ("impression_id", "t1")):
        changed = copy.deepcopy(original)
        changed[0][field] = value
        with pytest.raises(ValueError):
            run_training_experiment(*_change(sources, 2, changed))
    changed = copy.deepcopy(original)
    changed[0]["candidates"][1]["clicked"] = True
    with pytest.raises(ValueError, match="positive and negative"):
        run_training_experiment(*_change(sources, 2, changed))
    with pytest.raises(ValueError):
        run_training_experiment(sources[0] + b" " * MAX_SOURCE_BYTES, *sources[1:])
    with pytest.raises(ValueError):
        run_training_experiment(*sources[:3], sources[3] + b" " * MAX_PLAN_BYTES)
    import mosaicfeed.training_experiments as experiments

    monkeypatch.setattr(experiments, "MAX_WORK_UNITS", 1)
    with pytest.raises(ValueError, match="work limit"):
        run_training_experiment(*sources)


def test_plan_rejects_invalid_specs_and_untrusted_record(tmp_path: Path) -> None:
    sources = _sources()
    plan = json.loads(sources[3])
    plan["candidates"][1]["id"] = "p"
    with pytest.raises(ValueError, match="distinct"):
        TrainingExperimentPlan.from_bytes(_json(plan))
    plan["candidates"][1]["id"] = "l"
    plan["candidates"][1]["epochs"] = 101
    with pytest.raises(ValueError):
        TrainingExperimentPlan.from_bytes(_json(plan))
    record = run_training_experiment(*sources)
    path = write_training_experiment_record(tmp_path, record)
    changed = copy.deepcopy(record)
    changed["selected_candidate_id"] = "nonexistent"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError):
        read_selected_checkpoint(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(unknown=1),
        lambda p: p.update(schema_version=2),
        lambda p: p.update(feature_config=[]),
        lambda p: p.update(candidates={}),
        lambda p: p.update(selection_metric="precision@5"),
        lambda p: p.update(cutoff="2026-01-04T00:00:00"),
        lambda p: p["candidates"][0].update(id="Invalid ID"),
        lambda p: p["candidates"][0].update(objective="unknown"),
        lambda p: p["candidates"][0].update(seed=-1),
        lambda p: p["candidates"][0].update(extra=True),
    ],
)
def test_malformed_plan_rejected(mutate: object) -> None:
    plan = json.loads(_sources()[3])
    mutate(plan)
    with pytest.raises(ValueError):
        TrainingExperimentPlan.from_bytes(_json(plan))


def test_plan_utf8_and_direct_api_type_guards() -> None:
    with pytest.raises(ValueError, match="UTF-8"):
        TrainingExperimentPlan.from_bytes(b"\xff")
    with pytest.raises(ValueError):
        TrainingExperimentPlan.from_bytes(_json([]))
    with pytest.raises(ValueError):
        TrainingCandidate("bad name", "pairwise", 2, 0.2, 0.0, 5)
    with pytest.raises(ValueError):
        run_training_experiment("not bytes", *_sources()[1:])


@pytest.mark.parametrize(
    "change_index, mutate",
    [
        (1, lambda rows: rows[0].update(occurred_at="2026-01-05T00:00:00Z")),
        (1, lambda rows: rows[0].update(impression_id="t2")),
        (2, lambda rows: rows[0].update(candidates=rows[0]["candidates"] * 9)),
        (2, lambda rows: rows.clear()),
    ],
)
def test_row_boundaries_rejected(change_index: int, mutate: object) -> None:
    sources = _sources()
    rows = json.loads(sources[change_index])
    mutate(rows)
    with pytest.raises(ValueError):
        run_training_experiment(*_change(sources, change_index, rows))


def test_record_publication_and_reader_reject_bad_structures(tmp_path: Path) -> None:
    record = run_training_experiment(*_sources())
    with pytest.raises(ValueError):
        write_training_experiment_record(tmp_path, {**record, "protocol": "other"})
    with pytest.raises(ValueError, match="record digest"):
        write_training_experiment_record(tmp_path, {**record, "selection_metric": "mrr"})
    blocked = tmp_path / "blocked"
    blocked.write_text("file", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        write_training_experiment_record(blocked, record)
    path = write_training_experiment_record(tmp_path, record)
    for field, replacement in (
        ("outcomes", []),
        ("outcomes", [None, None]),
        ("selected_candidate_id", "unknown"),
    ):
        altered = copy.deepcopy(record)
        altered[field] = replacement
        _resign(altered)
        path.write_text(json.dumps(altered), encoding="utf-8")
        with pytest.raises(ValueError):
            read_selected_checkpoint(path)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda r: r["plan"].update(selection_metric="mrr"),
        lambda r: r["outcomes"][0].update(candidate_id="unknown"),
        lambda r: r["outcomes"][0].update(objective="pairwise"),
        lambda r: r["outcomes"][0]["checkpoint"]["parameters"].update(seed=6),
        lambda r: r["source_sha256"].update(validation="not-a-digest"),
    ],
)
def test_resigned_internal_inconsistency_is_rejected(tmp_path: Path, tamper: object) -> None:
    record = run_training_experiment(*_sources())
    path = write_training_experiment_record(tmp_path, record)
    altered = copy.deepcopy(record)
    tamper(altered)
    _resign(altered)
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        read_selected_checkpoint(path)


@pytest.mark.parametrize(
    "field, replacement",
    [
        ("split_sha256", {}),
        ("split_sha256", {"train": "x" * 64, "validation": "0" * 64}),
        ("work_units_upper_bound", True),
        ("work_units_upper_bound", 0),
        ("work_units_upper_bound", 20_000_001),
    ],
)
def test_resigned_split_and_work_shape_rejected(
    tmp_path: Path, field: str, replacement: object
) -> None:
    record = run_training_experiment(*_sources())
    path = write_training_experiment_record(tmp_path, record)
    altered = copy.deepcopy(record)
    altered[field] = replacement
    _resign(altered)
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError):
        read_selected_checkpoint(path)


def test_reader_shape_check_is_not_source_replay(tmp_path: Path) -> None:
    sources = _sources()
    record = run_training_experiment(*sources)
    path = write_training_experiment_record(tmp_path, record)
    altered = copy.deepcopy(record)
    altered["split_sha256"]["validation"] = "0" * 64
    _resign(altered)
    path.write_text(json.dumps(altered), encoding="utf-8")
    assert read_selected_checkpoint(path).is_fitted
    assert not verify_training_experiment_record(path, *sources)


def test_metrics_and_nonselected_checkpoint_are_integrity_checked(tmp_path: Path) -> None:
    sources = _sources()
    record = run_training_experiment(*sources)
    path = write_training_experiment_record(tmp_path, record)
    changed = copy.deepcopy(record)
    changed["outcomes"][0]["validation"]["auc"] = 0.123
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="record digest"):
        read_selected_checkpoint(path)
    assert not verify_training_experiment_record(path, *sources)


def test_concurrent_publication_has_exactly_one_winner(tmp_path: Path) -> None:
    record = run_training_experiment(*_sources())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(write_training_experiment_record, tmp_path, record) for _ in range(2)
        ]
        results = []
        for future in futures:
            with suppress(FileExistsError):
                results.append(future.result())
    assert len(results) == 1
    assert len(list(tmp_path.iterdir())) == 1


def test_bundled_synthetic_example_runs_as_documented(tmp_path: Path) -> None:
    examples = Path(__file__).resolve().parents[1] / "examples"
    sources = tuple(
        (examples / f"training_experiment_{name}.json").read_bytes()
        for name in ("articles", "train", "validation", "plan")
    )
    record = run_training_experiment(*sources)
    path = write_training_experiment_record(tmp_path, record)
    assert verify_training_experiment_record(path, *sources)


def test_cli_smoke_and_no_replace(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sources = _sources()
    paths = [
        tmp_path / name for name in ("articles.json", "train.json", "validation.json", "plan.json")
    ]
    for path, source in zip(paths, sources, strict=True):
        path.write_bytes(source)
    registry = tmp_path / "registry"
    args = [
        "run-training-experiment",
        "--articles",
        str(paths[0]),
        "--train",
        str(paths[1]),
        "--validation",
        str(paths[2]),
        "--plan",
        str(paths[3]),
        "--registry",
        str(registry),
    ]
    assert main(args) == 0
    assert "registered training experiment" in capsys.readouterr().out
    assert len(list(registry.glob("*.json"))) == 1
    assert main(args) == 2
    assert "already registered" in capsys.readouterr().err
