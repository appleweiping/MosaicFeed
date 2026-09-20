"""Bounded offline training experiments on caller-owned temporal MIND-shaped snapshots.

This is a validation selector, not hyperparameter search on a hidden test set or
an official MIND benchmark. Every candidate is trained independently on train.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from mosaicfeed.config import FeedConfig
from mosaicfeed.datasets import MindImpression
from mosaicfeed.io import json_text, load_articles_bytes, load_json_text, parse_datetime
from mosaicfeed.listwise import ListwiseImpressionRanker
from mosaicfeed.mind import (
    evaluate_mind_impressions,
    impression_to_dict,
    load_mind_impressions_bytes,
)
from mosaicfeed.pairwise import PairwiseImpressionRanker

PROTOCOL = "mosaicfeed-training-experiment-v1"
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_PLAN_BYTES = 64 * 1024
MAX_RECORD_BYTES = 4 * 1024 * 1024
MAX_ARTICLES = 2_000
MAX_TRAIN_ROWS = 200
MAX_VALIDATION_ROWS = 100
MAX_CANDIDATES_PER_ROW = 16
MAX_WORK_UNITS = 20_000_000
_METRICS = frozenset(("auc", "mrr", "ndcg@5", "ndcg@10"))
_ID = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")
Ranker = PairwiseImpressionRanker | ListwiseImpressionRanker


def _bytes(value: bytes, maximum: int, name: str) -> bytes:
    if type(value) is not bytes or len(value) > maximum:
        raise ValueError(f"{name} must be an immutable byte snapshot within {maximum} bytes")
    return value


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _ranker(objective: str, epochs: int, learning_rate: float, l2: float, seed: int) -> Ranker:
    if objective == "pairwise":
        return PairwiseImpressionRanker(
            epochs=epochs, learning_rate=learning_rate, l2=l2, seed=seed
        )
    if objective == "listwise":
        return ListwiseImpressionRanker(
            epochs=epochs, learning_rate=learning_rate, l2=l2, seed=seed
        )
    raise ValueError("objective must be pairwise or listwise")


@dataclass(frozen=True, slots=True)
class TrainingCandidate:
    id: str
    objective: Literal["pairwise", "listwise"]
    epochs: int
    learning_rate: float
    l2: float
    seed: int

    def __post_init__(self) -> None:
        if type(self.id) is not str or _ID.fullmatch(self.id) is None:
            raise ValueError("candidate id must be short lowercase ASCII")
        _ranker(self.objective, self.epochs, self.learning_rate, self.l2, self.seed)

    @classmethod
    def from_mapping(cls, data: object) -> TrainingCandidate:
        if not isinstance(data, dict) or set(data) != {
            "id",
            "objective",
            "epochs",
            "learning_rate",
            "l2",
            "seed",
        }:
            raise ValueError(
                "candidate must have exactly id, objective, epochs, learning_rate, l2, seed"
            )
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "objective": self.objective,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class TrainingExperimentPlan:
    cutoff: datetime
    feature_config: FeedConfig
    selection_metric: str
    candidates: tuple[TrainingCandidate, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.cutoff, datetime)
            or self.cutoff.tzinfo is None
            or self.cutoff.utcoffset() is None
        ):
            raise ValueError("cutoff must be timezone-aware")
        if not isinstance(self.feature_config, FeedConfig):
            raise ValueError("feature_config must be FeedConfig")
        if type(self.selection_metric) is not str or self.selection_metric not in _METRICS:
            raise ValueError("selection_metric is unsupported")
        if (
            type(self.candidates) is not tuple
            or not 2 <= len(self.candidates) <= 8
            or any(not isinstance(candidate, TrainingCandidate) for candidate in self.candidates)
            or len({c.id for c in self.candidates}) != len(self.candidates)
        ):
            raise ValueError("plan requires 2 to 8 distinct candidates")

    @classmethod
    def from_bytes(cls, source: bytes) -> TrainingExperimentPlan:
        _bytes(source, MAX_PLAN_BYTES, "plan")
        try:
            parsed = load_json_text(source.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise ValueError("plan must be UTF-8") from error
        if not isinstance(parsed, dict) or set(parsed) != {
            "schema_version",
            "cutoff",
            "feature_config",
            "selection_metric",
            "candidates",
        }:
            raise ValueError("plan has missing or unknown fields")
        if type(parsed["schema_version"]) is not int or parsed["schema_version"] != 1:
            raise ValueError("plan requires schema_version 1")
        if not isinstance(parsed["feature_config"], dict) or not isinstance(
            parsed["candidates"], list
        ):
            raise ValueError("feature_config and candidates must be object and array")
        return cls(
            parse_datetime(parsed["cutoff"], "cutoff"),
            FeedConfig.from_mapping(parsed["feature_config"]),
            parsed["selection_metric"],
            tuple(TrainingCandidate.from_mapping(row) for row in parsed["candidates"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "cutoff": self.cutoff.isoformat(),
            "feature_config": self.feature_config.to_dict(),
            "selection_metric": self.selection_metric,
            "candidates": [c.to_dict() for c in self.candidates],
        }


def _validate_rows(rows: tuple[MindImpression, ...], *, limit: int, label: str) -> None:
    if not 1 <= len(rows) <= limit:
        raise ValueError(f"{label} must have 1 to {limit} impressions")
    if any(not 2 <= len(row.candidates) <= MAX_CANDIDATES_PER_ROW for row in rows):
        raise ValueError(f"{label} candidate count is outside [2, {MAX_CANDIDATES_PER_ROW}]")


def _metric(report: dict[str, object], name: str) -> float:
    if name.startswith("ndcg@"):
        ndcg = report.get("ndcg")
        if not isinstance(ndcg, dict):
            raise ValueError("invalid validation nDCG mapping")
        value = ndcg.get(name.split("@")[1])
    else:
        value = report.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid validation metric")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError("invalid validation metric") from error
    if not 0.0 <= number <= 1.0:
        raise ValueError("invalid validation metric")
    return number


def select_candidate(outcomes: list[dict[str, object]], metric: str) -> str:
    """Select maximum validation metric, breaking exact ties by candidate id."""
    if type(metric) is not str or metric not in _METRICS or not outcomes:
        raise ValueError("selection requires a supported metric and outcomes")
    selected = min(
        outcomes,
        key=lambda item: (
            -_metric(cast(dict[str, object], item["validation"]), metric),
            str(item["candidate_id"]),
        ),
    )
    result = selected["candidate_id"]
    if not isinstance(result, str):
        raise ValueError("invalid candidate id")
    return result


def run_training_experiment(
    articles_source: bytes, train_source: bytes, validation_source: bytes, plan_source: bytes
) -> dict[str, object]:
    """Train each declared candidate using train labels and score only post-cutoff validation."""
    snapshots = {
        "articles": _bytes(articles_source, MAX_SOURCE_BYTES, "articles"),
        "train": _bytes(train_source, MAX_SOURCE_BYTES, "train"),
        "validation": _bytes(validation_source, MAX_SOURCE_BYTES, "validation"),
        "plan": _bytes(plan_source, MAX_PLAN_BYTES, "plan"),
    }
    plan = TrainingExperimentPlan.from_bytes(plan_source)
    articles = load_articles_bytes(articles_source)
    train = load_mind_impressions_bytes(train_source)
    validation = load_mind_impressions_bytes(validation_source)
    if not 1 <= len(articles) <= MAX_ARTICLES:
        raise ValueError("article catalog exceeds experiment limit")
    _validate_rows(train, limit=MAX_TRAIN_ROWS, label="train")
    _validate_rows(validation, limit=MAX_VALIDATION_ROWS, label="validation")
    if any(row.occurred_at > plan.cutoff for row in train):
        raise ValueError("train contains a row after cutoff")
    if any(row.occurred_at <= plan.cutoff for row in validation):
        raise ValueError("validation contains a row at or before cutoff")
    if max(row.occurred_at for row in train) >= min(row.occurred_at for row in validation):
        raise ValueError("temporal split is not strictly ordered")
    if {row.impression_id for row in train} & {row.impression_id for row in validation}:
        raise ValueError("train and validation impression ids overlap")
    if any(
        not any(c.clicked for c in row.candidates) or all(c.clicked for c in row.candidates)
        for row in validation
    ):
        raise ValueError("every validation row needs positive and negative candidates")
    # Include worst-case repeated history/profile/topic scans, candidate feature
    # construction, catalog verification and training updates for every model.
    train_cells = sum(len(row.candidates) for row in train)
    validation_cells = sum(len(row.candidates) for row in validation)
    topic_units = sum(len(article.topics) for article in articles)
    history_units = sum(sum(c.clicked for c in row.candidates) for row in train)
    work = 0
    for candidate in plan.candidates:
        pairs = sum(
            sum(c.clicked for c in row.candidates) * sum(not c.clicked for c in row.candidates)
            for row in train
        )
        updates = pairs if candidate.objective == "pairwise" else train_cells
        work += candidate.epochs * (updates + train_cells * (history_units + topic_units + 1))
        work += validation_cells * (history_units + topic_units + 1) + len(articles) * 4
    if work > MAX_WORK_UNITS:
        raise ValueError("training experiment exceeds aggregate work limit")
    fingerprints = {name: _sha(value) for name, value in snapshots.items()}
    train_fingerprint = _sha(_canonical([impression_to_dict(row) for row in train]))
    validation_fingerprint = _sha(_canonical([impression_to_dict(row) for row in validation]))
    outcomes: list[dict[str, object]] = []
    for candidate in sorted(plan.candidates, key=lambda c: c.id):
        model = _ranker(
            candidate.objective,
            candidate.epochs,
            candidate.learning_rate,
            candidate.l2,
            candidate.seed,
        )
        model.fit(
            articles,
            train,
            partition="train",
            cutoff=plan.cutoff,
            held_out_impression_ids=(row.impression_id for row in validation),
            source_sha256={"articles": fingerprints["articles"], "train": fingerprints["train"]},
            config=plan.feature_config,
        )
        state = model.to_state()
        restored = type(model).from_state(state)
        scores = restored.score_impressions(articles, validation, training_impressions=train)
        report = evaluate_mind_impressions(validation, scores).to_dict()
        outcomes.append(
            {
                "candidate_id": candidate.id,
                "objective": candidate.objective,
                "checkpoint_sha256": _sha(_canonical(state)),
                "checkpoint": state,
                "validation": report,
            }
        )
    identity = _sha(_canonical({"protocol": PROTOCOL, "sources": fingerprints}))
    record: dict[str, object] = {
        "protocol": PROTOCOL,
        "experiment_id": identity,
        "source_sha256": fingerprints,
        "split_sha256": {"train": train_fingerprint, "validation": validation_fingerprint},
        "work_units_upper_bound": work,
        "plan": plan.to_dict(),
        "selection_metric": plan.selection_metric,
        "selected_candidate_id": select_candidate(outcomes, plan.selection_metric),
        "outcomes": outcomes,
    }
    record["record_sha256"] = _sha(_canonical(record))
    if len(json_text(record).encode("utf-8")) > MAX_RECORD_BYTES:
        raise ValueError("training experiment record exceeds size limit")
    return record


def write_training_experiment_record(directory: str | Path, record: dict[str, object]) -> Path:
    """Publish one canonical experiment/checkpoint record atomically, without overwrite."""
    if not isinstance(record, dict) or record.get("protocol") != PROTOCOL:
        raise ValueError("invalid training experiment record")
    identity = record.get("experiment_id")
    if not isinstance(identity, str) or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
        raise ValueError("invalid experiment id")
    _check_record_digest(record)
    data = json_text(record).encode("utf-8")
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError("record exceeds size limit")
    registry = Path(directory)
    if registry.exists() and not registry.is_dir():
        raise ValueError("registry must be a directory")
    registry.mkdir(parents=True, exist_ok=True)
    target = registry / f"{identity}.json"
    descriptor, staged_name = tempfile.mkstemp(
        prefix=".training-experiment-", suffix=".tmp", dir=registry
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, target)
        if os.name == "posix":
            parent = os.open(registry, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        return target
    finally:
        staged.unlink(missing_ok=True)


def verify_training_experiment_record(
    path: str | Path,
    articles_source: bytes,
    train_source: bytes,
    validation_source: bytes,
    plan_source: bytes,
) -> bool:
    """Replay all candidates and compare every byte, including every checkpoint."""
    expected = run_training_experiment(
        articles_source, train_source, validation_source, plan_source
    )
    target = Path(path)
    if target.name != f"{expected['experiment_id']}.json":
        return False
    with target.open("rb") as stream:
        actual = stream.read(MAX_RECORD_BYTES + 1)
    return actual == json_text(expected).encode("utf-8")


def read_selected_checkpoint(path: str | Path) -> Ranker:
    """Load the selected state only after validating every recorded checkpoint."""
    target = Path(path)
    with target.open("rb") as stream:
        raw = stream.read(MAX_RECORD_BYTES + 1)
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError("record exceeds size limit")
    try:
        record: Any = load_json_text(raw.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError("record must be UTF-8") from error
    if not isinstance(record, dict) or record.get("protocol") != PROTOCOL:
        raise ValueError("invalid record protocol")
    _check_record_digest(record)
    if target.name != f"{record.get('experiment_id')}.json":
        raise ValueError("record filename differs from experiment id")
    outcomes = record.get("outcomes")
    if not isinstance(outcomes, list) or not 2 <= len(outcomes) <= 8:
        raise ValueError("invalid candidate outcomes")
    if any(not isinstance(row, dict) for row in outcomes):
        raise ValueError("invalid candidate outcome")
    plan_record = record.get("plan")
    if not isinstance(plan_record, dict):
        raise ValueError("invalid recorded plan")
    plan = TrainingExperimentPlan.from_bytes(_canonical(plan_record))
    sources = record.get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != {"articles", "train", "validation", "plan"}:
        raise ValueError("invalid source hashes")
    if record.get("experiment_id") != _sha(_canonical({"protocol": PROTOCOL, "sources": sources})):
        raise ValueError("experiment id does not match sources")
    selected = record.get("selected_candidate_id")
    metric = record.get("selection_metric")
    if metric != plan.selection_metric:
        raise ValueError("selection metric differs from plan")
    if len({row.get("candidate_id") for row in outcomes if isinstance(row, dict)}) != len(outcomes):
        raise ValueError("duplicate candidate id")
    specs = {candidate.id: candidate for candidate in plan.candidates}
    if {row.get("candidate_id") for row in outcomes} != set(specs):
        raise ValueError("outcomes differ from plan candidates")
    if not isinstance(metric, str) or select_candidate(outcomes, metric) != selected:
        raise ValueError("selection does not match recorded metrics")
    models: dict[str, Ranker] = {}
    for row in outcomes:
        if not isinstance(row, dict) or not isinstance(row.get("checkpoint"), dict):
            raise ValueError("invalid checkpoint")
        state = row["checkpoint"]
        if _sha(_canonical(state)) != row.get("checkpoint_sha256"):
            raise ValueError("checkpoint hash mismatch")
        candidate_id = row.get("candidate_id")
        objective = row.get("objective")
        if not isinstance(candidate_id, str) or objective not in ("pairwise", "listwise"):
            raise ValueError("invalid checkpoint objective or id")
        spec = specs[candidate_id]
        expected_parameters = {
            "epochs": spec.epochs,
            "learning_rate": spec.learning_rate,
            "l2": spec.l2,
            "seed": spec.seed,
        }
        if objective != spec.objective or state.get("parameters") != expected_parameters:
            raise ValueError("checkpoint differs from candidate specification")
        if state.get("feature_config") != plan.feature_config.to_dict():
            raise ValueError("checkpoint feature config differs from plan")
        summary = state.get("training")
        if not isinstance(summary, dict) or summary.get("cutoff") != plan.cutoff.isoformat():
            raise ValueError("checkpoint cutoff differs from plan")
        if summary.get("source_sha256") != {
            "articles": sources["articles"],
            "train": sources["train"],
        }:
            raise ValueError("checkpoint training sources differ from record")
        if objective == "pairwise":
            models[candidate_id] = PairwiseImpressionRanker.from_state(state)
        else:
            models[candidate_id] = ListwiseImpressionRanker.from_state(state)
    if selected not in models:
        raise ValueError("selected candidate is absent")
    return models[selected]


def _check_record_digest(record: dict[str, object]) -> None:
    if set(record) != {
        "protocol",
        "experiment_id",
        "source_sha256",
        "split_sha256",
        "work_units_upper_bound",
        "plan",
        "selection_metric",
        "selected_candidate_id",
        "outcomes",
        "record_sha256",
    }:
        raise ValueError("invalid record fields")
    digest = record["record_sha256"]
    if not isinstance(digest, str) or digest != _sha(
        _canonical({key: value for key, value in record.items() if key != "record_sha256"})
    ):
        raise ValueError("record digest mismatch")
