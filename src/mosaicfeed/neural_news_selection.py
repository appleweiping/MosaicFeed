"""Reproducible, one-factor ablations of the local neural-news baseline.

All candidates fit the same immutable training bytes. Validation labels only
select a checkpoint and calculate paired diagnostics; they never update it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from mosaicfeed.io import json_text, load_json_text, parse_datetime
from mosaicfeed.mind import (
    impression_auc,
    impression_mrr,
    impression_ndcg,
    load_mind_impressions_bytes,
)
from mosaicfeed.neural_news import (
    MAX_SOURCE_BYTES,
    NeuralNewsConfig,
    NeuralNewsRanker,
    run_neural_news_experiment,
)

PROTOCOL = "mosaicfeed.neural-news-selection"
SCOPE = "local synthetic/caller-owned ablation, not an official MIND result"
PAIRED_UNIT = "per-impression paired validation metric delta vs baseline"
MAX_PLAN_BYTES = 64 * 1024
MAX_SELECTION_SOURCE_BYTES = MAX_SOURCE_BYTES
MAX_RECORD_BYTES = 32 * 1024 * 1024
MAX_AGGREGATE_WORK = 60_000_000
_ID = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")
_FIELDS = frozenset(("dimension", "epochs", "learning_rate", "max_vocabulary", "max_history"))
_METRICS = frozenset(("auc", "mrr", "ndcg@5", "ndcg@10"))


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("neural selection record is not canonical JSON") from error


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object) -> str:
    return _sha(_canonical(value))


def _neural_state_digest(value: object) -> str:
    # The released single-run neural-news format uses json.dumps's ASCII default.
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _hex(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _unit(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("selection metric is not a unit-interval number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("selection metric is not a unit-interval number") from error
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("selection metric is not a unit-interval number")
    return result


def _signed_unit(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("selection paired diagnostic must be finite")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("selection paired diagnostic must be finite") from error
    if not math.isfinite(result) or not -1 <= result <= 1:
        raise ValueError("selection paired diagnostic must be bounded")
    return result


def _metric(report: object, name: str) -> float:
    if not isinstance(report, dict):
        raise ValueError("selection report is malformed")
    if name.startswith("ndcg@"):
        ndcg = report.get("ndcg")
        if not isinstance(ndcg, dict):
            raise ValueError("selection nDCG report is malformed")
        return _unit(ndcg.get(name.split("@")[1]))
    return _unit(report.get(name))


@dataclass(frozen=True, slots=True)
class NeuralAblation:
    id: str
    field: str
    value: int | float

    def __post_init__(self) -> None:
        if type(self.id) is not str or self.id == "baseline" or _ID.fullmatch(self.id) is None:
            raise ValueError("ablation id must be distinct short lowercase ASCII")
        if type(self.field) is not str or self.field not in _FIELDS:
            raise ValueError("ablation field is unsupported")

    def config(self, baseline: NeuralNewsConfig) -> NeuralNewsConfig:
        values = baseline.to_dict()
        if self.value == values[self.field] or type(self.value) is not type(values[self.field]):
            raise ValueError("ablation must change exactly its declared field")
        values[self.field] = self.value
        return NeuralNewsConfig.from_mapping(values)

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "field": self.field, "value": self.value}


@dataclass(frozen=True, slots=True)
class NeuralSelectionPlan:
    cutoff: datetime
    baseline: NeuralNewsConfig
    ablations: tuple[NeuralAblation, ...]
    selection_metric: str = "ndcg@5"
    bootstrap_samples: int = 100
    bootstrap_seed: int = 17

    def __post_init__(self) -> None:
        if (
            not isinstance(self.cutoff, datetime)
            or self.cutoff.tzinfo is None
            or self.cutoff.utcoffset() is None
        ):
            raise ValueError("selection cutoff must be timezone-aware")
        if not isinstance(self.baseline, NeuralNewsConfig):
            raise ValueError("selection baseline must be NeuralNewsConfig")
        if (
            type(self.ablations) is not tuple
            or not 1 <= len(self.ablations) <= 5
            or any(not isinstance(item, NeuralAblation) for item in self.ablations)
            or len({item.id for item in self.ablations}) != len(self.ablations)
            or len({item.field for item in self.ablations}) != len(self.ablations)
        ):
            raise ValueError("selection requires one to five distinct one-factor ablations")
        if type(self.selection_metric) is not str or self.selection_metric not in _METRICS:
            raise ValueError("selection metric is unsupported")
        if type(self.bootstrap_samples) is not int or not 40 <= self.bootstrap_samples <= 500:
            raise ValueError("bootstrap_samples must be in [40, 500]")
        if type(self.bootstrap_seed) is not int or not 0 <= self.bootstrap_seed <= 2**32 - 1:
            raise ValueError("bootstrap_seed is outside [0, 2^32-1]")
        for item in self.ablations:
            item.config(self.baseline)

    @classmethod
    def from_bytes(cls, source: bytes) -> NeuralSelectionPlan:
        if type(source) is not bytes or len(source) > MAX_PLAN_BYTES:
            raise ValueError("selection plan must be bounded immutable bytes")
        try:
            parsed = load_json_text(source.decode("utf-8", "strict"))
        except (UnicodeError, ValueError, RecursionError) as error:
            raise ValueError("selection plan must be strict JSON UTF-8") from error
        if not isinstance(parsed, dict) or set(parsed) != {
            "schema_version",
            "cutoff",
            "baseline",
            "ablations",
            "selection_metric",
            "bootstrap_samples",
            "bootstrap_seed",
        }:
            raise ValueError("selection plan fields are malformed")
        if type(parsed["schema_version"]) is not int or parsed["schema_version"] != 1:
            raise ValueError("unsupported selection plan schema")
        if not isinstance(parsed["ablations"], list) or not 1 <= len(parsed["ablations"]) <= 5:
            raise ValueError("selection ablations must be a bounded array")
        ablations = []
        for raw in parsed["ablations"]:
            if not isinstance(raw, dict) or set(raw) != {"id", "field", "value"}:
                raise ValueError("selection ablation fields are malformed")
            ablations.append(NeuralAblation(**raw))
        return cls(
            parse_datetime(parsed["cutoff"], "cutoff"),
            NeuralNewsConfig.from_mapping(parsed["baseline"]),
            tuple(ablations),
            parsed["selection_metric"],
            parsed["bootstrap_samples"],
            parsed["bootstrap_seed"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "cutoff": self.cutoff.isoformat(),
            "baseline": self.baseline.to_dict(),
            "ablations": [item.to_dict() for item in self.ablations],
            "selection_metric": self.selection_metric,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
        }


def _per_impression(
    validation_source: bytes, experiment: dict[str, object], metric: str
) -> tuple[float, ...]:
    rows = load_mind_impressions_bytes(validation_source)
    raw_scores = experiment["scores"]
    if not isinstance(raw_scores, list):
        raise ValueError("neural experiment scores are malformed")
    scores = {(row["impression_id"], row["article_id"]): row["score"] for row in raw_scores}
    result = []
    for row in rows:
        labels = tuple(item.clicked for item in row.candidates)
        values = tuple(scores[(row.impression_id, item.article_id)] for item in row.candidates)
        if metric == "auc":
            result.append(impression_auc(labels, values))
        elif metric == "mrr":
            result.append(impression_mrr(labels, values))
        else:
            result.append(impression_ndcg(labels, values, int(metric.split("@")[1])))
    return tuple(result)


def _paired(
    values: tuple[float, ...], baseline: tuple[float, ...], draws: tuple[tuple[int, ...], ...]
) -> dict[str, object]:
    differences = tuple(a - b for a, b in zip(values, baseline, strict=True))
    means = sorted(math.fsum(differences[index] for index in draw) / len(draw) for draw in draws)
    return {
        "mean": math.fsum(differences) / len(differences),
        "lower_95": means[math.floor(0.025 * (len(means) - 1))],
        "upper_95": means[math.ceil(0.975 * (len(means) - 1))],
        "unit": PAIRED_UNIT,
    }


def run_neural_selection(
    articles_source: bytes, train_source: bytes, validation_source: bytes, plan_source: bytes
) -> dict[str, object]:
    """Train on identical train bytes and select only by declared validation metric."""
    sources = {
        "articles": articles_source,
        "train": train_source,
        "validation": validation_source,
        "plan": plan_source,
    }
    if any(
        type(raw) is not bytes
        or len(raw) > (MAX_PLAN_BYTES if name == "plan" else MAX_SOURCE_BYTES)
        for name, raw in sources.items()
    ):
        raise ValueError("selection sources must be bounded immutable bytes")
    plan = NeuralSelectionPlan.from_bytes(plan_source)
    specs = [("baseline", None, plan.baseline)] + [
        (item.id, item.field, item.config(plan.baseline)) for item in plan.ablations
    ]
    outcomes: list[dict[str, object]] = []
    per_row: dict[str, tuple[float, ...]] = {}
    work = 0
    for identity, field, config in specs:
        experiment = run_neural_news_experiment(
            articles_source, train_source, validation_source, cutoff=plan.cutoff, config=config
        )
        units = experiment["work_units_upper_bound"]
        if type(units) is not int:
            raise ValueError("neural experiment work bound is malformed")
        work += units
        if work > MAX_AGGREGATE_WORK:
            raise ValueError("selection aggregate work limit exceeded")
        outcome: dict[str, object] = {
            "id": identity,
            "changed_field": field,
            "experiment": experiment,
        }
        outcomes.append(outcome)
        per_row[identity] = _per_impression(validation_source, experiment, plan.selection_metric)
    count = len(per_row["baseline"])
    rng = random.Random(plan.bootstrap_seed)  # nosec B311: reproducible audit sampling.
    draws = tuple(
        tuple(rng.randrange(count) for _ in range(count)) for _ in range(plan.bootstrap_samples)
    )
    paired = {
        identity: _paired(values, per_row["baseline"], draws)
        for identity, values in sorted(per_row.items())
    }
    selected = min(
        outcomes,
        key=lambda row: (
            -_metric(cast(dict[str, object], row["experiment"])["metrics"], plan.selection_metric),
            0 if row["id"] == "baseline" else 1,
            row["id"],
        ),
    )["id"]
    fingerprints = {name: _sha(raw) for name, raw in sorted(sources.items())}
    record: dict[str, object] = {
        "protocol": PROTOCOL,
        "schema_version": 1,
        "experiment_id": _digest({"protocol": PROTOCOL, "source_sha256": fingerprints}),
        "source_sha256": fingerprints,
        "plan": plan.to_dict(),
        "selection_metric": plan.selection_metric,
        "selected_id": selected,
        "work_units_upper_bound": work,
        "outcomes": outcomes,
        "paired_deltas": paired,
        "scope": SCOPE,
    }
    record["record_sha256"] = _digest(record)
    if len(json_text(record).encode("utf-8")) > MAX_RECORD_BYTES:
        raise ValueError("selection record exceeds byte limit")
    return record


def _selected_model(record: object) -> NeuralNewsRanker:
    if not isinstance(record, dict) or set(record) != {
        "protocol",
        "schema_version",
        "experiment_id",
        "source_sha256",
        "plan",
        "selection_metric",
        "selected_id",
        "work_units_upper_bound",
        "outcomes",
        "paired_deltas",
        "scope",
        "record_sha256",
    }:
        raise ValueError("selection record fields are malformed")
    if (
        record["protocol"] != PROTOCOL
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["scope"] != SCOPE
    ):
        raise ValueError("selection record protocol is unsupported")
    digest = record["record_sha256"]
    if not _hex(digest) or digest != _digest(
        {key: val for key, val in record.items() if key != "record_sha256"}
    ):
        raise ValueError("selection record digest mismatch")
    sources = record["source_sha256"]
    if (
        not isinstance(sources, dict)
        or set(sources) != {"articles", "train", "validation", "plan"}
        or any(not _hex(value) for value in sources.values())
    ):
        raise ValueError("selection source digests are malformed")
    if record["experiment_id"] != _digest({"protocol": PROTOCOL, "source_sha256": sources}):
        raise ValueError("selection experiment identity mismatch")
    plan = NeuralSelectionPlan.from_bytes(_canonical(record["plan"]))
    if record["selection_metric"] != plan.selection_metric:
        raise ValueError("selection metric differs from plan")
    work = record["work_units_upper_bound"]
    if type(work) is not int or not 0 <= work <= MAX_AGGREGATE_WORK:
        raise ValueError("selection work bound is malformed")
    outcomes = record["outcomes"]
    if not isinstance(outcomes, list) or len(outcomes) != 1 + len(plan.ablations):
        raise ValueError("selection outcome count is malformed")
    expected = {"baseline": (None, plan.baseline)} | {
        item.id: (item.field, item.config(plan.baseline)) for item in plan.ablations
    }
    models: dict[str, NeuralNewsRanker] = {}
    scores: list[tuple[str, float]] = []
    observed_work = 0
    for row in outcomes:
        if not isinstance(row, dict) or set(row) != {"id", "changed_field", "experiment"}:
            raise ValueError("selection outcome fields are malformed")
        identity = row["id"]
        if type(identity) is not str or identity not in expected or identity in models:
            raise ValueError("selection outcome ID is invalid or repeated")
        field, config = expected[identity]
        if row["changed_field"] != field:
            raise ValueError("selection ablation field mismatch")
        experiment = row["experiment"]
        if (
            not isinstance(experiment, dict)
            or experiment.get("format") != "mosaicfeed.neural-news-experiment"
            or set(experiment)
            != {
                "format",
                "schema_version",
                "cutoff",
                "source_sha256",
                "state_sha256",
                "model",
                "scores",
                "metrics",
                "work_units_upper_bound",
                "scope",
            }
            or type(experiment["schema_version"]) is not int
            or experiment["schema_version"] != 1
            or experiment["cutoff"] != plan.cutoff.isoformat()
        ):
            raise ValueError("selection outcome experiment is malformed")
        if experiment.get("source_sha256") != {
            name: sources[name] for name in ("articles", "train", "validation")
        }:
            raise ValueError("selection outcome sources mismatch")
        state = experiment.get("model")
        if not isinstance(state, dict) or experiment.get("state_sha256") != _neural_state_digest(
            state
        ):
            raise ValueError("selection checkpoint digest mismatch")
        model = NeuralNewsRanker.from_state(state)
        if model.config != config or model.cutoff != plan.cutoff:
            raise ValueError("selection checkpoint differs from plan")
        if experiment["work_units_upper_bound"] != model.work_units_upper_bound:
            raise ValueError("selection outcome work bound differs from checkpoint")
        models[identity] = model
        scores.append((identity, _metric(experiment.get("metrics"), plan.selection_metric)))
        observed_work += model.work_units_upper_bound
    if observed_work != work:
        raise ValueError("selection work bound differs from checkpoints")
    selected = min(scores, key=lambda pair: (-pair[1], 0 if pair[0] == "baseline" else 1, pair[0]))[
        0
    ]
    paired = record["paired_deltas"]
    if (
        record["selected_id"] != selected
        or not isinstance(paired, dict)
        or set(paired) != set(expected)
    ):
        raise ValueError("selection result or paired rows mismatch")
    for identity, diagnostic in paired.items():
        if (
            not isinstance(diagnostic, dict)
            or set(diagnostic) != {"mean", "lower_95", "upper_95", "unit"}
            or diagnostic["unit"] != PAIRED_UNIT
        ):
            raise ValueError("selection paired diagnostic is malformed")
        for name in ("mean", "lower_95", "upper_95"):
            _signed_unit(diagnostic[name])
        if identity == "baseline" and any(
            diagnostic[name] != 0 for name in ("mean", "lower_95", "upper_95")
        ):
            raise ValueError("selection baseline paired diagnostic must be zero")
    return models[selected]


def write_neural_selection_record(directory: str | Path, record: dict[str, object]) -> Path:
    """Atomically publish a content-addressed record without replacing a path."""
    _selected_model(record)
    raw = json_text(record).encode("utf-8")
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError("selection record exceeds byte limit")
    root = Path(directory)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError("selection registry must be a real directory")
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{record['experiment_id']}.json"
    descriptor, name = tempfile.mkstemp(prefix=".neural-selection-", suffix=".tmp", dir=root)
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, target)
        if os.name == "posix":
            parent = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        return target
    finally:
        staged.unlink(missing_ok=True)


def read_selected_neural_checkpoint(path: str | Path) -> NeuralNewsRanker:
    """Check internal record/state consistency; source authenticity needs replay."""
    target = Path(path)
    with target.open("rb") as stream:
        raw = stream.read(MAX_RECORD_BYTES + 1)
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError("selection record exceeds byte limit")
    try:
        record = load_json_text(raw.decode("utf-8", "strict"))
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError("selection record must be strict JSON UTF-8") from error
    model = _selected_model(record)
    parsed_record = cast(dict[str, object], record)
    if target.name != f"{parsed_record['experiment_id']}.json":
        raise ValueError("selection record filename differs from identity")
    return model


def verify_neural_selection_record(
    path: str | Path,
    articles_source: bytes,
    train_source: bytes,
    validation_source: bytes,
    plan_source: bytes,
) -> bool:
    """Replay all candidates against exact caller-provided bytes and compare bytes."""
    expected = run_neural_selection(articles_source, train_source, validation_source, plan_source)
    target = Path(path)
    if target.name != f"{expected['experiment_id']}.json":
        return False
    with target.open("rb") as stream:
        actual = stream.read(MAX_RECORD_BYTES + 1)
    return actual == json_text(expected).encode("utf-8")
