"""Bounded, content-addressed local ablation experiments.

The registry records deterministic offline evidence, not a trained-model
checkpoint, causal estimate, or an official MIND benchmark result.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

from mosaicfeed.benchmark import (
    _BOOTSTRAP_METRICS,
    _USER_MEAN_METRICS,
    _interval,
    _point_metrics,
    _resampled_metrics,
    bootstrap_mean,
    dataset_fingerprint,
)
from mosaicfeed.config import FeedConfig
from mosaicfeed.io import load_articles_bytes, load_events_bytes, load_json_text, parse_datetime
from mosaicfeed.logged import DEFAULT_RESAMPLES, MIN_USERS_FOR_INTERVAL
from mosaicfeed.metrics import EvaluationSamples, evaluate_leave_last_out_samples

MAX_EXPERIMENT_SOURCE_BYTES = 16 * 1024 * 1024
MAX_EXPERIMENT_PLAN_BYTES = 64 * 1024
MAX_EXPERIMENT_ROWS = 50_000
MAX_EXPERIMENT_USERS = 5_000
MAX_EVALUATION_WORK = 5_000_000
MAX_BOOTSTRAP_WORK = 8_000_000
MAX_RECORD_BYTES = 4 * 1024 * 1024
RUNNER_PROTOCOL = "mosaicfeed-ablation-v1"

ABLATIONS = (
    "without_interest",
    "without_freshness",
    "without_quality",
    "without_novelty",
    "without_popularity",
    "without_exploration",
    "without_slate_diversity",
)
_WEIGHT_FIELDS = {
    "without_interest": "interest_weight",
    "without_freshness": "freshness_weight",
    "without_quality": "quality_weight",
    "without_novelty": "novelty_weight",
    "without_popularity": "popularity_weight",
    "without_exploration": "exploration_weight",
}


def _snapshot(data: bytes, maximum: int, label: str) -> bytes:
    if type(data) is not bytes or len(data) > maximum:
        raise ValueError(f"{label} must be an immutable byte snapshot within {maximum} bytes")
    return data


def read_experiment_source(path: str | Path, *, maximum: int) -> bytes:
    """Capture one bounded path exactly once before parsing or hashing it."""

    if type(maximum) is not int or maximum < 1 or maximum > MAX_EXPERIMENT_SOURCE_BYTES:
        raise ValueError("experiment source maximum is invalid")
    with Path(path).open("rb") as stream:
        data = stream.read(maximum + 1)
    return _snapshot(data, maximum, "experiment source")


@dataclass(frozen=True, slots=True)
class AblationPlan:
    as_of: datetime
    config: FeedConfig
    k: int
    ablations: tuple[str, ...]
    bootstrap_samples: int = 100
    confidence: float = 0.95
    seed: int = 17

    def __post_init__(self) -> None:
        if (
            not isinstance(self.as_of, datetime)
            or self.as_of.tzinfo is None
            or self.as_of.utcoffset() is None
        ):
            raise ValueError("as_of must be timezone-aware")
        if not isinstance(self.config, FeedConfig) or self.config.size > 100:
            raise ValueError("config must be a FeedConfig with size at most 100")
        if type(self.k) is not int or not 1 <= self.k <= 100:
            raise ValueError("k must be an integer in [1, 100]")
        if (
            type(self.ablations) is not tuple
            or not 1 <= len(self.ablations) <= len(ABLATIONS)
            or any(type(name) is not str or name not in ABLATIONS for name in self.ablations)
            or len(set(self.ablations)) != len(self.ablations)
        ):
            raise ValueError("ablations must be distinct supported names")
        if type(self.bootstrap_samples) is not int or not 1 <= self.bootstrap_samples <= 2_000:
            raise ValueError("bootstrap_samples must be an integer in [1, 2000]")
        if type(self.seed) is not int or not -(2**63) <= self.seed < 2**63:
            raise ValueError("seed must be a signed 64-bit integer")
        bootstrap_mean((), samples=1, confidence=self.confidence, seed=self.seed)

    @classmethod
    def from_bytes(cls, source: bytes) -> AblationPlan:
        _snapshot(source, MAX_EXPERIMENT_PLAN_BYTES, "experiment plan")
        try:
            parsed = load_json_text(source.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise ValueError("experiment plan must be UTF-8") from error
        if not isinstance(parsed, dict):
            raise ValueError("experiment plan must be a JSON object")
        allowed = {
            "schema_version",
            "as_of",
            "config",
            "k",
            "ablations",
            "bootstrap_samples",
            "confidence",
            "seed",
        }
        required = {"schema_version", "as_of", "config", "k", "ablations"}
        if set(parsed) - allowed or not required <= set(parsed):
            raise ValueError("experiment plan has missing or unknown fields")
        if type(parsed["schema_version"]) is not int or parsed["schema_version"] != 1:
            raise ValueError("experiment plan requires schema_version 1")
        config = parsed["config"]
        names = parsed["ablations"]
        if not isinstance(config, dict) or not isinstance(names, list):
            raise ValueError("config must be an object and ablations must be an array")
        return cls(
            as_of=parse_datetime(parsed["as_of"], "as_of"),
            config=FeedConfig.from_mapping(config),
            k=parsed["k"],
            ablations=tuple(names),
            bootstrap_samples=parsed.get("bootstrap_samples", 100),
            confidence=parsed.get("confidence", 0.95),
            seed=parsed.get("seed", 17),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "as_of": self.as_of.isoformat(),
            "config": self.config.to_dict(),
            "k": self.k,
            "ablations": list(self.ablations),
            "bootstrap_samples": self.bootstrap_samples,
            "confidence": self.confidence,
            "seed": self.seed,
        }


def _variant_config(name: str, base: FeedConfig, catalog_size: int, k: int) -> FeedConfig:
    if name == "without_slate_diversity":
        return replace(
            base,
            mmr_lambda=1.0,
            calibration_weight=0.0,
            max_per_source=max(base.size, catalog_size, k),
        )
    return FeedConfig.from_mapping({**base.to_dict(), _WEIGHT_FIELDS[name]: 0.0})


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    runner_protocol: str
    experiment_id: str
    articles_sha256: str
    events_sha256: str
    plan_sha256: str
    dataset_sha256: str
    plan: AblationPlan
    outcomes: Mapping[str, Mapping[str, object]]
    paired_deltas: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        revision = (
            self.runner_protocol.removeprefix("mosaicfeed-ablation-v")
            if type(self.runner_protocol) is str
            else ""
        )
        if (
            type(self.runner_protocol) is not str
            or not self.runner_protocol.startswith("mosaicfeed-ablation-v")
            or not revision.isascii()
            or not revision.isdigit()
            or not revision
            or revision[0] == "0"
        ):
            raise ValueError("runner_protocol must declare an ablation revision")
        for value in (
            self.experiment_id,
            self.articles_sha256,
            self.events_sha256,
            self.plan_sha256,
            self.dataset_sha256,
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("experiment hashes must be lowercase SHA-256 digests")
        identity = hashlib.sha256(
            self.runner_protocol.encode("ascii")
            + b"\0"
            + b"".join(
                bytes.fromhex(value)
                for value in (self.articles_sha256, self.events_sha256, self.plan_sha256)
            )
        ).hexdigest()
        if self.experiment_id != identity:
            raise ValueError("experiment ID does not match its exact source digests")
        if not isinstance(self.plan, AblationPlan):
            raise ValueError("experiment plan must be an AblationPlan")
        expected = {"baseline", *self.plan.ablations}
        if set(self.outcomes) != expected or set(self.paired_deltas) != set(self.plan.ablations):
            raise ValueError("experiment outcome names do not match the plan")
        object.__setattr__(
            self,
            "outcomes",
            MappingProxyType({name: _freeze_json(value) for name, value in self.outcomes.items()}),
        )
        object.__setattr__(
            self,
            "paired_deltas",
            MappingProxyType(
                {name: _freeze_json(value) for name, value in self.paired_deltas.items()}
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "runner_protocol": self.runner_protocol,
            "experiment_id": self.experiment_id,
            "provenance": {
                "articles_sha256": self.articles_sha256,
                "events_sha256": self.events_sha256,
                "plan_sha256": self.plan_sha256,
                "dataset_sha256": self.dataset_sha256,
            },
            "plan": self.plan.to_dict(),
            "outcomes": {
                name: _thaw_json(self.outcomes[name]) for name in ("baseline", *self.plan.ablations)
            },
            "paired_deltas_from_baseline": {
                name: _thaw_json(self.paired_deltas[name]) for name in self.plan.ablations
            },
        }


def _source_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _experiment_id(articles: bytes, events: bytes, plan: bytes) -> str:
    material = (
        RUNNER_PROTOCOL.encode("ascii")
        + b"\0"
        + b"".join(hashlib.sha256(source).digest() for source in (articles, events, plan))
    )
    return hashlib.sha256(material).hexdigest()


def _record_bytes(run: ExperimentRun) -> bytes:
    serialized = json.dumps(
        run.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    data = (serialized + "\n").encode("utf-8")
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError("experiment record exceeds its byte limit")
    return data


def run_ablation_experiment(
    articles_source: bytes,
    events_source: bytes,
    plan_source: bytes,
    *,
    articles_name: str = "articles.json",
    events_name: str = "events.json",
) -> ExperimentRun:
    """Evaluate each declared one-factor ablation on shared temporal holdouts."""

    _snapshot(articles_source, MAX_EXPERIMENT_SOURCE_BYTES, "article source")
    _snapshot(events_source, MAX_EXPERIMENT_SOURCE_BYTES, "event source")
    plan = AblationPlan.from_bytes(plan_source)
    articles = load_articles_bytes(articles_source, source_name=articles_name)
    events = load_events_bytes(events_source, source_name=events_name)
    if len(articles) > MAX_EXPERIMENT_ROWS or len(events) > MAX_EXPERIMENT_ROWS:
        raise ValueError("experiment exceeds its article/event row limit")
    active_users = {event.user_id for event in events if event.occurred_at <= plan.as_of}
    variants = 1 + len(plan.ablations)
    if len(active_users) > MAX_EXPERIMENT_USERS:
        raise ValueError("experiment exceeds its active-user limit")
    # Every policy repeats event/profile scans and constructs a slate. MMR
    # revisits candidates against previous selections; calibrated reranking
    # evaluates a topic distribution for each candidate at each position,
    # including a scan of every reader topic. The global topic count bounds
    # any reader profile and any partial slate, without trusting unknown
    # per-user topic sparsity or a zero calibration weight to short-circuit.
    topic_units = sum(len(article.topics) for article in articles)
    slate_steps = min(plan.config.size, len(articles))
    event_profile_work = len(events) * (1 + topic_units)
    candidate_topic_work = len(articles) * topic_units * slate_steps * (slate_steps + 2)
    # EvaluationSamples.report() invokes the separate logged-policy summary
    # once per variant. When at least three users have logged propensities it
    # performs a fixed 1,000-draw clustered bootstrap, regardless of the
    # ablation plan's own bootstrap_samples setting.
    logged_users = {
        event.user_id
        for event in events
        if event.occurred_at <= plan.as_of and event.propensity is not None
    }
    logged_report_work = len(events)
    if len(logged_users) >= MIN_USERS_FOR_INTERVAL:
        logged_report_work += DEFAULT_RESAMPLES * (
            len(logged_users) + DEFAULT_RESAMPLES.bit_length()
        )
    per_variant_evaluation = (
        len(events)
        + len(articles)
        + logged_report_work
        + len(active_users) * (event_profile_work + len(articles) + candidate_topic_work)
    )
    if variants * per_variant_evaluation > MAX_EVALUATION_WORK:
        raise ValueError("experiment exceeds its evaluation work limit")
    # _resampled_metrics visits every selected user's five utility/diversity
    # values, every exposed slate ID, and every eligible catalog ID. Its Gini
    # statistic also sorts at most one exposure count per article. Counting
    # metrics alone would permit a large-catalog workload far beyond the cap.
    users = len(active_users)
    slate_width = min(plan.k, plan.config.size, len(articles))
    exposed_upper = min(len(articles), users * slate_width)
    per_variant_draw = (
        users * (1 + len(_USER_MEAN_METRICS) + slate_width + len(articles))
        + exposed_upper * (exposed_upper.bit_length() + 3)
        + len(_BOOTSTRAP_METRICS)
    )
    bootstrap_work = plan.bootstrap_samples * variants * per_variant_draw
    if bootstrap_work > MAX_BOOTSTRAP_WORK:
        raise ValueError("experiment exceeds its bootstrap work limit")
    configs = {"baseline": plan.config}
    for name in plan.ablations:
        try:
            configs[name] = _variant_config(name, plan.config, len(articles), plan.k)
        except ValueError as error:
            raise ValueError(f"ablation {name} produces invalid configuration: {error}") from error
    samples: dict[str, EvaluationSamples] = {}
    for name, config in configs.items():
        samples[name] = evaluate_leave_last_out_samples(
            articles, events, as_of=plan.as_of, config=config, k=plan.k
        )
    holdouts = tuple(
        (user.user_id, user.holdout_article_id, user.holdout_at)
        for user in samples["baseline"].users
    )
    if any(
        tuple((user.user_id, user.holdout_article_id, user.holdout_at) for user in sample.users)
        != holdouts
        for sample in samples.values()
    ):
        raise RuntimeError("ablation variants did not retain identical temporal holdouts")
    generator = random.Random(plan.seed)
    user_count = len(holdouts)
    draws: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in _BOOTSTRAP_METRICS} for name in configs
    }
    for _ in range(plan.bootstrap_samples):
        indices = tuple(generator.randrange(user_count) for _ in range(user_count))
        for name, sample in samples.items():
            values = _resampled_metrics(sample, indices)
            for metric in _BOOTSTRAP_METRICS:
                draws[name][metric].append(values[metric])
    outcomes: dict[str, dict[str, object]] = {}
    points: dict[str, dict[str, float]] = {}
    for name, sample in samples.items():
        report = sample.report()
        points[name] = _point_metrics(report)
        outcomes[name] = {
            "config": configs[name].to_dict(),
            "evaluation": report.to_dict(),
            "confidence_intervals": {
                metric: _interval(
                    points[name][metric],
                    draws[name][metric],
                    confidence=plan.confidence,
                    observations=user_count,
                ).to_dict()
                for metric in _BOOTSTRAP_METRICS
            },
        }
    paired: dict[str, dict[str, object]] = {}
    for name in plan.ablations:
        paired[name] = {
            metric: _interval(
                points["baseline"][metric] - points[name][metric],
                [
                    left - right
                    for left, right in zip(
                        draws["baseline"][metric], draws[name][metric], strict=True
                    )
                ],
                confidence=plan.confidence,
                observations=user_count,
            ).to_dict()
            for metric in _BOOTSTRAP_METRICS
        }
    return ExperimentRun(
        runner_protocol=RUNNER_PROTOCOL,
        experiment_id=_experiment_id(articles_source, events_source, plan_source),
        articles_sha256=_source_digest(articles_source),
        events_sha256=_source_digest(events_source),
        plan_sha256=_source_digest(plan_source),
        dataset_sha256=dataset_fingerprint(articles, events),
        plan=plan,
        outcomes=outcomes,
        paired_deltas=paired,
    )


def write_experiment_record(directory: str | Path, run: ExperimentRun) -> Path:
    """Atomically add one immutable content-addressed registry file, never replace."""

    if not isinstance(run, ExperimentRun):
        raise ValueError("run must be an ExperimentRun")
    data = _record_bytes(run)
    registry = Path(directory)
    if registry.exists() and not registry.is_dir():
        raise ValueError("experiment registry must be a directory")
    registry.mkdir(parents=True, exist_ok=True)
    destination = registry / f"{run.experiment_id}.json"
    descriptor, staged_name = tempfile.mkstemp(prefix=".experiment-", suffix=".tmp", dir=registry)
    staged = Path(staged_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, destination)
        if os.name == "posix":
            parent_descriptor = os.open(registry, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        return destination
    finally:
        staged.unlink(missing_ok=True)


def verify_experiment_record(
    path: str | Path,
    articles_source: bytes,
    events_source: bytes,
    plan_source: bytes,
    *,
    articles_name: str = "articles.json",
    events_name: str = "events.json",
) -> bool:
    """Replay exact snapshots and compare the complete canonical registered bytes."""

    expected = run_ablation_experiment(
        articles_source,
        events_source,
        plan_source,
        articles_name=articles_name,
        events_name=events_name,
    )
    record = Path(path)
    if record.name != f"{expected.experiment_id}.json":
        return False
    with record.open("rb") as stream:
        data = stream.read(MAX_RECORD_BYTES + 1)
    return len(data) <= MAX_RECORD_BYTES and data == _record_bytes(expected)
