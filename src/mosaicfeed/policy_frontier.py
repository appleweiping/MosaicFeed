"""Bounded, observational multi-policy comparisons on declared cohorts.

The Pareto set compares point estimates, not confidence intervals or causal
effects. Missing evidence never becomes a favourable zero-valued objective.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

from mosaicfeed.benchmark import dataset_fingerprint
from mosaicfeed.cohorts import (
    MAX_AUDIT_ROWS,
    MAX_USERS,
    CohortAudit,
    audit_cohorts,
)
from mosaicfeed.config import FeedConfig
from mosaicfeed.io import json_text, load_json_text, parse_datetime
from mosaicfeed.models import Article, Event, EventKind

MAX_POLICIES = 8
MAX_PLAN_BYTES = 64 * 1024
MAX_TOTAL_EVALUATION_WORK = 8_000_000
MAX_TOTAL_BOOTSTRAP_WORK = 8_000_000
MAX_REPORT_BYTES = 4 * 1024 * 1024
_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_OBJECTIVES = (
    "ndcg",
    "worst_cohort_ndcg",
    "ndcg_gap",
    "topic_calibration",
    "topic_calibration_gap",
)
_CONSTRAINT_FIELDS = (
    "minimum_ndcg",
    "minimum_worst_cohort_ndcg",
    "maximum_ndcg_gap",
    "maximum_topic_calibration",
    "maximum_topic_calibration_gap",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _positive(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class NamedPolicy:
    name: str
    config: FeedConfig

    def __post_init__(self) -> None:
        if type(self.name) is not str or _NAME.fullmatch(self.name) is None:
            raise ValueError("policy name must be a short lowercase identifier")
        if not isinstance(self.config, FeedConfig) or self.config.size > 100:
            raise ValueError("policy config must be a FeedConfig with size at most 100")

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "config": self.config.to_dict()}


@dataclass(frozen=True, slots=True)
class FrontierConstraints:
    minimum_ndcg: float | None = None
    minimum_worst_cohort_ndcg: float | None = None
    maximum_ndcg_gap: float | None = None
    maximum_topic_calibration: float | None = None
    maximum_topic_calibration_gap: float | None = None

    def __post_init__(self) -> None:
        for name in _CONSTRAINT_FIELDS:
            value = getattr(self, name)
            try:
                finite = math.isfinite(value) if type(value) in (int, float) else False
            except OverflowError:
                finite = False
            if value is not None and (
                type(value) not in (int, float)
                or not finite
                or value < 0.0
                or ((name.startswith("minimum") or name == "maximum_ndcg_gap") and value > 1.0)
            ):
                raise ValueError(f"{name} must be a finite non-negative bound")

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(value)
            for name in _CONSTRAINT_FIELDS
            if (value := getattr(self, name)) is not None
        }


@dataclass(frozen=True, slots=True)
class PolicyFrontierPlan:
    as_of: datetime
    k: int
    policies: tuple[NamedPolicy, ...]
    minimum_group_size: int = 5
    bootstrap_samples: int = 100
    confidence: float = 0.95
    seed: int = 17
    hard_constraints: FrontierConstraints = FrontierConstraints()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.as_of, datetime)
            or self.as_of.tzinfo is None
            or self.as_of.utcoffset() is None
        ):
            raise ValueError("as_of must be timezone-aware")
        _positive(self.k, "k", 100)
        _positive(self.minimum_group_size, "minimum_group_size", MAX_USERS)
        _positive(self.bootstrap_samples, "bootstrap_samples", 2_000)
        if type(self.policies) is not tuple or not 2 <= len(self.policies) <= MAX_POLICIES:
            raise ValueError("policies must be a tuple of 2 to 8 policies")
        if any(not isinstance(policy, NamedPolicy) for policy in self.policies):
            raise ValueError("policies must contain NamedPolicy values")
        if len({policy.name for policy in self.policies}) != len(self.policies):
            raise ValueError("policy names must be unique")
        if type(self.seed) is not int or not -(2**63) <= self.seed < 2**63:
            raise ValueError("seed must be a signed 64-bit integer")
        try:
            finite_confidence = (
                math.isfinite(self.confidence) if type(self.confidence) in (int, float) else False
            )
        except OverflowError:
            finite_confidence = False
        if not finite_confidence or not 0 < self.confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        if not isinstance(self.hard_constraints, FrontierConstraints):
            raise ValueError("hard_constraints must be FrontierConstraints")

    @classmethod
    def from_bytes(cls, data: bytes) -> PolicyFrontierPlan:
        if type(data) is not bytes or len(data) > MAX_PLAN_BYTES:
            raise ValueError("frontier plan exceeds the input size limit")
        try:
            value = load_json_text(data.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise ValueError("frontier plan must be strict UTF-8 JSON") from error
        if not isinstance(value, dict):
            raise ValueError("frontier plan must be an object")
        required = {"schema_version", "as_of", "k", "policies"}
        allowed = required | {
            "minimum_group_size",
            "bootstrap_samples",
            "confidence",
            "seed",
            "hard_constraints",
        }
        if (
            set(value) - allowed
            or not required <= set(value)
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
        ):
            raise ValueError("frontier plan requires schema_version 1 and known fields")
        items = value["policies"]
        if not isinstance(items, list) or not 2 <= len(items) <= MAX_POLICIES:
            raise ValueError("frontier plan requires 2 to 8 policies")
        policies: list[NamedPolicy] = []
        for item in items:
            if (
                not isinstance(item, dict)
                or set(item) != {"name", "config"}
                or not isinstance(item["config"], dict)
            ):
                raise ValueError("each policy requires only name and config")
            policies.append(NamedPolicy(item["name"], FeedConfig.from_mapping(item["config"])))
        bounds = value.get("hard_constraints", {})
        if not isinstance(bounds, dict) or set(bounds) - set(_CONSTRAINT_FIELDS):
            raise ValueError("hard_constraints has unknown fields")
        return cls(
            as_of=parse_datetime(value["as_of"], "as_of"),
            k=value["k"],
            policies=tuple(policies),
            minimum_group_size=value.get("minimum_group_size", 5),
            bootstrap_samples=value.get("bootstrap_samples", 100),
            confidence=value.get("confidence", 0.95),
            seed=value.get("seed", 17),
            hard_constraints=FrontierConstraints(**bounds),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "as_of": self.as_of.isoformat(),
            "k": self.k,
            "policies": [policy.to_dict() for policy in self.policies],
            "minimum_group_size": self.minimum_group_size,
            "bootstrap_samples": self.bootstrap_samples,
            "confidence": self.confidence,
            "seed": self.seed,
            "hard_constraints": self.hard_constraints.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PolicyOutcome:
    audit: CohortAudit
    objectives: Mapping[str, float | None]
    missing_objectives: tuple[str, ...]
    failed_constraints: tuple[str, ...]
    dominated_by: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.audit, CohortAudit)
            or not isinstance(self.objectives, Mapping)
            or set(self.objectives) != set(_OBJECTIVES)
        ):
            raise ValueError("outcome must contain all frontier objectives")
        if dict(self.objectives) != _objectives(self.audit):
            raise ValueError("outcome objectives do not match the cohort audit")
        if self.missing_objectives != tuple(
            name for name in _OBJECTIVES if self.objectives[name] is None
        ):
            raise ValueError("outcome missing-evidence flags do not match objectives")
        if any(name not in _CONSTRAINT_FIELDS for name in self.failed_constraints) or len(
            set(self.failed_constraints)
        ) != len(self.failed_constraints):
            raise ValueError("outcome has invalid constraint flags")
        if type(self.dominated_by) is not tuple or len(set(self.dominated_by)) != len(
            self.dominated_by
        ):
            raise ValueError("outcome has invalid dominance evidence")
        object.__setattr__(self, "objectives", MappingProxyType(dict(self.objectives)))

    def to_dict(self) -> dict[str, object]:
        return {
            "audit": self.audit.to_dict(),
            "objectives": dict(self.objectives),
            "missing_objectives": list(self.missing_objectives),
            "failed_constraints": list(self.failed_constraints),
            "dominated_by": list(self.dominated_by),
            "frontier_eligible": not self.missing_objectives and not self.failed_constraints,
        }


@dataclass(frozen=True, slots=True)
class PolicyFrontierReport:
    dataset_sha256: str
    cohorts_sha256: str
    plan_sha256: str
    holdouts_sha256: str
    source_sha256: Mapping[str, str] | None
    plan: PolicyFrontierPlan
    outcomes: Mapping[str, PolicyOutcome]
    pareto_frontier: tuple[str, ...]

    def __post_init__(self) -> None:
        for value in (
            self.dataset_sha256,
            self.cohorts_sha256,
            self.plan_sha256,
            self.holdouts_sha256,
        ):
            if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("report digests must be lowercase SHA-256 values")
        if not isinstance(self.plan, PolicyFrontierPlan) or self.plan_sha256 != _sha256(
            _canonical(self.plan.to_dict())
        ):
            raise ValueError("report plan digest does not match the plan")
        if (
            not isinstance(self.outcomes, Mapping)
            or set(self.outcomes) != {policy.name for policy in self.plan.policies}
            or any(
                not isinstance(outcome, PolicyOutcome)
                or outcome.audit.dataset_sha256 != self.dataset_sha256
                or outcome.audit.cohorts_sha256 != self.cohorts_sha256
                or outcome.audit.as_of != self.plan.as_of
                or outcome.audit.k != self.plan.k
                or outcome.audit.minimum_group_size != self.plan.minimum_group_size
                or outcome.audit.bootstrap_samples != self.plan.bootstrap_samples
                or outcome.audit.confidence != self.plan.confidence
                or outcome.audit.seed != self.plan.seed
                for outcome in self.outcomes.values()
            )
        ):
            raise ValueError("report outcomes do not match the shared plan and dataset")
        counts = {
            tuple(
                (label, item.active_users, item.users, item.users_skipped)
                for label, item in sorted(outcome.audit.cohorts.items())
            )
            for outcome in self.outcomes.values()
        }
        if len(counts) != 1:
            raise ValueError("report outcomes have different cohort holdout counts")
        eligible = {
            name
            for name, outcome in self.outcomes.items()
            if not outcome.missing_objectives and not outcome.failed_constraints
        }
        for name, outcome in self.outcomes.items():
            if outcome.failed_constraints != _failed_constraints(
                outcome.objectives, self.plan.hard_constraints
            ):
                raise ValueError("report constraint flags do not match objectives")
            dominators = tuple(
                other
                for other in sorted(eligible)
                if name in eligible
                and other != name
                and (
                    _dominates(self.outcomes[other].objectives, outcome.objectives)
                    or (self.outcomes[other].objectives == outcome.objectives and other < name)
                )
            )
            if outcome.dominated_by != dominators:
                raise ValueError("report Pareto dominance evidence is inconsistent")
        expected_frontier = tuple(
            name for name in sorted(eligible) if not self.outcomes[name].dominated_by
        )
        if self.pareto_frontier != expected_frontier:
            raise ValueError("report Pareto frontier is inconsistent")
        if self.source_sha256 is not None and (
            not isinstance(self.source_sha256, Mapping)
            or set(self.source_sha256) != {"articles", "events", "cohorts", "plan"}
            or any(
                type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                for digest in self.source_sha256.values()
            )
        ):
            raise ValueError("report source digests must be complete lowercase SHA-256 values")
        object.__setattr__(self, "outcomes", MappingProxyType(dict(self.outcomes)))
        if self.source_sha256 is not None:
            object.__setattr__(self, "source_sha256", MappingProxyType(dict(self.source_sha256)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "mosaicfeed.declared_cohort_policy_frontier",
            "dataset_sha256": self.dataset_sha256,
            "cohorts_sha256": self.cohorts_sha256,
            "plan_sha256": self.plan_sha256,
            "holdouts_sha256": self.holdouts_sha256,
            "source_sha256": None if self.source_sha256 is None else dict(self.source_sha256),
            "utility_metric": "ndcg",
            "objective_directions": {
                "ndcg": "maximize",
                "worst_cohort_ndcg": "maximize",
                "ndcg_gap": "minimize",
                "topic_calibration": "minimize",
                "topic_calibration_gap": "minimize",
            },
            "plan": self.plan.to_dict(),
            "outcomes": {
                name: outcome.to_dict() for name, outcome in sorted(self.outcomes.items())
            },
            "pareto_frontier": list(self.pareto_frontier),
        }


def _holdout_digest(articles: Sequence[Article], events: Sequence[Event], as_of: datetime) -> str:
    article_ids = {item.id for item in articles}
    selected: dict[str, Event] = {}
    for event in events:
        if (
            event.occurred_at > as_of
            or event.kind not in (EventKind.CLICK, EventKind.LIKE)
            or event.article_id not in article_ids
        ):
            continue
        previous = selected.get(event.user_id)
        if previous is None or (event.occurred_at, event.article_id) > (
            previous.occurred_at,
            previous.article_id,
        ):
            selected[event.user_id] = event
    return _sha256(
        _canonical(
            [
                [user, item.article_id, item.occurred_at.isoformat()]
                for user, item in sorted(selected.items())
            ]
        )
    )


def _objectives(audit: CohortAudit) -> dict[str, float | None]:
    summaries = tuple(audit.cohorts.values())
    users = sum(item.users for item in summaries)
    ndcg = math.fsum(item.users * item.metrics["ndcg"].mean for item in summaries) / users
    calibration = [item for item in summaries if item.topic_calibration is not None]
    calibration_count = sum(item.calibration_observations for item in calibration)
    return {
        "ndcg": ndcg,
        "worst_cohort_ndcg": min(item.metrics["ndcg"].mean for item in summaries),
        "ndcg_gap": audit.max_minus_min["ndcg"],
        "topic_calibration": (
            math.fsum(
                item.calibration_observations * item.topic_calibration.mean
                for item in calibration
                if item.topic_calibration is not None
            )
            / calibration_count
            if len(calibration) == len(summaries) and calibration_count
            else None
        ),
        "topic_calibration_gap": audit.max_minus_min["topic_calibration"],
    }


def _failed_constraints(
    values: Mapping[str, float | None], bounds: FrontierConstraints
) -> tuple[str, ...]:
    lookup = {
        "minimum_ndcg": ("ndcg", True),
        "minimum_worst_cohort_ndcg": ("worst_cohort_ndcg", True),
        "maximum_ndcg_gap": ("ndcg_gap", False),
        "maximum_topic_calibration": ("topic_calibration", False),
        "maximum_topic_calibration_gap": ("topic_calibration_gap", False),
    }
    failed = []
    for name, (objective, minimum) in lookup.items():
        bound = getattr(bounds, name)
        value = values[objective]
        if bound is not None and (value is None or (value < bound if minimum else value > bound)):
            failed.append(name)
    return tuple(failed)


def _dominates(left: Mapping[str, float | None], right: Mapping[str, float | None]) -> bool:
    direction = (1, 1, -1, -1, -1)
    comparisons: list[tuple[float, float]] = []
    for name, sign in zip(_OBJECTIVES, direction, strict=True):
        left_value, right_value = left[name], right[name]
        if left_value is None or right_value is None:
            raise ValueError("Pareto comparison requires complete evidence")
        comparisons.append((left_value * sign, right_value * sign))
    return all(a >= b for a, b in comparisons) and any(a > b for a, b in comparisons)


def compare_cohort_policies(
    articles: Sequence[Article],
    events: Sequence[Event],
    cohorts: Mapping[str, str],
    plan: PolicyFrontierPlan,
    *,
    source_sha256: Mapping[str, str] | None = None,
) -> PolicyFrontierReport:
    """Evaluate 2-8 declared configurations on one frozen temporal population.

    The optional ``source_sha256`` is caller-declared metadata, not recomputed
    from Python objects. The CLI alone computes it from each bounded raw byte
    snapshot before parsing; ``dataset_sha256`` and ``plan_sha256`` are always
    recomputed from the semantic inputs here.
    """

    if not isinstance(plan, PolicyFrontierPlan):
        raise ValueError("plan must be PolicyFrontierPlan")
    if not isinstance(cohorts, Mapping):
        raise ValueError("cohorts must be a declared user-to-group mapping")
    if len(articles) > MAX_AUDIT_ROWS or len(events) > MAX_AUDIT_ROWS:
        raise ValueError("frontier source exceeds article/event row limit")
    if len(cohorts) > MAX_USERS:
        raise ValueError("frontier exceeds declared user limit")
    articles = tuple(articles)
    events = tuple(events)
    cohorts = dict(cohorts)
    if any(not isinstance(item, Article) for item in articles) or any(
        not isinstance(item, Event) for item in events
    ):
        raise ValueError("frontier sources must contain Article and Event values")
    active = {event.user_id for event in events if event.occurred_at <= plan.as_of}
    count = len(plan.policies)
    if (
        len(active) > MAX_USERS
        or count * len(active) * (len(events) + len(articles)) > MAX_TOTAL_EVALUATION_WORK
    ):
        raise ValueError("frontier exceeds total evaluation work limit")
    if count * plan.bootstrap_samples * len(active) * 6 > MAX_TOTAL_BOOTSTRAP_WORK:
        raise ValueError("frontier exceeds total bootstrap work limit")
    if source_sha256 is not None and (
        not isinstance(source_sha256, Mapping)
        or set(source_sha256) != {"articles", "events", "cohorts", "plan"}
        or any(
            type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in source_sha256.values()
        )
    ):
        raise ValueError("source_sha256 must have four lowercase SHA-256 digests")
    audits = {
        policy.name: audit_cohorts(
            articles,
            events,
            cohorts,
            as_of=plan.as_of,
            config=policy.config,
            k=plan.k,
            minimum_group_size=plan.minimum_group_size,
            bootstrap_samples=plan.bootstrap_samples,
            confidence=plan.confidence,
            seed=plan.seed,
        )
        for policy in plan.policies
    }
    first = next(iter(audits.values()))
    counts = {
        label: (item.active_users, item.users, item.users_skipped)
        for label, item in first.cohorts.items()
    }
    if any(
        audit.dataset_sha256 != first.dataset_sha256
        or audit.cohorts_sha256 != first.cohorts_sha256
        or audit.k != first.k
        or {
            label: (item.active_users, item.users, item.users_skipped)
            for label, item in audit.cohorts.items()
        }
        != counts
        for audit in audits.values()
    ):
        raise ValueError("policy evaluations do not share the same cohort holdouts")
    values = {name: _objectives(audit) for name, audit in audits.items()}
    eligible = {
        name
        for name, objective in values.items()
        if all(value is not None for value in objective.values())
        and not _failed_constraints(objective, plan.hard_constraints)
    }
    dominated: dict[str, tuple[str, ...]] = {}
    for name in sorted(eligible):
        dominated[name] = tuple(
            other
            for other in sorted(eligible)
            if other != name
            and (
                _dominates(values[other], values[name])
                or (values[other] == values[name] and other < name)
            )
        )
    outcomes = {
        name: PolicyOutcome(
            audit,
            values[name],
            tuple(key for key in _OBJECTIVES if values[name][key] is None),
            _failed_constraints(values[name], plan.hard_constraints),
            dominated.get(name, ()),
        )
        for name, audit in audits.items()
    }
    return PolicyFrontierReport(
        dataset_sha256=dataset_fingerprint(articles, events),
        cohorts_sha256=first.cohorts_sha256,
        plan_sha256=_sha256(_canonical(plan.to_dict())),
        holdouts_sha256=_holdout_digest(articles, events, plan.as_of),
        source_sha256=source_sha256,
        plan=plan,
        outcomes=outcomes,
        pareto_frontier=tuple(name for name in sorted(eligible) if not dominated[name]),
    )


def write_frontier_report(path: str | Path, report: PolicyFrontierReport) -> None:
    """Publish one complete report without replacing an existing destination."""

    destination = Path(path)
    body = json_text(report.to_dict()).encode("utf-8")
    if len(body) > MAX_REPORT_BYTES:
        raise ValueError("frontier report exceeds the output size limit")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    staging = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, destination)
    finally:
        with suppress(FileNotFoundError):
            staging.unlink()
