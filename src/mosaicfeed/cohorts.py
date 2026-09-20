"""Point-in-time, declared-cohort diagnostics for a held-out feed evaluation.

These are observational group comparisons. Cohorts must be supplied by the
caller; this module does not infer demographic attributes or causal fairness.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

from mosaicfeed.benchmark import ConfidenceInterval, bootstrap_mean, dataset_fingerprint
from mosaicfeed.calibration import calibration_error, reader_distribution
from mosaicfeed.config import FeedConfig
from mosaicfeed.io import load_articles_bytes, load_events_bytes, load_json_text
from mosaicfeed.metrics import EvaluationSamples, UserEvaluation, evaluate_leave_last_out_samples
from mosaicfeed.models import Article, Event
from mosaicfeed.profile import build_profile

COHORT_METRICS = (
    "ndcg",
    "hit_rate",
    "reciprocal_rank",
    "intra_list_diversity",
    "source_diversity",
)
MAX_COHORTS = 64
MAX_USERS = 10_000
MAX_BOOTSTRAP_WORK = 8_000_000
MAX_COHORT_INPUT_BYTES = 4 * 1024 * 1024
MAX_AUDIT_SOURCE_BYTES = 16 * 1024 * 1024
MAX_AUDIT_ROWS = 100_000
MAX_EVALUATION_WORK = 5_000_000


@dataclass(frozen=True, slots=True)
class CohortSummary:
    active_users: int
    users: int
    users_skipped: int
    calibration_observations: int
    metrics: Mapping[str, ConfidenceInterval]
    topic_calibration: ConfidenceInterval | None

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not int or value < 0
                for value in (
                    self.active_users,
                    self.users,
                    self.users_skipped,
                    self.calibration_observations,
                )
            )
            or self.active_users != self.users + self.users_skipped
        ):
            raise ValueError("cohort counts are inconsistent")
        if self.users == 0 or self.calibration_observations > self.users:
            raise ValueError("cohort has no evaluable users or invalid calibration count")
        if not isinstance(self.metrics, Mapping) or set(self.metrics) != set(COHORT_METRICS):
            raise ValueError("cohort metrics must contain every declared metric")
        intervals = dict(self.metrics)
        if any(
            not isinstance(interval, ConfidenceInterval) or interval.observations != self.users
            for interval in intervals.values()
        ):
            raise ValueError("cohort metric intervals have inconsistent observations")
        if any(
            not 0.0 <= interval.lower <= interval.upper <= 1.0 or not 0.0 <= interval.mean <= 1.0
            for interval in intervals.values()
        ):
            raise ValueError("cohort utility and diversity metrics must be in [0, 1]")
        confidence = intervals[COHORT_METRICS[0]].confidence
        if any(interval.confidence != confidence for interval in intervals.values()):
            raise ValueError("cohort metric intervals must share one confidence level")
        if self.topic_calibration is not None and (
            not isinstance(self.topic_calibration, ConfidenceInterval)
            or self.calibration_observations == 0
            or self.topic_calibration.observations != self.calibration_observations
            or self.topic_calibration.confidence != confidence
        ):
            raise ValueError("cohort calibration interval has inconsistent observations")
        if self.topic_calibration is not None and (
            self.topic_calibration.lower < 0.0 or self.topic_calibration.mean < 0.0
        ):
            raise ValueError("cohort calibration error must be non-negative")
        object.__setattr__(self, "metrics", MappingProxyType(intervals))

    def to_dict(self) -> dict[str, object]:
        return {
            "active_users": self.active_users,
            "users": self.users,
            "users_skipped": self.users_skipped,
            "calibration_observations": self.calibration_observations,
            "metrics": {key: self.metrics[key].to_dict() for key in COHORT_METRICS},
            "topic_calibration": (
                None if self.topic_calibration is None else self.topic_calibration.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class CohortAudit:
    dataset_sha256: str
    cohorts_sha256: str
    as_of: datetime
    k: int
    users_skipped: int
    minimum_group_size: int
    bootstrap_samples: int
    confidence: float
    seed: int
    cohorts: Mapping[str, CohortSummary]
    max_minus_min: Mapping[str, float | None]

    def __post_init__(self) -> None:
        for name in ("dataset_sha256", "cohorts_sha256"):
            digest = getattr(self, name)
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if (
            not isinstance(self.as_of, datetime)
            or self.as_of.tzinfo is None
            or self.as_of.utcoffset() is None
        ):
            raise ValueError("audit as_of must be timezone-aware")
        _positive_int(self.k, "k")
        _positive_int(self.minimum_group_size, "minimum_group_size")
        _positive_int(self.bootstrap_samples, "bootstrap_samples")
        if type(self.users_skipped) is not int or self.users_skipped < 0:
            raise ValueError("audit users_skipped must be a non-negative integer")
        if type(self.seed) is not int:
            raise ValueError("audit seed must be an integer")
        if (
            type(self.confidence) is not float
            or not math.isfinite(self.confidence)
            or not (0.0 < self.confidence < 1.0)
        ):
            raise ValueError("audit confidence must be in (0, 1)")
        if (
            not isinstance(self.cohorts, Mapping)
            or not self.cohorts
            or any(
                not isinstance(label, str)
                or not label
                or len(label) > 64
                or label != label.strip()
                or any(not char.isprintable() for char in label)
                or not isinstance(summary, CohortSummary)
                for label, summary in self.cohorts.items()
            )
            or len(self.cohorts) > MAX_COHORTS
        ):
            raise ValueError("audit must contain declared cohort summaries")
        if any(
            summary.users < self.minimum_group_size
            or (summary.calibration_observations >= self.minimum_group_size)
            != (summary.topic_calibration is not None)
            or any(interval.confidence != self.confidence for interval in summary.metrics.values())
            or (
                summary.topic_calibration is not None
                and summary.topic_calibration.confidence != self.confidence
            )
            for summary in self.cohorts.values()
        ):
            raise ValueError("cohort interval or minimum-size metadata is inconsistent")
        if sum(summary.users_skipped for summary in self.cohorts.values()) != self.users_skipped:
            raise ValueError("audit skipped-user count is inconsistent")
        expected = {*COHORT_METRICS, "topic_calibration"}
        if not isinstance(self.max_minus_min, Mapping) or set(self.max_minus_min) != expected:
            raise ValueError("audit gaps must cover all cohort metrics")
        gaps = dict(self.max_minus_min)
        if any(
            value is not None
            and (type(value) is not float or not math.isfinite(value) or value < 0)
            for value in gaps.values()
        ):
            raise ValueError("audit gaps must be finite and non-negative")
        for metric in COHORT_METRICS:
            points = [summary.metrics[metric].mean for summary in self.cohorts.values()]
            expected_gap = max(points) - min(points) if len(points) >= 2 else None
            actual = gaps[metric]
            if (expected_gap is None) != (actual is None) or (
                expected_gap is not None
                and actual is not None
                and not math.isclose(actual, expected_gap, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError(f"audit {metric} gap does not match cohort means")
        calibration = [
            summary.topic_calibration.mean
            for summary in self.cohorts.values()
            if summary.topic_calibration is not None
        ]
        expected_calibration_gap = (
            max(calibration) - min(calibration)
            if len(calibration) >= 2 and len(calibration) == len(self.cohorts)
            else None
        )
        actual_calibration_gap = gaps["topic_calibration"]
        if (expected_calibration_gap is None) != (actual_calibration_gap is None) or (
            expected_calibration_gap is not None
            and actual_calibration_gap is not None
            and not math.isclose(
                actual_calibration_gap, expected_calibration_gap, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise ValueError("audit calibration gap does not match cohort means")
        object.__setattr__(self, "cohorts", MappingProxyType(dict(self.cohorts)))
        object.__setattr__(self, "max_minus_min", MappingProxyType(gaps))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "dataset_sha256": self.dataset_sha256,
            "cohorts_sha256": self.cohorts_sha256,
            "as_of": self.as_of.isoformat(),
            "k": self.k,
            "users_skipped": self.users_skipped,
            "minimum_group_size": self.minimum_group_size,
            "bootstrap_samples": self.bootstrap_samples,
            "confidence": self.confidence,
            "seed": self.seed,
            "cohorts": {key: value.to_dict() for key, value in sorted(self.cohorts.items())},
            "max_minus_min": dict(sorted(self.max_minus_min.items())),
        }


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _checked_cohorts(
    values: Mapping[str, str], active_users: set[str]
) -> tuple[dict[str, str], str]:
    if not isinstance(values, Mapping):
        raise ValueError("cohorts must be a user-to-cohort mapping")
    if set(values) != active_users:
        raise ValueError("cohort mapping must contain exactly the users active at as_of")
    result: dict[str, str] = {}
    for user, cohort in values.items():
        if not isinstance(user, str) or not user or not isinstance(cohort, str):
            raise ValueError("cohort user ids and labels must be non-empty strings")
        if (
            not cohort
            or len(cohort) > 64
            or cohort != cohort.strip()
            or any(not char.isprintable() for char in cohort)
        ):
            raise ValueError("cohort labels must be short printable strings")
        result[user] = cohort
    if len(set(result.values())) > MAX_COHORTS:
        raise ValueError("too many declared cohorts")
    canonical = json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return result, digest


def load_declared_cohorts(path: str | Path) -> dict[str, str]:
    """Read a bounded, strict JSON object from user ID to declared cohort."""

    with Path(path).open("rb") as source:
        data = source.read(MAX_COHORT_INPUT_BYTES + 1)
    if len(data) > MAX_COHORT_INPUT_BYTES:
        raise ValueError("cohort mapping exceeds the input size limit")
    try:
        payload = load_json_text(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("cohort mapping must be strict UTF-8 JSON") from error
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in payload.items()
    ):
        raise ValueError("cohort mapping must be a JSON object of string labels")
    return payload


def load_audit_sources(
    article_path: str | Path, event_path: str | Path
) -> tuple[list[Article], list[Event]]:
    """Load bounded immutable source snapshots for the audit CLI."""

    def snapshot(path: str | Path) -> bytes:
        with Path(path).open("rb") as source:
            raw = source.read(MAX_AUDIT_SOURCE_BYTES + 1)
        if len(raw) > MAX_AUDIT_SOURCE_BYTES:
            raise ValueError("cohort audit source exceeds the input size limit")
        return raw

    articles = load_articles_bytes(snapshot(article_path), source_name=str(article_path))
    events = load_events_bytes(snapshot(event_path), source_name=str(event_path))
    return articles, events


def _calibration_values(
    samples: EvaluationSamples,
    articles: Sequence[Article],
    events: Sequence[Event],
    config: FeedConfig,
) -> dict[str, float | None]:
    by_user: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        by_user[event.user_id].append(event)
    article_map = {article.id: article for article in articles}
    result: dict[str, float | None] = {}
    for user in samples.users:
        # Match the evaluator's strict pre-holdout barrier, including ties.
        training = [event for event in by_user[user.user_id] if event.occurred_at < user.holdout_at]
        profile = build_profile(
            user.user_id,
            training,
            article_map,
            as_of=user.holdout_at,
            config=config,
        )
        result[user.user_id] = (
            calibration_error(profile, user.ranked_ids, article_map)
            if user.ranked_ids and reader_distribution(profile)
            else None
        )
    return result


def audit_cohorts(
    articles: Sequence[Article],
    events: Sequence[Event],
    cohorts: Mapping[str, str],
    *,
    as_of: datetime,
    config: FeedConfig,
    k: int | None = None,
    minimum_group_size: int = 5,
    bootstrap_samples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 17,
) -> CohortAudit:
    """Audit user-mean outcomes across *declared* cohorts on identical holdouts.

    A group below ``minimum_group_size`` aborts the entire audit instead of
    silently omitting an inconvenient cohort. Calibration excludes readers
    without positive pre-holdout topic history or an exposed slate; ``None``
    denotes missing evidence rather than a perfect (zero-error) slate.
    """

    minimum = _positive_int(minimum_group_size, "minimum_group_size")
    draws = _positive_int(bootstrap_samples, "bootstrap_samples")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    # Reuse the public bootstrap contract for confidence validation.
    bootstrap_mean((), samples=draws, confidence=confidence, seed=seed)
    if not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    if len(articles) > MAX_AUDIT_ROWS or len(events) > MAX_AUDIT_ROWS:
        raise ValueError("cohort audit exceeds the article/event row limit")
    active = {event.user_id for event in events if event.occurred_at <= as_of}
    if not active:
        raise ValueError("cohort audit has no active users")
    if len(active) > MAX_USERS:
        raise ValueError("cohort audit exceeds the user limit")
    if len(active) * (len(events) + len(articles)) > MAX_EVALUATION_WORK:
        raise ValueError("cohort audit exceeds the evaluation work limit")
    declared, cohorts_digest = _checked_cohorts(cohorts, active)
    if draws * len(active) * (len(COHORT_METRICS) + 1) > MAX_BOOTSTRAP_WORK:
        raise ValueError("cohort bootstrap exceeds the work limit")
    samples = evaluate_leave_last_out_samples(articles, events, as_of=as_of, config=config, k=k)
    grouped: dict[str, list[UserEvaluation]] = defaultdict(list)
    for user in samples.users:
        grouped[declared[user.user_id]].append(user)
    for label in set(declared.values()):
        if len(grouped[label]) < minimum:
            raise ValueError(f"cohort {label!r} has fewer than minimum_group_size evaluable users")
    calibration = _calibration_values(samples, articles, events, config)
    active_counts: dict[str, int] = defaultdict(int)
    for label in declared.values():
        active_counts[label] += 1
    summaries: dict[str, CohortSummary] = {}
    for label in sorted(grouped):
        users = grouped[label]
        # A stable label-specific local seed makes each interval invariant to
        # input ordering and to adding another cohort to the audit.
        label_seed = int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "big")
        intervals = {
            metric: bootstrap_mean(
                [float(getattr(user, metric)) for user in users],
                samples=draws,
                confidence=confidence,
                seed=label_seed,
            )
            for metric in COHORT_METRICS
        }
        calibration_observed = [
            value for user in users if (value := calibration[user.user_id]) is not None
        ]
        calibration_interval = (
            bootstrap_mean(
                calibration_observed,
                samples=draws,
                confidence=confidence,
                seed=label_seed,
            )
            if len(calibration_observed) >= minimum
            else None
        )
        summaries[label] = CohortSummary(
            active_counts[label],
            len(users),
            active_counts[label] - len(users),
            len(calibration_observed),
            intervals,
            calibration_interval,
        )
    gaps: dict[str, float | None] = {}
    for metric in COHORT_METRICS:
        points = [summary.metrics[metric].mean for summary in summaries.values()]
        gaps[metric] = max(points) - min(points) if len(points) >= 2 else None
    calibration_points = [
        summary.topic_calibration.mean
        for summary in summaries.values()
        if summary.topic_calibration is not None
    ]
    gaps["topic_calibration"] = (
        max(calibration_points) - min(calibration_points)
        if len(calibration_points) >= 2 and len(calibration_points) == len(summaries)
        else None
    )
    return CohortAudit(
        dataset_sha256=dataset_fingerprint(articles, events),
        cohorts_sha256=cohorts_digest,
        as_of=as_of,
        k=samples.k,
        users_skipped=samples.users_skipped,
        minimum_group_size=minimum,
        bootstrap_samples=draws,
        confidence=confidence,
        seed=seed,
        cohorts=summaries,
        max_minus_min=gaps,
    )
