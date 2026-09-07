"""Candidate-lossless MIND impression I/O and official-style ranking metrics.

The formulas mirror the public MIND evaluator while keeping tie handling and
invalid-input behavior explicit.  No dataset is downloaded by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mosaicfeed.datasets import MindCandidate, MindImpression
from mosaicfeed.io import load_json_text, parse_datetime, write_json


def _records(path: str | Path, kind: str) -> list[dict[str, Any]]:
    source = Path(path)
    value = load_json_text(source.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(record, dict) for record in value):
        raise ValueError(f"{source} must contain a list of {kind} objects")
    return cast(list[dict[str, Any]], value)


def _strict(record: Mapping[str, object], fields: set[str], kind: str) -> None:
    actual = set(record)
    if actual != fields:
        unknown = actual - fields
        missing = fields - actual
        details = []
        if unknown:
            details.append(f"unknown fields: {', '.join(sorted(unknown))}")
        if missing:
            details.append(f"missing fields: {', '.join(sorted(missing))}")
        raise ValueError(f"invalid {kind} ({'; '.join(details)})")


def impression_to_dict(impression: MindImpression) -> dict[str, object]:
    """Serialize one impression without dropping order, labels, user, or time."""

    return {
        "impression_id": impression.impression_id,
        "user_id": impression.user_id,
        "occurred_at": impression.occurred_at.isoformat(),
        "candidates": [
            {"article_id": candidate.article_id, "clicked": candidate.clicked}
            for candidate in impression.candidates
        ],
    }


def write_mind_impressions(path: str | Path, impressions: Sequence[MindImpression]) -> None:
    """Write the version-independent canonical JSON interchange form."""

    _validate_impression_collection(impressions)
    write_json(path, [impression_to_dict(impression) for impression in impressions])


def _validate_impression_collection(impressions: Sequence[MindImpression]) -> None:
    seen: set[str] = set()
    for impression in impressions:
        if not isinstance(impression, MindImpression):
            raise ValueError("MIND impressions must contain MindImpression values")
        if impression.impression_id in seen:
            raise ValueError(f"duplicate MIND impression id: {impression.impression_id}")
        seen.add(impression.impression_id)


def load_mind_impressions(path: str | Path) -> tuple[MindImpression, ...]:
    """Load the strict JSON interchange form written by :func:`write_mind_impressions`."""

    result: list[MindImpression] = []
    seen: set[str] = set()
    for record in _records(path, "MIND impression"):
        _strict(record, {"impression_id", "user_id", "occurred_at", "candidates"}, "impression")
        raw_candidates = record["candidates"]
        if not isinstance(raw_candidates, list):
            raise ValueError("MIND impression candidates must be a list")
        candidates: list[MindCandidate] = []
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, dict):
                raise ValueError("MIND candidate must be an object")
            _strict(raw_candidate, {"article_id", "clicked"}, "candidate")
            article_id = raw_candidate["article_id"]
            clicked = raw_candidate["clicked"]
            if not isinstance(article_id, str):
                raise ValueError("MIND candidate article_id must be a string")
            if not isinstance(clicked, bool):
                raise ValueError("MIND candidate clicked must be a boolean")
            candidates.append(MindCandidate(article_id, clicked))
        impression_id = record["impression_id"]
        user_id = record["user_id"]
        if not isinstance(impression_id, str):
            raise ValueError("MIND impression_id must be a string")
        if not isinstance(user_id, str):
            raise ValueError("MIND impression user_id must be a string")
        impression = MindImpression(
            impression_id=impression_id,
            user_id=user_id,
            occurred_at=parse_datetime(record["occurred_at"], "occurred_at"),
            candidates=tuple(candidates),
        )
        if impression.impression_id in seen:
            raise ValueError(f"duplicate MIND impression id: {impression.impression_id}")
        seen.add(impression.impression_id)
        result.append(impression)
    return tuple(result)


def load_mind_scores(path: str | Path) -> dict[str, dict[str, float]]:
    """Load strict long-form score rows keyed by impression and candidate.

    Each record has exactly ``impression_id``, ``article_id`` and ``score``.
    Long form makes duplicate candidate scores detectable instead of letting a
    JSON object silently overwrite them.
    """

    result: dict[str, dict[str, float]] = {}
    for record in _records(path, "MIND score"):
        _strict(record, {"impression_id", "article_id", "score"}, "score")
        impression_id = record["impression_id"]
        article_id = record["article_id"]
        score = record["score"]
        if not isinstance(impression_id, str) or not impression_id.strip():
            raise ValueError("score impression_id must be a non-empty string")
        if not isinstance(article_id, str) or not article_id.strip():
            raise ValueError("score article_id must be a non-empty string")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("score must be a finite number")
        try:
            numeric_score = float(score)
        except OverflowError as error:
            raise ValueError("score must be a finite number") from error
        if not math.isfinite(numeric_score):
            raise ValueError("score must be a finite number")
        impression_scores = result.setdefault(impression_id, {})
        if article_id in impression_scores:
            raise ValueError(
                f"duplicate score for impression {impression_id}, article {article_id}"
            )
        impression_scores[article_id] = numeric_score
    return result


def _finite_score(score: object) -> float:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("scores must be finite numbers")
    try:
        value = float(score)
    except OverflowError as error:
        raise ValueError("scores must be finite numbers") from error
    if not math.isfinite(value):
        raise ValueError("scores must be finite numbers")
    return 0.0 if value == 0.0 else value


def _validate_labels_scores(labels: Sequence[bool], scores: Sequence[float]) -> tuple[float, ...]:
    if not labels or len(labels) != len(scores):
        raise ValueError("labels and scores must be non-empty and have equal length")
    if not all(isinstance(label, bool) for label in labels):
        raise ValueError("labels must be booleans")
    return tuple(_finite_score(score) for score in scores)


def impression_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Return pairwise ROC AUC, assigning half credit to tied scores."""

    normalized_scores = _validate_labels_scores(labels, scores)
    positives = [score for label, score in zip(labels, normalized_scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, normalized_scores, strict=True) if not label]
    if not positives or not negatives:
        raise ValueError("AUC requires at least one clicked and one unclicked candidate")
    credit = math.fsum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return credit / (len(positives) * len(negatives))


def impression_mrr(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Return the mean reciprocal rank of every clicked candidate."""

    normalized_scores = _validate_labels_scores(labels, scores)
    positives = sum(labels)
    if positives == 0:
        raise ValueError("MRR requires at least one clicked candidate")
    order = sorted(
        range(len(normalized_scores)), key=lambda index: (-normalized_scores[index], index)
    )
    return math.fsum(1.0 / rank for rank, index in enumerate(order, 1) if labels[index]) / positives


def impression_ndcg(labels: Sequence[bool], scores: Sequence[float], k: int) -> float:
    """Return binary nDCG@k using source order as the deterministic score-tie break."""

    normalized_scores = _validate_labels_scores(labels, scores)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    positives = sum(labels)
    if positives == 0:
        raise ValueError("nDCG requires at least one clicked candidate")
    order = sorted(
        range(len(normalized_scores)), key=lambda index: (-normalized_scores[index], index)
    )
    dcg = math.fsum(
        1.0 / math.log2(rank + 1) for rank, index in enumerate(order[:k], 1) if labels[index]
    )
    ideal = math.fsum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, positives) + 1))
    return dcg / ideal


@dataclass(frozen=True, slots=True)
class MindEvaluationReport:
    """Macro-averaged official-style MIND metrics with a replay fingerprint."""

    impressions: int
    candidates: int
    auc: float
    mrr: float
    ndcg: Mapping[int, float]
    impression_sha256: str
    score_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "impressions": self.impressions,
            "candidates": self.candidates,
            "auc": self.auc,
            "mrr": self.mrr,
            "ndcg": {str(k): value for k, value in sorted(self.ndcg.items())},
            "impression_sha256": self.impression_sha256,
            "score_sha256": self.score_sha256,
            "tie_break": "source candidate order",
            "averaging": "macro over impressions",
        }


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_mind_impressions(
    impressions: Sequence[MindImpression],
    scores: Mapping[str, Mapping[str, float]],
    *,
    cutoffs: Sequence[int] = (5, 10),
) -> MindEvaluationReport:
    """Evaluate exactly the candidates and labels retained from MIND.

    Score coverage is exact: every imported impression and candidate must have
    one score, and extra score rows are rejected.  An impression without both a
    positive and negative label is refused because official-style AUC is not
    defined for it.
    """

    if not impressions:
        raise ValueError("at least one MIND impression is required")
    if not cutoffs or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in cutoffs):
        raise ValueError("cutoffs must contain positive integers")
    normalized_cutoffs = tuple(sorted(set(cutoffs)))
    _validate_impression_collection(impressions)
    impression_ids = [impression.impression_id for impression in impressions]
    expected_ids = set(impression_ids)
    score_ids = set(scores)
    if score_ids != expected_ids:
        missing = sorted(expected_ids - score_ids)
        extra = sorted(score_ids - expected_ids)
        raise ValueError(f"score impression ids differ (missing={missing}, extra={extra})")

    auc_values: list[float] = []
    mrr_values: list[float] = []
    ndcg_values: dict[int, list[float]] = {k: [] for k in normalized_cutoffs}
    canonical_scores: list[dict[str, object]] = []
    candidate_count = 0
    for impression in impressions:
        impression_scores = scores[impression.impression_id]
        expected_articles = {candidate.article_id for candidate in impression.candidates}
        actual_articles = set(impression_scores)
        if actual_articles != expected_articles:
            missing = sorted(expected_articles - actual_articles)
            extra = sorted(actual_articles - expected_articles)
            raise ValueError(
                f"scores for impression {impression.impression_id} differ "
                f"(missing={missing}, extra={extra})"
            )
        labels = [candidate.clicked for candidate in impression.candidates]
        ordered_scores = [
            _finite_score(impression_scores[candidate.article_id])
            for candidate in impression.candidates
        ]
        auc_values.append(impression_auc(labels, ordered_scores))
        mrr_values.append(impression_mrr(labels, ordered_scores))
        for k in normalized_cutoffs:
            ndcg_values[k].append(impression_ndcg(labels, ordered_scores, k))
        candidate_count += len(labels)
        canonical_scores.append(
            {
                "impression_id": impression.impression_id,
                "scores": [
                    [candidate.article_id, ordered_scores[index]]
                    for index, candidate in enumerate(impression.candidates)
                ],
            }
        )

    return MindEvaluationReport(
        impressions=len(impressions),
        candidates=candidate_count,
        auc=math.fsum(auc_values) / len(auc_values),
        mrr=math.fsum(mrr_values) / len(mrr_values),
        ndcg={k: math.fsum(values) / len(values) for k, values in ndcg_values.items()},
        impression_sha256=_fingerprint(
            [impression_to_dict(impression) for impression in impressions]
        ),
        score_sha256=_fingerprint(canonical_scores),
    )
