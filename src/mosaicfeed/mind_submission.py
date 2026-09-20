"""Strict bounded subset of MIND truth/prediction text and rank export.

The frozen public evaluator accepts several ambiguous or unsafe inputs. This
module intentionally rejects them; it does not claim complete protocol parity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from mosaicfeed.io import load_json_text
from mosaicfeed.mind import load_mind_impressions_bytes

PROTOCOL = "mosaicfeed.mind-submission-v1"
MAX_TEXT_BYTES = 8 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_ROWS = 10_000
MAX_LINE_BYTES = 8 * 1024
MAX_CANDIDATES_PER_ROW = 1_000
MAX_TOTAL_CANDIDATES = 200_000
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z", flags=re.ASCII)


def _sha(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def _bounded(source: bytes, maximum: int, label: str) -> None:
    if type(source) is not bytes or len(source) > maximum:
        raise ValueError(f"{label} must be a byte snapshot within byte limit {maximum}")


def _identifier(identifier: object, label: str) -> str:
    if type(identifier) is not str or _ID.fullmatch(identifier) is None:
        raise ValueError(f"{label} must be a canonical ASCII impression id")
    return identifier


def _text_rows(source: bytes, kind: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    _bounded(source, MAX_TEXT_BYTES, kind)
    if not source or not source.endswith(b"\n") or b"\r" in source:
        raise ValueError(f"{kind} must be nonempty and LF-terminated without CR")
    try:
        content = source.decode("utf-8", "strict")
    except UnicodeError as error:
        raise ValueError(f"{kind} must be UTF-8") from error
    if content.startswith("\ufeff"):
        raise ValueError(f"{kind} must not contain a BOM")
    lines = content[:-1].split("\n")
    if len(lines) > MAX_ROWS:
        raise ValueError(f"{kind} exceeds row limit")
    seen: set[str] = set()
    result: list[tuple[str, tuple[int, ...]]] = []
    total = 0
    for index, line in enumerate(lines, 1):
        if len(line.encode("utf-8")) > MAX_LINE_BYTES:
            raise ValueError(f"{kind} line {index} exceeds byte limit")
        identity, separator, array = line.partition(" ")
        if not separator or not array.startswith("["):
            raise ValueError(f"{kind} line {index} is not canonical ID/array text")
        _identifier(identity, f"{kind} line {index}")
        if identity in seen:
            raise ValueError(f"{kind} contains duplicate impression id {identity}")
        seen.add(identity)
        try:
            values = load_json_text(array)
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"{kind} line {index} must contain a JSON array") from error
        if not isinstance(values, list):
            raise ValueError(f"{kind} line {index} must contain a JSON array")
        if len(values) > MAX_CANDIDATES_PER_ROW:
            raise ValueError(f"{kind} line {index} exceeds candidate limit")
        if any(type(value) is not int for value in values):
            raise ValueError(f"{kind} line {index} values must be integers")
        if kind == "truth" and any(value not in (0, 1) for value in values):
            raise ValueError(f"{kind} line {index} labels must be binary")
        if kind == "prediction" and any(value < 0 for value in values):
            raise ValueError(f"{kind} line {index} ranks must be nonnegative integers")
        if json.dumps(values, separators=(",", ":")) != array:
            raise ValueError(f"{kind} line {index} array must be canonical compact JSON")
        total += len(values)
        if total > MAX_TOTAL_CANDIDATES:
            raise ValueError(f"{kind} exceeds total candidate limit")
        result.append((identity, tuple(values)))
    return tuple(result)


def _aligned(
    truth: tuple[tuple[str, tuple[int, ...]], ...],
    prediction: tuple[tuple[str, tuple[int, ...]], ...],
) -> None:
    if len(truth) != len(prediction):
        raise ValueError("truth/prediction row count differs")
    for (truth_id, labels), (pred_id, ranks) in zip(truth, prediction, strict=True):
        if truth_id != pred_id:
            raise ValueError("truth/prediction impression id or row order differs")
        if not labels:
            if ranks and set(ranks) != set(range(1, len(ranks) + 1)):
                raise ValueError(
                    f"masked prediction ranks must be a 1..N permutation for {truth_id}"
                )
            continue
        if not ranks:
            raise ValueError(f"masked prediction invalid for nonmasked truth {truth_id}")
        if len(labels) != len(ranks):
            raise ValueError(f"truth/prediction candidate count differs for {truth_id}")
        if set(ranks) != set(range(1, len(ranks) + 1)):
            raise ValueError(f"prediction ranks must be a 1..N permutation for {truth_id}")


def _ndcg(labels: tuple[int, ...], ranks: tuple[int, ...], cutoff: int) -> float:
    positives = sum(labels)
    actual = math.fsum(
        1 / math.log2(rank + 1)
        for label, rank in zip(labels, ranks, strict=True)
        if label and rank <= cutoff
    )
    ideal = math.fsum(1 / math.log2(rank + 1) for rank in range(1, min(positives, cutoff) + 1))
    return actual / ideal


@dataclass(frozen=True, slots=True)
class MindSubmissionReport:
    """Aggregates only; exact input bytes remain caller-owned and private."""

    rows: int
    masked_rows: int
    scored_rows: int
    candidates: int
    auc: float
    mrr: float
    ndcg_5: float
    ndcg_10: float
    truth_sha256: str
    prediction_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": PROTOCOL,
            "rows": self.rows,
            "masked_rows": self.masked_rows,
            "scored_rows": self.scored_rows,
            "candidates": self.candidates,
            "auc": self.auc,
            "mrr": self.mrr,
            "ndcg_5": self.ndcg_5,
            "ndcg_10": self.ndcg_10,
            "truth_sha256": self.truth_sha256,
            "prediction_sha256": self.prediction_sha256,
        }


def evaluate_mind_submission(truth_source: bytes, prediction_source: bytes) -> MindSubmissionReport:
    """Evaluate exactly aligned, untied rank permutations in a strict text subset."""

    _bounded(truth_source, MAX_TEXT_BYTES, "truth")
    _bounded(prediction_source, MAX_TEXT_BYTES, "prediction")
    truth = _text_rows(truth_source, "truth")
    prediction = _text_rows(prediction_source, "prediction")
    _aligned(truth, prediction)
    auc: list[float] = []
    mrr: list[float] = []
    ndcg5: list[float] = []
    ndcg10: list[float] = []
    candidates = 0
    for (_, labels), (_, ranks) in zip(truth, prediction, strict=True):
        if not labels:
            continue
        positives = sum(labels)
        negatives = len(labels) - positives
        if not positives or not negatives:
            raise ValueError("unmasked impression requires both classes")
        candidates += len(labels)
        # Each rank occurs exactly once. Traversing the ranked labels counts
        # one correct clicked-before-unclicked pair for each earlier positive.
        positives_seen = 0
        wins = 0
        for index in sorted(range(len(ranks)), key=ranks.__getitem__):
            if labels[index]:
                positives_seen += 1
            else:
                wins += positives_seen
        auc.append(wins / (positives * negatives))
        mrr.append(
            math.fsum(1 / rank for label, rank in zip(labels, ranks, strict=True) if label)
            / positives
        )
        ndcg5.append(_ndcg(labels, ranks, 5))
        ndcg10.append(_ndcg(labels, ranks, 10))
    if not auc:
        raise ValueError("no unmasked impressions available for evaluation")
    return MindSubmissionReport(
        rows=len(truth),
        masked_rows=len(truth) - len(auc),
        scored_rows=len(auc),
        candidates=candidates,
        auc=math.fsum(auc) / len(auc),
        mrr=math.fsum(mrr) / len(mrr),
        ndcg_5=math.fsum(ndcg5) / len(ndcg5),
        ndcg_10=math.fsum(ndcg10) / len(ndcg10),
        truth_sha256=_sha(truth_source),
        prediction_sha256=_sha(prediction_source),
    )


@dataclass(frozen=True, slots=True)
class PreparedMindPredictions:
    """Candidate-order ranks plus source digests, without copying raw labels."""

    prediction: bytes
    impressions_sha256: str
    scores_sha256: str
    truth_sha256: str | None
    rows: int
    masked_rows: int
    candidates: int

    def to_dict(self) -> dict[str, object]:
        _bounded(self.prediction, MAX_TEXT_BYTES, "prediction")
        return {
            "protocol": PROTOCOL,
            "impressions_sha256": self.impressions_sha256,
            "scores_sha256": self.scores_sha256,
            "truth_sha256": self.truth_sha256,
            "prediction_sha256": _sha(self.prediction),
            "rows": self.rows,
            "masked_rows": self.masked_rows,
            "candidates": self.candidates,
            "tie_break": "source candidate order",
        }


def _score_rows(source: bytes) -> dict[tuple[str, str], float]:
    _bounded(source, MAX_JSON_BYTES, "scores")
    try:
        records = load_json_text(source.decode("utf-8", "strict"))
    except UnicodeError as error:
        raise ValueError("scores must be UTF-8 JSON") from error
    if not isinstance(records, list) or len(records) > MAX_TOTAL_CANDIDATES:
        raise ValueError("scores must be a bounded list of score rows")
    result: dict[tuple[str, str], float] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"impression_id", "article_id", "score"}:
            raise ValueError("score rows require exactly impression_id, article_id, score")
        identity = _identifier(record["impression_id"], "score impression_id")
        article = record["article_id"]
        if type(article) is not str or not article or len(article.encode("utf-8")) > 256:
            raise ValueError("score candidate article_id must be bounded nonempty text")
        score = record["score"]
        if type(score) not in (int, float):
            raise ValueError("score must be a finite number")
        try:
            value = float(score)
        except OverflowError as error:
            raise ValueError("score must be a finite number") from error
        if not math.isfinite(value):
            raise ValueError("score must be a finite number")
        pair = (identity, article)
        if pair in result:
            raise ValueError("duplicate score candidate pair")
        result[pair] = value
    return result


def prepare_mind_predictions(
    impressions_source: bytes,
    scores_source: bytes,
    truth_source: bytes | None = None,
) -> PreparedMindPredictions:
    """Export deterministic ranks after exact candidate and optional truth joins."""

    _bounded(impressions_source, MAX_JSON_BYTES, "impressions")
    _bounded(scores_source, MAX_JSON_BYTES, "scores")
    if truth_source is not None:
        _bounded(truth_source, MAX_TEXT_BYTES, "truth")
    impressions = load_mind_impressions_bytes(impressions_source)
    if not impressions or len(impressions) > MAX_ROWS:
        raise ValueError("impression row count exceeds bounded nonempty limit")
    observed: set[str] = set()
    count = 0
    for impression in impressions:
        _identifier(impression.impression_id, "impression")
        if impression.impression_id in observed:
            raise ValueError("duplicate impression id")
        observed.add(impression.impression_id)
        count += len(impression.candidates)
        if len(impression.candidates) > MAX_CANDIDATES_PER_ROW or count > MAX_TOTAL_CANDIDATES:
            raise ValueError("impression candidate count exceeds limit")
        for candidate in impression.candidates:
            if len(candidate.article_id.encode("utf-8")) > 256:
                raise ValueError("candidate article_id exceeds byte limit")
    scores = _score_rows(scores_source)
    expected_pairs = {
        (impression.impression_id, candidate.article_id)
        for impression in impressions
        for candidate in impression.candidates
    }
    if set(scores) != expected_pairs:
        raise ValueError("score candidate coverage differs from impressions")
    truth = _text_rows(truth_source, "truth") if truth_source is not None else None
    if truth is not None and len(truth) != len(impressions):
        raise ValueError("truth/impression row count differs")
    lines: list[str] = []
    masked = 0
    for index, impression in enumerate(impressions):
        labels = tuple(int(candidate.clicked) for candidate in impression.candidates)
        if truth is not None:
            truth_id, actual = truth[index]
            if truth_id != impression.impression_id:
                raise ValueError("truth/impression id or row order differs")
            if actual and len(actual) != len(labels):
                raise ValueError("truth/impression candidate count differs")
            if actual and actual != labels:
                raise ValueError("truth/impression labels differ")
            if not actual:
                masked += 1
                lines.append(f"{impression.impression_id} []\n")
                continue
            if not 0 < sum(actual) < len(actual):
                raise ValueError("unmasked impression requires both classes")
        values = [
            scores[(impression.impression_id, candidate.article_id)]
            for candidate in impression.candidates
        ]
        order = sorted(range(len(values)), key=lambda position: (-values[position], position))
        ranks = [0] * len(values)
        for rank, position in enumerate(order, 1):
            ranks[position] = rank
        lines.append(f"{impression.impression_id} [{','.join(map(str, ranks))}]\n")
    prediction = "".join(lines).encode("utf-8")
    _bounded(prediction, MAX_TEXT_BYTES, "prediction")
    return PreparedMindPredictions(
        prediction=prediction,
        impressions_sha256=_sha(impressions_source),
        scores_sha256=_sha(scores_source),
        truth_sha256=None if truth_source is None else _sha(truth_source),
        rows=len(impressions),
        masked_rows=masked,
        candidates=count,
    )


def read_snapshot(path: str | Path, maximum: int) -> bytes:
    """Read one file up to an explicit byte ceiling before any hashing/parsing."""

    with Path(path).open("rb") as stream:
        source = stream.read(maximum + 1)
    if len(source) > maximum:
        raise ValueError("source exceeds byte limit")
    return source


def create_only(path: str | Path, source: bytes) -> None:
    """Publish a bounded artifact without replacing an existing filesystem entry."""

    _bounded(source, MAX_TEXT_BYTES, "output")
    destination = Path(path)
    if not destination.parent.is_dir():
        raise ValueError("output parent directory does not exist")
    descriptor = -1
    staged: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        staged = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            if stream.write(source) != len(source):
                raise OSError("short output write")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, destination)
        if os.name == "posix":
            directory = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except FileExistsError:
        raise ValueError("output already exists") from None
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        if staged is not None:
            with suppress(OSError):
                staged.unlink(missing_ok=True)
