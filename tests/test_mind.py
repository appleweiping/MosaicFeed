from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mosaicfeed.cli import main
from mosaicfeed.datasets import MindCandidate, MindImpression
from mosaicfeed.mind import (
    evaluate_mind_impressions,
    impression_auc,
    impression_mrr,
    impression_ndcg,
    load_mind_impressions,
    load_mind_scores,
    write_mind_impressions,
)


def _impression(
    impression_id: str = "i1",
    labels: tuple[bool, ...] = (True, False, True, False),
) -> MindImpression:
    return MindImpression(
        impression_id,
        "u1",
        datetime(2019, 11, 15, tzinfo=UTC),
        tuple(MindCandidate(f"n{index}", label) for index, label in enumerate(labels, 1)),
    )


def test_official_style_metric_golden_case() -> None:
    labels = [True, False, True, False]
    scores = [0.9, 0.8, 0.7, 0.6]
    assert impression_auc(labels, scores) == 0.75
    assert impression_mrr(labels, scores) == pytest.approx((1 + 1 / 3) / 2)
    assert impression_ndcg(labels, scores, 2) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert impression_ndcg(labels, scores, 4) == pytest.approx(
        (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    )


def _independent_untied_mind_metrics(
    labels: list[bool], scores: list[float], cutoff: int
) -> tuple[float, float, float]:
    """Compute untied MIND metrics without calling production helpers."""
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count

    # This rank-sum AUC is algebraically independent of production's pairwise counter.
    ascending = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    positive_rank_sum = math.fsum(
        rank for rank, (_, clicked) in enumerate(ascending, start=1) if clicked
    )
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )

    ranked_labels = [
        clicked
        for _, clicked in sorted(
            zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True
        )
    ]
    mrr = (
        math.fsum(1 / rank for rank, clicked in enumerate(ranked_labels, start=1) if clicked)
        / positive_count
    )
    dcg = math.fsum(
        1 / math.log2(rank + 1)
        for rank, clicked in enumerate(ranked_labels[:cutoff], start=1)
        if clicked
    )
    ideal_dcg = math.fsum(
        1 / math.log2(rank + 1) for rank in range(1, min(positive_count, cutoff) + 1)
    )
    return auc, mrr, dcg / ideal_dcg


def test_untied_metrics_match_independent_mind_reference_randomized() -> None:
    # Formulas cross-checked against the commit-pinned official MIND evaluator:
    # https://github.com/msnews/MIND/blob/47fb9852c97814d80ccf9c658d6b81f5c930b510/evaluate.py
    rng = random.Random(20260907)

    for case_index in range(200):
        candidate_count = rng.randint(2, 20)
        positive_count = rng.randint(1, candidate_count - 1)
        labels = [True] * positive_count + [False] * (candidate_count - positive_count)
        rng.shuffle(labels)
        scores = [float(value) for value in rng.sample(range(-10_000, 10_001), candidate_count)]
        cutoff = rng.randint(1, candidate_count + 3)

        expected_auc, expected_mrr, expected_ndcg = _independent_untied_mind_metrics(
            labels, scores, cutoff
        )
        assert impression_auc(labels, scores) == pytest.approx(expected_auc), case_index
        assert impression_mrr(labels, scores) == pytest.approx(expected_mrr), case_index
        assert impression_ndcg(labels, scores, cutoff) == pytest.approx(expected_ndcg), case_index


def test_auc_assigns_half_credit_to_ties_and_ndcg_uses_source_order() -> None:
    assert impression_auc([True, False], [0.5, 0.5]) == 0.5
    assert impression_mrr([False, True], [0.5, 0.5]) == 0.5
    assert impression_ndcg([False, True], [0.5, 0.5], 2) == pytest.approx(1 / math.log2(3))


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([], [], "non-empty"),
        ([True], [], "equal length"),
        ([True], [float("nan")], "finite"),
        ([True, True], [1.0, 0.0], "clicked and one unclicked"),
        ([False, False], [1.0, 0.0], "clicked and one unclicked"),
    ],
)
def test_auc_rejects_undefined_or_invalid_inputs(
    labels: list[bool], scores: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        impression_auc(labels, scores)


def test_rank_metrics_require_a_positive_and_valid_cutoff() -> None:
    with pytest.raises(ValueError, match="clicked"):
        impression_mrr([False], [1.0])
    with pytest.raises(ValueError, match="clicked"):
        impression_ndcg([False], [1.0], 1)
    with pytest.raises(ValueError, match="positive integer"):
        impression_ndcg([True], [1.0], 0)


def test_impression_round_trip_preserves_candidate_order_and_labels(tmp_path: Path) -> None:
    path = tmp_path / "impressions.json"
    source = (_impression(), _impression("i2", (False, True)))
    write_mind_impressions(path, source)
    assert load_mind_impressions(path) == source


def test_impression_writer_rejects_duplicate_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate MIND impression"):
        write_mind_impressions(tmp_path / "impressions.json", (_impression(), _impression()))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "list"),
        ([{"impression_id": "i", "user_id": "u", "occurred_at": "x"}], "missing"),
        (
            [
                {
                    "impression_id": "i",
                    "user_id": "u",
                    "occurred_at": "2019-01-01T00:00:00Z",
                    "candidates": [{"article_id": "n", "clicked": 1}],
                }
            ],
            "boolean",
        ),
    ],
)
def test_impression_loader_is_strict(tmp_path: Path, payload: object, message: str) -> None:
    path = tmp_path / "impressions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_mind_impressions(path)


def test_score_loader_rejects_duplicate_non_finite_and_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "scores.json"
    rows = [
        {"impression_id": "i", "article_id": "n", "score": 1.0},
        {"impression_id": "i", "article_id": "n", "score": 2.0},
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate score"):
        load_mind_scores(path)
    path.write_text('[{"impression_id":"i","article_id":"n","score":NaN}]', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        load_mind_scores(path)
    rows = [{"impression_id": "i", "article_id": "n", "score": 1, "extra": 2}]
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_mind_scores(path)


def test_evaluation_macro_averages_and_fingerprints() -> None:
    impressions = (_impression(), _impression("i2", (False, True)))
    scores = {
        "i1": {"n1": 0.9, "n2": 0.8, "n3": 0.7, "n4": 0.6},
        "i2": {"n1": 0.1, "n2": 0.9},
    }
    report = evaluate_mind_impressions(impressions, scores, cutoffs=(10, 2, 2))
    assert report.impressions == 2
    assert report.candidates == 6
    assert report.auc == pytest.approx(0.875)
    assert report.mrr == pytest.approx(((1 + 1 / 3) / 2 + 1) / 2)
    assert set(report.ndcg) == {2, 10}
    payload = report.to_dict()
    assert payload["averaging"] == "macro over impressions"
    assert len(str(payload["impression_sha256"])) == 64
    assert len(str(payload["score_sha256"])) == 64


def test_evaluation_normalizes_numeric_scores_before_fingerprinting() -> None:
    impression = (_impression(labels=(True, False)),)
    integer_report = evaluate_mind_impressions(impression, {"i1": {"n1": 1, "n2": 0}}, cutoffs=(1,))
    float_report = evaluate_mind_impressions(
        impression, {"i1": {"n1": 1.0, "n2": 0.0}}, cutoffs=(1,)
    )
    assert integer_report == float_report

    positive_zero = evaluate_mind_impressions(
        impression, {"i1": {"n1": 1.0, "n2": 0.0}}, cutoffs=(1,)
    )
    negative_zero = evaluate_mind_impressions(
        impression, {"i1": {"n1": 1.0, "n2": -0.0}}, cutoffs=(1,)
    )
    assert positive_zero == negative_zero


def test_public_metrics_and_evaluator_reject_unrepresentable_integers() -> None:
    huge = 10**10_000
    with pytest.raises(ValueError, match="finite"):
        impression_auc([True, False], [huge, 0])
    with pytest.raises(ValueError, match="finite"):
        evaluate_mind_impressions(
            (_impression(labels=(True, False)),),
            {"i1": {"n1": huge, "n2": 0}},
            cutoffs=(1,),
        )


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ({}, "impression ids differ"),
        ({"i1": {"n1": 1.0}}, "scores for impression"),
        ({"i1": {"n1": 1.0, "n2": 0.0, "n3": 0.5, "n4": 0.2}, "extra": {}}, "ids"),
    ],
)
def test_evaluation_requires_exact_score_coverage(
    scores: dict[str, dict[str, float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_mind_impressions((_impression(),), scores)


def test_evaluate_mind_cli_writes_versioned_report(tmp_path: Path) -> None:
    impressions_path = tmp_path / "impressions.json"
    scores_path = tmp_path / "scores.json"
    output = tmp_path / "report.json"
    write_mind_impressions(impressions_path, (_impression("i", (False, True)),))
    scores_path.write_text(
        json.dumps(
            [
                {"impression_id": "i", "article_id": "n1", "score": 0.1},
                {"impression_id": "i", "article_id": "n2", "score": 0.9},
            ]
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "evaluate-mind",
                "--impressions",
                str(impressions_path),
                "--scores",
                str(scores_path),
                "--cutoff",
                "1",
                "--cutoff",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["auc"] == payload["mrr"] == 1.0
    assert payload["ndcg"] == {"1": 1.0, "2": 1.0}
