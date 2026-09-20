"""Independent metric and text-protocol oracles for a strict MIND subset."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

import mosaicfeed.mind_submission as submission
from mosaicfeed.cli import main
from mosaicfeed.mind_submission import evaluate_mind_submission, prepare_mind_predictions

TRUTH = b"1 [1,0,1,0]\n2 []\n3 [0,1]\n"
PREDICTION = b"1 [1,2,3,4]\n2 []\n3 [2,1]\n"


def _impressions() -> bytes:
    return json.dumps(
        [
            {
                "impression_id": "1",
                "user_id": "u",
                "occurred_at": "2020-01-01T00:00:00Z",
                "candidates": [
                    {"article_id": "A", "clicked": True},
                    {"article_id": "B", "clicked": False},
                    {"article_id": "C", "clicked": True},
                    {"article_id": "D", "clicked": False},
                ],
            },
            {
                "impression_id": "2",
                "user_id": "u",
                "occurred_at": "2020-01-01T01:00:00Z",
                "candidates": [
                    {"article_id": "E", "clicked": False},
                    {"article_id": "F", "clicked": False},
                ],
            },
        ],
        separators=(",", ":"),
    ).encode()


def _scores() -> bytes:
    return json.dumps(
        [
            {"impression_id": identifier, "article_id": article, "score": score}
            for identifier, article, score in (
                ("1", "A", 0.9),
                ("1", "B", 0.8),
                ("1", "C", 0.7),
                ("1", "D", 0.6),
                ("2", "E", 0.1),
                ("2", "F", 0.9),
            )
        ],
        separators=(",", ":"),
    ).encode()


def test_hand_auc_mrr_ndcg_with_masked_row_and_exact_source_hashes() -> None:
    report = evaluate_mind_submission(TRUTH, PREDICTION).to_dict()
    assert report["protocol"] == "mosaicfeed.mind-submission-v1"
    assert (report["rows"], report["masked_rows"], report["scored_rows"], report["candidates"]) == (
        3,
        1,
        2,
        6,
    )
    # First impression has three of four clicked/unclicked pairs correctly
    # ordered; second unmasked impression has its positive ranked first.
    assert report["auc"] == pytest.approx((3 / 4 + 1) / 2)
    assert report["mrr"] == pytest.approx(((1 + 1 / 3) / 2 + 1) / 2)
    first_ndcg = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert report["ndcg_5"] == pytest.approx((first_ndcg + 1) / 2)
    assert report["ndcg_10"] == pytest.approx((first_ndcg + 1) / 2)
    assert report["truth_sha256"] == hashlib.sha256(TRUTH).hexdigest()
    assert report["prediction_sha256"] == hashlib.sha256(PREDICTION).hexdigest()
    assert "1 [" not in json.dumps(report)


def test_permutation_ranks_score_by_inverse_rank_not_source_order() -> None:
    report = evaluate_mind_submission(b"x [0,1,0,1]\n", b"x [3,1,4,2]\n")
    assert report.auc == 1.0
    assert report.mrr == pytest.approx((1 + 1 / 2) / 2)
    assert report.ndcg_5 == 1.0 and report.ndcg_10 == 1.0


def test_masked_truth_consumes_one_ranked_prediction_row_without_scoring_it() -> None:
    report = evaluate_mind_submission(
        b"hidden []\nvisible [0,1]\n", b"hidden [2,1,3]\nvisible [2,1]\n"
    )
    assert (report.rows, report.masked_rows, report.scored_rows, report.candidates) == (2, 1, 1, 2)
    assert report.auc == report.mrr == report.ndcg_5 == report.ndcg_10 == 1.0
    with pytest.raises(ValueError, match="permutation"):
        evaluate_mind_submission(b"hidden []\nvisible [0,1]\n", b"hidden [1,1]\nvisible [2,1]\n")


@pytest.mark.parametrize(
    ("truth", "prediction", "reason"),
    [
        (b"1 [1,0]\n", b"2 [1,2]\n", "impression id"),
        (b"1 [1,0]\n2 [0,1]\n", b"1 [1,2]\n", "row count"),
        (b"1 [1,0]\n", b"1 [1,2]\n2 [1,2]\n", "row count"),
        (b"1 [1,0]\n1 [0,1]\n", b"1 [1,2]\n1 [2,1]\n", "duplicate"),
        (b"1 [1,0]\n", b"1 [1,1]\n", "permutation"),
        (b"1 [1,0]\n", b"1 [0,2]\n", "permutation"),
        (b"1 [1,0]\n", b"1 [1,3]\n", "permutation"),
        (b"1 [1,0]\n", b"1 [1]\n", "candidate count"),
        (b"1 []\n2 [1,0]\n", b"1 [1,1]\n2 [1,2]\n", "permutation"),
        (b"1 [1,0]\n", b"1 []\n", "masked"),
        (b"1 [1,1]\n", b"1 [1,2]\n", "both classes"),
        (b"1 [0,0]\n", b"1 [1,2]\n", "both classes"),
        (b"1 []\n", b"1 []\n", "no unmasked"),
        (b"1 [1,true]\n", b"1 [1,2]\n", "integer"),
        (b"1 [1,2]\n", b"1 [1,2]\n", "binary"),
        (b"1 [1,0]\n", b"1 [1.0,2]\n", "integer"),
        (b"1 [1,0]\n", b"1 [-1,2]\n", "nonnegative"),
        (b"1 [1,0]\n", b"1 [01,2]\n", "JSON"),
        (b"1 [1,0]\n", b"1 [1, 2]\n", "canonical"),
        (b"1  [1,0]\n", b"1 [1,2]\n", "canonical"),
        (b"bad id [1,0]\n", b"bad id [1,2]\n", "canonical"),
        (b"1 [1,0]", b"1 [1,2]\n", "LF"),
        (b"\xef\xbb\xbf1 [1,0]\n", b"1 [1,2]\n", "BOM"),
        (b"1 [1,0]\r\n", b"1 [1,2]\r\n", "LF"),
    ],
)
def test_strict_protocol_rejects_malformed_rows(
    truth: bytes, prediction: bytes, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        evaluate_mind_submission(truth, prediction)


def test_deterministic_export_aligns_scores_and_mask_without_using_test_labels() -> None:
    prepared = prepare_mind_predictions(_impressions(), _scores(), b"1 [1,0,1,0]\n2 []\n")
    assert prepared.prediction == b"1 [1,2,3,4]\n2 []\n"
    assert prepared.to_dict()["impressions_sha256"] == hashlib.sha256(_impressions()).hexdigest()
    assert prepared.to_dict()["scores_sha256"] == hashlib.sha256(_scores()).hexdigest()
    assert prepared.to_dict()["truth_sha256"] == hashlib.sha256(b"1 [1,0,1,0]\n2 []\n").hexdigest()
    assert (
        prepared.to_dict()["prediction_sha256"] == hashlib.sha256(prepared.prediction).hexdigest()
    )
    assert evaluate_mind_submission(b"1 [1,0,1,0]\n2 []\n", prepared.prediction).scored_rows == 1
    unmasked = prepare_mind_predictions(_impressions(), _scores())
    assert unmasked.prediction == b"1 [1,2,3,4]\n2 [2,1]\n"


def test_score_ties_follow_candidate_source_order_and_gold_change_does_not_change_ranks() -> None:
    scores = _scores().replace(b'"score":0.8', b'"score":0.9')
    first = prepare_mind_predictions(_impressions(), scores)
    changed = _impressions().replace(b'"clicked":true', b'"clicked":false')
    second = prepare_mind_predictions(changed, scores)
    assert first.prediction == second.prediction
    assert first.prediction.startswith(b"1 [1,2,3,4]\n")


def test_export_rejects_unscorable_unmasked_truth_but_allows_hidden_test_rows() -> None:
    all_negative = _impressions().replace(b'"clicked":true', b'"clicked":false')
    with pytest.raises(ValueError, match="both classes"):
        prepare_mind_predictions(all_negative, _scores(), b"1 [0,0,0,0]\n2 []\n")
    hidden = prepare_mind_predictions(_impressions(), _scores(), b"1 []\n2 []\n")
    assert hidden.prediction == b"1 []\n2 []\n"
    assert hidden.masked_rows == 2


@pytest.mark.parametrize(
    ("impressions", "scores", "truth", "reason"),
    [
        (b"[]", _scores(), None, "impression"),
        (_impressions(), b"[]", None, "score"),
        (
            _impressions(),
            _scores().replace(b'"article_id":"A"', b'"article_id":"Z"'),
            None,
            "candidate",
        ),
        (_impressions(), _scores() + b" ", b"1 [0,0,1,0]\n2 []\n", "labels"),
        (_impressions(), _scores(), b"2 []\n1 [1,0,1,0]\n", "impression id"),
        (_impressions(), _scores(), b"1 [1,0,1]\n2 []\n", "candidate count"),
    ],
)
def test_export_rejects_missing_excess_mismatch_and_label_drift(
    impressions: bytes, scores: bytes, truth: bytes | None, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        prepare_mind_predictions(impressions, scores, truth)


@pytest.mark.parametrize(
    ("scores", "reason"),
    [
        (b"{}", "bounded list"),
        (b"[{}]", "exactly"),
        (b'[{"impression_id":"1","article_id":"A","score":0.3,"extra":1}]', "exactly"),
        (_scores().replace(b'"impression_id":"1"', b'"impression_id":"bad id"'), "ASCII"),
        (_scores().replace(b'"article_id":"A"', b'"article_id":""'), "article_id"),
        (_scores().replace(b'"score":0.9', b'"score":true'), "finite"),
        (_scores().replace(b'"score":0.9', b'"score":1e999'), "finite"),
        (_scores().replace(b'"score":0.9', b'"score":"0.9"'), "finite"),
        (_scores()[:-1] + b"," + _scores()[1:], "duplicate score"),
    ],
)
def test_export_rejects_invalid_score_rows(scores: bytes, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        prepare_mind_predictions(_impressions(), scores)


def test_export_rejects_invalid_utf8_and_canonical_impression_id() -> None:
    with pytest.raises(ValueError, match="UTF-8"):
        prepare_mind_predictions(_impressions(), b"\xff")
    with pytest.raises(ValueError, match="ASCII"):
        prepare_mind_predictions(
            _impressions().replace(b'"impression_id":"1"', b'"impression_id":"bad id"'),
            _scores(),
        )


def test_read_snapshot_and_create_only_bounds_and_existing_file(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_bytes(b"abcd")
    assert submission.read_snapshot(input_path, 4) == b"abcd"
    with pytest.raises(ValueError, match="byte limit"):
        submission.read_snapshot(input_path, 3)
    output = tmp_path / "out.txt"
    submission.create_only(output, b"first")
    with pytest.raises(ValueError, match="already exists"):
        submission.create_only(output, b"second")
    assert output.read_bytes() == b"first"
    with pytest.raises(ValueError, match="parent directory"):
        submission.create_only(tmp_path / "absent" / "out.txt", b"value")


@pytest.mark.parametrize("oversized_truth", [True, False])
def test_direct_api_resource_preflight_before_hash(
    monkeypatch: pytest.MonkeyPatch, oversized_truth: bool
) -> None:
    too_large = b"x" * (submission.MAX_TEXT_BYTES + 1)

    def unexpected_hash(_raw: bytes) -> str:
        pytest.fail("oversized input reached SHA-256")

    monkeypatch.setattr(submission, "_sha", unexpected_hash)
    truth, pred = (too_large, PREDICTION) if oversized_truth else (TRUTH, too_large)
    with pytest.raises(ValueError, match="byte limit"):
        evaluate_mind_submission(truth, pred)


@pytest.mark.parametrize("oversized_impressions", [True, False])
def test_export_preflights_all_json_sources_before_parsing(
    monkeypatch: pytest.MonkeyPatch, oversized_impressions: bool
) -> None:
    def unexpected_parser(_source: bytes) -> object:
        pytest.fail("oversized source reached impression parser")

    monkeypatch.setattr(submission, "load_mind_impressions_bytes", unexpected_parser)
    giant = b"x" * (submission.MAX_JSON_BYTES + 1)
    impressions, scores = (giant, _scores()) if oversized_impressions else (_impressions(), giant)
    with pytest.raises(ValueError, match="byte limit"):
        prepare_mind_predictions(impressions, scores)


def test_byte_and_row_limits_and_create_only_enforce_resource_caps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="byte limit"):
        evaluate_mind_submission(TRUTH, b"x" * (submission.MAX_TEXT_BYTES + 1))
    with pytest.raises(ValueError, match="row limit"):
        evaluate_mind_submission(b"x []\n" * (submission.MAX_ROWS + 1), b"x []\n")
    with pytest.raises(ValueError, match="candidate limit"):
        evaluate_mind_submission(
            b"x [" + b"1," * submission.MAX_CANDIDATES_PER_ROW + b"0]\n",
            b"x [1,2]\n",
        )
    output = tmp_path / "too-large.txt"
    with pytest.raises(ValueError, match="byte limit"):
        submission.create_only(output, b"x" * (submission.MAX_TEXT_BYTES + 1))
    assert not output.exists()


def test_cli_create_only_reports_and_alias_rejection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    truth = tmp_path / "truth.txt"
    prediction = tmp_path / "prediction.txt"
    impressions = tmp_path / "impressions.json"
    scores = tmp_path / "scores.json"
    report = tmp_path / "report.json"
    truth.write_bytes(b"1 [1,0,1,0]\n2 []\n")
    impressions.write_bytes(_impressions())
    scores.write_bytes(_scores())
    assert (
        main(
            [
                "export-mind-predictions",
                "--impressions",
                str(impressions),
                "--scores",
                str(scores),
                "--truth",
                str(truth),
                "--output",
                str(prediction),
            ]
        )
        == 0
    )
    assert prediction.read_bytes() == b"1 [1,2,3,4]\n2 []\n"
    assert (
        json.loads(capsys.readouterr().out)["prediction_sha256"]
        == hashlib.sha256(prediction.read_bytes()).hexdigest()
    )
    assert (
        main(
            [
                "evaluate-mind-submission",
                "--truth",
                str(truth),
                "--prediction",
                str(prediction),
                "--output",
                str(report),
            ]
        )
        == 0
    )
    assert json.loads(report.read_bytes())["auc"] == 0.75
    original = prediction.read_bytes()
    assert (
        main(
            [
                "export-mind-predictions",
                "--impressions",
                str(impressions),
                "--scores",
                str(scores),
                "--truth",
                str(truth),
                "--output",
                str(prediction),
            ]
        )
        == 2
    )
    assert prediction.read_bytes() == original
    assert (
        main(
            [
                "evaluate-mind-submission",
                "--truth",
                str(truth),
                "--prediction",
                str(prediction),
                "--output",
                str(truth),
            ]
        )
        == 2
    )
    assert truth.read_bytes() == b"1 [1,0,1,0]\n2 []\n"


def test_cli_rejects_source_drift_before_create_only_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    truth = tmp_path / "truth.txt"
    prediction = tmp_path / "prediction.txt"
    truth.write_bytes(b"x [1,0]\n")
    prediction.write_bytes(b"x [1,2]\n")
    original = submission.read_snapshot
    count = 0

    def drift(path: str | Path, maximum: int) -> bytes:
        nonlocal count
        count += 1
        raw = original(path, maximum)
        return raw + b" " if count == 3 else raw

    monkeypatch.setattr(submission, "read_snapshot", drift)
    destination = tmp_path / "report.json"
    assert (
        main(
            [
                "evaluate-mind-submission",
                "--truth",
                str(truth),
                "--prediction",
                str(prediction),
                "--output",
                str(destination),
            ]
        )
        == 2
    )
    assert not destination.exists()
