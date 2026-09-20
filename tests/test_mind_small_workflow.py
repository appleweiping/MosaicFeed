from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import pytest

from mosaicfeed.mind_small_workflow import main, run_mind_small
from mosaicfeed.pairwise import PairwiseImpressionRanker

NEWS_TRAIN = (
    "N1\tnews\tworld\tFirst headline\tWorld news\thttps://example.test/1\t[]\t[]\n"
    "N2\tsports\tfootball\tSecond headline\tMatch news\thttps://example.test/2\t[]\t[]\n"
    "N3\tnews\tworld\tThird headline\tAnother story\thttps://example.test/3\t[]\t[]\n"
)
NEWS_DEV = NEWS_TRAIN + (
    "N4\ttechnology\tai\tFourth headline\tAI news\thttps://example.test/4\t[]\t[]\n"
)
TRAIN = (
    "1\tU1\t11/14/2019 9:00:00 AM\tN1\tN1-1 N2-0 N3-0\n"
    "2\tU1\t11/14/2019 10:00:00 AM\tN1\tN1-0 N2-1 N3-0\n"
)
DEV = (
    "1\tU1\t11/15/2019 9:00:00 AM\tN1 N2\tN1-0 N2-0 N4-1\n"
    "2\tU2\t11/15/2019 10:00:00 AM\tN1\tN4-0 N2-1 N3-0\n"
    "3\tU1\t11/15/2019 11:00:00 AM\tN1\tN1-0 N4-0\n"
)
CATALOG = datetime(2019, 1, 1, tzinfo=UTC)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _files(tmp_path: Path, *, zipped: bool = False) -> tuple[Path, Path, dict[str, object]]:
    sources = []
    hashes: dict[str, object] = {}
    for label, news, behaviors in (("train", NEWS_TRAIN, TRAIN), ("validation", NEWS_DEV, DEV)):
        folder = tmp_path / label
        folder.mkdir()
        for name, value in (
            ("news.tsv", news.encode()),
            ("behaviors.tsv", behaviors.encode()),
            ("entity_embedding.vec", b"Q1 0.1\n"),
            ("relation_embedding.vec", b"R1 0.1\n"),
        ):
            (folder / name).write_bytes(value)
        hashes[f"{label}_news_sha256"] = _sha(news.encode())
        hashes[f"{label}_behaviors_sha256"] = _sha(behaviors.encode())
        if zipped:
            archive = tmp_path / f"{label}.zip"
            with ZipFile(archive, "w") as writer:
                for member in sorted(folder.iterdir()):
                    writer.write(member, member.name)
            hashes[f"{label}_archive_sha256"] = _sha(archive.read_bytes())
            sources.append(archive)
        else:
            sources.append(folder)
    return sources[0], sources[1], hashes


def _run(
    train: Path, validation: Path, output: Path, hashes: dict[str, object]
) -> dict[str, object]:
    return run_mind_small(
        train,
        validation,
        output,
        **hashes,  # type: ignore[arg-type]
        catalog_published_at=CATALOG,
        behavior_utc_offset=0,
        max_train_impressions=2,
        max_validation_impressions=3,
        epochs=1,
    )


@pytest.mark.parametrize("zipped", [False, True])
def test_offline_mind_small_train_dev_bundle_is_reproducible(tmp_path: Path, zipped: bool) -> None:
    train, validation, hashes = _files(tmp_path, zipped=zipped)
    first = tmp_path / "first"
    second = tmp_path / "second"
    report = _run(train, validation, first, hashes)
    assert _run(train, validation, second, hashes) == report
    for filename in ("model.json", "scores.json", "report.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert PairwiseImpressionRanker.load(first / "model.json").training.impressions == 2
    assert report["selection"]["selected_train_impressions"] == 2  # type: ignore[index]
    assert report["validation"]["mixed_label_impressions"] == 2  # type: ignore[index]
    assert report["validation"]["skipped_all_negative"] == 1  # type: ignore[index]
    assert report["source"]["train"]["behaviors_sha256"] == hashes["train_behaviors_sha256"]  # type: ignore[index]
    rows = json.loads((first / "scores.json").read_text(encoding="utf-8"))
    assert len(rows) == 8
    assert {row["impression_id"] for row in rows} == {"dev:1", "dev:2", "dev:3"}
    assert {item["article_id"] for item in rows} == {"N1", "N2", "N3", "N4"}
    with pytest.raises(ValueError, match="distinct"):
        _run(train, validation, first, hashes)


def test_hash_mismatch_fails_before_output_and_changed_dev_labels_do_not_change_scores(
    tmp_path: Path,
) -> None:
    train, validation, hashes = _files(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _run(train, validation, tmp_path / "bad", {**hashes, "train_news_sha256": "0" * 64})
    assert not (tmp_path / "bad").exists()
    _run(train, validation, tmp_path / "original", hashes)
    changed = (
        (validation / "behaviors.tsv")
        .read_text(encoding="utf-8")
        .replace("N1-0 N2-0 N4-1", "N1-1 N2-0 N4-0")
    )
    (validation / "behaviors.tsv").write_bytes(changed.encode())
    changed_hashes = {**hashes, "validation_behaviors_sha256": _sha(changed.encode())}
    _run(train, validation, tmp_path / "changed", changed_hashes)
    first = json.loads((tmp_path / "original" / "scores.json").read_text(encoding="utf-8"))
    second = json.loads((tmp_path / "changed" / "scores.json").read_text(encoding="utf-8"))
    assert first == second
    assert (tmp_path / "original" / "model.json").read_bytes() == (
        tmp_path / "changed" / "model.json"
    ).read_bytes()


def test_rejects_non_temporal_splits_and_conflicting_catalog(tmp_path: Path) -> None:
    train, validation, hashes = _files(tmp_path)
    overlapping = DEV.replace("11/15/2019", "11/14/2019")
    (validation / "behaviors.tsv").write_bytes(overlapping.encode())
    with pytest.raises(ValueError, match="temporal split"):
        _run(
            train,
            validation,
            tmp_path / "overlap",
            {**hashes, "validation_behaviors_sha256": _sha(overlapping.encode())},
        )
    (validation / "behaviors.tsv").write_bytes(DEV.encode())
    conflicting = NEWS_DEV.replace("First headline", "Edited headline")
    (validation / "news.tsv").write_bytes(conflicting.encode())
    with pytest.raises(ValueError, match="disagree on article metadata"):
        _run(
            train,
            validation,
            tmp_path / "conflict",
            {**hashes, "validation_news_sha256": _sha(conflicting.encode())},
        )
    assert not (tmp_path / "conflict").exists()


def test_rejects_archive_member_injection_and_tamper(tmp_path: Path) -> None:
    train, validation, hashes = _files(tmp_path, zipped=True)
    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        _run(train, validation, tmp_path / "bad-hash", {**hashes, "train_archive_sha256": "0" * 64})
    with ZipFile(validation, "a") as writer:
        writer.writestr("../unexpected.tsv", "bad")
    with pytest.raises(ValueError, match="exactly four root MIND files"):
        _run(
            train,
            validation,
            tmp_path / "bad-zip",
            {**hashes, "validation_archive_sha256": _sha(validation.read_bytes())},
        )
    assert not (tmp_path / "bad-zip").exists()


def test_rejects_missing_file_unbounded_source_and_invalid_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, validation, hashes = _files(tmp_path)
    (validation / "relation_embedding.vec").unlink()
    with pytest.raises(ValueError, match="four regular MIND files"):
        _run(train, validation, tmp_path / "missing", hashes)
    (validation / "relation_embedding.vec").write_bytes(b"R1 0.1\n")
    with pytest.raises(ValueError, match="max_train_impressions"):
        run_mind_small(
            train,
            validation,
            tmp_path / "limit",
            **hashes,  # type: ignore[arg-type]
            catalog_published_at=CATALOG,
            behavior_utc_offset=0,
            max_train_impressions=0,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _run(
            train, validation, tmp_path / "invalid-hash", {**hashes, "train_news_sha256": "A" * 64}
        )
    monkeypatch.setattr("mosaicfeed.mind_small_workflow.MAX_SOURCE_BYTES", 8)
    with pytest.raises(ValueError, match="exceeds 8 bytes"):
        _run(train, validation, tmp_path / "too-large", hashes)
    assert not any((tmp_path / name).exists() for name in ("limit", "invalid-hash", "too-large"))


def test_rejects_noncomparable_validation_before_publication(tmp_path: Path) -> None:
    train, validation, hashes = _files(tmp_path)
    all_negative = DEV.replace("N4-1", "N4-0").replace("N2-1", "N2-0")
    (validation / "behaviors.tsv").write_bytes(all_negative.encode())
    with pytest.raises(ValueError, match="no mixed-label impressions"):
        _run(
            train,
            validation,
            tmp_path / "unusable",
            {**hashes, "validation_behaviors_sha256": _sha(all_negative.encode())},
        )
    assert not (tmp_path / "unusable").exists()


def test_rejects_raw_row_budget_before_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, validation, hashes = _files(tmp_path)
    monkeypatch.setattr("mosaicfeed.mind_small_workflow.MAX_RAW_IMPRESSIONS", 2)
    with pytest.raises(ValueError, match="raw row limit"):
        _run(train, validation, tmp_path / "rows", hashes)
    assert not (tmp_path / "rows").exists()


def test_selected_candidate_and_score_output_budgets_prevent_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, validation, hashes = _files(tmp_path)
    monkeypatch.setattr("mosaicfeed.mind_small_workflow.MAX_SELECTED_CANDIDATES", 8)
    with pytest.raises(ValueError, match="selected train/dev candidates"):
        _run(train, validation, tmp_path / "candidate-limit", hashes)
    monkeypatch.setattr("mosaicfeed.mind_small_workflow.MAX_SELECTED_CANDIDATES", 100_000)
    monkeypatch.setattr("mosaicfeed.mind_small_workflow.MAX_SCORE_OUTPUT_BYTES", 8)
    with pytest.raises(ValueError, match="score output exceeds"):
        _run(train, validation, tmp_path / "score-limit", hashes)
    assert not (tmp_path / "candidate-limit").exists()
    assert not (tmp_path / "score-limit").exists()


def test_preflight_rejects_dense_candidates_and_cr_only_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, validation, hashes = _files(tmp_path)
    too_many = TRAIN.replace("N1-1 N2-0 N3-0", " ".join(["N1-0"] * 4_097))
    (train / "behaviors.tsv").write_bytes(too_many.encode())
    with pytest.raises(ValueError, match="raw per-impression candidate limit"):
        _run(
            train,
            validation,
            tmp_path / "dense",
            {**hashes, "train_behaviors_sha256": _sha(too_many.encode())},
        )
    (train / "behaviors.tsv").write_bytes(TRAIN.encode())
    monkeypatch.setattr("mosaicfeed.mind_small_workflow.MAX_RAW_CANDIDATES", 5)
    with pytest.raises(ValueError, match="cumulative token limit"):
        _run(train, validation, tmp_path / "cumulative", hashes)
    monkeypatch.setattr("mosaicfeed.mind_small_workflow.MAX_RAW_CANDIDATES", 2_000_000)
    cr_only = TRAIN.replace("\n", "\r")
    (train / "behaviors.tsv").write_bytes(cr_only.encode())
    with pytest.raises(ValueError, match="bare CR"):
        _run(
            train,
            validation,
            tmp_path / "cr-only",
            {**hashes, "train_behaviors_sha256": _sha(cr_only.encode())},
        )
    assert not any((tmp_path / name).exists() for name in ("dense", "cumulative", "cr-only"))


def test_oversized_raw_slate_is_excluded_whole_before_eligible_prefix(tmp_path: Path) -> None:
    train, validation, hashes = _files(tmp_path)
    extra_ids = [f"N{index}" for index in range(5, 262)]
    extra_news = "".join(
        f"{article_id}\tnews\tworld\tHeadline\tSummary\thttps://example.test/{article_id}\t[]\t[]\n"
        for article_id in extra_ids
    )
    news = NEWS_TRAIN + extra_news
    behavior = TRAIN.replace("N1-1 N2-0 N3-0", " ".join(f"{item}-0" for item in extra_ids))
    (train / "news.tsv").write_bytes(news.encode())
    (train / "behaviors.tsv").write_bytes(behavior.encode())
    report = _run(
        train,
        validation,
        tmp_path / "eligible",
        {
            **hashes,
            "train_news_sha256": _sha(news.encode()),
            "train_behaviors_sha256": _sha(behavior.encode()),
        },
    )
    selection = report["selection"]
    assert isinstance(selection, dict)
    assert selection["raw_train_oversized_excluded"] == 1
    assert selection["selected_train_impressions"] == 1
    assert selection["selected_catalog_articles"] == 4


def test_concurrent_output_creation_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, validation, hashes = _files(tmp_path)
    destination = tmp_path / "raced"
    original_mkdir = Path.mkdir

    def racing_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        if path == destination:
            original_mkdir(path)
            (path / "user-file").write_bytes(b"preserve")
        original_mkdir(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", racing_mkdir)
    with pytest.raises(FileExistsError):
        _run(train, validation, destination, hashes)
    assert (destination / "user-file").read_bytes() == b"preserve"
    assert not (destination / "report.json").exists()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        ("same-source", "distinct"),
        ("missing-parent", "parent directory"),
        ("naive-time", "timezone-aware"),
        ("late-catalog", "temporal split"),
        ("bad-validation-limit", "max_validation_impressions"),
        ("directory-archive-hash", "only valid for a ZIP"),
    ],
)
def test_rejects_invalid_protocol_before_output(tmp_path: Path, mutator: str, message: str) -> None:
    train, validation, hashes = _files(tmp_path)
    kwargs: dict[str, object] = {
        **hashes,
        "catalog_published_at": CATALOG,
        "behavior_utc_offset": 0,
        "max_train_impressions": 2,
        "max_validation_impressions": 3,
        "epochs": 1,
    }
    output = tmp_path / "invalid"
    if mutator == "same-source":
        validation = train
    elif mutator == "missing-parent":
        output = tmp_path / "absent" / "invalid"
    elif mutator == "naive-time":
        kwargs["catalog_published_at"] = datetime(2019, 1, 1)
    elif mutator == "late-catalog":
        kwargs["catalog_published_at"] = datetime(2019, 11, 14, 12, tzinfo=UTC)
    elif mutator == "bad-validation-limit":
        kwargs["max_validation_impressions"] = 0
    elif mutator == "directory-archive-hash":
        kwargs["train_archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=message):
        run_mind_small(train, validation, output, **kwargs)  # type: ignore[arg-type]
    assert not output.exists()


def test_rejects_invalid_zip_shapes_and_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, validation, hashes = _files(tmp_path, zipped=True)
    with pytest.raises(ValueError, match="requires a declared archive SHA-256"):
        _run(train, validation, tmp_path / "undeclared", {**hashes, "train_archive_sha256": None})
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a ZIP")
    with pytest.raises(ValueError, match="not a valid MIND ZIP"):
        _run(
            train,
            corrupt,
            tmp_path / "corrupt-out",
            {**hashes, "validation_archive_sha256": _sha(corrupt.read_bytes())},
        )
    monkeypatch.setattr("mosaicfeed.mind_small_workflow.MAX_ARCHIVE_BYTES", 16)
    with pytest.raises(ValueError, match="exceeds 16 bytes"):
        _run(train, validation, tmp_path / "archive-limit", hashes)


def test_script_entrypoint_accepts_declared_hashes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    train, validation, hashes = _files(tmp_path)
    args = [
        "--train",
        str(train),
        "--validation",
        str(validation),
        "--catalog-published-at",
        CATALOG.isoformat(),
        "--behavior-utc-offset",
        "0",
        "--max-train-impressions",
        "2",
        "--max-validation-impressions",
        "3",
        "--epochs",
        "1",
        "--output",
        str(tmp_path / "out"),
    ]
    for key, value in hashes.items():
        args.extend(("--" + key.replace("_", "-"), str(value)))
    assert main(args) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["validation"]["mixed_label_impressions"] == 2
