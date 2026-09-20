"""Offline, bounded MIND-small train/dev experiment with exact-source provenance.

This is a sampled linear baseline, not the official neural benchmark result.
No dataset is downloaded, bundled, or accepted without caller-declared hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from mosaicfeed.datasets import MAX_MIND_SOURCE_BYTES, MindDataset, MindImpression, fixed_offset
from mosaicfeed.io import json_text, parse_datetime
from mosaicfeed.mind import evaluate_mind_impressions, load_mind_scores
from mosaicfeed.models import Article
from mosaicfeed.pairwise import MAX_ARTICLES, MAX_CANDIDATES, PairwiseImpressionRanker

MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_SOURCE_BYTES = min(MAX_MIND_SOURCE_BYTES, 128 * 1024 * 1024)
MAX_RAW_IMPRESSIONS = 250_000
MAX_RAW_NEWS = 250_000
MAX_RAW_CANDIDATES = 2_000_000
MAX_RAW_CANDIDATES_PER_IMPRESSION = 4_096
MAX_RAW_HISTORY_IDS = 20_000_000
MAX_RAW_LINE_BYTES = 64 * 1024
MAX_SELECTED_IMPRESSIONS = 2_000
MAX_SELECTED_CANDIDATES = 100_000
MAX_SCORE_OUTPUT_BYTES = 128 * 1024 * 1024
MIND_MEMBERS = frozenset(
    {"news.tsv", "behaviors.tsv", "entity_embedding.vec", "relation_embedding.vec"}
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _expected(value: str, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _read_bounded(path: Path, maximum: int) -> bytes:
    with path.open("rb") as stream:
        value = stream.read(maximum + 1)
    if len(value) > maximum:
        raise ValueError(f"MIND input exceeds {maximum} bytes: {path}")
    return value


def _preflight_tsv(data: bytes, *, maximum: int, name: str, behaviors: bool) -> int:
    """Bound parser work before materializing its full article/impression graph."""

    rows = candidates = history_ids = 0
    for line in io.BytesIO(data):
        rows += 1
        if rows > maximum:
            raise ValueError(f"{name} exceeds its raw row limit")
        if len(line) > MAX_RAW_LINE_BYTES:
            raise ValueError(f"{name} contains an oversized row")
        content = line.rstrip(b"\n").rstrip(b"\r")
        if b"\r" in content:
            raise ValueError(f"{name} contains a bare CR line ending")
        if not behaviors or not content:
            continue
        fields = content.split(b"\t", 4)
        if len(fields) != 5:
            # The existing strict parser will provide the schema error.
            continue
        candidate_count = len(fields[4].split())
        if candidate_count > MAX_RAW_CANDIDATES_PER_IMPRESSION:
            raise ValueError(f"{name} exceeds the raw per-impression candidate limit")
        candidates += candidate_count
        history_ids += len(fields[3].split())
        if candidates > MAX_RAW_CANDIDATES or history_ids > MAX_RAW_HISTORY_IDS:
            raise ValueError(f"{name} exceeds its cumulative token limit")
    return rows


def _source(
    path: Path,
    *,
    label: str,
    expected_news: str,
    expected_behaviors: str,
    expected_archive: str | None,
    catalog_time: datetime,
    offset_hours: float,
) -> tuple[MindDataset, dict[str, object]]:
    """Snapshot exactly the relevant TSVs, verifying caller-supplied digests first."""

    _expected(expected_news, f"{label} news hash")
    _expected(expected_behaviors, f"{label} behaviors hash")
    if path.is_dir():
        if expected_archive is not None:
            raise ValueError(f"{label} archive hash is only valid for a ZIP source")
        files = {name: path / name for name in MIND_MEMBERS}
        if any(not file.is_file() or file.is_symlink() for file in files.values()):
            raise ValueError(f"{label} must contain four regular MIND files")
        news = _read_bounded(files["news.tsv"], MAX_SOURCE_BYTES)
        behaviors = _read_bounded(files["behaviors.tsv"], MAX_SOURCE_BYTES)
        container: dict[str, object] = {"kind": "directory", "archive_sha256": None}
    else:
        if expected_archive is None:
            raise ValueError(f"{label} ZIP requires a declared archive SHA-256")
        _expected(expected_archive, f"{label} archive hash")
        archive = _read_bounded(path, MAX_ARCHIVE_BYTES)
        if _digest(archive) != expected_archive:
            raise ValueError(f"{label} archive SHA-256 mismatch")
        try:
            with ZipFile(io.BytesIO(archive)) as source:
                members = source.infolist()
                names = [member.filename for member in members]
                if len(names) != len(set(names)) or set(names) != MIND_MEMBERS:
                    raise ValueError(f"{label} ZIP must contain exactly four root MIND files")
                if any(member.is_dir() or member.flag_bits & 1 for member in members):
                    raise ValueError(f"{label} ZIP contains a directory or encrypted member")
                info = {member.filename: member for member in members}
                if any(
                    info[name].file_size > MAX_SOURCE_BYTES
                    for name in ("news.tsv", "behaviors.tsv")
                ):
                    raise ValueError(f"{label} ZIP TSV exceeds source byte limit")
                with source.open("news.tsv") as stream:
                    news = stream.read(MAX_SOURCE_BYTES + 1)
                with source.open("behaviors.tsv") as stream:
                    behaviors = stream.read(MAX_SOURCE_BYTES + 1)
        except BadZipFile as error:
            raise ValueError(f"{label} is not a valid MIND ZIP") from error
        if len(news) > MAX_SOURCE_BYTES or len(behaviors) > MAX_SOURCE_BYTES:
            raise ValueError(f"{label} ZIP TSV exceeds source byte limit")
        container = {"kind": "zip", "archive_sha256": expected_archive}
    if _digest(news) != expected_news or _digest(behaviors) != expected_behaviors:
        raise ValueError(f"{label} TSV SHA-256 mismatch")
    news_rows = _preflight_tsv(news, maximum=MAX_RAW_NEWS, name=f"{label} news", behaviors=False)
    behavior_rows = _preflight_tsv(
        behaviors, maximum=MAX_RAW_IMPRESSIONS, name=f"{label} behaviors", behaviors=True
    )
    dataset = MindDataset(
        news,
        behaviors,
        catalog_published_at=catalog_time,
        behavior_timezone=fixed_offset(offset_hours),
        max_source_bytes=MAX_SOURCE_BYTES,
    )
    if not dataset.impression_records or len(dataset.impression_records) > MAX_RAW_IMPRESSIONS:
        raise ValueError(f"{label} has no impressions or exceeds the raw impression limit")
    return dataset, {
        **container,
        "news_sha256": dataset.news_sha256,
        "behaviors_sha256": dataset.behaviors_sha256,
        "news_bytes": len(news),
        "behaviors_bytes": len(behaviors),
        "physical_news_rows": news_rows,
        "physical_behavior_rows": behavior_rows,
        "parsed_news_rows": len(dataset.articles),
        "parsed_behavior_rows": len(dataset.impression_records),
    }


def _selected(
    records: Sequence[MindImpression], *, prefix: str, maximum: int
) -> tuple[tuple[MindImpression, ...], int]:
    eligible = [item for item in records if len(item.candidates) <= MAX_CANDIDATES]
    selected = tuple(
        MindImpression(
            f"{prefix}:{item.impression_id}", item.user_id, item.occurred_at, item.candidates
        )
        for item in sorted(eligible, key=lambda value: (value.occurred_at, value.impression_id))[
            :maximum
        ]
    )
    if not selected:
        raise ValueError(f"{prefix} has no impressions within the selected candidate limit")
    return selected, len(records) - len(eligible)


def _catalog(
    training: MindDataset, validation: MindDataset, wanted: set[str]
) -> tuple[Article, ...]:
    combined: dict[str, Article] = {}
    for article in chain(training.articles, validation.articles):
        if article.id not in wanted:
            continue
        earlier = combined.get(article.id)
        if earlier is not None and earlier != article:
            raise ValueError(f"train/dev disagree on article metadata: {article.id}")
        combined[article.id] = article
    if len(combined) != len(wanted) or len(combined) > MAX_ARTICLES:
        raise ValueError("selected article catalog is incomplete or exceeds model limit")
    return tuple(combined[key] for key in sorted(combined))


def _scores_rows(
    records: Sequence[MindImpression], scores: Mapping[str, Mapping[str, float]]
) -> list[dict[str, object]]:
    return [
        {
            "impression_id": item.impression_id,
            "article_id": candidate.article_id,
            "score": scores[item.impression_id][candidate.article_id],
        }
        for item in records
        for candidate in item.candidates
    ]


def run_mind_small(
    train: str | Path,
    validation: str | Path,
    output: str | Path,
    *,
    train_news_sha256: str,
    train_behaviors_sha256: str,
    validation_news_sha256: str,
    validation_behaviors_sha256: str,
    train_archive_sha256: str | None = None,
    validation_archive_sha256: str | None = None,
    catalog_published_at: datetime,
    behavior_utc_offset: float,
    max_train_impressions: int = 1_000,
    max_validation_impressions: int = 1_000,
    epochs: int = 5,
    seed: int = 17,
) -> dict[str, object]:
    """Train a sampled pairwise baseline and publish a no-overwrite report bundle."""

    if (
        type(max_train_impressions) is not int
        or not 1 <= max_train_impressions <= MAX_SELECTED_IMPRESSIONS
    ):
        raise ValueError("max_train_impressions must be an integer in [1, 2000]")
    if (
        type(max_validation_impressions) is not int
        or not 1 <= max_validation_impressions <= MAX_SELECTED_IMPRESSIONS
    ):
        raise ValueError("max_validation_impressions must be an integer in [1, 2000]")
    destination = Path(output)
    train_path, validation_path = Path(train), Path(validation)
    resolved = [path.resolve(strict=False) for path in (train_path, validation_path, destination)]
    if len(set(resolved)) != 3 or destination.exists() or destination.is_symlink():
        raise ValueError("train, validation, and new output directory must be distinct")
    if not destination.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    if (
        not isinstance(catalog_published_at, datetime)
        or catalog_published_at.tzinfo is None
        or catalog_published_at.utcoffset() is None
    ):
        raise ValueError("catalog_published_at must be timezone-aware")
    catalog_time = catalog_published_at.astimezone(UTC)
    training, train_provenance = _source(
        train_path,
        label="train",
        expected_news=train_news_sha256,
        expected_behaviors=train_behaviors_sha256,
        expected_archive=train_archive_sha256,
        catalog_time=catalog_time,
        offset_hours=behavior_utc_offset,
    )
    development, validation_provenance = _source(
        validation_path,
        label="validation",
        expected_news=validation_news_sha256,
        expected_behaviors=validation_behaviors_sha256,
        expected_archive=validation_archive_sha256,
        catalog_time=catalog_time,
        offset_hours=behavior_utc_offset,
    )
    train_times = [item.occurred_at for item in training.impression_records]
    dev_times = [item.occurred_at for item in development.impression_records]
    if catalog_time > min(train_times) or max(train_times) >= min(dev_times):
        raise ValueError("declared catalog time or MIND train/dev temporal split is invalid")
    train_records, train_oversized = _selected(
        training.impression_records, prefix="train", maximum=max_train_impressions
    )
    dev_records, dev_oversized = _selected(
        development.impression_records, prefix="dev", maximum=max_validation_impressions
    )
    selected_candidates = sum(len(item.candidates) for item in chain(train_records, dev_records))
    if selected_candidates > MAX_SELECTED_CANDIDATES:
        raise ValueError("selected train/dev candidates exceed workflow limit")
    wanted = {
        candidate.article_id
        for item in (*train_records, *dev_records)
        for candidate in item.candidates
    }
    articles = _catalog(training, development, wanted)
    sources = {
        "train_news": training.news_sha256,
        "train_behaviors": training.behaviors_sha256,
        "dev_news": development.news_sha256,
    }
    cutoff = max(item.occurred_at for item in train_records)
    model = PairwiseImpressionRanker(epochs=epochs, seed=seed).fit(
        articles,
        train_records,
        partition="mind-small-train-prefix",
        cutoff=cutoff,
        held_out_impression_ids=(item.impression_id for item in dev_records),
        source_sha256=sources,
    )
    model_text = json_text(model.to_state())
    restored = PairwiseImpressionRanker.from_state(json.loads(model_text))
    scores = restored.score_impressions(articles, dev_records, training_impressions=train_records)
    score_text = json_text(_scores_rows(dev_records, scores))
    score_bytes = score_text.encode("utf-8")
    if len(score_bytes) > MAX_SCORE_OUTPUT_BYTES:
        raise ValueError("validation score output exceeds workflow byte limit")
    comparable = tuple(
        item
        for item in dev_records
        if any(candidate.clicked for candidate in item.candidates)
        and any(not candidate.clicked for candidate in item.candidates)
    )
    if not comparable:
        raise ValueError("selected validation subset contains no mixed-label impressions")
    with tempfile.TemporaryDirectory(prefix="mosaicfeed-mind-scores-") as temporary:
        score_path = Path(temporary) / "scores.json"
        score_path.write_text(score_text, encoding="utf-8")
        reloaded_scores = load_mind_scores(score_path)
    metrics = evaluate_mind_impressions(
        comparable, {item.impression_id: reloaded_scores[item.impression_id] for item in comparable}
    )
    report: dict[str, object] = {
        "format": "mosaicfeed.mind-small-local-baseline",
        "schema_version": 1,
        "dataset_claim": (
            "caller-supplied MIND-small-shaped files; hashes do not prove official authenticity"
        ),
        "benchmark_claim": (
            "sampled local linear pairwise baseline; not an official full MIND-small result"
        ),
        "model_objective": "pairwise-impression",
        "score_semantics": "raw logit; not probability",
        "selection": {
            "rule": (
                "earliest (occurred_at, impression_id) prefix of <=256-candidate records "
                "independently in train and dev"
            ),
            "max_train_impressions": max_train_impressions,
            "max_validation_impressions": max_validation_impressions,
            "selected_train_impressions": len(train_records),
            "selected_validation_impressions": len(dev_records),
            "selected_candidate_cells": selected_candidates,
            "raw_train_oversized_excluded": train_oversized,
            "raw_validation_oversized_excluded": dev_oversized,
            "selected_catalog_articles": len(articles),
            "train_cutoff": cutoff.isoformat(),
        },
        "source": {"train": train_provenance, "validation": validation_provenance},
        "assumptions": {
            "catalog_published_at": catalog_time.isoformat(),
            "behavior_utc_offset_hours": behavior_utc_offset,
            "dev_article_metadata": (
                "transductive catalog known before fit; no dev labels or clicks used"
            ),
            "history": "undated MIND history is ignored by the pairwise model",
        },
        "training": restored.training.to_dict(),
        "validation": {
            "mixed_label_impressions": len(comparable),
            "skipped_all_positive": sum(
                all(c.clicked for c in item.candidates) for item in dev_records
            ),
            "skipped_all_negative": sum(
                not any(c.clicked for c in item.candidates) for item in dev_records
            ),
            "scores_sha256": _digest(score_bytes),
            "model_sha256": _digest(model_text.encode("utf-8")),
            "metrics": metrics.to_dict(),
        },
    }
    if destination.exists() or destination.is_symlink():
        raise ValueError("output directory was created during the run")
    with tempfile.TemporaryDirectory(prefix=".mind-small-", dir=destination.parent) as staging:
        staged = Path(staging)
        for filename, body in (
            ("model.json", model_text),
            ("scores.json", score_text),
            ("report.json", json_text(report)),
        ):
            with (staged / filename).open("wb") as stream:
                stream.write(body.encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
        # mkdir and hard-link publication are both exclusive: on POSIX too,
        # a concurrently-created destination cannot be replaced by rename.
        destination.mkdir()
        for filename in ("model.json", "scores.json", "report.json"):
            os.link(staged / filename, destination / filename)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", required=True, help="user-owned MINDsmall_train ZIP or directory"
    )
    parser.add_argument(
        "--validation", required=True, help="user-owned MINDsmall_dev ZIP or directory"
    )
    parser.add_argument("--train-news-sha256", required=True)
    parser.add_argument("--train-behaviors-sha256", required=True)
    parser.add_argument("--validation-news-sha256", required=True)
    parser.add_argument("--validation-behaviors-sha256", required=True)
    parser.add_argument("--train-archive-sha256")
    parser.add_argument("--validation-archive-sha256")
    parser.add_argument("--catalog-published-at", required=True)
    parser.add_argument("--behavior-utc-offset", required=True, type=float)
    parser.add_argument("--max-train-impressions", type=int, default=1_000)
    parser.add_argument("--max-validation-impressions", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True, help="new directory for model/scores/report")
    args = parser.parse_args(argv)
    report = run_mind_small(
        args.train,
        args.validation,
        args.output,
        train_news_sha256=args.train_news_sha256,
        train_behaviors_sha256=args.train_behaviors_sha256,
        validation_news_sha256=args.validation_news_sha256,
        validation_behaviors_sha256=args.validation_behaviors_sha256,
        train_archive_sha256=args.train_archive_sha256,
        validation_archive_sha256=args.validation_archive_sha256,
        catalog_published_at=parse_datetime(args.catalog_published_at, "catalog_published_at"),
        behavior_utc_offset=args.behavior_utc_offset,
        max_train_impressions=args.max_train_impressions,
        max_validation_impressions=args.max_validation_impressions,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(json.dumps({"output": args.output, "validation": report["validation"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
