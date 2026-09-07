from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mosaicfeed.cli import main
from mosaicfeed.datasets import fixed_offset, load_mind
from mosaicfeed.models import EventKind

NEWS = (
    "N1\tnews\tworld\tFirst title\tFirst abstract\thttps://example.test/1\t[]\t[]\n"
    "N2\tsports\tfootball\tSecond title\t\thttps://example.test/2\t[]\t[]\n"
)
BEHAVIORS = "1\tU1\t11/15/2019 9:55:12 AM\tN1\tN1-0 N2-1\n"


def _files(tmp_path: Path, news: str = NEWS, behaviors: str = BEHAVIORS) -> tuple[Path, Path]:
    news_path = tmp_path / "news.tsv"
    behavior_path = tmp_path / "behaviors.tsv"
    news_path.write_text(news, encoding="utf-8")
    behavior_path.write_text(behaviors, encoding="utf-8")
    return news_path, behavior_path


def test_load_mind_preserves_declared_information_and_clicks(tmp_path: Path) -> None:
    news, behaviors = _files(tmp_path)
    catalog_time = datetime(2019, 1, 1, tzinfo=UTC)
    dataset = load_mind(
        news,
        behaviors,
        catalog_published_at=catalog_time,
        behavior_timezone=fixed_offset(-8),
    )

    assert len(dataset.articles) == 2
    assert dataset.articles[0].published_at == catalog_time
    assert dataset.articles[0].source == "news"
    assert dataset.articles[0].topics == ("news", "world")
    assert len(dataset.events) == 1
    assert dataset.events[0].article_id == "N2"
    assert dataset.events[0].kind is EventKind.CLICK
    offset = dataset.events[0].occurred_at.utcoffset()
    assert offset is not None and offset.total_seconds() == -8 * 3_600
    assert dataset.ignored_history_items == 1
    assert dataset.impressions == len(dataset.impression_records) == 1
    impression = dataset.impression_records[0]
    assert impression.impression_id == "1"
    assert impression.user_id == "U1"
    assert [(item.article_id, item.clicked) for item in impression.candidates] == [
        ("N1", False),
        ("N2", True),
    ]
    assert len(dataset.news_sha256) == len(dataset.behaviors_sha256) == 64
    assert dataset.news_sha256 == hashlib.sha256(news.read_bytes()).hexdigest()
    assert dataset.behaviors_sha256 == hashlib.sha256(behaviors.read_bytes()).hexdigest()


@pytest.mark.parametrize("offset", [True, float("nan"), 15, 10**1_000, 0.0001])
def test_fixed_offset_rejects_invalid_values(offset: object) -> None:
    with pytest.raises(ValueError, match="offset"):
        fixed_offset(offset)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("news", "behaviors", "message"),
    [
        ("too\tfew\tfields\n", BEHAVIORS, "8 tab-separated"),
        (NEWS + NEWS.splitlines()[0] + "\n", BEHAVIORS, "duplicate MIND news"),
        (NEWS, "1\tU\tbad\t\tN1-1\n", "behavior time"),
        (NEWS, "1\tU\t11/15/2019 9:55:12 AM\t\tN3-1\n", "unknown news"),
        (NEWS, "1\tU\t11/15/2019 9:55:12 AM\t\tN1-x\n", "impression token"),
        (NEWS, BEHAVIORS + BEHAVIORS, "duplicate MIND impression"),
        (NEWS, "1\tU\t11/15/2019 9:55:12 AM\tN3\tN1-1\n", "history"),
        (NEWS, "\tU\t11/15/2019 9:55:12 AM\t\tN1-1\n", "impression id is empty"),
        (NEWS, "1\t\t11/15/2019 9:55:12 AM\t\tN1-1\n", "user id is empty"),
        (NEWS, "1\tU\t11/15/2019 9:55:12 AM\t\t\n", "impressions are empty"),
        (
            NEWS,
            "1\tU\t11/15/2019 9:55:12 AM\t\tN1-0 N1-1\n",
            "duplicate MIND impression article",
        ),
    ],
)
def test_load_mind_rejects_malformed_or_inconsistent_rows(
    tmp_path: Path, news: str, behaviors: str, message: str
) -> None:
    news_path, behavior_path = _files(tmp_path, news, behaviors)
    with pytest.raises(ValueError, match=message):
        load_mind(
            news_path,
            behavior_path,
            catalog_published_at=datetime(2019, 1, 1, tzinfo=UTC),
            behavior_timezone=UTC,
        )


def test_load_mind_requires_aware_catalog_time_and_timezone(tmp_path: Path) -> None:
    news, behaviors = _files(tmp_path)
    with pytest.raises(ValueError, match="timezone-aware"):
        load_mind(
            news,
            behaviors,
            catalog_published_at=datetime(2019, 1, 1),
            behavior_timezone=UTC,
        )
    with pytest.raises(ValueError, match="tzinfo"):
        load_mind(
            news,
            behaviors,
            catalog_published_at=datetime(2019, 1, 1, tzinfo=UTC),
            behavior_timezone="UTC",  # type: ignore[arg-type]
        )


def test_import_mind_cli_writes_auditable_conversion(tmp_path: Path) -> None:
    news, behaviors = _files(tmp_path)
    output = tmp_path / "converted"

    code = main(
        [
            "import-mind",
            "--news",
            str(news),
            "--behaviors",
            str(behaviors),
            "--catalog-published-at",
            "2019-01-01T00:00:00Z",
            "--behavior-utc-offset",
            "-8",
            "--directory",
            str(output),
        ]
    )

    assert code == 0
    assert len(json.loads((output / "articles.json").read_text(encoding="utf-8"))) == 2
    assert len(json.loads((output / "events.json").read_text(encoding="utf-8"))) == 1
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["adapter"] == "mind-tsv-v2"
    assert len(metadata["limitations"]) == 3
    assert len(metadata["news_sha256"]) == len(metadata["behaviors_sha256"]) == 64
    assert metadata["catalog_published_at"] == "2019-01-01T00:00:00+00:00"
    assert metadata["behavior_utc_offset_hours"] == -8.0
    impressions = json.loads((output / "impressions.json").read_text(encoding="utf-8"))
    assert impressions[0]["candidates"] == [
        {"article_id": "N1", "clicked": False},
        {"article_id": "N2", "clicked": True},
    ]

    second_output = tmp_path / "converted-with-other-assumptions"
    assert (
        main(
            [
                "import-mind",
                "--news",
                str(news),
                "--behaviors",
                str(behaviors),
                "--catalog-published-at",
                "2020-01-01T00:00:00Z",
                "--behavior-utc-offset",
                "2",
                "--directory",
                str(second_output),
            ]
        )
        == 0
    )
    second_metadata = json.loads((second_output / "metadata.json").read_text(encoding="utf-8"))
    assert second_metadata["news_sha256"] == metadata["news_sha256"]
    assert second_metadata["behaviors_sha256"] == metadata["behaviors_sha256"]
    assert second_metadata["catalog_published_at"] != metadata["catalog_published_at"]
    assert second_metadata["behavior_utc_offset_hours"] != metadata["behavior_utc_offset_hours"]
