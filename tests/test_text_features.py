from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import UTC, datetime, timedelta

import pytest

from mosaicfeed.cli import _article_record, main
from mosaicfeed.config import FeedConfig
from mosaicfeed.datasets import MindDataset
from mosaicfeed.io import load_articles, load_articles_bytes
from mosaicfeed.learning import FEATURE_NAMES, TEXT_FEATURE_NAMES, PointwiseLogisticRanker
from mosaicfeed.models import MISSING_MIND_TITLE, Article, Event, EventKind
from mosaicfeed.pipeline import build_feed
from mosaicfeed.profile import build_profile
from mosaicfeed.report import render_feed_html
from mosaicfeed.text_features import NewsTextEncoder, TextFeatureConfig

T0 = datetime(2019, 1, 1, tzinfo=UTC)


def article(article_id: str, title: str, *, published_at: datetime = T0) -> Article:
    return Article(article_id, title, "", ("news", "world"), "news", published_at)


def training() -> tuple[list[Article], list[Event]]:
    articles = [article("red", "red red"), article("blue", "blue")]
    events = [
        Event("u", "red", EventKind.CLICK, T0 + timedelta(days=1)),
        Event("u", "blue", EventKind.VIEW, T0 + timedelta(days=2)),
        Event("u", "red", EventKind.CLICK, T0 + timedelta(days=3)),
        Event("u", "blue", EventKind.VIEW, T0 + timedelta(days=4)),
    ]
    return articles, events


def test_news_tfidf_matches_independent_two_document_oracle() -> None:
    articles, _ = training()
    encoder = NewsTextEncoder.fit(reversed(articles), as_of=T0)
    assert encoder.to_state() == NewsTextEncoder.fit(articles, as_of=T0).to_state()
    assert encoder.vocabulary == (
        "category:news",
        "subcategory:world",
        "title:blue",
        "title:red",
    )
    rare_idf = 1.0 + math.log(3.0 / 2.0)
    assert encoder.inverse_document_frequency == pytest.approx((1.0, 1.0, rare_idf, rare_idf))
    expected_red_weight = (1.0 + math.log(2.0)) * rare_idf
    norm = math.sqrt(2.0 + expected_red_weight**2)
    assert encoder.vectorize(articles[0]) == pytest.approx(
        {
            "category:news": 1.0 / norm,
            "subcategory:world": 1.0 / norm,
            "title:red": expected_red_weight / norm,
        }
    )
    cold_red = article("new-red", "red")
    red = encoder.vectorize(articles[0])
    cold = encoder.vectorize(cold_red)
    oracle = sum(red.get(name, 0.0) * value for name, value in cold.items())
    assert encoder.affinity(cold_red, red) == pytest.approx(oracle, abs=1e-15)
    assert "title:red" in cold
    assert "title:new" not in encoder.vectorize(article("unseen", "new"))


def test_sparse_projection_does_not_scan_the_full_8192_word_vocabulary() -> None:
    names = tuple(f"title:t{index:04d}" for index in range(8_192))
    encoder = NewsTextEncoder(TextFeatureConfig(), T0, "0" * 64, names, (1.0,) * len(names))

    class NonIterableVocabulary(tuple):
        def __iter__(self):
            raise AssertionError("projection must not traverse the vocabulary")

    object.__setattr__(encoder, "vocabulary", NonIterableVocabulary(names))
    unseen = article("new", "t4096")
    assert encoder.vectorize(unseen) == {"title:t4096": 1.0}
    assert encoder.affinity(unseen, {"title:t4096": 1.0}) == 1.0


def test_text_feature_resource_limits_and_schema_fail_closed() -> None:
    articles, _ = training()
    with pytest.raises(ValueError, match="max_vocabulary"):
        TextFeatureConfig(max_vocabulary=0)
    with pytest.raises(ValueError, match="max_articles"):
        NewsTextEncoder.fit(articles, as_of=T0, config=TextFeatureConfig(max_articles=1))
    with pytest.raises(ValueError, match="title exceeds"):
        NewsTextEncoder.fit(
            articles,
            as_of=T0,
            config=TextFeatureConfig(max_title_characters=3),
        )
    with pytest.raises(ValueError, match="token limit"):
        NewsTextEncoder.fit(
            articles,
            as_of=T0,
            config=TextFeatureConfig(max_title_tokens=1),
        )
    with pytest.raises(ValueError, match="feature occurrence"):
        NewsTextEncoder.fit(
            articles,
            as_of=T0,
            config=TextFeatureConfig(max_feature_occurrences=1),
        )
    encoder = NewsTextEncoder.fit(articles, as_of=T0)
    state = encoder.to_state()
    assert NewsTextEncoder.from_state(state).to_state() == state
    damaged = copy.deepcopy(state)
    damaged["vocabulary"] = ["title:illegal space"]
    with pytest.raises(ValueError, match="vocabulary"):
        NewsTextEncoder.from_state(damaged)
    damaged = copy.deepcopy(state)
    damaged["inverse_document_frequency"] = [float("nan")] * 4
    with pytest.raises(ValueError, match="IDF"):
        NewsTextEncoder.from_state(damaged)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"schema_version": 2}, "unsupported"),
        ({"training_articles_sha256": "bad"}, "digest"),
        ({"source_kind": "unknown"}, "source"),
        ({"vocabulary": ["title:bad space"]}, "vocabulary"),
        ({"vocabulary": ["title:red", "title:red"]}, "vocabulary"),
        ({"inverse_document_frequency": [0.5, 1.0, 1.0, 1.0]}, "IDF"),
        ({"inverse_document_frequency": [1.0]}, "IDF"),
        ({"config": {}}, "configuration"),
    ],
)
def test_text_state_rejects_mutated_schema_even_without_model_checksum(change, message) -> None:
    encoder = NewsTextEncoder.fit(training()[0], as_of=T0)
    state = encoder.to_state()
    state.update(change)
    with pytest.raises(ValueError, match=message):
        NewsTextEncoder.from_state(state)


def test_text_encoder_rejects_invisible_duplicate_and_empty_vocabulary() -> None:
    articles, _ = training()
    with pytest.raises(ValueError, match="unique"):
        NewsTextEncoder.fit([articles[0], articles[0]], as_of=T0)
    with pytest.raises(ValueError, match="not yet published"):
        NewsTextEncoder.fit(
            [article("later", "red", published_at=T0 + timedelta(days=1))], as_of=T0
        )
    with pytest.raises(ValueError, match="at least one"):
        NewsTextEncoder.fit([], as_of=T0)
    with pytest.raises(ValueError, match="no vocabulary"):
        NewsTextEncoder.fit(
            articles,
            as_of=T0,
            config=TextFeatureConfig(min_document_frequency=3),
        )
    with pytest.raises(ValueError, match="max_articles"):
        TextFeatureConfig(max_articles=1, min_document_frequency=2)


def test_text_vector_fail_closed_for_invalid_utf8_and_unseen_content() -> None:
    encoder = NewsTextEncoder.fit(training()[0], as_of=T0)
    assert encoder.vectorize(Article("oov", "unseen", "", ("other",), "other", T0)) == {}
    assert encoder.affinity(article("new", "red"), {}) == 0.0
    with pytest.raises(ValueError, match="UTF-8"):
        encoder.vectorize(article("bad", "bad\ud800"))
    with pytest.raises(ValueError, match="categories"):
        encoder.vectorize(
            Article("many", "red", "", tuple(f"topic-{index}" for index in range(17)), "other", T0)
        )
    with pytest.raises(ValueError, match="history vector"):
        encoder.affinity(article("new", "red"), {"unknown": 1.0})
    with pytest.raises(ValueError, match="history vector"):
        encoder.affinity(article("new", "red"), {"title:red": float("nan")})


def test_text_schema_and_sparse_vector_reject_edge_shapes() -> None:
    articles, _ = training()
    encoder = NewsTextEncoder.fit(articles, as_of=T0)
    with pytest.raises(ValueError, match="timezone-aware"):
        NewsTextEncoder.fit(articles, as_of=datetime(2019, 1, 1))
    with pytest.raises(ValueError, match="unique Article"):
        NewsTextEncoder.fit([object()], as_of=T0)
    with pytest.raises(ValueError, match="vocabulary bounds"):
        encoder.affinity(article("new", "red"), {f"extra:{index}": 1.0 for index in range(5)})
    long_title_token = article("long-token", "x" * 300)
    assert not any(name.startswith("title:") for name in encoder.vectorize(long_title_token))
    with pytest.raises(ValueError, match="categories"):
        encoder.vectorize(Article("long-category", "red", "", ("x" * 257,), "other", T0))

    state = encoder.to_state()
    with pytest.raises(ValueError, match="malformed"):
        NewsTextEncoder.from_state([])
    damaged = copy.deepcopy(state)
    damaged["format"] = "unknown"
    with pytest.raises(ValueError, match="unsupported"):
        NewsTextEncoder.from_state(damaged)
    damaged = copy.deepcopy(state)
    damaged["vocabulary"] = "not-an-array"
    with pytest.raises(ValueError, match="arrays"):
        NewsTextEncoder.from_state(damaged)
    damaged = copy.deepcopy(state)
    damaged["trained_as_of"] = "2019-01-01"
    with pytest.raises(ValueError, match="timezone"):
        NewsTextEncoder.from_state(damaged)


def test_missing_mind_title_is_lossless_in_export_and_excluded_from_tokens(tmp_path) -> None:
    news = tmp_path / "news.tsv"
    behaviors = tmp_path / "behaviors.tsv"
    news.write_text("N1\t\tworld\t\t\thttps://example.test\t[]\t[]\n", encoding="utf-8")
    behaviors.write_text("1\tU1\t11/15/2019 9:55:12 AM\t\tN1-1\n", encoding="utf-8")
    dataset = MindDataset(
        news.read_bytes(), behaviors.read_bytes(), catalog_published_at=T0, behavior_timezone=UTC
    )
    assert dataset.articles[0].title == MISSING_MIND_TITLE
    assert dataset.articles[0].title_missing
    assert dataset.articles[0].topics == ("world",)
    assert dataset.articles[0].mind_category == ""
    assert dataset.articles[0].mind_subcategory == "world"
    encoder = NewsTextEncoder.fit(dataset.articles, as_of=T0)
    assert not any(name.startswith("title:") for name in encoder.vocabulary)
    assert "category:uncategorized" not in encoder.vocabulary
    assert encoder.vectorize(dataset.articles[0])
    output = tmp_path / "imported"
    assert (
        main(
            [
                "import-mind",
                "--news",
                str(news),
                "--behaviors",
                str(behaviors),
                "--catalog-published-at",
                T0.isoformat(),
                "--behavior-utc-offset",
                "0",
                "--directory",
                str(output),
            ]
        )
        == 0
    )
    record = json.loads((output / "articles.json").read_text(encoding="utf-8"))[0]
    assert record["title"] == ""
    assert record["title_missing"] is True
    assert record["category_missing"] is True
    assert record["mind_category"] == ""
    assert record["mind_subcategory"] == "world"
    assert load_articles(output / "articles.json") == list(dataset.articles)
    rendered = render_feed_html(
        build_feed("u", dataset.articles, (), as_of=T0),
        {item.id: item for item in dataset.articles},
    )
    assert MISSING_MIND_TITLE not in rendered
    assert '<h2 data-title-missing="true">Title unavailable (source field empty)</h2>' in rendered

    impossible = MindDataset(
        news.read_bytes(),
        behaviors.read_bytes(),
        catalog_published_at=datetime(2020, 1, 1, tzinfo=UTC),
        behavior_timezone=UTC,
    )
    assert impossible.impression_records[0].occurred_at < impossible.articles[0].published_at
    with pytest.raises(ValueError, match="predates"):
        PointwiseLogisticRanker().fit(
            impossible.articles,
            impossible.events,
            as_of=datetime(2020, 1, 2, tzinfo=UTC),
            text_features=True,
        )


def test_mind_equal_category_and_blank_category_keep_source_field_identity() -> None:
    news = (
        b"N1\tsports\tsports\tMatch\t\thttps://example.test\t[]\t[]\n"
        b"N2\t\tworld\tWorld\t\thttps://example.test\t[]\t[]\n"
    )
    behaviors = b"1\tU1\t11/15/2019 9:55:12 AM\t\tN1-1 N2-0\n"
    dataset = MindDataset(news, behaviors, catalog_published_at=T0, behavior_timezone=UTC)
    same, blank = dataset.articles
    assert same.topics == ("sports",)
    assert same.mind_category == same.mind_subcategory == "sports"
    round_trip = load_articles_bytes(json.dumps([_article_record(same)]).encode("utf-8"))
    assert round_trip == [same]
    assert blank.topics == ("world",)
    assert blank.category_missing and not blank.subcategory_missing
    encoder = NewsTextEncoder.fit(dataset.articles, as_of=T0)
    assert {"category:sports", "subcategory:sports", "subcategory:world"}.issubset(
        encoder.vocabulary
    )
    assert "category:world" not in encoder.vocabulary
    assert "category:uncategorized" not in encoder.vocabulary
    event = Event("u", blank.id, EventKind.CLICK, T0 + timedelta(days=1))
    profile = build_profile(
        "u",
        [event],
        {item.id: item for item in dataset.articles},
        as_of=event.occurred_at,
        config=FeedConfig(),
    )
    assert set(profile.topic_weights) == {"world"}


def test_text_ranker_schema_two_cold_start_and_old_default_compatibility(tmp_path) -> None:
    articles, events = training()
    cutoff = T0 + timedelta(days=4)
    legacy = PointwiseLogisticRanker(epochs=100, learning_rate=0.1).fit(
        articles, events, as_of=cutoff
    )
    assert legacy.to_state()["schema_version"] == 1
    assert list(legacy.weights) == list(FEATURE_NAMES)
    assert "text_encoder" not in legacy.to_state()
    model = PointwiseLogisticRanker(epochs=100, learning_rate=0.1).fit(
        articles, events, as_of=cutoff, text_features=True
    )
    assert model.to_state()["schema_version"] == 2
    assert list(model.weights) == list(TEXT_FEATURE_NAMES)
    assert model.weights["text_affinity"] != 0
    cold = [article("cold-red", "red"), article("cold-blue", "blue")]
    ranking = model.rank_for_user(
        "u", [*articles, *cold], events, as_of=cutoff, candidate_ids=[item.id for item in cold]
    )
    assert {item.article_id for item in ranking} == {"cold-red", "cold-blue"}
    assert ranking[0].probability != ranking[1].probability
    path = tmp_path / "text-model.json"
    model.save(path)
    restored = PointwiseLogisticRanker.load(path)
    assert restored.to_state() == model.to_state()
    assert (
        restored.rank_for_user(
            "u", [*articles, *cold], events, as_of=cutoff, candidate_ids=[item.id for item in cold]
        )
        == ranking
    )
    damaged = copy.deepcopy(model.to_state())
    damaged["text_encoder"]["vocabulary"].append("title:test")
    with pytest.raises(ValueError, match="checksum"):
        PointwiseLogisticRanker.from_state(damaged)


def test_declared_multi_document_snapshot_matches_independent_idf_oracle() -> None:
    articles, events = training()
    model = PointwiseLogisticRanker().fit(
        articles,
        events,
        as_of=T0 + timedelta(days=4),
        text_features=True,
        text_vocabulary_articles=reversed(articles),
    )
    encoder = model.text_encoder
    assert encoder is not None
    assert encoder.source_kind == "declared-training-snapshot"
    assert encoder.vocabulary == NewsTextEncoder.fit(articles, as_of=T0).vocabulary
    idf = dict(zip(encoder.vocabulary, encoder.inverse_document_frequency, strict=True))
    assert idf["title:red"] == pytest.approx(1.0 + math.log(3.0 / 2.0))
    assert idf["category:news"] == pytest.approx(1.0)
    assert (
        encoder.training_articles_sha256
        == NewsTextEncoder.fit(articles, as_of=T0).training_articles_sha256
    )


def test_declared_snapshot_rejects_unavailable_or_mismatched_news() -> None:
    articles, events = training()
    late = article("late", "later", published_at=T0 + timedelta(days=2))
    with pytest.raises(ValueError, match="not yet published"):
        PointwiseLogisticRanker().fit(
            [*articles, late],
            events,
            as_of=T0 + timedelta(days=4),
            text_features=True,
            text_vocabulary_articles=[articles[0], late],
        )
    with pytest.raises(ValueError, match="must match"):
        PointwiseLogisticRanker().fit(
            articles,
            events,
            as_of=T0 + timedelta(days=4),
            text_features=True,
            text_vocabulary_articles=[article("red", "changed")],
        )
    with pytest.raises(ValueError, match="requires enabled"):
        PointwiseLogisticRanker().fit(
            articles,
            events,
            as_of=T0 + timedelta(days=4),
            text_vocabulary_articles=articles,
        )


def test_text_training_and_ranking_limit_inputs_and_reject_future_encoder_state() -> None:
    articles, events = training()
    cutoff = T0 + timedelta(days=4)
    with pytest.raises(ValueError, match="text_config requires"):
        PointwiseLogisticRanker().fit(
            articles, events, as_of=cutoff, text_config=TextFeatureConfig()
        )
    with pytest.raises(ValueError, match="max_articles"):
        PointwiseLogisticRanker().fit(
            articles,
            events,
            as_of=cutoff,
            text_features=True,
            text_config=TextFeatureConfig(max_articles=1),
        )
    with pytest.raises(ValueError, match="max_events"):
        PointwiseLogisticRanker().fit(
            articles,
            events,
            as_of=cutoff,
            text_features=True,
            text_config=TextFeatureConfig(max_events=2),
        )
    model = PointwiseLogisticRanker().fit(
        articles,
        events,
        as_of=cutoff,
        text_features=True,
        text_config=TextFeatureConfig(max_articles=2),
    )
    with pytest.raises(ValueError, match="max_articles"):
        model.rank_for_user("u", [*articles, article("third", "red")], events, as_of=cutoff)
    with pytest.raises(ValueError, match="max_articles"):
        model.rank_for_user(
            "u", articles, events, as_of=cutoff, candidate_ids=["red", "blue", "third"]
        )

    state = model.to_state()
    state["text_encoder"]["trained_as_of"] = (cutoff + timedelta(days=1)).isoformat()
    payload = {name: value for name, value in state.items() if name != "state_sha256"}
    state["state_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
    with pytest.raises(ValueError, match="after model training cutoff"):
        PointwiseLogisticRanker.from_state(state)


def test_text_vocabulary_ignores_future_and_unlogged_test_news() -> None:
    articles, events = training()
    cutoff = T0 + timedelta(days=4)
    baseline = (
        PointwiseLogisticRanker(seed=5)
        .fit(articles, events, as_of=cutoff, text_features=True)
        .to_state()
    )
    future_news = article("future", "secret futureword", published_at=cutoff + timedelta(days=1))
    unlogged_test_news = article("held-out", "secret heldoutword")
    future_event = Event("u", "future", EventKind.CLICK, cutoff + timedelta(days=2))
    extended = (
        PointwiseLogisticRanker(seed=5)
        .fit(
            [*articles, future_news, unlogged_test_news],
            [*events, future_event],
            as_of=cutoff,
            text_features=True,
        )
        .to_state()
    )
    assert extended == baseline
    assert "title:futureword" not in extended["text_encoder"]["vocabulary"]
    assert "title:heldoutword" not in extended["text_encoder"]["vocabulary"]


@pytest.mark.parametrize("declared", [False, True])
def test_later_training_event_before_cutoff_cannot_change_earlier_vocabulary(
    declared: bool,
) -> None:
    articles, events = training()
    cutoff = T0 + timedelta(days=4)
    snapshot = articles if declared else None
    baseline = PointwiseLogisticRanker().fit(
        articles,
        events,
        as_of=cutoff,
        text_features=True,
        text_vocabulary_articles=snapshot,
    )
    later_news = article("later", "laterword")
    later_event = Event("other", later_news.id, EventKind.VIEW, T0 + timedelta(days=3))
    extended = PointwiseLogisticRanker().fit(
        [*articles, later_news],
        [*events, later_event],
        as_of=cutoff,
        text_features=True,
        text_vocabulary_articles=snapshot,
    )
    assert extended.text_encoder is not None
    assert baseline.text_encoder is not None
    assert extended.text_encoder.to_state() == baseline.text_encoder.to_state()
    assert "title:laterword" not in extended.text_encoder.vocabulary


def test_rank_text_history_excludes_future_interactions_and_future_news() -> None:
    articles, events = training()
    cutoff = T0 + timedelta(days=4)
    model = PointwiseLogisticRanker(epochs=50).fit(
        articles, events, as_of=cutoff, text_features=True
    )
    cold = article("cold", "red")
    future = article("not-yet-visible", "red", published_at=cutoff + timedelta(days=2))
    ranking = model.rank_for_user("u", [*articles, cold, future], events, as_of=cutoff)
    future_interaction = Event("u", "blue", EventKind.CLICK, cutoff + timedelta(days=1))
    repeated = model.rank_for_user(
        "u", [*articles, cold, future], [*events, future_interaction], as_of=cutoff
    )
    assert repeated == ranking
    assert all(item.article_id != future.id for item in ranking)

    late_visible = article("late", "red", published_at=cutoff - timedelta(days=1))
    eligible = [*articles, cold, late_visible]
    baseline = model.rank_for_user("u", eligible, events, as_of=cutoff, candidate_ids=[cold.id])
    impossible_history = Event("u", late_visible.id, EventKind.CLICK, cutoff - timedelta(days=2))
    assert (
        model.rank_for_user(
            "u", eligible, [*events, impossible_history], as_of=cutoff, candidate_ids=[cold.id]
        )
        == baseline
    )


def test_text_model_size_budget_fails_before_replacing_state(tmp_path, monkeypatch) -> None:
    import mosaicfeed.learning as learning_module

    articles, events = training()
    model = PointwiseLogisticRanker().fit(
        articles, events, as_of=T0 + timedelta(days=4), text_features=True
    )
    path = tmp_path / "existing.json"
    path.write_bytes(b"old-state")
    monkeypatch.setattr(learning_module, "MAX_MODEL_STATE_BYTES", 128)
    with pytest.raises(ValueError, match="size limit"):
        model.save(path)
    assert path.read_bytes() == b"old-state"
    large = tmp_path / "large.json"
    large.write_bytes(b"x" * 129)
    with pytest.raises(ValueError, match="size limit"):
        PointwiseLogisticRanker.load(large)


def test_text_ranker_cli_round_trip(tmp_path, capsys) -> None:
    from mosaicfeed.io import write_json

    articles, events = training()
    articles_path = tmp_path / "articles.json"
    vocabulary_path = tmp_path / "training-news.json"
    events_path = tmp_path / "events.json"
    model_path = tmp_path / "model.json"
    write_json(
        articles_path,
        [
            {
                "id": item.id,
                "title": item.title,
                "topics": list(item.topics),
                "source": item.source,
                "published_at": item.published_at.isoformat(),
            }
            for item in articles
        ],
    )
    write_json(
        events_path,
        [
            {
                "user_id": item.user_id,
                "article_id": item.article_id,
                "kind": item.kind.value,
                "occurred_at": item.occurred_at.isoformat(),
            }
            for item in events
        ],
    )
    write_json(vocabulary_path, json.loads(articles_path.read_text(encoding="utf-8")))
    assert (
        main(
            [
                "train-click-model",
                "--articles",
                str(articles_path),
                "--events",
                str(events_path),
                "--as-of",
                (T0 + timedelta(days=4)).isoformat(),
                "--output",
                str(model_path),
                "--text-features",
                "--text-vocabulary-articles",
                str(vocabulary_path),
                "--epochs",
                "50",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["text_vocabulary_size"] == 4
    assert report["text_vocabulary_source"] == "declared-training-snapshot"
    assert "text_affinity" in report["weights"]
    assert (
        main(
            [
                "rank-click-model",
                "--model",
                str(model_path),
                "--articles",
                str(articles_path),
                "--events",
                str(events_path),
                "--user",
                "new",
                "--as-of",
                (T0 + timedelta(days=4)).isoformat(),
                "--k",
                "2",
            ]
        )
        == 0
    )
    assert len(json.loads(capsys.readouterr().out)["predictions"]) == 2
