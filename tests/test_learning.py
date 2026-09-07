from __future__ import annotations

import copy
import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mosaicfeed.cli import main
from mosaicfeed.config import FeedConfig
from mosaicfeed.io import write_json
from mosaicfeed.learning import ClickPrediction, PointwiseLogisticRanker
from mosaicfeed.models import Article, Event, EventKind, UserProfile

ORIGIN = datetime(2026, 1, 1, tzinfo=UTC)


def catalog() -> list[Article]:
    return [
        Article("positive", "Strong", "", ("useful",), "A", ORIGIN, 0.95, 0.4),
        Article("negative", "Weak", "", ("noise",), "B", ORIGIN, 0.05, 0.4),
        Article("candidate-a", "Candidate A", "", ("useful",), "C", ORIGIN, 0.85, 0.3),
        Article("candidate-b", "Candidate B", "", ("noise",), "D", ORIGIN, 0.15, 0.3),
    ]


def labels() -> list[Event]:
    return [
        Event("u1", "positive", EventKind.CLICK, ORIGIN + timedelta(days=1)),
        Event("u1", "negative", EventKind.VIEW, ORIGIN + timedelta(days=2)),
        Event("u2", "positive", EventKind.LIKE, ORIGIN + timedelta(days=1, hours=1)),
        Event("u2", "negative", EventKind.HIDE, ORIGIN + timedelta(days=2, hours=1)),
        Event("u3", "positive", EventKind.CLICK, ORIGIN + timedelta(days=1, hours=2)),
        Event("u3", "negative", EventKind.VIEW, ORIGIN + timedelta(days=2, hours=2)),
    ]


def fitted(**kwargs: object) -> PointwiseLogisticRanker:
    return PointwiseLogisticRanker(
        epochs=kwargs.get("epochs", 80),  # type: ignore[arg-type]
        learning_rate=kwargs.get("learning_rate", 0.1),  # type: ignore[arg-type]
        l2=kwargs.get("l2", 0.001),  # type: ignore[arg-type]
        seed=kwargs.get("seed", 7),  # type: ignore[arg-type]
    ).fit(
        catalog(),
        labels(),
        as_of=ORIGIN + timedelta(days=3),
        config=FeedConfig(exploration_weight=0.0),
    )


def article_record(article: Article) -> dict[str, object]:
    return {
        "id": article.id,
        "title": article.title,
        "summary": article.summary,
        "topics": list(article.topics),
        "source": article.source,
        "published_at": article.published_at.isoformat(),
        "quality": article.quality,
        "popularity": article.popularity,
    }


def event_record(event: Event) -> dict[str, object]:
    return {
        "user_id": event.user_id,
        "article_id": event.article_id,
        "kind": event.kind.value,
        "occurred_at": event.occurred_at.isoformat(),
    }


def test_click_prediction_validates_and_serializes() -> None:
    prediction = ClickPrediction("a", 0.25, 1)
    assert prediction.to_dict() == {"article_id": "a", "probability": 0.25, "rank": 1}
    with pytest.raises(ValueError):
        ClickPrediction("", 0.5, 1)
    with pytest.raises(ValueError):
        ClickPrediction("a", 1.1, 1)
    with pytest.raises(ValueError):
        ClickPrediction("a", 0.5, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"epochs": True},
        {"learning_rate": 0},
        {"learning_rate": float("inf")},
        {"l2": -1},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_invalid_hyperparameters_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PointwiseLogisticRanker(**kwargs)  # type: ignore[arg-type]


def test_not_fitted_access_is_rejected() -> None:
    model = PointwiseLogisticRanker()
    with pytest.raises(ValueError, match="not been fitted"):
        _ = model.weights
    with pytest.raises(ValueError, match="not been fitted"):
        _ = model.training_examples


def test_training_is_deterministic_and_does_not_touch_global_random() -> None:
    random.seed(12345)
    before = random.getstate()
    first = fitted().to_state()
    assert random.getstate() == before
    assert first == fitted().to_state()


def test_different_seed_changes_the_sgd_path() -> None:
    assert fitted(seed=1).weights != fitted(seed=2).weights


def test_training_learns_positive_quality_evidence() -> None:
    model = fitted()
    profile = UserProfile("new")
    articles = {article.id: article for article in catalog()}
    positive = model.predict(articles["candidate-a"], profile, as_of=ORIGIN + timedelta(days=3))
    negative = model.predict(articles["candidate-b"], profile, as_of=ORIGIN + timedelta(days=3))
    assert positive > negative
    assert model.weights["quality"] > 0


def test_future_events_do_not_change_the_fitted_state() -> None:
    cutoff = ORIGIN + timedelta(days=3)
    baseline = PointwiseLogisticRanker(seed=4).fit(catalog(), labels(), as_of=cutoff).to_state()
    future = Event("u4", "positive", EventKind.CLICK, cutoff + timedelta(seconds=1))
    extended = (
        PointwiseLogisticRanker(seed=4).fit(catalog(), [*labels(), future], as_of=cutoff).to_state()
    )
    assert baseline == extended


def test_training_digest_identifies_the_exact_eligible_examples() -> None:
    first = fitted()
    second = fitted()
    assert first.training_sha256 == second.training_sha256
    assert len(first.training_sha256) == 64
    changed_labels = labels()
    changed_labels[0] = Event("u1", "positive", EventKind.VIEW, ORIGIN + timedelta(days=1))
    changed = PointwiseLogisticRanker(seed=7).fit(
        catalog(),
        changed_labels,
        as_of=ORIGIN + timedelta(days=3),
        config=FeedConfig(exploration_weight=0.0),
    )
    assert changed.training_sha256 != first.training_sha256


def test_input_order_breaks_equal_timestamp_history_ties_deterministically() -> None:
    same_time = ORIGIN + timedelta(days=1)
    events = [
        Event("u", "positive", EventKind.CLICK, same_time),
        Event("u", "negative", EventKind.VIEW, same_time),
        Event("v", "positive", EventKind.CLICK, same_time),
        Event("v", "negative", EventKind.VIEW, same_time),
    ]
    first = PointwiseLogisticRanker(seed=3).fit(
        catalog(), events, as_of=same_time, config=FeedConfig(exploration_weight=0)
    )
    second = PointwiseLogisticRanker(seed=3).fit(
        catalog(), events, as_of=same_time, config=FeedConfig(exploration_weight=0)
    )
    assert first.to_state() == second.to_state()


def test_fit_rejects_invalid_data_contracts() -> None:
    cutoff = ORIGIN + timedelta(days=3)
    with pytest.raises(ValueError, match="at least one Article"):
        PointwiseLogisticRanker().fit([], labels(), as_of=cutoff)
    with pytest.raises(ValueError, match="unique"):
        PointwiseLogisticRanker().fit([catalog()[0], catalog()[0]], labels(), as_of=cutoff)
    with pytest.raises(ValueError, match="Event"):
        PointwiseLogisticRanker().fit(catalog(), [object()], as_of=cutoff)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="unknown article"):
        PointwiseLogisticRanker().fit(
            catalog(),
            [*labels(), Event("u", "missing", EventKind.VIEW, cutoff)],
            as_of=cutoff,
        )
    future_article = Article("late", "Late", "", ("x",), "L", cutoff + timedelta(days=1), 0.5, 0.5)
    with pytest.raises(ValueError, match="predates"):
        PointwiseLogisticRanker().fit(
            [*catalog(), future_article],
            [*labels(), Event("u", "late", EventKind.VIEW, cutoff)],
            as_of=cutoff,
        )
    with pytest.raises(ValueError, match="positive and one negative"):
        PointwiseLogisticRanker().fit(
            catalog(),
            [Event("u", "positive", EventKind.CLICK, cutoff)],
            as_of=cutoff,
        )
    with pytest.raises(ValueError, match="eligible"):
        PointwiseLogisticRanker().fit(catalog(), labels(), as_of=ORIGIN)


def test_rank_for_user_is_stable_and_respects_seen_and_future_items() -> None:
    model = fitted()
    future = Article("future", "Future", "", ("useful",), "Z", ORIGIN + timedelta(days=5), 1.0, 1.0)
    first = model.rank_for_user(
        "u1", [*catalog(), future], labels(), as_of=ORIGIN + timedelta(days=3), k=10
    )
    second = model.rank_for_user(
        "u1", [*catalog(), future], labels(), as_of=ORIGIN + timedelta(days=3), k=10
    )
    assert first == second
    assert {item.article_id for item in first}.isdisjoint({"positive", "negative", "future"})
    assert [item.rank for item in first] == list(range(1, len(first) + 1))


def test_rank_ties_use_article_id_and_validate_inputs() -> None:
    model = fitted()
    state = model.to_state()
    state["weights"] = [0.0] * 7
    tied = PointwiseLogisticRanker.from_state(state)
    ranked = tied.rank_for_user(
        "new", list(reversed(catalog())), [], as_of=ORIGIN + timedelta(days=3), k=4
    )
    assert [item.article_id for item in ranked] == sorted(article.id for article in catalog())
    with pytest.raises(ValueError, match="positive integer"):
        tied.rank_for_user("new", catalog(), [], as_of=ORIGIN, k=0)
    with pytest.raises(ValueError, match="unique"):
        tied.rank_for_user("new", [catalog()[0], catalog()[0]], [], as_of=ORIGIN, k=1)


def test_state_and_file_round_trip_are_exact(tmp_path: Path) -> None:
    model = fitted()
    restored = PointwiseLogisticRanker.from_state(model.to_state())
    assert restored.to_state() == model.to_state()
    path = tmp_path / "nested" / "model.json"
    model.save(path)
    assert PointwiseLogisticRanker.load(path).to_state() == model.to_state()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: state.update(extra=True),
        lambda state: state.update(format="other"),
        lambda state: state.update(schema_version=2),
        lambda state: state.update(feature_names=["bias"]),
        lambda state: state.update(weights=[0.0]),
        lambda state: state["weights"].__setitem__(0, float("nan")),
        lambda state: state["training"].update(examples=99),
        lambda state: state["training"].update(as_of="not-a-time"),
        lambda state: state["training"].update(examples_sha256="NOT-A-DIGEST"),
    ],
)
def test_tampered_state_is_rejected(mutation: object) -> None:
    state = copy.deepcopy(fitted().to_state())
    mutation(state)  # type: ignore[operator]
    with pytest.raises(ValueError):
        PointwiseLogisticRanker.from_state(state)


def test_cli_trains_and_ranks_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    articles_path = tmp_path / "articles.json"
    events_path = tmp_path / "events.json"
    model_path = tmp_path / "click-model.json"
    output_path = tmp_path / "ranked.json"
    write_json(articles_path, [article_record(article) for article in catalog()])
    write_json(events_path, [event_record(event) for event in labels()])
    as_of = (ORIGIN + timedelta(days=3)).isoformat()

    assert (
        main(
            [
                "train-click-model",
                "--articles",
                str(articles_path),
                "--events",
                str(events_path),
                "--as-of",
                as_of,
                "--output",
                str(model_path),
                "--epochs",
                "10",
            ]
        )
        == 0
    )
    training = json.loads(capsys.readouterr().out)
    assert training["training_examples"] == len(labels())
    assert len(training["training_sha256"]) == 64
    assert model_path.exists()

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
                as_of,
                "--k",
                "2",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["user_id"] == "new"
    assert len(payload["predictions"]) == 2
