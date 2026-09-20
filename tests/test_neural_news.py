"""Independent mathematical and temporal checks for the local neural-news baseline."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mosaicfeed.cli import main
from mosaicfeed.datasets import MindCandidate, MindImpression
from mosaicfeed.models import Article
from mosaicfeed.neural_news import (
    NeuralNewsConfig,
    NeuralNewsRanker,
    run_neural_news_experiment,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
CUTOFF = START + timedelta(days=5)


def _articles() -> tuple[Article, ...]:
    return (
        Article("a", "Grid solar", "", ("energy",), "s1", START),
        Article("b", "Match sports", "", ("sports",), "s2", START),
        Article("c", "Grid power", "", ("energy",), "s3", START),
        Article("d", "Sports game", "", ("sports",), "s4", START),
    )


def _row(
    identity: str, user: str, day: int, first: str, second: str, clicked_first: bool
) -> MindImpression:
    return MindImpression(
        identity,
        user,
        START + timedelta(days=day),
        (MindCandidate(first, clicked_first), MindCandidate(second, not clicked_first)),
    )


def _train() -> tuple[MindImpression, ...]:
    return (
        _row("t1", "u", 1, "a", "b", True),
        _row("t2", "u", 2, "c", "d", True),
        _row("t3", "u", 3, "a", "d", True),
    )


def _validation() -> tuple[MindImpression, ...]:
    return (_row("v1", "u", 6, "c", "b", True),)


def test_neural_score_matches_independent_tanh_dot_oracle() -> None:
    model = NeuralNewsRanker.from_state(
        {
            "format": "mosaicfeed.neural-news",
            "schema_version": 1,
            "config": NeuralNewsConfig(dimension=2, epochs=1, learning_rate=0.1, seed=1).to_dict(),
            "cutoff": CUTOFF.isoformat(),
            "vocabulary": ["<unk>", "grid", "solar"],
            "embeddings": [[0.0, 0.0], [0.4, 0.2], [0.2, -0.2]],
            "user_bias": [0.1, -0.1],
            "histories": {"u": ["a"]},
            "train_impression_ids": ["t1"],
            "training_history_lengths": {"t1": 0},
            "work_units_upper_bound": 0,
        }
    )
    candidate = Article("x", "Grid solar", "", ("energy",), "s", START)
    validation = (_row("v1", "u", 6, "x", "b", True),)
    scores = model.score_impressions((*_articles(), candidate), validation)
    x = math.tanh(0.3)
    y = math.tanh(0.0)
    expected = (0.1 + x) * x + (-0.1 + y) * y
    assert scores["v1"]["x"] == pytest.approx(expected, abs=1e-15)


def test_fit_is_deterministic_and_dev_labels_never_change_model_or_scores() -> None:
    config = NeuralNewsConfig(dimension=4, epochs=3, learning_rate=0.2, seed=7)
    first = NeuralNewsRanker.fit(_articles(), _train(), cutoff=CUTOFF, config=config)
    second = NeuralNewsRanker.fit(_articles(), _train(), cutoff=CUTOFF, config=config)
    assert first.to_state() == second.to_state()
    assert NeuralNewsRanker.from_state(first.to_state()).to_state() == first.to_state()
    held_out = _validation()
    flipped = (_row("v1", "u", 6, "c", "b", False),)
    assert first.score_impressions(_articles(), held_out) == first.score_impressions(
        _articles(), flipped
    )
    assert first.to_state()["histories"] == {"u": ["a", "c", "a"]}
    assert any(abs(value) > 0.0 for value in first.to_state()["embeddings"][1][1:])


def test_same_timestamp_training_clicks_are_not_prior_history() -> None:
    simultaneous = (
        _row("t1", "u", 1, "a", "b", True),
        _row("t2", "u", 1, "c", "d", True),
        _row("t3", "u", 2, "a", "d", True),
    )
    model = NeuralNewsRanker.fit(
        _articles(), simultaneous, cutoff=CUTOFF, config=NeuralNewsConfig(epochs=1)
    )
    assert model.training_history_lengths == {"t1": 0, "t2": 0, "t3": 2}


def _sources() -> tuple[bytes, bytes, bytes]:
    articles = json.dumps(
        [
            {
                "id": article.id,
                "title": article.title,
                "topics": list(article.topics),
                "source": article.source,
                "published_at": article.published_at.isoformat(),
            }
            for article in _articles()
        ]
    ).encode()
    train = json.dumps(
        [
            {
                "impression_id": row.impression_id,
                "user_id": row.user_id,
                "occurred_at": row.occurred_at.isoformat(),
                "candidates": [
                    {"article_id": c.article_id, "clicked": c.clicked} for c in row.candidates
                ],
            }
            for row in _train()
        ]
    ).encode()
    validation = json.dumps(
        [
            {
                "impression_id": row.impression_id,
                "user_id": row.user_id,
                "occurred_at": row.occurred_at.isoformat(),
                "candidates": [
                    {"article_id": c.article_id, "clicked": c.clicked} for c in row.candidates
                ],
            }
            for row in _validation()
        ]
    ).encode()
    return articles, train, validation


def test_runner_keeps_source_provenance_and_complete_candidate_scores() -> None:
    articles, train, validation = _sources()
    output = run_neural_news_experiment(
        articles, train, validation, cutoff=CUTOFF, config=NeuralNewsConfig(epochs=2)
    )
    assert output["format"] == "mosaicfeed.neural-news-experiment"
    assert len(output["scores"]) == 2
    assert {row["article_id"] for row in output["scores"]} == {"b", "c"}
    assert output["metrics"]["impressions"] == 1
    assert output["source_sha256"] == {
        "articles": hashlib.sha256(articles).hexdigest(),
        "train": hashlib.sha256(train).hexdigest(),
        "validation": hashlib.sha256(validation).hexdigest(),
    }


def test_validation_label_flip_changes_only_evaluation_and_source_hash() -> None:
    articles, train, validation = _sources()
    flipped_rows = json.loads(validation)
    for candidate in flipped_rows[0]["candidates"]:
        candidate["clicked"] = not candidate["clicked"]
    flipped = json.dumps(flipped_rows).encode()
    config = NeuralNewsConfig(epochs=2, seed=11)
    original = run_neural_news_experiment(articles, train, validation, cutoff=CUTOFF, config=config)
    changed = run_neural_news_experiment(articles, train, flipped, cutoff=CUTOFF, config=config)
    assert changed["model"] == original["model"]
    assert changed["state_sha256"] == original["state_sha256"]
    assert changed["scores"] == original["scores"]
    assert changed["source_sha256"]["validation"] != original["source_sha256"]["validation"]
    assert changed["metrics"]["impression_sha256"] != original["metrics"]["impression_sha256"]


def test_single_step_gradient_matches_finite_difference_oracle() -> None:
    base = NeuralNewsRanker.fit(
        _articles(),
        _train()[:1],
        cutoff=CUTOFF,
        config=NeuralNewsConfig(dimension=2, epochs=1, learning_rate=0.01),
    ).to_state()
    base["vocabulary"] = ["<unk>", "grid", "solar", "match", "sports"]
    base["embeddings"] = [[0.0, 0.0], [0.2, 0.1], [0.1, -0.1], [-0.2, 0.1], [0.1, 0.2]]
    base["user_bias"] = [0.3, -0.2]
    model = NeuralNewsRanker.from_state(base)
    row = _train()[0]
    token_ids = model._token_ids({article.id: article for article in _articles()}, model.vocabulary)

    def loss(state: dict[str, object]) -> float:
        embeddings = state["embeddings"]
        bias = state["user_bias"]
        total = 0.0
        for candidate in row.candidates:
            ids = token_ids[candidate.article_id]
            encoded = [
                math.tanh(sum(embeddings[token][axis] for token in ids) / len(ids))
                for axis in range(2)
            ]
            logit = sum(bias[axis] * encoded[axis] for axis in range(2))
            total += math.log1p(math.exp(logit)) - float(candidate.clicked) * logit
        return total

    epsilon = 1e-6
    oracle: dict[tuple[str, int, int], float] = {}
    for token in range(1, 5):
        for axis in range(2):
            plus = copy.deepcopy(base)
            minus = copy.deepcopy(base)
            plus["embeddings"][token][axis] += epsilon
            minus["embeddings"][token][axis] -= epsilon
            oracle[("embedding", token, axis)] = (loss(plus) - loss(minus)) / (2 * epsilon)
    for axis in range(2):
        plus = copy.deepcopy(base)
        minus = copy.deepcopy(base)
        plus["user_bias"][axis] += epsilon
        minus["user_bias"][axis] -= epsilon
        oracle[("bias", 0, axis)] = (loss(plus) - loss(minus)) / (2 * epsilon)
    model._train_row(row, (), token_ids)
    trained = model.to_state()
    for (kind, token, axis), derivative in oracle.items():
        before = base["embeddings"][token][axis] if kind == "embedding" else base["user_bias"][axis]
        after = (
            trained["embeddings"][token][axis]
            if kind == "embedding"
            else trained["user_bias"][axis]
        )
        assert (before - after) / model.config.learning_rate == pytest.approx(derivative, abs=1e-8)
    assert loss(trained) < loss(base)


def test_rejects_temporal_overlap_future_publication_and_unknown_candidate() -> None:
    with pytest.raises(ValueError, match="after cutoff"):
        NeuralNewsRanker.fit(_articles(), _validation(), cutoff=CUTOFF)
    with pytest.raises(ValueError, match="overlap"):
        NeuralNewsRanker.fit(_articles(), _train(), cutoff=CUTOFF, held_out_impression_ids=["t2"])
    model = NeuralNewsRanker.fit(_articles(), _train(), cutoff=CUTOFF)
    with pytest.raises(ValueError, match="not after cutoff"):
        model.score_impressions(_articles(), (_row("v1", "u", 5, "a", "b", True),))
    with pytest.raises(ValueError, match="overlap"):
        model.score_impressions(_articles(), (_row("t2", "u", 6, "a", "b", True),))
    with pytest.raises(ValueError, match="unknown article"):
        model.score_impressions(_articles(), (_row("v1", "u", 6, "z", "a", True),))
    future = Article("z", "Future title", "", ("future",), "s", START + timedelta(days=7))
    with pytest.raises(ValueError, match="predates article publication"):
        model.score_impressions((*_articles(), future), (_row("v1", "u", 6, "z", "a", True),))


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("format", "wrong", "format"),
        ("vocabulary", ["<unk>", "grid", "grid"], "vocabulary"),
        ("embeddings", [[0.0]], "embedding shape"),
        ("user_bias", "nan-first-element", "weight"),
        ("histories", {"u": ["a"] * 33}, "history"),
        ("work_units_upper_bound", 20_000_001, "work units"),
    ],
)
def test_state_rejects_corruption(field: str, replacement: object, error: str) -> None:
    state = NeuralNewsRanker.fit(_articles(), _train(), cutoff=CUTOFF).to_state()
    if field == "user_bias":
        state["user_bias"][0] = float("nan")
    else:
        state[field] = replacement
    with pytest.raises(ValueError, match=error):
        NeuralNewsRanker.from_state(state)


def test_limits_reject_oversized_sources_histories_and_work_before_training() -> None:
    _, train, validation = _sources()
    with pytest.raises(ValueError, match="bounded immutable bytes"):
        run_neural_news_experiment(b"x" * (8 * 1024 * 1024 + 1), train, validation, cutoff=CUTOFF)
    with pytest.raises(ValueError, match="history"):
        NeuralNewsRanker.fit(
            _articles(),
            _train(),
            cutoff=CUTOFF,
            config=NeuralNewsConfig(max_history=2),
        )
    many_titles = tuple(
        Article(str(index), " ".join(f"word{part}" for part in range(24)), "", ("x",), "s", START)
        for index in range(16)
    )
    many_rows = tuple(
        MindImpression(
            str(day),
            "u",
            START + timedelta(hours=day + 1),
            tuple(MindCandidate(str(index), index == 0) for index in range(16)),
        )
        for day in range(30)
    )
    with pytest.raises(ValueError, match="work limit"):
        NeuralNewsRanker.fit(
            many_titles,
            many_rows,
            cutoff=CUTOFF,
            config=NeuralNewsConfig(dimension=32, epochs=10),
        )


def test_cli_writes_complete_neural_experiment_and_rejects_source_alias(tmp_path: Path) -> None:
    sources = _sources()
    paths = [tmp_path / name for name in ("articles.json", "train.json", "dev.json")]
    for path, content in zip(paths, sources, strict=True):
        path.write_bytes(content)
    destination = tmp_path / "neural.json"
    arguments = [
        "run-neural-news",
        "--articles",
        str(paths[0]),
        "--train",
        str(paths[1]),
        "--validation",
        str(paths[2]),
        "--cutoff",
        CUTOFF.isoformat(),
        "--epochs",
        "2",
        "--output",
        str(destination),
    ]
    assert main(arguments) == 0
    result = json.loads(destination.read_text(encoding="utf-8"))
    assert result["format"] == "mosaicfeed.neural-news-experiment"
    assert len(result["scores"]) == 2
    assert main([*arguments[:-1], str(paths[0])]) == 2
    assert paths[0].read_bytes() == sources[0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dimension", 1),
        ("epochs", 11),
        ("learning_rate", float("inf")),
        ("learning_rate", 0.0),
        ("seed", -1),
        ("max_vocabulary", 1),
        ("max_history", 33),
    ],
)
def test_configuration_rejects_unbounded_and_nonfinite_values(field: str, value: object) -> None:
    options = NeuralNewsConfig().to_dict()
    options[field] = value
    with pytest.raises(ValueError, match=field):
        NeuralNewsConfig.from_mapping(options)


def test_rejects_catalog_and_title_limits_before_allocating_parameters() -> None:
    with pytest.raises(ValueError, match="catalog"):
        NeuralNewsRanker.fit((), _train(), cutoff=CUTOFF)
    with pytest.raises(ValueError, match="catalog"):
        NeuralNewsRanker.fit((_articles()[0],) * 2_001, _train(), cutoff=CUTOFF)
    with pytest.raises(ValueError, match="duplicate"):
        NeuralNewsRanker.fit((_articles()[0], _articles()[0]), _train(), cutoff=CUTOFF)
    too_many_tokens = Article(
        "a", " ".join(f"word{index}" for index in range(25)), "", ("energy",), "s", START
    )
    with pytest.raises(ValueError, match="title token count"):
        NeuralNewsRanker.fit((too_many_tokens, *_articles()[1:]), _train(), cutoff=CUTOFF)
    huge_token = Article("a", "x" * 65, "", ("energy",), "s", START)
    with pytest.raises(ValueError, match="title token count"):
        NeuralNewsRanker.fit((huge_token, *_articles()[1:]), _train(), cutoff=CUTOFF)


def test_validation_oov_and_missing_training_history_are_explicit() -> None:
    model = NeuralNewsRanker.fit(_articles(), _train(), cutoff=CUTOFF)
    unknown = Article(
        "x", "Previously unseen vocabulary", "", ("x",), "sx", START + timedelta(days=6)
    )
    scores = model.score_impressions(
        (*_articles(), unknown), (_row("v1", "fresh", 6, "x", "a", True),)
    )
    assert scores["v1"]["x"] == 0.0
    state = model.to_state()
    state["histories"]["u"] = ["absent"]
    tampered = NeuralNewsRanker.from_state(state)
    with pytest.raises(ValueError, match="history article absent"):
        tampered.score_impressions(_articles(), _validation())


def test_standalone_state_does_not_bind_caller_supplied_catalog_titles() -> None:
    state = NeuralNewsRanker.fit(
        _articles(), _train(), cutoff=CUTOFF, config=NeuralNewsConfig(dimension=2)
    ).to_state()
    state["vocabulary"] = ["<unk>", "grid"]
    state["embeddings"] = [[0.0, 0.0], [0.4, 0.2]]
    state["user_bias"] = [0.1, 0.1]
    state["histories"] = {}
    model = NeuralNewsRanker.from_state(state)
    row = (_row("v1", "fresh", 6, "c", "b", True),)
    original = model.score_impressions(_articles(), row)["v1"]["c"]
    changed_article = Article("c", "Previously unseen", "", ("energy",), "s3", START)
    changed = model.score_impressions((*_articles()[:2], changed_article, _articles()[3]), row)[
        "v1"
    ]["c"]
    assert original > 0.0
    assert changed == 0.0
    assert model.to_state() == state


def test_runner_rejects_overlap_and_out_of_order_validation_before_fitting() -> None:
    articles, train, validation = _sources()
    overlap = json.loads(validation)
    overlap[0]["impression_id"] = "t2"
    with pytest.raises(ValueError, match="overlap"):
        run_neural_news_experiment(articles, train, json.dumps(overlap).encode(), cutoff=CUTOFF)
    early = json.loads(validation)
    early[0]["occurred_at"] = CUTOFF.isoformat()
    with pytest.raises(ValueError, match="cutoff"):
        run_neural_news_experiment(articles, train, json.dumps(early).encode(), cutoff=CUTOFF)
    with pytest.raises(ValueError, match="timezone-aware"):
        run_neural_news_experiment(articles, train, validation, cutoff=datetime(2026, 1, 4))
