from __future__ import annotations

import copy
import json
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mosaicfeed import listwise as listwise_module
from mosaicfeed.cli import main
from mosaicfeed.config import FeedConfig
from mosaicfeed.datasets import MindCandidate, MindImpression
from mosaicfeed.io import write_json
from mosaicfeed.learning import FEATURE_NAMES, PointwiseLogisticRanker
from mosaicfeed.listwise import ListwiseImpressionRanker
from mosaicfeed.mind import load_mind_impressions, load_mind_scores, write_mind_impressions
from mosaicfeed.models import Article, UserProfile
from mosaicfeed.pairwise import PairwiseImpressionRanker, _digest

T0 = datetime(2026, 1, 1, tzinfo=UTC)
T1 = T0 + timedelta(days=1)
T2 = T0 + timedelta(days=2)


def catalog() -> tuple[Article, ...]:
    return (
        Article("a", "A", "", ("technology",), "s1", T0, 0.9, 0.3),
        Article("b", "B", "", ("sports",), "s2", T0, 0.1, 0.3),
        Article("c", "C", "", ("technology",), "s3", T0, 0.8, 0.1),
        Article("d", "D", "", ("sports",), "s4", T0, 0.2, 0.2),
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


def impression(
    ident: str,
    *,
    user: str = "u",
    time: datetime = T1,
    labels: tuple[tuple[str, bool], ...] = (("a", True), ("b", False)),
) -> MindImpression:
    return MindImpression(
        ident,
        user,
        time,
        tuple(MindCandidate(article_id, clicked) for article_id, clicked in labels),
    )


def fit(
    records: tuple[MindImpression, ...] | None = None,
    **kwargs: object,
) -> ListwiseImpressionRanker:
    return ListwiseImpressionRanker(epochs=1, learning_rate=0.2, l2=0.0).fit(
        catalog(),
        (impression("train"),) if records is None else records,
        partition="train",
        cutoff=T1,
        **kwargs,
    )


def cold_vectors() -> dict[str, tuple[float, ...]]:
    cold = UserProfile("u")
    config = FeedConfig()
    return {
        article.id: PointwiseLogisticRanker._features(article, cold, as_of=T1, config=config)
        for article in catalog()
    }


def test_one_slate_hand_gradient_and_raw_logit_semantics() -> None:
    model = fit()
    vectors = cold_vectors()
    # At zero weights, p=(1/2,1/2), q=(1,0); one SGD step is
    # eta * (x_positive - (x_positive+x_negative)/2).
    expected = {
        name: 0.1 * (vectors["a"][index] - vectors["b"][index])
        for index, name in enumerate(FEATURE_NAMES)
    }
    assert model.weights == pytest.approx(expected)
    assert model.weights["bias"] == 0.0
    assert model.training.training_candidates == 2
    assert model.training.updates == 1
    assert model.to_state()["score_semantics"] == "raw logit; not probability"
    assert model.to_state()["objective"] == "listwise-impression"


def test_multi_positive_three_candidate_oracle_and_order_invariance() -> None:
    labels = (("a", True), ("b", False), ("c", True))
    model = fit((impression("train", labels=labels),))
    reversed_model = fit((impression("train", labels=tuple(reversed(labels))),))
    vectors = cold_vectors()
    expected = [
        0.2
        * (
            (vectors["a"][index] + vectors["c"][index]) / 2
            - (vectors["a"][index] + vectors["b"][index] + vectors["c"][index]) / 3
        )
        for index in range(len(FEATURE_NAMES))
    ]
    assert tuple(model.weights.values()) == pytest.approx(expected)
    assert model.to_state() == reversed_model.to_state()


def test_same_time_profiles_are_cold_and_sgd_matches_independent_oracle() -> None:
    records = (
        impression("z-second", labels=(("c", True), ("d", False))),
        impression("a-first", labels=(("a", True), ("b", False))),
    )
    model = fit(records)
    assert model.to_state() == fit(tuple(reversed(records))).to_state()
    vectors = cold_vectors()
    slates = [(("a", "b"), (1.0, 0.0)), (("c", "d"), (1.0, 0.0))]
    order = list(range(2))
    random.Random(17).shuffle(order)
    oracle = [0.0] * len(FEATURE_NAMES)
    for index in order:
        ids, target = slates[index]
        logits = [sum(w * x for w, x in zip(oracle, vectors[key], strict=True)) for key in ids]
        total = sum(math.exp(value - max(logits)) for value in logits)
        probabilities = [math.exp(value - max(logits)) / total for value in logits]
        oracle = [
            weight
            + 0.2
            * sum(
                (q - p) * vectors[key][feature_index]
                for q, p, key in zip(target, probabilities, ids, strict=True)
            )
            for feature_index, weight in enumerate(oracle)
        ]
    assert tuple(model.weights.values()) == pytest.approx(oracle)


def test_noncomparable_counts_and_empty_signal_rejection() -> None:
    records = (
        impression("positive", labels=(("a", True), ("b", True))),
        impression("negative", labels=(("c", False), ("d", False))),
        impression("mixed"),
    )
    summary = fit(records).training
    assert (
        summary.impressions,
        summary.comparable_impressions,
        summary.skipped_all_positive,
        summary.skipped_all_negative,
        summary.training_candidates,
    ) == (3, 1, 1, 1, 2)
    with pytest.raises(ValueError, match="mixed-label"):
        fit(records[:2])


@pytest.mark.parametrize(
    ("articles", "records", "message"),
    [
        (catalog(), (impression("x", labels=(("missing", True), ("b", False))),), "unknown"),
        (catalog(), (impression("x"), impression("x")), "duplicate impression"),
        (
            (*catalog(), Article("future", "F", "", ("news",), "s", T2, 0.5, 0.5)),
            (impression("x", labels=(("future", True), ("b", False))),),
            "availability",
        ),
    ],
)
def test_training_rejects_unknown_duplicate_and_future_catalog(
    articles: tuple[Article, ...], records: tuple[MindImpression, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ListwiseImpressionRanker().fit(articles, records, partition="train", cutoff=T1)


def test_cutoff_heldout_and_source_bounds_preserve_old_fit() -> None:
    model = fit()
    old = model.to_state()
    with pytest.raises(ValueError, match="cutoff"):
        model.fit(catalog(), (impression("late", time=T2),), partition="train", cutoff=T1)
    with pytest.raises(ValueError, match="overlap"):
        model.fit(
            catalog(),
            (impression("train"),),
            partition="train",
            cutoff=T1,
            held_out_impression_ids=("train",),
        )
    with pytest.raises(ValueError, match="held-out impression is not after"):
        model.fit(
            catalog(),
            (impression("train"),),
            partition="train",
            cutoff=T1,
            held_out_impressions=(impression("test", time=T1),),
        )
    with pytest.raises(ValueError, match="source_sha256"):
        model.fit(
            catalog(),
            (impression("train"),),
            partition="train",
            cutoff=T1,
            source_sha256={"input": "bad"},
        )
    assert model.to_state() == old


def test_post_cutoff_scoring_checks_history_and_ignores_target_labels() -> None:
    model = fit()
    later = impression("test", time=T2)
    scores = model.score_impressions(
        catalog(), (later,), training_impressions=(impression("train"),)
    )
    relabeled = impression("test", time=T2, labels=(("a", False), ("b", True)))
    assert scores == model.score_impressions(
        catalog(), (relabeled,), training_impressions=(impression("train"),)
    )
    with pytest.raises(ValueError, match="history differs"):
        model.score_impressions(
            catalog(),
            (later,),
            training_impressions=(impression("train", labels=(("a", False), ("b", True))),),
        )
    with pytest.raises(ValueError, match="after training cutoff"):
        model.score_impressions(
            catalog(),
            (impression("test", time=T1),),
            training_impressions=(impression("train"),),
        )
    with pytest.raises(ValueError, match="overlap"):
        model.score_impressions(
            catalog(),
            (impression("train", time=T2),),
            training_impressions=(impression("train"),),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda state: state.update({"objective": "pairwise-impression"}), "unsupported"),
        (lambda state: state.update({"weights": [float("nan")] * len(FEATURE_NAMES)}), "canonical"),
        (lambda state: state["training"].update({"updates": 99}), "update count"),
        (lambda state: state["training"].update({"training_candidates": 1}), "candidate count"),
        (lambda state: state["training"].update({"comparable_impressions": 2}), "counts"),
        (lambda state: state["parameters"].update({"epochs": 0}), "epochs"),
    ],
)
def test_state_rejects_even_rechecksummed_malformed_fields(mutation: object, message: str) -> None:
    state = copy.deepcopy(fit().to_state())
    mutation(state)
    if message != "canonical":
        state["state_sha256"] = _digest({k: v for k, v in state.items() if k != "state_sha256"})
    with pytest.raises(ValueError, match=message):
        ListwiseImpressionRanker.from_state(state)


def test_checksum_roundtrip_and_objective_isolation(tmp_path: Path) -> None:
    model = fit()
    path = tmp_path / "listwise.json"
    model.save(path)
    assert ListwiseImpressionRanker.load(path).to_state() == model.to_state()
    with pytest.raises(ValueError):
        PairwiseImpressionRanker.load(path)
    bad = json.loads(path.read_text(encoding="utf-8"))
    bad["weights"][0] += 1
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        ListwiseImpressionRanker.load(path)


def test_resource_bounds_and_divergence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(listwise_module, "MAX_TRAINING_CANDIDATES", 1)
    with pytest.raises(ValueError, match="candidate count"):
        fit()
    monkeypatch.setattr(listwise_module, "MAX_TRAINING_CANDIDATES", 1_000_000)
    monkeypatch.setattr(listwise_module, "MAX_UPDATES", 0)
    with pytest.raises(ValueError, match="update count"):
        fit()
    monkeypatch.setattr(listwise_module, "MAX_UPDATES", 5_000_000)
    monkeypatch.setattr(listwise_module, "MAX_PROFILE_SCANS", 0)
    with pytest.raises(ValueError, match="profile scans"):
        ListwiseImpressionRanker(epochs=1).fit(
            catalog(),
            (impression("one"), impression("two", time=T2)),
            partition="train",
            cutoff=T2,
        )
    monkeypatch.setattr(listwise_module, "MAX_PROFILE_SCANS", 5_000_000)
    monkeypatch.setattr(listwise_module, "MAX_TOPIC_VISITS", 0)
    with pytest.raises(ValueError, match="topic visits"):
        fit()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"partition": " "}, "partition"),
        ({"config": "not-config"}, "config"),
        ({"held_out_impression_ids": ("",)}, "held-out ids"),
        ({"held_out_impression_ids": ("same", "same")}, "held-out ids"),
        ({"source_sha256": []}, "source hashes"),
        ({"source_sha256": {"": "0" * 64}}, "source hash key"),
    ],
)
def test_training_rejects_invalid_contracts_before_mutation(
    extra: dict[str, object], message: str
) -> None:
    model = fit()
    old = model.to_state()
    args: dict[str, object] = {
        "partition": "train",
        "cutoff": T1,
        **extra,
    }
    with pytest.raises(ValueError, match=message):
        model.fit(catalog(), (impression("new"),), **args)
    assert model.to_state() == old


def test_heldout_modes_are_exclusive_and_invalid_ids_bounded() -> None:
    with pytest.raises(ValueError, match="either held-out ids"):
        fit(
            held_out_impression_ids=("test",),
            held_out_impressions=(impression("test", time=T2),),
        )
    with pytest.raises(ValueError, match="held-out ids"):
        fit(held_out_impression_ids=("x" * 257,))
    assert fit(held_out_impressions=(impression("test", time=T2),)).is_fitted


def test_score_rejects_catalog_change_and_bounds_profile_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = fit()
    train = (impression("train"),)
    target = (impression("test", time=T2),)
    changed = (*catalog()[:-1], Article("d", "Changed", "", ("sports",), "s4", T0, 0.2, 0.2))
    with pytest.raises(ValueError, match="catalog differs"):
        model.score_impressions(changed, target, training_impressions=train)
    monkeypatch.setattr(listwise_module, "MAX_PROFILE_SCANS", 0)
    with pytest.raises(ValueError, match="profile scans"):
        model.score_impressions(catalog(), target, training_impressions=train)
    monkeypatch.setattr(listwise_module, "MAX_PROFILE_SCANS", 5_000_000)
    monkeypatch.setattr(listwise_module, "MAX_TOPIC_VISITS", 0)
    with pytest.raises(ValueError, match="topic visits"):
        model.score_impressions(catalog(), target, training_impressions=train)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda state: state.update({"unexpected": 1}), "missing or unknown"),
        (lambda state: state.update({"parameters": []}), "parameters"),
        (lambda state: state.update({"feature_names": []}), "feature schema"),
        (lambda state: state.update({"feature_config": []}), "feature config"),
        (lambda state: state.update({"weights": []}), "weights"),
        (lambda state: state.update({"training": []}), "training summary"),
        (lambda state: state["training"].update({"partition": ""}), "partition"),
        (lambda state: state["training"].update({"skipped_all_positive": -1}), "skipped"),
        (lambda state: state["training"].update({"source_sha256": []}), "source hashes"),
        (
            lambda state: state["training"].update({"source_sha256": {"": "0" * 64}}),
            "source hash key",
        ),
    ],
)
def test_state_strict_schema_even_with_recomputed_checksum(change: object, message: str) -> None:
    state = copy.deepcopy(fit().to_state())
    change(state)
    state["state_sha256"] = _digest({k: v for k, v in state.items() if k != "state_sha256"})
    with pytest.raises(ValueError, match=message):
        ListwiseImpressionRanker.from_state(state)


def test_unfitted_state_and_state_size_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ListwiseImpressionRanker()
    with pytest.raises(ValueError, match="not been fitted"):
        model.to_state()
    fitted = fit()
    monkeypatch.setattr(listwise_module, "MAX_STATE_BYTES", 1)
    path = tmp_path / "state.json"
    with pytest.raises(ValueError, match="size limit"):
        fitted.save(path)
    assert not path.exists()
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="size limit"):
        ListwiseImpressionRanker.load(path)
    monkeypatch.setattr(listwise_module, "MAX_STATE_BYTES", 4 * 1024 * 1024)
    with pytest.raises(ValueError, match="JSON object"):
        path.write_text("[]", encoding="utf-8")
        ListwiseImpressionRanker.load(path)


def test_bounded_hyperparameters_and_unfitted_summary() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        ListwiseImpressionRanker(learning_rate=0)
    with pytest.raises(ValueError, match="l2"):
        ListwiseImpressionRanker(l2=2)
    with pytest.raises(ValueError, match="seed"):
        ListwiseImpressionRanker(seed=-1)
    model = fit()
    model._summary = None
    with pytest.raises(ValueError, match="inconsistent training summary"):
        _ = model.training


@pytest.mark.parametrize(
    "invalid_field", ["partition", "source-key", "held-out-id", "impression-id", "article-title"]
)
def test_invalid_utf8_input_fails_before_replacing_a_fitted_model(invalid_field: str) -> None:
    model = fit()
    old = model.to_state()
    articles = catalog()
    records = (impression("new"),)
    options: dict[str, object] = {"partition": "train", "cutoff": T1}
    invalid = "bad\ud800"
    if invalid_field == "partition":
        options["partition"] = invalid
    elif invalid_field == "source-key":
        options["source_sha256"] = {invalid: "0" * 64}
    elif invalid_field == "held-out-id":
        options["held_out_impression_ids"] = (invalid,)
    elif invalid_field == "impression-id":
        records = (impression(invalid),)
    else:
        articles = (
            Article("a", invalid, "", ("technology",), "s1", T0, 0.9, 0.3),
            *articles[1:],
        )
    with pytest.raises(ValueError, match="canonical UTF-8 JSON"):
        model.fit(articles, records, **options)
    assert model.to_state() == old


def test_invalid_utf8_partition_cannot_create_unserializable_new_model() -> None:
    model = ListwiseImpressionRanker(epochs=1)
    with pytest.raises(ValueError, match="canonical UTF-8 JSON"):
        model.fit(catalog(), (impression("new"),), partition="\ud800", cutoff=T1)
    assert model.is_fitted is False


def test_cli_train_rank_evaluate_and_atomic_rejection(tmp_path: Path) -> None:
    article_path = tmp_path / "articles.json"
    train_path = tmp_path / "train.json"
    test_path = tmp_path / "test.json"
    model_path = tmp_path / "model.json"
    scores_path = tmp_path / "scores.json"
    report_path = tmp_path / "report.json"
    write_json(article_path, [article_record(article) for article in catalog()])
    write_mind_impressions(train_path, (impression("train"),))
    records = (
        impression("test", time=T2, labels=(("c", False), ("a", True), ("b", False))),
        impression("all-positive", time=T2, labels=(("a", True), ("b", True))),
    )
    write_mind_impressions(test_path, records)
    assert (
        main(
            [
                "train-listwise-model",
                "--articles",
                str(article_path),
                "--impressions",
                str(train_path),
                "--held-out-impressions",
                str(test_path),
                "--partition",
                "local",
                "--cutoff",
                T1.isoformat(),
                "--epochs",
                "1",
                "--output",
                str(model_path),
            ]
        )
        == 0
    )
    shared = [
        "--model",
        str(model_path),
        "--articles",
        str(article_path),
        "--training-impressions",
        str(train_path),
        "--impressions",
        str(test_path),
        "--scores-output",
        str(scores_path),
    ]
    assert main(["rank-listwise-model", *shared, "--report-output", str(report_path)]) == 0
    assert len(load_mind_scores(scores_path)) == 2
    assert json.loads(report_path.read_text(encoding="utf-8"))["objective"] == "listwise-impression"
    assert main(["evaluate-listwise-model", *shared, "--output", str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["metrics"]["impressions"] == 1
    assert report["skipped_all_positive"] == 1
    assert [row.impression_id for row in load_mind_impressions(test_path)] == [
        "test",
        "all-positive",
    ]
    before_scores = scores_path.read_bytes()
    before_report = report_path.read_bytes()
    too_early = tmp_path / "early.json"
    write_mind_impressions(too_early, (impression("early", time=T1),))
    failed = [
        "--model",
        str(model_path),
        "--articles",
        str(article_path),
        "--training-impressions",
        str(train_path),
        "--impressions",
        str(too_early),
        "--scores-output",
        str(scores_path),
        "--output",
        str(report_path),
    ]
    assert main(["evaluate-listwise-model", *failed]) == 2
    assert scores_path.read_bytes() == before_scores
    assert report_path.read_bytes() == before_report
