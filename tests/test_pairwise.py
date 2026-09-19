from __future__ import annotations

import copy
import json
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mosaicfeed import pairwise as pairwise_module
from mosaicfeed.cli import main
from mosaicfeed.config import FeedConfig
from mosaicfeed.datasets import MindCandidate, MindImpression
from mosaicfeed.io import write_json
from mosaicfeed.learning import FEATURE_NAMES, PointwiseLogisticRanker
from mosaicfeed.mind import (
    evaluate_mind_impressions,
    load_mind_impressions,
    load_mind_scores,
    write_mind_impressions,
)
from mosaicfeed.models import Article, UserProfile
from mosaicfeed.pairwise import (
    MAX_CANDIDATES,
    MAX_EPOCHS,
    MAX_IMPRESSIONS,
    MAX_PAIRS,
    MAX_STATE_BYTES,
    MAX_UPDATES,
    PairwiseImpressionRanker,
)

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


def fit(records: tuple[MindImpression, ...] | None = None) -> PairwiseImpressionRanker:
    return PairwiseImpressionRanker(epochs=1, learning_rate=0.2, l2=0.0).fit(
        catalog(),
        (impression("train"),) if records is None else records,
        partition="train",
        cutoff=T1,
    )


def test_independent_one_pair_one_step_oracle() -> None:
    model = fit()
    articles = {article.id: article for article in catalog()}
    profile = UserProfile("u")
    config = FeedConfig()
    positive = PointwiseLogisticRanker._features(articles["a"], profile, as_of=T1, config=config)
    negative = PointwiseLogisticRanker._features(articles["b"], profile, as_of=T1, config=config)
    # At zero weights sigmoid(-margin)=1/2 and initial L2 penalty is zero.
    expected = {
        name: 0.1 * (p - n) for name, p, n in zip(FEATURE_NAMES, positive, negative, strict=True)
    }
    assert model.weights == pytest.approx(expected)
    assert model.weights["bias"] == 0.0
    assert model.training.pairs == model.training.updates == 1


def test_exhaustive_tiny_pair_oracle_and_counterfactual_order() -> None:
    labels = (("a", True), ("b", False), ("c", True), ("d", False))
    model = fit((impression("train", labels=labels),))
    reversed_model = fit((impression("train", labels=tuple(reversed(labels))),))
    assert model.training.pairs == 4
    assert model.weights == reversed_model.weights
    assert model.training.training_sha256 == reversed_model.training.training_sha256
    # The independent cartesian product contains exactly four ordered pairs.
    expected = {(p, n) for p in ("a", "c") for n in ("b", "d")}
    assert expected == {("a", "b"), ("a", "d"), ("c", "b"), ("c", "d")}
    articles = {article.id: article for article in catalog()}
    cold = UserProfile("u")
    vectors = {
        article_id: PointwiseLogisticRanker._features(
            articles[article_id], cold, as_of=T1, config=FeedConfig()
        )
        for article_id in ("a", "b", "c", "d")
    }
    pairs = [(p, n) for p in sorted(("a", "c")) for n in sorted(("b", "d"))]
    order = list(range(len(pairs)))
    random.Random(17).shuffle(order)
    oracle = [0.0] * len(FEATURE_NAMES)
    for index in order:
        p, n = pairs[index]
        delta = [left - right for left, right in zip(vectors[p], vectors[n], strict=True)]
        margin = sum(weight * difference for weight, difference in zip(oracle, delta, strict=True))
        factor = 1.0 / (1.0 + math.exp(margin))
        oracle = [
            weight + 0.2 * factor * difference
            for weight, difference in zip(oracle, delta, strict=True)
        ]
    assert tuple(model.weights.values()) == pytest.approx(oracle)


def test_same_timestamp_user_history_is_not_visible_between_impressions() -> None:
    first = impression("a-first", labels=(("a", True), ("b", False)))
    second = impression("z-second", labels=(("c", True), ("d", False)))
    model = fit((second, first))
    # Both first-time profiles are cold; their gradients are the sum of two
    # independent zero-margin steps except for sequential SGD interaction.
    assert model.training.impressions == 2
    assert model.training.pairs == 2
    assert model.to_state() == fit((first, second)).to_state()
    articles = {article.id: article for article in catalog()}
    cold = UserProfile("u")
    vectors = {
        article_id: PointwiseLogisticRanker._features(
            articles[article_id], cold, as_of=T1, config=FeedConfig()
        )
        for article_id in ("a", "b", "c", "d")
    }
    pairs = [("a", "b"), ("c", "d")]
    order = [0, 1]
    random.Random(17).shuffle(order)
    oracle = [0.0] * len(FEATURE_NAMES)
    for index in order:
        p, n = pairs[index]
        delta = [left - right for left, right in zip(vectors[p], vectors[n], strict=True)]
        margin = sum(weight * difference for weight, difference in zip(oracle, delta, strict=True))
        factor = 1.0 / (1.0 + math.exp(margin))
        oracle = [
            weight + 0.2 * factor * difference
            for weight, difference in zip(oracle, delta, strict=True)
        ]
    assert tuple(model.weights.values()) == pytest.approx(oracle)
    later = impression("later", time=T2, labels=(("a", True), ("b", False)))
    scores = model.score_impressions(catalog(), (later,), training_impressions=(second, first))
    assert set(scores["later"]) == {"a", "b"}
    relabeled = impression("later", time=T2, labels=(("a", False), ("b", True)))
    assert (
        model.score_impressions(catalog(), (relabeled,), training_impressions=(first, second))
        == scores
    )


def test_skipped_counts_and_noncomparable_rejection() -> None:
    records = (
        impression("positive", labels=(("a", True), ("b", True))),
        impression("negative", labels=(("c", False), ("d", False))),
        impression("mixed"),
    )
    summary = fit(records).training
    assert (
        summary.skipped_all_positive,
        summary.skipped_all_negative,
        summary.comparable_impressions,
        summary.pairs,
    ) == (1, 1, 1, 1)
    with pytest.raises(ValueError, match="mixed-label"):
        fit(records[:2])


def test_validation_before_update_and_heldout_disjointness() -> None:
    future = Article("future", "Future", "", ("topic",), "s", T2)
    invalid_cases = [
        (catalog(), (impression("x", labels=(("unseen", True), ("b", False))),)),
        ((*catalog(), future), (impression("x", labels=(("future", True), ("b", False))),)),
        (catalog(), (impression("x"), impression("x"))),
    ]
    for articles, records in invalid_cases:
        with pytest.raises(ValueError):
            PairwiseImpressionRanker().fit(articles, records, partition="train", cutoff=T1)
    with pytest.raises(ValueError, match="overlap"):
        PairwiseImpressionRanker().fit(
            catalog(),
            (impression("x"),),
            partition="train",
            cutoff=T1,
            held_out_impression_ids=("x",),
        )
    with pytest.raises(ValueError, match="after cutoff"):
        PairwiseImpressionRanker().fit(
            catalog(), (impression("later", time=T2),), partition="train", cutoff=T1
        )
    with pytest.raises(ValueError, match="held-out impression is not after"):
        PairwiseImpressionRanker().fit(
            catalog(),
            (impression("train"),),
            partition="train",
            cutoff=T1,
            held_out_impressions=(impression("heldout", time=T1),),
        )
    with pytest.raises(ValueError, match="unknown article"):
        PairwiseImpressionRanker().fit(
            catalog(),
            (impression("train"),),
            partition="train",
            cutoff=T1,
            held_out_impressions=(
                impression("heldout", time=T2, labels=(("unseen", True), ("b", False))),
            ),
        )
    with pytest.raises(ValueError, match="availability"):
        PairwiseImpressionRanker().fit(
            (*catalog(), Article("future", "Future", "", ("topic",), "s", T2)),
            (impression("train"),),
            partition="train",
            cutoff=T1,
            held_out_impressions=(
                impression(
                    "heldout", time=T1 + timedelta(hours=1), labels=(("future", True), ("b", False))
                ),
            ),
        )
    for partition in ("", " "):
        with pytest.raises(ValueError):
            PairwiseImpressionRanker().fit(
                catalog(), (impression("x"),), partition=partition, cutoff=T1
            )


def test_scoring_rejects_overlap_cutoff_and_history_tamper() -> None:
    model = fit()
    with pytest.raises(ValueError, match="overlap"):
        model.score_impressions(
            catalog(), (impression("train", time=T2),), training_impressions=(impression("train"),)
        )
    with pytest.raises(ValueError, match="after training cutoff"):
        model.score_impressions(
            catalog(), (impression("test", time=T1),), training_impressions=(impression("train"),)
        )
    with pytest.raises(ValueError, match="history differs"):
        model.score_impressions(
            catalog(),
            (impression("test", time=T2),),
            training_impressions=(impression("train", labels=(("a", False), ("b", True))),),
        )
    changed = list(catalog())
    changed[0] = Article("a", "Changed", "", ("technology",), "s1", T0, 0.9, 0.3)
    with pytest.raises(ValueError, match="catalog differs"):
        model.score_impressions(
            changed, (impression("test", time=T2),), training_impressions=(impression("train"),)
        )


@pytest.mark.parametrize("field", ["objective", "score_semantics", "weights", "training"])
def test_strict_checksummed_state_rejects_mutation(field: str) -> None:
    model = fit()
    state = copy.deepcopy(model.to_state())
    state[field] = "corrupt"
    with pytest.raises(ValueError, match="checksum"):
        PairwiseImpressionRanker.from_state(state)
    assert PairwiseImpressionRanker.from_state(model.to_state()).weights == model.weights


def test_corrupt_state_even_with_recomputed_checksum_is_rejected() -> None:
    from mosaicfeed.pairwise import _digest

    for mutation in (
        lambda state: state.update({"objective": "pointwise"}),
        lambda state: state.update({"feature_names": ["bias"]}),
        lambda state: state["training"].update({"pairs": 0}),
        lambda state: state["training"].update({"updates": 2}),
        lambda state: state.update({"weights": [1e9] * len(FEATURE_NAMES)}),
    ):
        state = copy.deepcopy(fit().to_state())
        mutation(state)
        state["state_sha256"] = _digest({k: v for k, v in state.items() if k != "state_sha256"})
        with pytest.raises(ValueError):
            PairwiseImpressionRanker.from_state(state)


def test_resource_bounds_and_model_file_limit(tmp_path: Path) -> None:
    for kwargs in (
        {"epochs": MAX_EPOCHS + 1},
        {"seed": -1},
        {"learning_rate": math.inf},
        {"l2": 2},
    ):
        with pytest.raises(ValueError):
            PairwiseImpressionRanker(**kwargs)
    many_candidates = tuple(MindCandidate("a", True) for _ in range(MAX_CANDIDATES + 1))
    with pytest.raises(ValueError):
        MindImpression("x", "u", T1, many_candidates)
    assert MAX_PAIRS * MAX_EPOCHS > MAX_UPDATES
    assert MAX_IMPRESSIONS > 0
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (MAX_STATE_BYTES + 1))
    with pytest.raises(ValueError, match="size limit"):
        PairwiseImpressionRanker.load(oversized)


def test_collection_and_compute_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="catalog"):
        PairwiseImpressionRanker().fit((), (impression("x"),), partition="train", cutoff=T1)
    monkeypatch.setattr(pairwise_module, "MAX_ARTICLES", 1)
    with pytest.raises(ValueError, match="catalog"):
        fit()
    monkeypatch.setattr(pairwise_module, "MAX_ARTICLES", 100_000)
    monkeypatch.setattr(pairwise_module, "MAX_IMPRESSIONS", 1)
    with pytest.raises(ValueError, match="impressions"):
        fit((impression("x"), impression("y")))
    monkeypatch.setattr(pairwise_module, "MAX_IMPRESSIONS", 10_000)
    monkeypatch.setattr(pairwise_module, "MAX_CANDIDATES", 1)
    with pytest.raises(ValueError, match="candidate count"):
        fit()
    monkeypatch.setattr(pairwise_module, "MAX_CANDIDATES", 256)
    monkeypatch.setattr(pairwise_module, "MAX_FEATURE_CELLS", 7)
    with pytest.raises(ValueError, match="feature cells"):
        fit()
    monkeypatch.setattr(pairwise_module, "MAX_FEATURE_CELLS", 3_000_000)
    monkeypatch.setattr(pairwise_module, "MAX_PAIRS", 1)
    with pytest.raises(ValueError, match="pair count"):
        fit((impression("x", labels=(("a", True), ("b", False), ("c", True))),))
    monkeypatch.setattr(pairwise_module, "MAX_PAIRS", 1_000_000)
    monkeypatch.setattr(pairwise_module, "MAX_UPDATES", 1)
    with pytest.raises(ValueError, match="update count"):
        PairwiseImpressionRanker(epochs=2).fit(
            catalog(), (impression("x"),), partition="train", cutoff=T1
        )


def test_profile_and_topic_work_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    second_time = T1 + timedelta(hours=1)
    records = (impression("first"), impression("second", time=second_time))
    monkeypatch.setattr(pairwise_module, "MAX_PROFILE_SCANS", 0)
    with pytest.raises(ValueError, match="training profile scans"):
        PairwiseImpressionRanker().fit(catalog(), records, partition="train", cutoff=second_time)
    monkeypatch.setattr(pairwise_module, "MAX_PROFILE_SCANS", 5_000_000)
    monkeypatch.setattr(pairwise_module, "MAX_TOPIC_VISITS", 3)
    with pytest.raises(ValueError, match="training topic visits"):
        fit()
    monkeypatch.setattr(pairwise_module, "MAX_TOPIC_VISITS", 10_000_000)
    model = fit()
    monkeypatch.setattr(pairwise_module, "MAX_PROFILE_SCANS", 0)
    with pytest.raises(ValueError, match="scoring profile scans"):
        model.score_impressions(
            catalog(), (impression("test", time=T2),), training_impressions=(impression("train"),)
        )
    monkeypatch.setattr(pairwise_module, "MAX_PROFILE_SCANS", 5_000_000)
    monkeypatch.setattr(pairwise_module, "MAX_TOPIC_VISITS", 3)
    with pytest.raises(ValueError, match="scoring topic visits"):
        model.score_impressions(
            catalog(), (impression("test", time=T2),), training_impressions=(impression("train"),)
        )


def test_training_source_hashes_are_defensively_frozen() -> None:
    sources = {"articles": "a" * 64}
    model = PairwiseImpressionRanker(epochs=1).fit(
        catalog(),
        (impression("train"),),
        partition="train",
        cutoff=T1,
        source_sha256=sources,
    )
    sources["articles"] = "b" * 64
    assert model.training.source_sha256["articles"] == "a" * 64
    with pytest.raises(TypeError):
        model.training.source_sha256["articles"] = "c" * 64  # type: ignore[index]
    assert model.to_state()["training"]["source_sha256"]["articles"] == "a" * 64  # type: ignore[index]


def test_invalid_sources_split_and_catalog_inputs() -> None:
    with pytest.raises(ValueError, match="unique bounded"):
        PairwiseImpressionRanker().fit(
            catalog(),
            (impression("x"),),
            partition="train",
            cutoff=T1,
            held_out_impression_ids=("y", "y"),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        PairwiseImpressionRanker().fit(
            catalog(),
            (impression("x"),),
            partition="train",
            cutoff=T1,
            source_sha256={"articles": "bad"},
        )
    with pytest.raises(ValueError, match="source hashes"):
        PairwiseImpressionRanker().fit(
            catalog(),
            (impression("x"),),
            partition="train",
            cutoff=T1,
            source_sha256={str(index): "a" * 64 for index in range(9)},
        )
    with pytest.raises(ValueError, match="key is invalid"):
        PairwiseImpressionRanker().fit(
            catalog(),
            (impression("x"),),
            partition="train",
            cutoff=T1,
            source_sha256={"": "a" * 64},
        )
    with pytest.raises(ValueError, match="duplicate article"):
        PairwiseImpressionRanker().fit(
            (*catalog(), catalog()[0]), (impression("x"),), partition="train", cutoff=T1
        )
    with pytest.raises(ValueError, match="Article"):
        PairwiseImpressionRanker().fit(
            (*catalog(), "not-an-article"),
            (impression("x"),),  # type: ignore[arg-type]
            partition="train",
            cutoff=T1,
        )
    with pytest.raises(ValueError, match="MindImpression"):
        PairwiseImpressionRanker().fit(
            catalog(),
            ("not-an-impression",),  # type: ignore[arg-type]
            partition="train",
            cutoff=T1,
        )
    with pytest.raises(ValueError, match="non-empty"):
        PairwiseImpressionRanker().fit(
            catalog(),
            (impression("x"),),
            partition="train",
            cutoff=T1,
            held_out_impression_ids=("",),
        )


def test_unfitted_invalid_parameters_and_state_input(tmp_path: Path) -> None:
    model = PairwiseImpressionRanker()
    with pytest.raises(ValueError, match="not been fitted"):
        _ = model.weights
    with pytest.raises(ValueError, match="not been fitted"):
        _ = model.training
    for kwargs in (
        {"epochs": True},
        {"learning_rate": 0},
        {"learning_rate": 2},
        {"l2": -1},
        {"seed": True},
    ):
        with pytest.raises(ValueError):
            PairwiseImpressionRanker(**kwargs)
    with pytest.raises(ValueError, match="missing or unknown"):
        PairwiseImpressionRanker.from_state({})
    not_object = tmp_path / "not-object.json"
    not_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        PairwiseImpressionRanker.load(not_object)


def test_re_signed_state_structural_validation() -> None:
    from mosaicfeed.pairwise import _digest

    mutators = (
        lambda state: state.update({"schema_version": True}),
        lambda state: state.update({"score_semantics": "probability"}),
        lambda state: state.update({"parameters": {"epochs": 1}}),
        lambda state: state.update({"feature_config": "bad"}),
        lambda state: state.update({"weights": [0.0]}),
        lambda state: state["training"].update({"impressions": 2}),
        lambda state: state["training"].update({"partition": ""}),
        lambda state: state["training"].update({"catalog_sha256": "bad"}),
        lambda state: state["training"].update({"source_sha256": {"x": "bad"}}),
    )
    for mutate in mutators:
        state = copy.deepcopy(fit().to_state())
        mutate(state)
        state["state_sha256"] = _digest(
            {key: value for key, value in state.items() if key != "state_sha256"}
        )
        with pytest.raises(ValueError):
            PairwiseImpressionRanker.from_state(state)
    state = fit().to_state()
    state["unexpected"] = 1
    with pytest.raises(ValueError, match="missing or unknown"):
        PairwiseImpressionRanker.from_state(state)


def _article_record(article: Article) -> dict[str, object]:
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


def test_cli_candidate_lossless_mind_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    articles_path = tmp_path / "articles.json"
    train_path = tmp_path / "train.json"
    test_path = tmp_path / "test.json"
    model_path = tmp_path / "model.json"
    scores_path = tmp_path / "scores.json"
    rank_report_path = tmp_path / "rank-report.json"
    evaluation_path = tmp_path / "evaluation.json"
    write_json(articles_path, [_article_record(article) for article in catalog()])
    write_mind_impressions(train_path, (impression("train"),))
    test_records = (
        impression("test1", time=T2, labels=(("b", False), ("a", True), ("c", False), ("d", True))),
        impression("test2", time=T2, labels=(("c", False), ("d", True), ("a", True), ("b", False))),
        impression(
            "all-negative", time=T2, labels=(("a", False), ("b", False), ("c", False), ("d", False))
        ),
    )
    write_mind_impressions(test_path, test_records)
    # Compact interchange is valid and can expand when score rows are emitted.
    for source in (articles_path, train_path, test_path):
        source.write_text(
            json.dumps(json.loads(source.read_text(encoding="utf-8")), separators=(",", ":")),
            encoding="utf-8",
        )
    assert (
        main(
            [
                "train-pairwise-model",
                "--articles",
                str(articles_path),
                "--impressions",
                str(train_path),
                "--partition",
                "local-train",
                "--cutoff",
                T1.isoformat(),
                "--held-out-impressions",
                str(test_path),
                "--output",
                str(model_path),
                "--epochs",
                "1",
            ]
        )
        == 0
    )
    shared = [
        "--model",
        str(model_path),
        "--articles",
        str(articles_path),
        "--training-impressions",
        str(train_path),
        "--impressions",
        str(test_path),
        "--scores-output",
        str(scores_path),
    ]
    assert main(["rank-pairwise-model", *shared, "--report-output", str(rank_report_path)]) == 0
    rows = json.loads(scores_path.read_text(encoding="utf-8"))
    assert [(row["impression_id"], row["article_id"]) for row in rows] == [
        (item.impression_id, candidate.article_id)
        for item in test_records
        for candidate in item.candidates
    ]
    loaded = load_mind_scores(scores_path)
    assert set(loaded) == {"test1", "test2", "all-negative"}
    assert main(["evaluate-pairwise-model", *shared, "--output", str(evaluation_path)]) == 0
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["impressions"] == 2
    assert payload["skipped_all_negative"] == 1
    assert payload["training"]["partition"] == "local-train"
    assert payload["score_semantics"] == "raw logit; not probability"
    assert (
        evaluate_mind_impressions(
            load_mind_impressions(test_path)[:2],
            {key: loaded[key] for key in ("test1", "test2")},
        ).to_dict()
        == payload["metrics"]
    )
    changed_test = tmp_path / "changed-test.json"
    write_mind_impressions(changed_test, (impression("different", time=T2),))
    changed_shared = [
        "--model",
        str(model_path),
        "--articles",
        str(articles_path),
        "--training-impressions",
        str(train_path),
        "--impressions",
        str(changed_test),
        "--scores-output",
        str(tmp_path / "changed-scores.json"),
    ]
    assert main(["rank-pairwise-model", *changed_shared]) == 2
    # Score JSON can legitimately be larger than every bounded input file.
    input_ceiling = (
        max(articles_path.stat().st_size, train_path.stat().st_size, test_path.stat().st_size) + 1
    )
    assert scores_path.stat().st_size > input_ceiling
    monkeypatch.setattr("mosaicfeed.cli.MAX_PAIRWISE_INPUT_BYTES", input_ceiling)
    large_scores = tmp_path / "large-scores.json"
    large_report = tmp_path / "large-report.json"
    larger_output_args = [
        "--model",
        str(model_path),
        "--articles",
        str(articles_path),
        "--training-impressions",
        str(train_path),
        "--impressions",
        str(test_path),
        "--scores-output",
        str(large_scores),
    ]
    assert (
        main(["rank-pairwise-model", *larger_output_args, "--report-output", str(large_report)])
        == 0
    )
    assert large_scores.stat().st_size > input_ceiling
    assert load_mind_scores(large_scores) == loaded
    failed_scores = tmp_path / "failed-scores.json"
    failed_report = tmp_path / "failed-report.json"
    failed_args = [
        "--model",
        str(model_path),
        "--articles",
        str(articles_path),
        "--training-impressions",
        str(train_path),
        "--impressions",
        str(test_path),
        "--scores-output",
        str(failed_scores),
    ]
    assert (
        main(
            [
                "evaluate-pairwise-model",
                *failed_args,
                "--cutoff",
                "0",
                "--output",
                str(failed_report),
            ]
        )
        == 2
    )
    assert not failed_scores.exists()
    assert not failed_report.exists()
    bad_holdout = tmp_path / "bad-holdout.json"
    bad_model = tmp_path / "bad-model.json"
    write_mind_impressions(bad_holdout, (impression("too-early", time=T1),))
    assert (
        main(
            [
                "train-pairwise-model",
                "--articles",
                str(articles_path),
                "--impressions",
                str(train_path),
                "--partition",
                "train",
                "--cutoff",
                T1.isoformat(),
                "--held-out-impressions",
                str(bad_holdout),
                "--output",
                str(bad_model),
            ]
        )
        == 2
    )
    assert not bad_model.exists()
