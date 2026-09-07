"""Matching a reader's topic proportions, which is not the same as diversity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mosaicfeed.calibration import (
    SMOOTHING,
    calibration_error,
    marginal_objective,
    reader_distribution,
    slate_distribution,
    topic_mass,
)
from mosaicfeed.config import RERANK_STRATEGIES, FeedConfig
from mosaicfeed.metrics import intra_list_diversity
from mosaicfeed.models import Article, Event, EventKind, UserProfile
from mosaicfeed.pipeline import build_feed
from mosaicfeed.profile import build_profile
from mosaicfeed.rerank import rerank
from mosaicfeed.scoring import score_candidates

AS_OF = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
TOPICS = ("politics", "sports", "tech", "science", "culture")


def _article(article_id: str, topic: str, *, source: str = "src", quality: float = 0.6) -> Article:
    return Article(
        id=article_id,
        title=f"{topic} {article_id}",
        summary=f"about {topic}",
        topics=(topic,),
        source=source,
        published_at=AS_OF - timedelta(hours=4),
        quality=quality,
    )


def _corpus(per_topic: int = 10) -> list[Article]:
    articles: list[Article] = []
    index = 0
    for topic in TOPICS:
        for _ in range(per_topic):
            articles.append(_article(f"a{index:03d}", topic, source=f"s{index % 5}"))
            index += 1
    return articles


def _skewed_history(articles: list[Article], politics: int = 16, sports: int = 4) -> list[Event]:
    by_topic: dict[str, list[Article]] = {}
    for article in articles:
        by_topic.setdefault(article.topics[0], []).append(article)
    events: list[Event] = []
    for count, topic in ((politics, "politics"), (sports, "sports")):
        for index in range(count):
            events.append(
                Event(
                    user_id="u1",
                    article_id=by_topic[topic][index % len(by_topic[topic])].id,
                    kind=EventKind.CLICK,
                    occurred_at=AS_OF - timedelta(hours=index + 2),
                )
            )
    return events


def _profile(articles: list[Article], events: list[Event]) -> UserProfile:
    return build_profile(
        "u1",
        events,
        {article.id: article for article in articles},
        as_of=AS_OF,
        config=FeedConfig(),
    )


# ---------------------------------------------------------------------------
# The distributions themselves.
# ---------------------------------------------------------------------------


def test_a_reader_distribution_normalizes_positive_weights() -> None:
    profile = UserProfile(user_id="u", topic_weights={"a": 0.9, "b": 0.3})
    assert reader_distribution(profile) == pytest.approx({"a": 0.75, "b": 0.25})


def test_pushed_away_topics_take_no_share() -> None:
    # A negative weight records what to avoid. Giving it a share of the
    # distribution would ask the slate to serve a topic the reader rejected.
    profile = UserProfile(user_id="u", topic_weights={"a": 1.0, "b": -0.8})
    assert reader_distribution(profile) == pytest.approx({"a": 1.0})


def test_a_reader_with_no_positive_history_has_no_distribution() -> None:
    assert reader_distribution(UserProfile(user_id="u", topic_weights={"a": -1.0})) == {}
    assert reader_distribution(UserProfile(user_id="u")) == {}


def test_an_article_divides_one_vote_among_its_topics() -> None:
    single = Article(id="s", title="t", summary="s", topics=("a",), source="x", published_at=AS_OF)
    multi = Article(
        id="m", title="t", summary="s", topics=("a", "b"), source="x", published_at=AS_OF
    )
    assert topic_mass(single) == pytest.approx({"a": 1.0})
    assert topic_mass(multi) == pytest.approx({"a": 0.5, "b": 0.5})
    assert slate_distribution(["s", "m"], {"s": single, "m": multi}) == pytest.approx(
        {"a": 0.75, "b": 0.25}
    )


def test_an_article_always_has_a_topic_to_vote_with() -> None:
    # `topic_mass` has no voteless case because the model refuses to build one.
    with pytest.raises(ValueError, match="at least one non-empty value"):
        Article(id="b", title="t", summary="s", topics=(), source="x", published_at=AS_OF)


def test_a_slate_referencing_an_unknown_article_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown article"):
        slate_distribution(["missing"], {})


# ---------------------------------------------------------------------------
# The error itself.
# ---------------------------------------------------------------------------


def test_a_matching_slate_scores_zero() -> None:
    articles = {f"p{index}": _article(f"p{index}", "politics") for index in range(4)}
    articles.update({"s0": _article("s0", "sports")})
    profile = UserProfile(user_id="u", topic_weights={"politics": 0.8, "sports": 0.2})
    ranked = ["p0", "p1", "p2", "p3", "s0"]
    assert calibration_error(profile, ranked, articles) == pytest.approx(0.0, abs=1e-9)


def test_a_missing_topic_costs_more_than_a_thin_one() -> None:
    articles = {"p": _article("p", "politics"), "s": _article("s", "sports")}
    profile = UserProfile(user_id="u", topic_weights={"politics": 0.5, "sports": 0.5})
    thin = calibration_error(profile, ["p", "p", "s"], articles)
    absent = calibration_error(profile, ["p"], articles)
    assert absent > thin > 0.0


def test_the_error_stays_finite_when_a_topic_is_absent() -> None:
    # Without smoothing this divergence is infinite, which cannot rank two
    # imperfect slates against each other.
    articles = {"p": _article("p", "politics")}
    profile = UserProfile(user_id="u", topic_weights={"politics": 0.5, "sports": 0.5})
    error = calibration_error(profile, ["p"], articles)
    assert 0.0 < error < float("inf")


def test_less_smoothing_punishes_an_absent_topic_harder() -> None:
    articles = {"p": _article("p", "politics")}
    profile = UserProfile(user_id="u", topic_weights={"politics": 0.5, "sports": 0.5})
    assert calibration_error(profile, ["p"], articles, smoothing=0.001) > calibration_error(
        profile, ["p"], articles, smoothing=0.2
    )


def test_nothing_to_calibrate_against_scores_zero() -> None:
    articles = {"p": _article("p", "politics")}
    blank = UserProfile(user_id="u")
    assert calibration_error(blank, ["p"], articles) == 0.0
    profile = UserProfile(user_id="u", topic_weights={"politics": 1.0})
    assert calibration_error(profile, [], articles) == 0.0


@pytest.mark.parametrize("smoothing", [0.0, 1.0, -0.1, 1.5])
def test_the_smoothing_weight_is_validated(smoothing: float) -> None:
    profile = UserProfile(user_id="u", topic_weights={"a": 1.0})
    with pytest.raises(ValueError, match="smoothing"):
        calibration_error(profile, [], {}, smoothing=smoothing)


def test_the_default_smoothing_is_small_but_present() -> None:
    assert 0.0 < SMOOTHING < 0.1


def test_the_marginal_objective_prefers_the_under_served_topic() -> None:
    reader = {"politics": 0.5, "sports": 0.5}
    mass = {"politics": 3.0}
    politics = _article("p", "politics")
    sports = _article("s", "sports")
    # Equal relevance, so only the divergence decides.
    assert marginal_objective(reader, mass, sports, 0.5, weight=0.5) > marginal_objective(
        reader, mass, politics, 0.5, weight=0.5
    )


def test_a_zero_weight_marginal_objective_is_pure_relevance() -> None:
    reader = {"politics": 1.0}
    assert marginal_objective(
        reader, {}, _article("s", "sports"), 0.7, weight=0.0
    ) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# What it does to a slate.
# ---------------------------------------------------------------------------


def test_calibration_beats_mmr_at_matching_the_reader() -> None:
    """The reason the strategy exists, measured rather than asserted.

    MMR rewards a slate whose items are unlike each other, which on a skewed
    reader pulls in topics they have never engaged with at all.
    """

    articles = _corpus()
    by_id = {article.id: article for article in articles}
    events = _skewed_history(articles)
    profile = _profile(articles, events)

    def slate(strategy: str, weight: float = 0.5) -> list[str]:
        config = FeedConfig(
            size=10,
            rerank_strategy=strategy,
            calibration_weight=weight,
            max_per_source=10,
            exclude_seen=False,
        )
        feed = build_feed("u1", articles, events, as_of=AS_OF, config=config)
        return [item.article_id for item in feed.recommendations]

    mmr = slate("mmr")
    calibrated = slate("calibrated")
    mmr_error = calibration_error(profile, mmr, by_id)
    calibrated_error = calibration_error(profile, calibrated, by_id)

    # A fifth is comfortably inside the eight-fold reduction observed, and
    # does not pin the test to one corpus.
    assert calibrated_error < mmr_error / 5.0, (mmr_error, calibrated_error)
    # MMR reaches topics the reader has no history for; calibration does not.
    unseen = set(TOPICS) - set(reader_distribution(profile))
    assert {by_id[item].topics[0] for item in mmr} & unseen
    assert not {by_id[item].topics[0] for item in calibrated} & unseen


def test_the_trade_against_diversity_is_real_and_visible() -> None:
    # Calibration is not a free improvement: MMR scores higher on its own
    # measure. Reporting both is what makes the trade reviewable.
    articles = _corpus()
    by_id = {article.id: article for article in articles}
    events = _skewed_history(articles)

    def slate(strategy: str) -> list[str]:
        config = FeedConfig(
            size=10, rerank_strategy=strategy, max_per_source=10, exclude_seen=False
        )
        feed = build_feed("u1", articles, events, as_of=AS_OF, config=config)
        return [item.article_id for item in feed.recommendations]

    assert intra_list_diversity(slate("mmr"), by_id) > intra_list_diversity(
        slate("calibrated"), by_id
    )


def test_the_source_cap_still_binds_under_calibration() -> None:
    articles = [_article(f"p{index}", "politics", source="only") for index in range(5)]
    events = [
        Event(
            user_id="u1",
            article_id="p0",
            kind=EventKind.CLICK,
            occurred_at=AS_OF - timedelta(hours=3),
        )
    ]
    config = FeedConfig(size=5, rerank_strategy="calibrated", max_per_source=2, exclude_seen=False)
    feed = build_feed("u1", articles, events, as_of=AS_OF, config=config)
    assert len(feed.recommendations) == 2


def test_the_default_strategy_is_unchanged() -> None:
    assert FeedConfig().rerank_strategy == "mmr"


def test_calibrated_reranking_requires_a_profile() -> None:
    # The two strategies build different slates, so falling back silently would
    # change the output with nothing in it to say so.
    articles = _corpus(per_topic=2)
    config = FeedConfig(rerank_strategy="calibrated")
    scores = score_candidates(articles, UserProfile(user_id="u1"), as_of=AS_OF, config=FeedConfig())
    with pytest.raises(ValueError, match="requires a user profile"):
        rerank(articles, scores, config=config)


def test_mmr_does_not_require_a_profile() -> None:
    articles = _corpus(per_topic=2)
    scores = score_candidates(articles, UserProfile(user_id="u1"), as_of=AS_OF, config=FeedConfig())
    assert rerank(articles, scores, config=FeedConfig(size=3))


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------


def test_the_strategy_names_are_the_documented_pair() -> None:
    assert set(RERANK_STRATEGIES) == {"mmr", "calibrated"}


@pytest.mark.parametrize("strategy", ["MMR", "steck", "", None, 3])
def test_an_unknown_strategy_is_refused(strategy: object) -> None:
    with pytest.raises(ValueError, match="rerank_strategy"):
        FeedConfig(rerank_strategy=strategy)  # type: ignore[arg-type]


@pytest.mark.parametrize("weight", [-0.1, 1.5, float("nan"), float("inf")])
def test_the_calibration_weight_is_validated(weight: float) -> None:
    with pytest.raises(ValueError):
        FeedConfig(calibration_weight=weight)


def test_the_settings_round_trip_through_a_mapping() -> None:
    config = FeedConfig.from_mapping({"rerank_strategy": "calibrated", "calibration_weight": 0.3})
    assert config.rerank_strategy == "calibrated"
    assert config.calibration_weight == pytest.approx(0.3)
    assert config.to_dict()["rerank_strategy"] == "calibrated"
