"""Deterministic pointwise click-probability learning for feed candidates."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

from mosaicfeed.config import FeedConfig
from mosaicfeed.io import load_json_text, parse_datetime
from mosaicfeed.models import Article, Event, EventKind, UserProfile
from mosaicfeed.profile import build_profile
from mosaicfeed.scoring import score_article

MODEL_FORMAT = "mosaicfeed.pointwise-logistic"
MODEL_SCHEMA_VERSION = 1
FEATURE_NAMES = (
    "bias",
    "interest",
    "freshness",
    "quality",
    "novelty",
    "popularity",
    "exploration",
)
POSITIVE_KINDS = frozenset({EventKind.CLICK, EventKind.LIKE})


def _clock(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return numeric


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


@dataclass(frozen=True, slots=True)
class ClickPrediction:
    """One learned candidate probability and deterministic rank."""

    article_id: str
    probability: float
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.article_id, str) or not self.article_id.strip():
            raise ValueError("article_id must be a non-empty string")
        probability = _finite(self.probability, "probability", minimum=0.0)
        if probability > 1.0:
            raise ValueError("probability must be at most 1")
        object.__setattr__(self, "probability", probability)
        _positive_int(self.rank, "rank")

    def to_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


class PointwiseLogisticRanker:
    """Learn click/like probability from leakage-safe historical score features.

    Each training event is featurized from only events that precede it in
    timestamp/input order.  Clicks and likes are positive labels; views and
    hides are negative labels.  The model is intentionally pointwise: without
    complete logged candidate sets it does not claim a pairwise or listwise
    objective.
    """

    def __init__(
        self,
        *,
        epochs: int = 20,
        learning_rate: float = 0.05,
        l2: float = 0.001,
        seed: int = 17,
    ) -> None:
        self.epochs = _positive_int(epochs, "epochs")
        self.learning_rate = _finite(learning_rate, "learning_rate", minimum=0.0)
        if self.learning_rate == 0.0:
            raise ValueError("learning_rate must be positive")
        self.l2 = _finite(l2, "l2", minimum=0.0)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.seed = seed
        self._weights: tuple[float, ...] = ()
        self._config: FeedConfig | None = None
        self._trained_as_of: datetime | None = None
        self._examples = 0
        self._positives = 0
        self._training_sha256 = ""

    @property
    def is_fitted(self) -> bool:
        return bool(self._weights)

    @property
    def weights(self) -> Mapping[str, float]:
        self._require_fitted()
        return dict(zip(FEATURE_NAMES, self._weights, strict=True))

    @property
    def training_examples(self) -> int:
        self._require_fitted()
        return self._examples

    @property
    def training_sha256(self) -> str:
        """Digest of the exact event-time feature/label examples used by SGD."""

        self._require_fitted()
        return self._training_sha256

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError("PointwiseLogisticRanker has not been fitted")

    def _fitted_config(self) -> FeedConfig:
        self._require_fitted()
        if self._config is None:
            raise ValueError("pointwise model has inconsistent fitted configuration")
        return self._config

    @staticmethod
    def _features(
        article: Article,
        profile: UserProfile,
        *,
        as_of: datetime,
        config: FeedConfig,
    ) -> tuple[float, ...]:
        evidence = score_article(article, profile, as_of=as_of, config=config)
        return (
            1.0,
            evidence.interest,
            evidence.freshness,
            evidence.quality,
            evidence.novelty,
            evidence.popularity,
            evidence.exploration,
        )

    def fit(
        self,
        articles: Iterable[Article],
        events: Iterable[Event],
        *,
        as_of: datetime,
        config: FeedConfig | None = None,
    ) -> Self:
        """Fit with event-time feature construction and model-local shuffling."""

        cutoff = _clock(as_of, "as_of")
        active_config = config or FeedConfig()
        article_list = tuple(articles)
        if not article_list or any(not isinstance(article, Article) for article in article_list):
            raise ValueError("articles must contain at least one Article")
        article_map = {article.id: article for article in article_list}
        if len(article_map) != len(article_list):
            raise ValueError("articles must have unique ids")
        event_list = tuple(events)
        if any(not isinstance(event, Event) for event in event_list):
            raise ValueError("events must contain Event values")

        history: dict[str, list[Event]] = defaultdict(list)
        examples: list[tuple[tuple[float, ...], float]] = []
        ordered = sorted(enumerate(event_list), key=lambda pair: (pair[1].occurred_at, pair[0]))
        for _, event in ordered:
            if event.occurred_at > cutoff:
                continue
            article = article_map.get(event.article_id)
            if article is None:
                raise ValueError(f"event references unknown article: {event.article_id}")
            if article.published_at > event.occurred_at:
                raise ValueError(
                    f"event for {event.article_id} predates that article's publication time"
                )
            profile = build_profile(
                event.user_id,
                history[event.user_id],
                article_map,
                as_of=event.occurred_at,
                config=active_config,
            )
            examples.append(
                (
                    self._features(
                        article,
                        profile,
                        as_of=event.occurred_at,
                        config=active_config,
                    ),
                    1.0 if event.kind in POSITIVE_KINDS else 0.0,
                )
            )
            history[event.user_id].append(event)
        positives = sum(int(label) for _, label in examples)
        if not examples:
            raise ValueError("no eligible training events exist at or before as_of")
        if positives == 0 or positives == len(examples):
            raise ValueError("training requires at least one positive and one negative event")

        canonical_examples = json.dumps(
            [[*features, label] for features, label in examples],
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        training_sha256 = hashlib.sha256(canonical_examples).hexdigest()

        weights = [0.0] * len(FEATURE_NAMES)
        generator = random.Random(self.seed)
        indices = list(range(len(examples)))
        for _ in range(self.epochs):
            generator.shuffle(indices)
            for index in indices:
                features, label = examples[index]
                prediction = _sigmoid(
                    math.fsum(w * x for w, x in zip(weights, features, strict=True))
                )
                error = prediction - label
                for feature_index, feature in enumerate(features):
                    penalty = 0.0 if feature_index == 0 else self.l2 * weights[feature_index]
                    updated = weights[feature_index] - self.learning_rate * (
                        error * feature + penalty
                    )
                    if not math.isfinite(updated):
                        raise ValueError("pointwise logistic training diverged")
                    weights[feature_index] = updated
        self._weights = tuple(weights)
        self._config = active_config
        self._trained_as_of = cutoff
        self._examples = len(examples)
        self._positives = positives
        self._training_sha256 = training_sha256
        return self

    def predict(self, article: Article, profile: UserProfile, *, as_of: datetime) -> float:
        """Predict a finite probability for one article/profile pair."""

        active_config = self._fitted_config()
        if not isinstance(article, Article) or not isinstance(profile, UserProfile):
            raise ValueError("article and profile must be MosaicFeed model values")
        clock = _clock(as_of, "as_of")
        features = self._features(article, profile, as_of=clock, config=active_config)
        return _sigmoid(
            math.fsum(
                weight * feature for weight, feature in zip(self._weights, features, strict=True)
            )
        )

    def rank_for_user(
        self,
        user_id: str,
        articles: Iterable[Article],
        events: Iterable[Event],
        *,
        as_of: datetime,
        k: int = 10,
    ) -> tuple[ClickPrediction, ...]:
        """Build a point-in-time profile and rank eligible fitted-feature candidates."""

        active_config = self._fitted_config()
        _positive_int(k, "k")
        clock = _clock(as_of, "as_of")
        article_list = tuple(articles)
        if any(not isinstance(article, Article) for article in article_list):
            raise ValueError("articles must contain Article values")
        article_map = {article.id: article for article in article_list}
        if len(article_map) != len(article_list):
            raise ValueError("articles must have unique ids")
        event_list = tuple(events)
        if any(not isinstance(event, Event) for event in event_list):
            raise ValueError("events must contain Event values")
        profile = build_profile(
            user_id,
            event_list,
            article_map,
            as_of=clock,
            config=active_config,
        )
        scored = [
            (article.id, self.predict(article, profile, as_of=clock))
            for article in article_list
            if article.published_at <= clock
            and (not active_config.exclude_seen or article.id not in profile.seen_article_ids)
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return tuple(
            ClickPrediction(article_id=article_id, probability=probability, rank=rank)
            for rank, (article_id, probability) in enumerate(scored[:k], start=1)
        )

    def to_state(self) -> dict[str, object]:
        """Return a complete strict-JSON-compatible model state."""

        active_config = self._fitted_config()
        trained_as_of = self._trained_as_of
        if trained_as_of is None:
            raise ValueError("pointwise model has inconsistent fitted timestamp")
        return {
            "format": MODEL_FORMAT,
            "schema_version": MODEL_SCHEMA_VERSION,
            "parameters": {
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "l2": self.l2,
                "seed": self.seed,
            },
            "feature_names": list(FEATURE_NAMES),
            "feature_config": active_config.to_dict(),
            "weights": list(self._weights),
            "training": {
                "as_of": trained_as_of.isoformat(),
                "examples": self._examples,
                "positives": self._positives,
                "negatives": self._examples - self._positives,
                "examples_sha256": self._training_sha256,
            },
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> Self:
        """Restore a model after validating every persisted field."""

        expected = {
            "format",
            "schema_version",
            "parameters",
            "feature_names",
            "feature_config",
            "weights",
            "training",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("pointwise model state has missing or unknown fields")
        if state["format"] != MODEL_FORMAT or state["schema_version"] != MODEL_SCHEMA_VERSION:
            raise ValueError("unsupported pointwise model format or schema version")
        parameters = state["parameters"]
        if not isinstance(parameters, Mapping) or set(parameters) != {
            "epochs",
            "learning_rate",
            "l2",
            "seed",
        }:
            raise ValueError("pointwise model parameters are malformed")
        instance = cls(
            epochs=parameters["epochs"],
            learning_rate=parameters["learning_rate"],
            l2=parameters["l2"],
            seed=parameters["seed"],
        )
        if state["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("pointwise model feature schema is incompatible")
        raw_config = state["feature_config"]
        if not isinstance(raw_config, dict):
            raise ValueError("pointwise model feature_config must be an object")
        config = FeedConfig.from_mapping(dict(raw_config))
        raw_weights = state["weights"]
        if not isinstance(raw_weights, list) or len(raw_weights) != len(FEATURE_NAMES):
            raise ValueError("pointwise model weights are malformed")
        weights = tuple(_finite(value, "weight") for value in raw_weights)
        training = state["training"]
        if not isinstance(training, Mapping) or set(training) != {
            "as_of",
            "examples",
            "positives",
            "negatives",
            "examples_sha256",
        }:
            raise ValueError("pointwise model training summary is malformed")
        examples = _positive_int(training["examples"], "training.examples")
        positives = _positive_int(training["positives"], "training.positives")
        negatives = _positive_int(training["negatives"], "training.negatives")
        if positives + negatives != examples:
            raise ValueError("pointwise model training counts are inconsistent")
        examples_sha256 = training["examples_sha256"]
        if (
            not isinstance(examples_sha256, str)
            or len(examples_sha256) != 64
            or examples_sha256 != examples_sha256.lower()
            or any(character not in "0123456789abcdef" for character in examples_sha256)
        ):
            raise ValueError("pointwise model training digest is malformed")
        trained_as_of = parse_datetime(training["as_of"], "training.as_of")
        instance._weights = weights
        instance._config = config
        instance._trained_as_of = trained_as_of
        instance._examples = examples
        instance._positives = positives
        instance._training_sha256 = examples_sha256
        return instance

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_state(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @classmethod
    def load(cls, path: str | Path) -> PointwiseLogisticRanker:
        source = Path(path)
        payload = load_json_text(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("pointwise model file must contain a JSON object")
        return cls.from_state(payload)
