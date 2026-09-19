"""Deterministic pointwise click-probability learning for feed candidates."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Self

from mosaicfeed.config import FeedConfig
from mosaicfeed.io import atomic_write_text, load_json_text, parse_datetime
from mosaicfeed.models import Article, Event, EventKind, UserProfile
from mosaicfeed.profile import build_profile, event_signal
from mosaicfeed.scoring import score_article
from mosaicfeed.text_features import NewsTextEncoder, TextFeatureConfig

MODEL_FORMAT = "mosaicfeed.pointwise-logistic"
MODEL_SCHEMA_VERSION = 1
TEXT_MODEL_SCHEMA_VERSION = 2
MAX_MODEL_STATE_BYTES = 16 * 1024 * 1024
FEATURE_NAMES = (
    "bias",
    "interest",
    "freshness",
    "quality",
    "novelty",
    "popularity",
    "exploration",
)
TEXT_FEATURE_NAMES = (*FEATURE_NAMES, "text_affinity")
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


def _finite_dot(weights: Iterable[float], features: Iterable[float], name: str) -> float:
    try:
        value = math.fsum(
            weight * feature for weight, feature in zip(weights, features, strict=True)
        )
    except OverflowError as error:
        raise ValueError(f"{name} is not finite") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} is not finite")
    return value


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
        self._text_encoder: NewsTextEncoder | None = None

    @property
    def is_fitted(self) -> bool:
        return bool(self._weights)

    @property
    def weights(self) -> Mapping[str, float]:
        self._require_fitted()
        names = TEXT_FEATURE_NAMES if self._text_encoder is not None else FEATURE_NAMES
        return dict(zip(names, self._weights, strict=True))

    @property
    def text_encoder(self) -> NewsTextEncoder | None:
        """The frozen training-only news vocabulary, if this opt-in model uses one."""

        self._require_fitted()
        return self._text_encoder

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
        deadline_check: Callable[[], None] | None = None,
        text_affinity: float | None = None,
    ) -> tuple[float, ...]:
        evidence = score_article(
            article,
            profile,
            as_of=as_of,
            config=config,
            deadline_check=deadline_check,
        )
        base = (
            1.0,
            evidence.interest,
            evidence.freshness,
            evidence.quality,
            evidence.novelty,
            evidence.popularity,
            evidence.exploration,
        )
        return base if text_affinity is None else (*base, _finite(text_affinity, "text affinity"))

    def fit(
        self,
        articles: Iterable[Article],
        events: Iterable[Event],
        *,
        as_of: datetime,
        config: FeedConfig | None = None,
        text_features: bool = False,
        text_config: TextFeatureConfig | None = None,
        text_vocabulary_articles: Iterable[Article] | None = None,
    ) -> Self:
        """Fit with event-time feature construction and model-local shuffling."""

        cutoff = _clock(as_of, "as_of")
        active_config = config or FeedConfig()
        if not isinstance(text_features, bool) or (text_config is not None and not text_features):
            raise ValueError("text_config requires enabled text_features")
        if text_vocabulary_articles is not None and not text_features:
            raise ValueError("text_vocabulary_articles requires enabled text_features")
        if text_config is not None and not isinstance(text_config, TextFeatureConfig):
            raise ValueError("text_config must be TextFeatureConfig")
        active_text_config = (
            (
                TextFeatureConfig()
                if text_config is None
                else TextFeatureConfig(**asdict(text_config))
            )
            if text_features
            else None
        )
        article_list = (
            tuple(islice(articles, active_text_config.max_articles + 1))
            if active_text_config is not None
            else tuple(articles)
        )
        if active_text_config is not None and len(article_list) > active_text_config.max_articles:
            raise ValueError("text training exceeds max_articles")
        if not article_list or any(not isinstance(article, Article) for article in article_list):
            raise ValueError("articles must contain at least one Article")
        article_map = {article.id: article for article in article_list}
        if len(article_map) != len(article_list):
            raise ValueError("articles must have unique ids")
        event_list = (
            tuple(islice(events, active_text_config.max_events + 1))
            if active_text_config is not None
            else tuple(events)
        )
        if active_text_config is not None and len(event_list) > active_text_config.max_events:
            raise ValueError("text training exceeds max_events")
        if any(not isinstance(event, Event) for event in event_list):
            raise ValueError("events must contain Event values")

        history: dict[str, list[Event]] = defaultdict(list)
        text_history: dict[str, dict[str, float]] = defaultdict(dict)
        examples: list[tuple[tuple[float, ...], float]] = []
        ordered = sorted(enumerate(event_list), key=lambda pair: (pair[1].occurred_at, pair[0]))
        eligible_events = [event for _, event in ordered if event.occurred_at <= cutoff]
        text_encoder: NewsTextEncoder | None = None
        if text_features and eligible_events:
            for event in eligible_events:
                candidate = article_map.get(event.article_id)
                if candidate is None:
                    raise ValueError(f"event references unknown article: {event.article_id}")
                if candidate.published_at > event.occurred_at:
                    raise ValueError(
                        f"event for {event.article_id} predates that article's publication time"
                    )
            first_time = eligible_events[0].occurred_at
            # Event IDs after the first example are not known at that example's
            # cutoff, even when their articles were already in the catalog.
            vocabulary_articles: tuple[Article, ...]
            if text_vocabulary_articles is None:
                vocabulary_articles = (article_map[eligible_events[0].article_id],)
                vocabulary_source = "first-event-seed"
            else:
                if active_text_config is None:
                    raise ValueError("text vocabulary configuration is unavailable")
                vocabulary_articles = tuple(
                    islice(text_vocabulary_articles, active_text_config.max_articles + 1)
                )
                if len(vocabulary_articles) > active_text_config.max_articles:
                    raise ValueError("text vocabulary snapshot exceeds max_articles")
                for snapshot_article in vocabulary_articles:
                    if not isinstance(snapshot_article, Article) or (
                        article_map.get(snapshot_article.id) != snapshot_article
                    ):
                        raise ValueError(
                            "text vocabulary snapshot must match articles in the training catalog"
                        )
                vocabulary_source = "declared-training-snapshot"
            text_encoder = NewsTextEncoder.fit(
                vocabulary_articles,
                as_of=first_time,
                config=active_text_config,
                source_kind=vocabulary_source,
            )
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
            affinity = (
                text_encoder.affinity(article, text_history[event.user_id])
                if text_encoder is not None
                else None
            )
            examples.append(
                (
                    self._features(
                        article,
                        profile,
                        as_of=event.occurred_at,
                        config=active_config,
                        text_affinity=affinity,
                    ),
                    1.0 if event.kind in POSITIVE_KINDS else 0.0,
                )
            )
            history[event.user_id].append(event)
            if text_encoder is not None:
                strength = event_signal(event.kind, active_config) * event.weight
                for name, value in text_encoder.vectorize(article).items():
                    updated = text_history[event.user_id].get(name, 0.0) + strength * value
                    if not math.isfinite(updated):
                        raise ValueError("text history accumulation is not finite")
                    text_history[event.user_id][name] = updated
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

        weights = [0.0] * (
            len(TEXT_FEATURE_NAMES) if text_encoder is not None else len(FEATURE_NAMES)
        )
        generator = random.Random(self.seed)
        indices = list(range(len(examples)))
        for _ in range(self.epochs):
            generator.shuffle(indices)
            for index in indices:
                features, label = examples[index]
                prediction = _sigmoid(_finite_dot(weights, features, "training logit"))
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
        self._text_encoder = text_encoder
        return self

    def predict(
        self,
        article: Article,
        profile: UserProfile,
        *,
        as_of: datetime,
        deadline_check: Callable[[], None] | None = None,
        text_profile: Mapping[str, float] | None = None,
    ) -> float:
        """Predict a finite probability for one article/profile pair."""

        active_config = self._fitted_config()
        if not isinstance(article, Article) or not isinstance(profile, UserProfile):
            raise ValueError("article and profile must be MosaicFeed model values")
        clock = _clock(as_of, "as_of")
        features = self._features(
            article,
            profile,
            as_of=clock,
            config=active_config,
            deadline_check=deadline_check,
            text_affinity=(
                self._text_encoder.affinity(article, text_profile or {})
                if self._text_encoder is not None
                else None
            ),
        )
        return _sigmoid(_finite_dot(self._weights, features, "prediction logit"))

    def rank_for_user(
        self,
        user_id: str,
        articles: Iterable[Article],
        events: Iterable[Event],
        *,
        as_of: datetime,
        k: int = 10,
        candidate_ids: Iterable[str] | None = None,
        deadline_check: Callable[[], None] | None = None,
    ) -> tuple[ClickPrediction, ...]:
        """Build a point-in-time profile and rank eligible fitted-feature candidates.

        ``articles`` is always the complete catalog used to derive the profile.  An
        optional ``candidate_ids`` subset restricts only the items scored, so profile
        history remains correct even when an earlier interaction is not a candidate.
        """

        active_config = self._fitted_config()
        _positive_int(k, "k")
        clock = _clock(as_of, "as_of")
        if type(articles) is tuple:
            article_list = articles
        else:
            article_values: list[Article] = []
            for index, article in enumerate(articles):
                if deadline_check is not None and index % 64 == 0:
                    deadline_check()
                article_values.append(article)
                if (
                    self._text_encoder is not None
                    and len(article_values) > self._text_encoder.config.max_articles
                ):
                    raise ValueError("text ranking exceeds max_articles")
            article_list = tuple(article_values)
        if (
            self._text_encoder is not None
            and len(article_list) > self._text_encoder.config.max_articles
        ):
            raise ValueError("text ranking exceeds max_articles")
        for index, article in enumerate(article_list):
            if deadline_check is not None and index % 64 == 0:
                deadline_check()
            if not isinstance(article, Article):
                raise ValueError("articles must contain Article values")
        article_map: dict[str, Article] = {}
        for index, article in enumerate(article_list):
            if deadline_check is not None and index % 64 == 0:
                deadline_check()
            if article.id in article_map:
                raise ValueError("articles must have unique ids")
            article_map[article.id] = article
        if type(events) is tuple:
            event_list = events
        else:
            event_values: list[Event] = []
            for index, event in enumerate(events):
                if deadline_check is not None and index % 64 == 0:
                    deadline_check()
                event_values.append(event)
                if (
                    self._text_encoder is not None
                    and len(event_values) > self._text_encoder.config.max_events
                ):
                    raise ValueError("text ranking exceeds max_events")
            event_list = tuple(event_values)
        if (
            self._text_encoder is not None
            and len(event_list) > self._text_encoder.config.max_events
        ):
            raise ValueError("text ranking exceeds max_events")
        for index, event in enumerate(event_list):
            if deadline_check is not None and index % 64 == 0:
                deadline_check()
            if not isinstance(event, Event):
                raise ValueError("events must contain Event values")
        if candidate_ids is None:
            candidates = article_list
        else:
            selected_values: list[str] = []
            selected_set: set[str] = set()
            for index, article_id in enumerate(candidate_ids):
                if deadline_check is not None and index % 64 == 0:
                    deadline_check()
                if not isinstance(article_id, str) or not article_id.strip():
                    raise ValueError("candidate_ids must contain non-empty strings")
                if article_id in selected_set:
                    raise ValueError("candidate_ids must be unique")
                selected_values.append(article_id)
                if (
                    self._text_encoder is not None
                    and len(selected_values) > self._text_encoder.config.max_articles
                ):
                    raise ValueError("text ranking exceeds max_articles")
                selected_set.add(article_id)
            selected_ids = tuple(selected_values)
            unknown_ids = sorted(selected_set - set(article_map))
            if unknown_ids:
                raise ValueError(
                    f"candidate_ids reference unknown articles: {', '.join(unknown_ids)}"
                )
            candidates = tuple(article_map[article_id] for article_id in selected_ids)
        profile_events = event_list
        if self._text_encoder is not None:
            visible_events: list[Event] = []
            for index, event in enumerate(event_list):
                if deadline_check is not None and index % 64 == 0:
                    deadline_check()
                source = article_map.get(event.article_id)
                if source is None or source.published_at <= event.occurred_at:
                    visible_events.append(event)
            profile_events = tuple(visible_events)
        profile = build_profile(
            user_id,
            profile_events,
            article_map,
            as_of=clock,
            config=active_config,
            deadline_check=deadline_check,
        )
        text_profile: dict[str, float] = {}
        if self._text_encoder is not None:
            for index, event in enumerate(event_list):
                if deadline_check is not None and index % 64 == 0:
                    deadline_check()
                history_article = article_map.get(event.article_id)
                if (
                    event.user_id != user_id
                    or event.occurred_at > clock
                    or history_article is None
                    or history_article.published_at > event.occurred_at
                ):
                    continue
                strength = event_signal(event.kind, active_config) * event.weight
                for name, value in self._text_encoder.vectorize(history_article).items():
                    updated = text_profile.get(name, 0.0) + strength * value
                    if not math.isfinite(updated):
                        raise ValueError("text history accumulation is not finite")
                    text_profile[name] = updated
        scored: list[tuple[str, float]] = []
        for index, article in enumerate(candidates):
            if deadline_check is not None and index % 64 == 0:
                deadline_check()
            if article.published_at > clock or (
                active_config.exclude_seen and article.id in profile.seen_article_ids
            ):
                continue
            scored.append(
                (
                    article.id,
                    self.predict(
                        article,
                        profile,
                        as_of=clock,
                        deadline_check=deadline_check,
                        text_profile=text_profile,
                    ),
                )
            )
        if deadline_check is not None:
            deadline_check()
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        if deadline_check is not None:
            deadline_check()
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
        state: dict[str, object] = {
            "format": MODEL_FORMAT,
            "schema_version": (
                TEXT_MODEL_SCHEMA_VERSION
                if self._text_encoder is not None
                else MODEL_SCHEMA_VERSION
            ),
            "parameters": {
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "l2": self.l2,
                "seed": self.seed,
            },
            "feature_names": list(
                TEXT_FEATURE_NAMES if self._text_encoder is not None else FEATURE_NAMES
            ),
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
        if self._text_encoder is not None:
            state["text_encoder"] = self._text_encoder.to_state()
            state["state_sha256"] = hashlib.sha256(
                json.dumps(
                    state,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        return state

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
        if not isinstance(state, Mapping):
            raise ValueError("pointwise model state has missing or unknown fields")
        schema_version = state.get("schema_version")
        if schema_version == TEXT_MODEL_SCHEMA_VERSION and type(schema_version) is int:
            expected.update({"text_encoder", "state_sha256"})
        if set(state) != expected:
            raise ValueError("pointwise model state has missing or unknown fields")
        if schema_version == TEXT_MODEL_SCHEMA_VERSION:
            integrity = state["state_sha256"]
            if (
                not isinstance(integrity, str)
                or len(integrity) != 64
                or any(character not in "0123456789abcdef" for character in integrity)
            ):
                raise ValueError("pointwise text state checksum is malformed")
            payload = {key: value for key, value in state.items() if key != "state_sha256"}
            try:
                actual = hashlib.sha256(
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
            except (TypeError, ValueError, UnicodeError, OverflowError) as error:
                raise ValueError("pointwise text state checksum is malformed") from error
            if not hmac.compare_digest(integrity, actual):
                raise ValueError("pointwise text state checksum mismatch")
        if (
            state["format"] != MODEL_FORMAT
            or isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version not in {MODEL_SCHEMA_VERSION, TEXT_MODEL_SCHEMA_VERSION}
        ):
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
        expected_names = (
            TEXT_FEATURE_NAMES if schema_version == TEXT_MODEL_SCHEMA_VERSION else FEATURE_NAMES
        )
        if state["feature_names"] != list(expected_names):
            raise ValueError("pointwise model feature schema is incompatible")
        raw_config = state["feature_config"]
        if not isinstance(raw_config, dict):
            raise ValueError("pointwise model feature_config must be an object")
        config = FeedConfig.from_mapping(dict(raw_config))
        raw_weights = state["weights"]
        if not isinstance(raw_weights, list) or len(raw_weights) != len(expected_names):
            raise ValueError("pointwise model weights are malformed")
        weights = tuple(_finite(value, "weight") for value in raw_weights)
        _finite_dot(tuple(abs(weight) for weight in weights), (1.0,) * len(weights), "weight norm")
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
        if schema_version == TEXT_MODEL_SCHEMA_VERSION:
            instance._text_encoder = NewsTextEncoder.from_state(state["text_encoder"])
            if instance._text_encoder.trained_as_of > trained_as_of:
                raise ValueError("text vocabulary was fitted after model training cutoff")
        return instance

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        encoded = (
            json.dumps(
                self.to_state(),
                indent=2,
                sort_keys=True,
                ensure_ascii=self._text_encoder is None,
                allow_nan=False,
            )
            + "\n"
        )
        if len(encoded.encode("utf-8")) > MAX_MODEL_STATE_BYTES:
            raise ValueError("pointwise model state exceeds size limit")
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(destination, encoded)

    @classmethod
    def load(cls, path: str | Path) -> PointwiseLogisticRanker:
        source = Path(path)
        with source.open("rb") as stream:
            raw = stream.read(MAX_MODEL_STATE_BYTES + 1)
        if len(raw) > MAX_MODEL_STATE_BYTES:
            raise ValueError("pointwise model state exceeds size limit")
        payload = load_json_text(raw.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("pointwise model file must contain a JSON object")
        return cls.from_state(payload)
