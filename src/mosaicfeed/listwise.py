"""Candidate-lossless, point-in-time listwise impression ranking.

This model optimizes a distribution over *displayed* candidates, not a
calibrated probability of clicking an individual article.
"""

from __future__ import annotations

import hmac
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby, islice
from pathlib import Path
from types import MappingProxyType
from typing import Self

from mosaicfeed.config import FeedConfig
from mosaicfeed.datasets import MindImpression
from mosaicfeed.io import atomic_write_text, load_json_text, parse_datetime
from mosaicfeed.learning import FEATURE_NAMES, PointwiseLogisticRanker, _finite_dot
from mosaicfeed.models import Article, Event
from mosaicfeed.pairwise import (
    MAX_CANDIDATES,
    MAX_EPOCHS,
    MAX_IMPRESSIONS,
    MAX_MAGNITUDE,
    MAX_PROFILE_SCANS,
    MAX_STATE_BYTES,
    MAX_TOPIC_VISITS,
    MAX_UPDATES,
    _aware,
    _bounded_int,
    _catalog,
    _catalog_digest,
    _clicked_events,
    _digest,
    _history_digest,
    _impressions,
    _number,
    _sha,
)
from mosaicfeed.profile import build_profile

LISTWISE_FORMAT = "mosaicfeed.listwise-impression"
LISTWISE_SCHEMA_VERSION = 1
MAX_TRAINING_CANDIDATES = 1_000_000


@dataclass(frozen=True, slots=True)
class ListwiseTrainingSummary:
    partition: str
    cutoff: datetime
    impressions: int
    comparable_impressions: int
    skipped_all_positive: int
    skipped_all_negative: int
    training_candidates: int
    updates: int
    training_sha256: str
    history_sha256: str
    catalog_sha256: str
    source_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_sha256", MappingProxyType(dict(self.source_sha256)))

    def to_dict(self) -> dict[str, object]:
        return {
            "partition": self.partition,
            "cutoff": self.cutoff.isoformat(),
            "impressions": self.impressions,
            "comparable_impressions": self.comparable_impressions,
            "skipped_all_positive": self.skipped_all_positive,
            "skipped_all_negative": self.skipped_all_negative,
            "training_candidates": self.training_candidates,
            "updates": self.updates,
            "training_sha256": self.training_sha256,
            "history_sha256": self.history_sha256,
            "catalog_sha256": self.catalog_sha256,
            "source_sha256": dict(self.source_sha256),
        }


class ListwiseImpressionRanker:
    """SGD on softmax cross-entropy with uniform clicked-item targets."""

    def __init__(
        self, *, epochs: int = 20, learning_rate: float = 0.05, l2: float = 0.001, seed: int = 17
    ) -> None:
        self.epochs = _bounded_int(epochs, "epochs", MAX_EPOCHS)
        self.learning_rate = _number(learning_rate, "learning_rate")
        if self.learning_rate == 0 or self.learning_rate > 1:
            raise ValueError("learning_rate must be in (0, 1]")
        self.l2 = _number(l2, "l2")
        if self.l2 > 1:
            raise ValueError("l2 must be in [0, 1]")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be a bounded non-negative integer")
        self.seed = seed
        self._weights: tuple[float, ...] = ()
        self._config: FeedConfig | None = None
        self._summary: ListwiseTrainingSummary | None = None

    @property
    def is_fitted(self) -> bool:
        return bool(self._weights)

    @property
    def weights(self) -> Mapping[str, float]:
        self._fitted()
        return dict(zip(FEATURE_NAMES, self._weights, strict=True))

    @property
    def training(self) -> ListwiseTrainingSummary:
        self._fitted()
        if self._summary is None:
            raise ValueError("listwise model has inconsistent training summary")
        return self._summary

    def _fitted(self) -> FeedConfig:
        if not self.is_fitted or self._config is None:
            raise ValueError("ListwiseImpressionRanker has not been fitted")
        return self._config

    def fit(
        self,
        articles: Iterable[Article],
        impressions: Iterable[MindImpression],
        *,
        partition: str,
        cutoff: datetime,
        held_out_impression_ids: Iterable[str] = (),
        held_out_impressions: Iterable[MindImpression] | None = None,
        source_sha256: Mapping[str, str] | None = None,
        config: FeedConfig | None = None,
    ) -> Self:
        """Validate the entire training split before modifying fitted state."""

        if not isinstance(partition, str) or not partition.strip() or len(partition) > 256:
            raise ValueError("partition must be a non-empty bounded string")
        clock = _aware(cutoff, "cutoff")
        if config is not None and not isinstance(config, FeedConfig):
            raise ValueError("config must be FeedConfig")
        active_config = config or FeedConfig()
        catalog = _catalog(articles)
        records = _impressions(impressions, catalog)
        if any(item.occurred_at > clock for item in records):
            raise ValueError("training impression occurs after cutoff")
        held_out = tuple(islice(held_out_impression_ids, MAX_IMPRESSIONS + 1))
        if held_out_impressions is not None:
            if held_out:
                raise ValueError("provide either held-out ids or held-out impressions")
            held_out_records = _impressions(held_out_impressions, catalog)
            if any(item.occurred_at <= clock for item in held_out_records):
                raise ValueError("held-out impression is not after training cutoff")
            held_out = tuple(item.impression_id for item in held_out_records)
        if (
            len(held_out) > MAX_IMPRESSIONS
            or any(
                not isinstance(item, str) or not item.strip() or len(item) > 256
                for item in held_out
            )
            or len(held_out) != len(set(held_out))
        ):
            raise ValueError("held-out ids must be unique bounded non-empty strings")
        if set(held_out).intersection(item.impression_id for item in records):
            raise ValueError("training and held-out impression ids overlap")
        raw_sources = {} if source_sha256 is None else source_sha256
        if not isinstance(raw_sources, Mapping) or len(raw_sources) > 8:
            raise ValueError("source hashes must be a bounded mapping")
        sources: dict[str, str] = {}
        for name, value in raw_sources.items():
            if not isinstance(name, str) or not name.strip() or len(name) > 64:
                raise ValueError("source hash key is invalid")
            sources[name] = _sha(value, f"source_sha256.{name}")
        try:
            _digest([partition, held_out, sources, active_config.to_dict()])
        except (TypeError, ValueError, OverflowError, UnicodeError) as error:
            raise ValueError("training metadata must be canonical UTF-8 JSON") from error

        history: dict[str, list[Event]] = defaultdict(list)
        history_topics: dict[str, int] = defaultdict(int)
        slates: list[tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]] = []
        skipped_positive = 0
        skipped_negative = 0
        profile_scans = 0
        topic_visits = 0
        training_candidates = 0
        fingerprint: list[object] = []
        for when, group in groupby(records, key=lambda item: item.occurred_at):
            same_time = tuple(group)
            for item in same_time:
                profile_scans += len(history[item.user_id])
                if profile_scans > MAX_PROFILE_SCANS:
                    raise ValueError("training profile scans exceed limit")
                topic_visits += history_topics[item.user_id] + 2 * sum(
                    len(catalog[candidate.article_id].topics) for candidate in item.candidates
                )
                if topic_visits > MAX_TOPIC_VISITS:
                    raise ValueError("training topic visits exceed limit")
                profile = build_profile(
                    item.user_id, history[item.user_id], catalog, as_of=when, config=active_config
                )
                vectors: dict[str, tuple[float, ...]] = {}
                for candidate in item.candidates:
                    features = PointwiseLogisticRanker._features(
                        catalog[candidate.article_id], profile, as_of=when, config=active_config
                    )
                    if any(
                        not math.isfinite(value) or abs(value) > MAX_MAGNITUDE for value in features
                    ):
                        raise ValueError("listwise feature magnitude exceeds limit")
                    vectors[candidate.article_id] = features
                clicked = {
                    candidate.article_id for candidate in item.candidates if candidate.clicked
                }
                ordered_ids = sorted(vectors)
                fingerprint.append(
                    [
                        item.impression_id,
                        item.user_id,
                        when.isoformat(),
                        [[key, vectors[key], key in clicked] for key in ordered_ids],
                    ]
                )
                if not clicked:
                    skipped_negative += 1
                elif len(clicked) == len(vectors):
                    skipped_positive += 1
                else:
                    training_candidates += len(vectors)
                    if training_candidates > MAX_TRAINING_CANDIDATES:
                        raise ValueError("training candidate count exceeds limit")
                    slates.append(
                        (
                            tuple(
                                1.0 / len(clicked) if key in clicked else 0.0 for key in ordered_ids
                            ),
                            tuple(vectors[key] for key in ordered_ids),
                        )
                    )
            # The complete timestamp group used the prior history snapshot.
            for event in _clicked_events(same_time):
                history[event.user_id].append(event)
                history_topics[event.user_id] += len(catalog[event.article_id].topics)
        if not slates:
            raise ValueError("training requires at least one mixed-label impression")
        if len(slates) * self.epochs > MAX_UPDATES:
            raise ValueError("training update count exceeds limit")
        try:
            training_digest = _digest(fingerprint)
            history_digest = _history_digest(records)
            catalog_digest = _catalog_digest(catalog)
        except (TypeError, ValueError, OverflowError, UnicodeError) as error:
            raise ValueError("training data must be canonical UTF-8 JSON") from error

        weights = [0.0] * len(FEATURE_NAMES)
        indices = list(range(len(slates)))
        generator = random.Random(self.seed)
        for _ in range(self.epochs):
            generator.shuffle(indices)
            for index in indices:
                targets, slate_vectors = slates[index]
                logits = [
                    _finite_dot(weights, vector, "listwise training logit")
                    for vector in slate_vectors
                ]
                shift = max(logits)
                exponentials = [math.exp(value - shift) for value in logits]
                normalizer = sum(exponentials)
                if not math.isfinite(normalizer) or normalizer <= 0:
                    raise ValueError("listwise softmax normalization diverged")
                for feature_index in range(len(weights)):
                    gradient = math.fsum(
                        (target - exponential / normalizer) * vector[feature_index]
                        for target, exponential, vector in zip(
                            targets, exponentials, slate_vectors, strict=True
                        )
                    )
                    updated = weights[feature_index] + self.learning_rate * (
                        gradient - self.l2 * weights[feature_index]
                    )
                    if not math.isfinite(updated) or abs(updated) > MAX_MAGNITUDE:
                        raise ValueError("listwise training diverged")
                    weights[feature_index] = updated
        summary = ListwiseTrainingSummary(
            partition,
            clock,
            len(records),
            len(slates),
            skipped_positive,
            skipped_negative,
            training_candidates,
            len(slates) * self.epochs,
            training_digest,
            history_digest,
            catalog_digest,
            sources,
        )
        self._weights = tuple(weights)
        self._config = active_config
        self._summary = summary
        return self

    def score_impressions(
        self,
        articles: Iterable[Article],
        impressions: Iterable[MindImpression],
        *,
        training_impressions: Iterable[MindImpression],
    ) -> dict[str, dict[str, float]]:
        """Score a disjoint post-cutoff split using verified training clicks only."""

        config = self._fitted()
        catalog = _catalog(articles)
        if not hmac.compare_digest(_catalog_digest(catalog), self.training.catalog_sha256):
            raise ValueError("article catalog differs from fitted catalog")
        history_records = _impressions(training_impressions, catalog)
        if not hmac.compare_digest(_history_digest(history_records), self.training.history_sha256):
            raise ValueError("training impression history differs from fitted history")
        targets = _impressions(impressions, catalog)
        history_ids = {item.impression_id for item in history_records}
        if any(item.impression_id in history_ids for item in targets):
            raise ValueError("training and scored impression ids overlap")
        if any(item.occurred_at <= self.training.cutoff for item in targets):
            raise ValueError("scored impressions must occur after training cutoff")
        by_user: dict[str, list[Event]] = defaultdict(list)
        by_user_topics: dict[str, int] = defaultdict(int)
        for event in _clicked_events(history_records):
            by_user[event.user_id].append(event)
            by_user_topics[event.user_id] += len(catalog[event.article_id].topics)
        result: dict[str, dict[str, float]] = {}
        profile_scans = 0
        topic_visits = 0
        for item in targets:
            profile_scans += len(by_user[item.user_id])
            if profile_scans > MAX_PROFILE_SCANS:
                raise ValueError("scoring profile scans exceed limit")
            topic_visits += by_user_topics[item.user_id] + 2 * sum(
                len(catalog[candidate.article_id].topics) for candidate in item.candidates
            )
            if topic_visits > MAX_TOPIC_VISITS:
                raise ValueError("scoring topic visits exceed limit")
            visible = [
                event for event in by_user[item.user_id] if event.occurred_at < item.occurred_at
            ]
            profile = build_profile(
                item.user_id, visible, catalog, as_of=item.occurred_at, config=config
            )
            scores: dict[str, float] = {}
            for candidate in item.candidates:
                features = PointwiseLogisticRanker._features(
                    catalog[candidate.article_id], profile, as_of=item.occurred_at, config=config
                )
                scores[candidate.article_id] = _finite_dot(
                    self._weights, features, "listwise score logit"
                )
            result[item.impression_id] = scores
        return result

    def to_state(self) -> dict[str, object]:
        config = self._fitted()
        state: dict[str, object] = {
            "format": LISTWISE_FORMAT,
            "schema_version": LISTWISE_SCHEMA_VERSION,
            "objective": "listwise-impression",
            "score_semantics": "raw logit; not probability",
            "parameters": {
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "l2": self.l2,
                "seed": self.seed,
            },
            "feature_names": list(FEATURE_NAMES),
            "feature_config": config.to_dict(),
            "weights": list(self._weights),
            "training": self.training.to_dict(),
        }
        state["state_sha256"] = _digest(state)
        return state

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> Self:
        expected = {
            "format",
            "schema_version",
            "objective",
            "score_semantics",
            "parameters",
            "feature_names",
            "feature_config",
            "weights",
            "training",
            "state_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("listwise model state has missing or unknown fields")
        integrity = _sha(state["state_sha256"], "state_sha256")
        try:
            actual = _digest({key: value for key, value in state.items() if key != "state_sha256"})
        except (TypeError, ValueError, OverflowError, UnicodeError) as error:
            raise ValueError("listwise model state is not canonical JSON") from error
        if not hmac.compare_digest(integrity, actual):
            raise ValueError("listwise model state checksum mismatch")
        if (
            state["format"] != LISTWISE_FORMAT
            or type(state["schema_version"]) is not int
            or state["schema_version"] != LISTWISE_SCHEMA_VERSION
            or state["objective"] != "listwise-impression"
            or state["score_semantics"] != "raw logit; not probability"
        ):
            raise ValueError("unsupported listwise objective, semantics, or schema")
        parameters = state["parameters"]
        if not isinstance(parameters, Mapping) or set(parameters) != {
            "epochs",
            "learning_rate",
            "l2",
            "seed",
        }:
            raise ValueError("listwise parameters are malformed")
        model = cls(
            epochs=parameters["epochs"],
            learning_rate=parameters["learning_rate"],
            l2=parameters["l2"],
            seed=parameters["seed"],
        )
        if state["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("listwise feature schema is incompatible")
        raw_config = state["feature_config"]
        if not isinstance(raw_config, dict):
            raise ValueError("listwise feature config is malformed")
        config = FeedConfig.from_mapping(raw_config)
        raw_weights = state["weights"]
        if not isinstance(raw_weights, list) or len(raw_weights) != len(FEATURE_NAMES):
            raise ValueError("listwise weights are malformed")
        weights = tuple(_number(value, "weight", minimum=-MAX_MAGNITUDE) for value in raw_weights)
        raw_training = state["training"]
        fields = {
            "partition",
            "cutoff",
            "impressions",
            "comparable_impressions",
            "skipped_all_positive",
            "skipped_all_negative",
            "training_candidates",
            "updates",
            "training_sha256",
            "history_sha256",
            "catalog_sha256",
            "source_sha256",
        }
        if not isinstance(raw_training, Mapping) or set(raw_training) != fields:
            raise ValueError("listwise training summary is malformed")
        partition = raw_training["partition"]
        if not isinstance(partition, str) or not partition.strip() or len(partition) > 256:
            raise ValueError("listwise partition is malformed")
        impressions = _bounded_int(raw_training["impressions"], "impressions", MAX_IMPRESSIONS)
        comparable = _bounded_int(
            raw_training["comparable_impressions"], "comparable_impressions", MAX_IMPRESSIONS
        )
        skipped_positive = raw_training["skipped_all_positive"]
        skipped_negative = raw_training["skipped_all_negative"]
        if any(
            type(value) is not int or value < 0 or value > MAX_IMPRESSIONS
            for value in (skipped_positive, skipped_negative)
        ):
            raise ValueError("listwise skipped counts are malformed")
        if comparable + skipped_positive + skipped_negative != impressions:
            raise ValueError("listwise impression counts are inconsistent")
        candidate_count = _bounded_int(
            raw_training["training_candidates"], "training_candidates", MAX_TRAINING_CANDIDATES
        )
        if not 2 * comparable <= candidate_count <= MAX_CANDIDATES * comparable:
            raise ValueError("listwise candidate count is inconsistent")
        updates = _bounded_int(raw_training["updates"], "updates", MAX_UPDATES)
        if updates != comparable * model.epochs:
            raise ValueError("listwise update count is inconsistent")
        sources = raw_training["source_sha256"]
        if not isinstance(sources, Mapping) or len(sources) > 8:
            raise ValueError("listwise source hashes are malformed")
        normalized_sources: dict[str, str] = {}
        for key, value in sources.items():
            if not isinstance(key, str) or not key.strip() or len(key) > 64:
                raise ValueError("listwise source hash key is malformed")
            normalized_sources[key] = _sha(value, "source hash")
        model._weights = weights
        model._config = config
        model._summary = ListwiseTrainingSummary(
            partition,
            parse_datetime(raw_training["cutoff"], "cutoff"),
            impressions,
            comparable,
            skipped_positive,
            skipped_negative,
            candidate_count,
            updates,
            _sha(raw_training["training_sha256"], "training_sha256"),
            _sha(raw_training["history_sha256"], "history_sha256"),
            _sha(raw_training["catalog_sha256"], "catalog_sha256"),
            normalized_sources,
        )
        return model

    def save(self, path: str | Path) -> None:
        encoded = (
            json.dumps(
                self.to_state(), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
            )
            + "\n"
        )
        if len(encoded.encode("utf-8")) > MAX_STATE_BYTES:
            raise ValueError("listwise model state exceeds size limit")
        atomic_write_text(path, encoded)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_STATE_BYTES + 1)
        if len(raw) > MAX_STATE_BYTES:
            raise ValueError("listwise model state exceeds size limit")
        payload = load_json_text(raw.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("listwise model must contain a JSON object")
        return cls.from_state(payload)
