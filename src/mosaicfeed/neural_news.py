"""Bounded, dependency-free trainable title embeddings for offline news ranking.

This is a local neural baseline, not NRMS or an official MIND implementation.
Only earlier training clicks form user histories; held-out labels are never
consulted by the encoder or scorer.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby, islice
from typing import Self

from mosaicfeed.datasets import MindImpression
from mosaicfeed.io import json_text, load_articles_bytes, parse_datetime
from mosaicfeed.mind import evaluate_mind_impressions, load_mind_impressions_bytes
from mosaicfeed.models import Article

FORMAT = "mosaicfeed.neural-news"
EXPERIMENT_FORMAT = "mosaicfeed.neural-news-experiment"
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_ARTICLES = 2_000
MAX_TRAIN_ROWS = 200
MAX_VALIDATION_ROWS = 100
MAX_CANDIDATES = 16
MAX_TITLE_CHARACTERS = 512
MAX_TITLE_TOKENS = 24
MAX_VOCABULARY = 4_096
MAX_HISTORY = 32
MAX_WORK_UNITS = 20_000_000
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _bounded_int(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _bounded_real(value: object, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in ({low}, {high}]")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or not low < result <= high:
        raise ValueError(f"{name} must be a finite number in ({low}, {high}]")
    return result


@dataclass(frozen=True, slots=True)
class NeuralNewsConfig:
    """Small explicit budgets for deterministic CPU-only embedding training."""

    dimension: int = 8
    epochs: int = 4
    learning_rate: float = 0.1
    seed: int = 17
    max_vocabulary: int = 2_048
    max_history: int = MAX_HISTORY

    def __post_init__(self) -> None:
        _bounded_int(self.dimension, "dimension", 2, 32)
        _bounded_int(self.epochs, "epochs", 1, 10)
        object.__setattr__(
            self,
            "learning_rate",
            _bounded_real(self.learning_rate, "learning_rate", 0.0, 1.0),
        )
        _bounded_int(self.seed, "seed", 0, 2**32 - 1)
        _bounded_int(self.max_vocabulary, "max_vocabulary", 2, MAX_VOCABULARY)
        _bounded_int(self.max_history, "max_history", 1, MAX_HISTORY)

    def to_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "max_vocabulary": self.max_vocabulary,
            "max_history": self.max_history,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("invalid neural-news configuration fields")
        return cls(**value)


def _catalog(values: Iterable[Article]) -> dict[str, Article]:
    articles = tuple(islice(values, MAX_ARTICLES + 1))
    if not 1 <= len(articles) <= MAX_ARTICLES:
        raise ValueError("article catalog is empty or exceeds neural-news limit")
    result: dict[str, Article] = {}
    for article in articles:
        if not isinstance(article, Article):
            raise ValueError("catalog must contain Article values")
        if article.id in result:
            raise ValueError("duplicate neural-news article id")
        if len(article.id) > 256 or len(article.title) > MAX_TITLE_CHARACTERS:
            raise ValueError("article id or title exceeds neural-news limit")
        result[article.id] = article
    return result


def _title_tokens(article: Article) -> tuple[str, ...]:
    if article.title_missing:
        return ()
    tokens = tuple(_TOKEN.findall(article.title.casefold()))
    if len(tokens) > MAX_TITLE_TOKENS or any(len(token) > 64 for token in tokens):
        raise ValueError("article title token count exceeds neural-news limit")
    return tokens


def _rows(
    values: Iterable[MindImpression],
    catalog: Mapping[str, Article],
    *,
    maximum: int,
    label: str,
) -> tuple[MindImpression, ...]:
    rows = tuple(islice(values, maximum + 1))
    if not 1 <= len(rows) <= maximum:
        raise ValueError(f"{label} impression count exceeds neural-news limit")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, MindImpression):
            raise ValueError(f"{label} must contain MindImpression values")
        if row.impression_id in seen or len(row.impression_id) > 256 or len(row.user_id) > 256:
            raise ValueError(f"{label} has duplicate or oversized impression/user id")
        seen.add(row.impression_id)
        if not 2 <= len(row.candidates) <= MAX_CANDIDATES:
            raise ValueError(f"{label} candidate count exceeds neural-news limit")
        for candidate in row.candidates:
            article = catalog.get(candidate.article_id)
            if article is None:
                raise ValueError(f"{label} references unknown article")
            if article.published_at > row.occurred_at:
                raise ValueError(f"{label} candidate predates article publication")
    return tuple(sorted(rows, key=lambda row: (row.occurred_at, row.impression_id)))


def _contexts(
    rows: Sequence[MindImpression], max_history: int
) -> tuple[list[tuple[MindImpression, tuple[str, ...]]], dict[str, tuple[str, ...]]]:
    histories: dict[str, list[str]] = defaultdict(list)
    contexts: list[tuple[MindImpression, tuple[str, ...]]] = []
    for _, group in groupby(rows, key=lambda row: row.occurred_at):
        simultaneous = tuple(group)
        # Snapshot every row before applying *any* click at this timestamp.
        for row in simultaneous:
            contexts.append((row, tuple(histories[row.user_id])))
        for row in simultaneous:
            history = histories[row.user_id]
            history.extend(
                candidate.article_id for candidate in row.candidates if candidate.clicked
            )
            if len(history) > max_history:
                raise ValueError("clicked training history exceeds neural-news limit")
    return contexts, {user: tuple(items) for user, items in histories.items()}


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _news(token_ids: Sequence[int], embeddings: Sequence[Sequence[float]]) -> list[float]:
    dimension = len(embeddings[0])
    inverse = 1.0 / len(token_ids)
    return [
        math.tanh(sum(embeddings[token][axis] for token in token_ids) * inverse)
        for axis in range(dimension)
    ]


def _add_gradient(
    gradients: dict[int, list[float]],
    token_ids: Sequence[int],
    encoded: Sequence[float],
    upstream: Sequence[float],
) -> None:
    inverse = 1.0 / len(token_ids)
    for token in token_ids:
        if token == 0:  # OOV/missing-title vector is fixed at zero.
            continue
        row = gradients.setdefault(token, [0.0] * len(encoded))
        for axis, value in enumerate(encoded):
            row[axis] += upstream[axis] * (1.0 - value * value) * inverse


class NeuralNewsRanker:
    """Trainable tanh title encoder + prior-click mean + dot-product logit."""

    def __init__(
        self,
        config: NeuralNewsConfig,
        cutoff: datetime,
        vocabulary: tuple[str, ...],
        embeddings: list[list[float]],
        user_bias: list[float],
        histories: dict[str, tuple[str, ...]],
        train_ids: tuple[str, ...],
        history_lengths: dict[str, int],
        work_units: int,
    ) -> None:
        self.config = config
        self.cutoff = cutoff
        self.vocabulary = vocabulary
        self._embeddings = embeddings
        self._user_bias = user_bias
        self._histories = histories
        self._train_ids = train_ids
        self._history_lengths = history_lengths
        self.work_units_upper_bound = work_units

    @property
    def training_history_lengths(self) -> dict[str, int]:
        return dict(self._history_lengths)

    @staticmethod
    def _token_ids(
        catalog: Mapping[str, Article], vocabulary: Sequence[str]
    ) -> dict[str, tuple[int, ...]]:
        lookup = {token: index for index, token in enumerate(vocabulary)}
        return {
            identity: tuple(lookup.get(token, 0) for token in _title_tokens(article)) or (0,)
            for identity, article in catalog.items()
        }

    @classmethod
    def fit(
        cls,
        articles: Iterable[Article],
        impressions: Iterable[MindImpression],
        *,
        cutoff: datetime,
        config: NeuralNewsConfig | None = None,
        held_out_impression_ids: Iterable[str] = (),
    ) -> Self:
        config = NeuralNewsConfig() if config is None else config
        if not isinstance(config, NeuralNewsConfig):
            raise ValueError("config must be NeuralNewsConfig")
        cutoff = _aware(cutoff, "cutoff")
        catalog = _catalog(articles)
        rows = _rows(impressions, catalog, maximum=MAX_TRAIN_ROWS, label="train")
        if any(row.occurred_at > cutoff for row in rows):
            raise ValueError("training impression occurs after cutoff")
        train_ids = tuple(sorted(row.impression_id for row in rows))
        held_out = tuple(islice(held_out_impression_ids, MAX_VALIDATION_ROWS + 1))
        if len(held_out) > MAX_VALIDATION_ROWS or len(held_out) != len(set(held_out)):
            raise ValueError("held-out impression ids exceed limit or repeat")
        if set(train_ids) & set(held_out):
            raise ValueError("train and held-out impression ids overlap")
        contexts, histories = _contexts(rows, config.max_history)
        training_article_ids = {c.article_id for row in rows for c in row.candidates}
        frequencies = Counter(
            token for identity in training_article_ids for token in _title_tokens(catalog[identity])
        )
        vocabulary = (
            "<unk>",
            *(
                token
                for token, _ in sorted(frequencies.items(), key=lambda pair: (-pair[1], pair[0]))[
                    : config.max_vocabulary - 1
                ]
            ),
        )
        token_ids = cls._token_ids(catalog, vocabulary)
        work = 0
        for row, history in contexts:
            history_tokens = sum(len(token_ids[identity]) for identity in history)
            for candidate in row.candidates:
                work += (
                    config.epochs
                    * config.dimension
                    * 8
                    * (history_tokens + len(token_ids[candidate.article_id]) + 1)
                )
        if work > MAX_WORK_UNITS:
            raise ValueError("neural-news training exceeds aggregate work limit")
        rng = random.Random(config.seed)
        embeddings = [[0.0] * config.dimension] + [
            [rng.uniform(-0.05, 0.05) for _ in range(config.dimension)] for _ in vocabulary[1:]
        ]
        bias = [rng.uniform(-0.05, 0.05) for _ in range(config.dimension)]
        model = cls(
            config,
            cutoff,
            vocabulary,
            embeddings,
            bias,
            histories,
            train_ids,
            {row.impression_id: len(history) for row, history in contexts},
            work,
        )
        for _ in range(config.epochs):
            for row, history in contexts:
                model._train_row(row, history, token_ids)
        return model

    def _train_row(
        self,
        row: MindImpression,
        history: tuple[str, ...],
        token_ids: Mapping[str, tuple[int, ...]],
    ) -> None:
        dimension = self.config.dimension
        history_vectors = [_news(token_ids[identity], self._embeddings) for identity in history]
        user = self._user_bias.copy()
        if history_vectors:
            for vector in history_vectors:
                for axis in range(dimension):
                    user[axis] += vector[axis] / len(history_vectors)
        gradients: dict[int, list[float]] = {}
        bias_gradient = [0.0] * dimension
        for candidate in row.candidates:
            ids = token_ids[candidate.article_id]
            news = _news(ids, self._embeddings)
            logit = sum(user[axis] * news[axis] for axis in range(dimension))
            error = _sigmoid(logit) - float(candidate.clicked)
            _add_gradient(gradients, ids, news, [error * value for value in user])
            for axis in range(dimension):
                bias_gradient[axis] += error * news[axis]
            if history_vectors:
                for history_id, vector in zip(history, history_vectors, strict=True):
                    _add_gradient(
                        gradients,
                        token_ids[history_id],
                        vector,
                        [error * value / len(history_vectors) for value in news],
                    )
        for token, axes in gradients.items():
            for axis, gradient in enumerate(axes):
                step = self.config.learning_rate * max(-1.0, min(1.0, gradient))
                self._embeddings[token][axis] = max(
                    -10.0, min(10.0, self._embeddings[token][axis] - step)
                )
        for axis, gradient in enumerate(bias_gradient):
            step = self.config.learning_rate * max(-1.0, min(1.0, gradient))
            self._user_bias[axis] = max(-10.0, min(10.0, self._user_bias[axis] - step))

    def score_impressions(
        self, articles: Iterable[Article], impressions: Iterable[MindImpression]
    ) -> dict[str, dict[str, float]]:
        catalog = _catalog(articles)
        rows = _rows(impressions, catalog, maximum=MAX_VALIDATION_ROWS, label="validation")
        if any(row.occurred_at <= self.cutoff for row in rows):
            raise ValueError("validation impression is not after cutoff")
        if set(self._train_ids) & {row.impression_id for row in rows}:
            raise ValueError("train and validation impression ids overlap")
        token_ids = self._token_ids(catalog, self.vocabulary)
        vectors = {identity: _news(ids, self._embeddings) for identity, ids in token_ids.items()}
        scores: dict[str, dict[str, float]] = {}
        for row in rows:
            history = self._histories.get(row.user_id, ())
            user = self._user_bias.copy()
            for identity in history:
                if identity not in vectors:
                    raise ValueError("training history article absent from scoring catalog")
                for axis, value in enumerate(vectors[identity]):
                    user[axis] += value / len(history)
            scores[row.impression_id] = {
                candidate.article_id: sum(
                    user[axis] * vectors[candidate.article_id][axis]
                    for axis in range(self.config.dimension)
                )
                for candidate in row.candidates
            }
        return scores

    def to_state(self) -> dict[str, object]:
        return {
            "format": FORMAT,
            "schema_version": 1,
            "config": self.config.to_dict(),
            "cutoff": self.cutoff.isoformat(),
            "vocabulary": list(self.vocabulary),
            "embeddings": [row.copy() for row in self._embeddings],
            "user_bias": self._user_bias.copy(),
            "histories": {user: list(items) for user, items in sorted(self._histories.items())},
            "train_impression_ids": list(self._train_ids),
            "training_history_lengths": dict(sorted(self._history_lengths.items())),
            "work_units_upper_bound": self.work_units_upper_bound,
        }

    @classmethod
    def from_state(cls, value: object) -> Self:
        if not isinstance(value, dict) or set(value) != {
            "format",
            "schema_version",
            "config",
            "cutoff",
            "vocabulary",
            "embeddings",
            "user_bias",
            "histories",
            "train_impression_ids",
            "training_history_lengths",
            "work_units_upper_bound",
        }:
            raise ValueError("invalid neural-news state fields")
        if (
            value["format"] != FORMAT
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
        ):
            raise ValueError("unsupported neural-news state format")
        config = NeuralNewsConfig.from_mapping(value["config"])
        cutoff = _aware(parse_datetime(value["cutoff"], "cutoff"), "cutoff")
        words = value["vocabulary"]
        if (
            not isinstance(words, list)
            or not 1 <= len(words) <= config.max_vocabulary
            or words[0] != "<unk>"
            or any(type(word) is not str or not word or len(word) > 64 for word in words)
            or len(set(words)) != len(words)
        ):
            raise ValueError("invalid neural-news vocabulary")
        raw_embeddings = value["embeddings"]
        raw_bias = value["user_bias"]
        if not isinstance(raw_embeddings, list) or len(raw_embeddings) != len(words):
            raise ValueError("invalid neural-news embedding shape")
        if not isinstance(raw_bias, list) or len(raw_bias) != config.dimension:
            raise ValueError("invalid neural-news user bias")
        embeddings = [_state_vector(row, config.dimension) for row in raw_embeddings]
        bias = _state_vector(raw_bias, config.dimension)
        if any(embeddings[0]):
            raise ValueError("neural-news OOV embedding must be zero")
        raw_ids = value["train_impression_ids"]
        raw_histories = value["histories"]
        raw_lengths = value["training_history_lengths"]
        if (
            not isinstance(raw_ids, list)
            or not 1 <= len(raw_ids) <= MAX_TRAIN_ROWS
            or any(
                type(identity) is not str or not identity or len(identity) > 256
                for identity in raw_ids
            )
            or raw_ids != sorted(set(raw_ids))
            or not isinstance(raw_histories, dict)
            or len(raw_histories) > MAX_TRAIN_ROWS
            or not isinstance(raw_lengths, dict)
            or set(raw_lengths) != set(raw_ids)
        ):
            raise ValueError("invalid neural-news training identity state")
        histories: dict[str, tuple[str, ...]] = {}
        for user, items in raw_histories.items():
            if (
                type(user) is not str
                or not user
                or len(user) > 256
                or not isinstance(items, list)
                or len(items) > config.max_history
                or any(type(item) is not str or not item or len(item) > 256 for item in items)
            ):
                raise ValueError("invalid neural-news history state")
            histories[user] = tuple(items)
        lengths = {
            identity: _bounded_int(length, "training history length", 0, config.max_history)
            for identity, length in raw_lengths.items()
        }
        work = _bounded_int(value["work_units_upper_bound"], "work units", 0, MAX_WORK_UNITS)
        return cls(
            config,
            cutoff,
            tuple(words),
            embeddings,
            bias,
            histories,
            tuple(raw_ids),
            lengths,
            work,
        )


def _state_vector(value: object, dimension: int) -> list[float]:
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError("invalid neural-news vector shape")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("neural-news weight must be finite and bounded")
        try:
            number = float(item)
        except OverflowError as error:
            raise ValueError("neural-news weight must be finite and bounded") from error
        if not math.isfinite(number) or abs(number) > 10.0:
            raise ValueError("neural-news weight must be finite and bounded")
        result.append(number)
    return result


def run_neural_news_experiment(
    articles_source: bytes,
    train_source: bytes,
    validation_source: bytes,
    *,
    cutoff: datetime,
    config: NeuralNewsConfig | None = None,
) -> dict[str, object]:
    """Fit on train only, score complete post-cutoff slates, then evaluate labels."""
    sources = {"articles": articles_source, "train": train_source, "validation": validation_source}
    if any(
        type(source) is not bytes or len(source) > MAX_SOURCE_BYTES for source in sources.values()
    ):
        raise ValueError("neural-news source must be bounded immutable bytes")
    cutoff = _aware(cutoff, "cutoff")
    catalog = _catalog(load_articles_bytes(articles_source))
    train = _rows(
        load_mind_impressions_bytes(train_source), catalog, maximum=MAX_TRAIN_ROWS, label="train"
    )
    validation = _rows(
        load_mind_impressions_bytes(validation_source),
        catalog,
        maximum=MAX_VALIDATION_ROWS,
        label="validation",
    )
    if any(row.occurred_at > cutoff for row in train) or any(
        row.occurred_at <= cutoff for row in validation
    ):
        raise ValueError("train/validation rows cross the declared cutoff")
    if max(row.occurred_at for row in train) >= min(row.occurred_at for row in validation):
        raise ValueError("train and validation are not temporally disjoint")
    if {row.impression_id for row in train} & {row.impression_id for row in validation}:
        raise ValueError("train and validation impression ids overlap")
    model = NeuralNewsRanker.fit(
        catalog.values(),
        train,
        cutoff=cutoff,
        config=config,
        held_out_impression_ids=(row.impression_id for row in validation),
    )
    state = model.to_state()
    scores = NeuralNewsRanker.from_state(state).score_impressions(catalog.values(), validation)
    metrics = evaluate_mind_impressions(validation, scores).to_dict()
    rows = [
        {
            "impression_id": row.impression_id,
            "article_id": candidate.article_id,
            "score": scores[row.impression_id][candidate.article_id],
        }
        for row in validation
        for candidate in row.candidates
    ]
    state_bytes = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result: dict[str, object] = {
        "format": EXPERIMENT_FORMAT,
        "schema_version": 1,
        "cutoff": cutoff.isoformat(),
        "source_sha256": {
            name: hashlib.sha256(source).hexdigest() for name, source in sources.items()
        },
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "model": state,
        "scores": rows,
        "metrics": metrics,
        "work_units_upper_bound": model.work_units_upper_bound,
        "scope": "synthetic/local train-dev only; not NRMS or official MIND reproduction",
    }
    if len(json_text(result).encode()) > MAX_OUTPUT_BYTES:
        raise ValueError("neural-news experiment output exceeds limit")
    return result
