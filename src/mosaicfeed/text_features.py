"""Versioned, bounded TF-IDF news vectors for opt-in cold-start ranking."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from types import MappingProxyType

from mosaicfeed.io import parse_datetime
from mosaicfeed.models import Article

TEXT_FORMAT = "mosaicfeed.news-tfidf"
TEXT_SCHEMA_VERSION = 1
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_HEX = frozenset("0123456789abcdef")


def _bounded_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _valid_feature_name(value: object, maximum_token_characters: int) -> bool:
    if not isinstance(value, str):
        return False
    prefix, separator, token = value.partition(":")
    return (
        separator == ":"
        and prefix in {"title", "category", "subcategory"}
        and 0 < len(token) <= maximum_token_characters
        and token == token.casefold()
        and _TOKEN.fullmatch(token) is not None
    )


@dataclass(frozen=True, slots=True)
class TextFeatureConfig:
    max_articles: int = 100_000
    max_events: int = 1_000_000
    max_article_id_characters: int = 256
    max_title_characters: int = 4_096
    max_category_characters: int = 256
    max_title_tokens: int = 256
    max_vocabulary: int = 8_192
    min_document_frequency: int = 1
    max_feature_occurrences: int = 1_000_000

    def __post_init__(self) -> None:
        limits = {
            "max_articles": (1, 100_000),
            "max_events": (1, 1_000_000),
            "max_article_id_characters": (1, 256),
            "max_title_characters": (1, 4_096),
            "max_category_characters": (1, 256),
            "max_title_tokens": (1, 256),
            "max_vocabulary": (1, 8_192),
            "min_document_frequency": (1, 100_000),
            "max_feature_occurrences": (1, 1_000_000),
        }
        for name, (minimum, maximum) in limits.items():
            object.__setattr__(
                self, name, _bounded_int(getattr(self, name), name, minimum, maximum)
            )
        if self.min_document_frequency > self.max_articles:
            raise ValueError("min_document_frequency exceeds max_articles")

    @classmethod
    def from_state(cls, value: object) -> TextFeatureConfig:
        if not isinstance(value, dict) or set(value) != set(asdict(cls())):
            raise ValueError("text feature configuration is malformed")
        return cls(**value)


def _raw(article: Article, config: TextFeatureConfig) -> dict[str, float]:
    if not isinstance(article, Article):
        raise ValueError("text features require Article values")
    if len(article.id) > config.max_article_id_characters:
        raise ValueError("news id exceeds text feature character limit")
    if len(article.title) > config.max_title_characters:
        raise ValueError("news title exceeds text feature character limit")
    try:
        article.id.encode("utf-8", "strict")
        article.title.encode("utf-8", "strict")
        for topic in article.topics:
            topic.encode("utf-8", "strict")
        if article.mind_category is not None and article.mind_subcategory is not None:
            article.mind_category.encode("utf-8", "strict")
            article.mind_subcategory.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise ValueError("news text must be UTF-8 encodable") from error
    if len(article.topics) > 16 or any(
        len(topic) > config.max_category_characters for topic in article.topics
    ):
        raise ValueError("news categories exceed text feature limits")
    values: dict[str, float] = {}
    if not article.title_missing:
        counts: Counter[str] = Counter()
        for index, match in enumerate(_TOKEN.finditer(article.title.casefold()), start=1):
            if index > config.max_title_tokens:
                raise ValueError("news title exceeds text feature token limit")
            token = match.group()
            if len(token) <= config.max_category_characters:
                counts[token] += 1
        values.update({f"title:{name}": 1.0 + math.log(count) for name, count in counts.items()})
    if article.mind_category is not None and article.mind_subcategory is not None:
        topic_fields: tuple[tuple[str, str], ...] = (
            ("category", article.mind_category),
            ("subcategory", article.mind_subcategory),
        )
    else:
        topic_fields = tuple(
            ("category" if index == 0 else "subcategory", topic)
            for index, topic in enumerate(article.topics)
            if not (
                (index == 0 and article.category_missing)
                or (index > 0 and article.subcategory_missing)
            )
        )
    for prefix, topic in topic_fields:
        if not topic.strip():
            continue
        if len(topic) > config.max_category_characters:
            raise ValueError("news categories exceed text feature limits")
        for match in _TOKEN.finditer(topic.casefold()):
            if len(match.group()) > config.max_category_characters:
                raise ValueError("news category token exceeds text feature limit")
            values[f"{prefix}:{match.group()}"] = 1.0
    return values


@dataclass(frozen=True, slots=True)
class NewsTextEncoder:
    """An immutable training-only vocabulary; unseen news project without refitting."""

    config: TextFeatureConfig
    trained_as_of: datetime
    training_articles_sha256: str
    vocabulary: tuple[str, ...]
    inverse_document_frequency: tuple[float, ...]
    source_kind: str = "direct-corpus"
    _idf_by_name: Mapping[str, float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, TextFeatureConfig):
            raise ValueError("text config must be TextFeatureConfig")
        if not isinstance(self.source_kind, str) or self.source_kind not in {
            "direct-corpus",
            "first-event-seed",
            "declared-training-snapshot",
        }:
            raise ValueError("text vocabulary source is malformed")
        object.__setattr__(self, "config", TextFeatureConfig(**asdict(self.config)))
        _aware(self.trained_as_of, "text trained_as_of")
        digest = self.training_articles_sha256
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in _HEX for c in digest):
            raise ValueError("text training digest is malformed")
        names = self.vocabulary
        if (
            not isinstance(names, tuple)
            or not names
            or len(names) > self.config.max_vocabulary
            or any(
                not _valid_feature_name(name, self.config.max_category_characters) for name in names
            )
            or names != tuple(sorted(set(names)))
        ):
            raise ValueError("text vocabulary is malformed")
        if not isinstance(self.inverse_document_frequency, tuple) or len(
            self.inverse_document_frequency
        ) != len(names):
            raise ValueError("text IDF is malformed")
        maximum = math.log(self.config.max_articles + 1) + 1.0 + 1e-12
        for value in self.inverse_document_frequency:
            try:
                valid = (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                    and 1.0 <= value <= maximum
                )
            except (OverflowError, ValueError):
                valid = False
            if not valid:
                raise ValueError("text IDF is malformed")
        object.__setattr__(self, "vocabulary", tuple(names))
        object.__setattr__(
            self, "inverse_document_frequency", tuple(self.inverse_document_frequency)
        )
        object.__setattr__(
            self,
            "_idf_by_name",
            MappingProxyType(dict(zip(names, self.inverse_document_frequency, strict=True))),
        )

    @classmethod
    def fit(
        cls,
        articles: Iterable[Article],
        *,
        as_of: datetime,
        config: TextFeatureConfig | None = None,
        source_kind: str = "direct-corpus",
    ) -> NewsTextEncoder:
        cutoff = _aware(as_of, "text as_of")
        settings = TextFeatureConfig() if config is None else TextFeatureConfig(**asdict(config))
        values: list[Article] = []
        seen: set[str] = set()
        for article in articles:
            if len(values) >= settings.max_articles:
                raise ValueError("text training exceeds max_articles")
            if not isinstance(article, Article) or article.id in seen:
                raise ValueError("text training requires unique Article ids")
            if article.published_at > cutoff:
                raise ValueError("text training article is not yet published")
            seen.add(article.id)
            values.append(article)
        if not values:
            raise ValueError("text training requires at least one visible article")
        ordered = sorted(values, key=lambda article: article.id)
        frequencies: Counter[str] = Counter()
        occurrences = 0
        for article in ordered:
            document = _raw(article, settings)
            occurrences += len(document)
            if occurrences > settings.max_feature_occurrences:
                raise ValueError("text training exceeds feature occurrence limit")
            frequencies.update(document.keys())
        eligible = [
            name
            for name, frequency in frequencies.items()
            if frequency >= settings.min_document_frequency
        ]
        eligible.sort(key=lambda name: (-frequencies[name], name))
        vocabulary = tuple(sorted(eligible[: settings.max_vocabulary]))
        if not vocabulary:
            raise ValueError("text training selected no vocabulary")
        idf = tuple(
            math.log((1.0 + len(ordered)) / (1.0 + frequencies[name])) + 1.0 for name in vocabulary
        )
        canonical = json.dumps(
            [
                [
                    article.id,
                    article.title if not article.title_missing else "",
                    list(article.topics),
                    article.category_missing,
                    article.subcategory_missing,
                    article.mind_category,
                    article.mind_subcategory,
                    article.published_at.isoformat(),
                ]
                for article in ordered
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            settings, cutoff, hashlib.sha256(canonical).hexdigest(), vocabulary, idf, source_kind
        )

    def vectorize(self, article: Article) -> dict[str, float]:
        """Return a sparse unit vector; OOV-only news maps to the zero vector."""

        raw = _raw(article, self.config)
        weighted = {
            name: value * idf
            for name, value in raw.items()
            if (idf := self._idf_by_name.get(name)) is not None
        }
        norm = math.sqrt(math.fsum(value * value for value in weighted.values()))
        if not math.isfinite(norm):
            raise ValueError("news vector norm is not finite")
        return {name: value / norm for name, value in weighted.items()} if norm else {}

    def affinity(self, article: Article, history_vector: Mapping[str, float]) -> float:
        if not isinstance(history_vector, Mapping) or len(history_vector) > len(self.vocabulary):
            raise ValueError("text history vector exceeds vocabulary bounds")
        try:
            if any(
                name not in self._idf_by_name
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for name, value in history_vector.items()
            ):
                raise ValueError("text history vector is invalid")
        except (OverflowError, TypeError) as error:
            raise ValueError("text history vector is invalid") from error
        candidate = self.vectorize(article)
        try:
            norm = math.sqrt(math.fsum(value * value for value in history_vector.values()))
        except (OverflowError, ValueError) as error:
            raise ValueError("text history norm is not finite") from error
        if not math.isfinite(norm):
            raise ValueError("text history norm is not finite")
        if not norm or not candidate:
            return 0.0
        similarity = (
            math.fsum(value * history_vector.get(name, 0.0) for name, value in candidate.items())
            / norm
        )
        return min(1.0, max(-1.0, similarity))

    def to_state(self) -> dict[str, object]:
        return {
            "format": TEXT_FORMAT,
            "schema_version": TEXT_SCHEMA_VERSION,
            "config": asdict(self.config),
            "trained_as_of": self.trained_as_of.isoformat(),
            "training_articles_sha256": self.training_articles_sha256,
            "vocabulary": list(self.vocabulary),
            "inverse_document_frequency": list(self.inverse_document_frequency),
            "source_kind": self.source_kind,
        }

    @classmethod
    def from_state(cls, state: object) -> NewsTextEncoder:
        if not isinstance(state, dict) or set(state) != {
            "format",
            "schema_version",
            "config",
            "trained_as_of",
            "training_articles_sha256",
            "vocabulary",
            "inverse_document_frequency",
            "source_kind",
        }:
            raise ValueError("text encoder state is malformed")
        if (
            state["format"] != TEXT_FORMAT
            or type(state["schema_version"]) is not int
            or state["schema_version"] != TEXT_SCHEMA_VERSION
        ):
            raise ValueError("unsupported text feature schema")
        names = state["vocabulary"]
        idf = state["inverse_document_frequency"]
        if not isinstance(names, list) or not isinstance(idf, list):
            raise ValueError("text vocabulary and IDF must be arrays")
        return cls(
            TextFeatureConfig.from_state(state["config"]),
            parse_datetime(state["trained_as_of"], "trained_as_of"),
            state["training_articles_sha256"],
            tuple(names),
            tuple(idf),
            state["source_kind"],
        )
