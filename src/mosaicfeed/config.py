"""Configuration for profiling, ranking, and reranking."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from mosaicfeed.io import load_json_text


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


@dataclass(frozen=True, slots=True)
class FeedConfig:
    size: int = 10
    interest_weight: float = 0.45
    freshness_weight: float = 0.20
    quality_weight: float = 0.15
    novelty_weight: float = 0.10
    popularity_weight: float = 0.10
    exploration_weight: float = 0.02
    article_half_life_hours: float = 72.0
    profile_half_life_days: float = 30.0
    mmr_lambda: float = 0.78
    max_per_source: int = 2
    minimum_score: float = 0.0
    exclude_seen: bool = True
    view_signal: float = 0.2
    click_signal: float = 0.7
    like_signal: float = 1.0
    hide_signal: float = -1.2

    def __post_init__(self) -> None:
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 1:
            raise ValueError("size must be positive")
        if (
            isinstance(self.max_per_source, bool)
            or not isinstance(self.max_per_source, int)
            or self.max_per_source < 1
        ):
            raise ValueError("max_per_source must be positive")
        numeric_values = {
            **self.ranking_weights(),
            "article_half_life_hours": self.article_half_life_hours,
            "profile_half_life_days": self.profile_half_life_days,
            "mmr_lambda": self.mmr_lambda,
            "minimum_score": self.minimum_score,
            "view_signal": self.view_signal,
            "click_signal": self.click_signal,
            "like_signal": self.like_signal,
            "hide_signal": self.hide_signal,
        }
        if any(not _is_finite_number(value) for value in numeric_values.values()):
            raise ValueError("numeric configuration values must be finite numbers")
        if not isinstance(self.exclude_seen, bool):
            raise ValueError("exclude_seen must be a boolean")
        if self.article_half_life_hours <= 0.0 or self.profile_half_life_days <= 0.0:
            raise ValueError("half-lives must be positive")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between 0 and 1")
        if not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError("minimum_score must be between 0 and 1")
        weights = self.ranking_weights()
        if any(value < 0.0 for value in weights.values()):
            raise ValueError("ranking weights must not be negative")
        try:
            weight_total = math.fsum(weights.values())
        except OverflowError as error:
            raise ValueError("sum of ranking weights must be finite") from error
        if not math.isfinite(weight_total):
            raise ValueError("sum of ranking weights must be finite")
        if weight_total <= 0.0:
            raise ValueError("at least one ranking weight must be positive")

    def ranking_weights(self) -> dict[str, float]:
        return {
            "interest": self.interest_weight,
            "freshness": self.freshness_weight,
            "quality": self.quality_weight,
            "novelty": self.novelty_weight,
            "popularity": self.popularity_weight,
            "exploration": self.exploration_weight,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> FeedConfig:
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown config fields: {', '.join(sorted(unknown))}")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError(f"invalid configuration value: {error}") from error

    @classmethod
    def from_json(cls, path: str | Path) -> FeedConfig:
        parsed = load_json_text(Path(path).read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("configuration must be a JSON object")
        return cls.from_mapping(cast(dict[str, Any], parsed))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
