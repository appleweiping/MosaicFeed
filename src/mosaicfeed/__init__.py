"""Public package surface for MosaicFeed."""

from mosaicfeed.config import FeedConfig
from mosaicfeed.models import Article, Event, EventKind, Feed, Recommendation, UserProfile
from mosaicfeed.pipeline import build_feed

__all__ = [
    "Article",
    "Event",
    "EventKind",
    "Feed",
    "FeedConfig",
    "Recommendation",
    "UserProfile",
    "build_feed",
]

__version__ = "0.1.0"
