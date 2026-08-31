"""Command-line workflows for ranking, evaluation, and simulation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from mosaicfeed.config import FeedConfig
from mosaicfeed.io import feed_to_dict, load_articles, load_events, parse_datetime, write_json
from mosaicfeed.metrics import evaluate_leave_last_out
from mosaicfeed.pipeline import build_feed
from mosaicfeed.report import render_feed_report
from mosaicfeed.simulation import generate_synthetic


def _clock(value: str | None) -> datetime:
    return datetime.now(UTC) if value is None else parse_datetime(value, "as_of")


def _config(path: str | None) -> FeedConfig:
    return FeedConfig() if path is None else FeedConfig.from_json(path)


def _article_record(article: object) -> dict[str, object]:
    from mosaicfeed.models import Article

    if not isinstance(article, Article):
        raise TypeError("article must be an Article")
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


def _event_record(event: object) -> dict[str, object]:
    from mosaicfeed.models import Event

    if not isinstance(event, Event):
        raise TypeError("event must be an Event")
    result: dict[str, object] = {
        "user_id": event.user_id,
        "article_id": event.article_id,
        "kind": event.kind.value,
        "occurred_at": event.occurred_at.isoformat(),
    }
    if event.propensity is not None:
        result["propensity"] = event.propensity
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mosaicfeed", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate", help="strictly validate catalog and event files")
    validate.add_argument("--articles", required=True)
    validate.add_argument("--events", required=True)

    recommend = subcommands.add_parser("recommend", help="generate one explainable feed")
    recommend.add_argument("--articles", required=True)
    recommend.add_argument("--events", required=True)
    recommend.add_argument("--user", required=True)
    recommend.add_argument("--as-of")
    recommend.add_argument("--config")
    recommend.add_argument("--output")
    recommend.add_argument("--html")

    evaluate = subcommands.add_parser("evaluate", help="run temporal leave-last-out evaluation")
    evaluate.add_argument("--articles", required=True)
    evaluate.add_argument("--events", required=True)
    evaluate.add_argument("--as-of")
    evaluate.add_argument("--config")
    evaluate.add_argument("--k", type=int)
    evaluate.add_argument("--output")

    simulate = subcommands.add_parser("simulate", help="create a reproducible synthetic dataset")
    simulate.add_argument("--directory", required=True)
    simulate.add_argument("--seed", type=int, default=17)
    simulate.add_argument("--users", type=int, default=20)
    simulate.add_argument("--articles", type=int, default=100)
    simulate.add_argument("--events-per-user", type=int, default=15)
    return parser


def _run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        articles = load_articles(args.articles)
        events = load_events(args.events)
        article_map = {article.id: article for article in articles}
        unknown = sorted({event.article_id for event in events} - set(article_map))
        if unknown:
            raise ValueError(f"events reference unknown articles: {', '.join(unknown)}")
        print(json.dumps({"articles": len(articles), "events": len(events), "valid": True}))
        return 0
    if args.command == "recommend":
        articles = load_articles(args.articles)
        events = load_events(args.events)
        feed = build_feed(
            args.user,
            articles,
            events,
            as_of=_clock(args.as_of),
            config=_config(args.config),
        )
        payload = feed_to_dict(feed)
        if args.output:
            write_json(args.output, payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        if args.html:
            article_map = {article.id: article for article in articles}
            render_feed_report(feed, article_map, output=args.html)
        return 0
    if args.command == "evaluate":
        report = evaluate_leave_last_out(
            load_articles(args.articles),
            load_events(args.events),
            as_of=_clock(args.as_of),
            config=_config(args.config),
            k=args.k,
        )
        evaluation_payload = report.to_dict()
        if args.output:
            write_json(args.output, evaluation_payload)
        else:
            print(json.dumps(evaluation_payload, indent=2, sort_keys=True))
        return 0
    if args.command == "simulate":
        dataset = generate_synthetic(
            seed=args.seed,
            users=args.users,
            articles=args.articles,
            events_per_user=args.events_per_user,
        )
        directory = Path(args.directory)
        directory.mkdir(parents=True, exist_ok=True)
        article_records = [_article_record(article) for article in dataset.articles]
        write_json(directory / "articles.json", article_records)
        write_json(directory / "events.json", [_event_record(event) for event in dataset.events])
        write_json(
            directory / "metadata.json",
            {
                "as_of": dataset.as_of.isoformat(),
                "seed": args.seed,
                "users": list(dataset.user_ids),
            },
        )
        print(
            f"wrote {len(dataset.articles)} articles and "
            f"{len(dataset.events)} events to {directory}"
        )
        return 0
    raise AssertionError("unreachable command")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(argv)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
