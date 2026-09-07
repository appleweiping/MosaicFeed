"""Command-line workflows for ranking, evaluation, and simulation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from mosaicfeed import __version__
from mosaicfeed.benchmark import run_policy_benchmark, write_benchmark_html
from mosaicfeed.config import FeedConfig
from mosaicfeed.datasets import fixed_offset, load_mind
from mosaicfeed.io import feed_to_dict, load_articles, load_events, parse_datetime, write_json
from mosaicfeed.learning import PointwiseLogisticRanker
from mosaicfeed.metrics import evaluate_leave_last_out
from mosaicfeed.mind import (
    evaluate_mind_impressions,
    load_mind_impressions,
    load_mind_scores,
    write_mind_impressions,
)
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
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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

    benchmark = subcommands.add_parser(
        "benchmark", help="compare MosaicFeed with declared point-in-time baselines"
    )
    benchmark.add_argument("--articles", required=True)
    benchmark.add_argument("--events", required=True)
    benchmark.add_argument("--as-of")
    benchmark.add_argument("--config")
    benchmark.add_argument("--k", type=int)
    benchmark.add_argument("--bootstrap-samples", type=int, default=1_000)
    benchmark.add_argument("--confidence", type=float, default=0.95)
    benchmark.add_argument("--seed", type=int, default=17)
    benchmark.add_argument("--output")
    benchmark.add_argument("--html")

    simulate = subcommands.add_parser("simulate", help="create a reproducible synthetic dataset")
    simulate.add_argument("--directory", required=True)
    simulate.add_argument("--seed", type=int, default=17)
    simulate.add_argument("--users", type=int, default=20)
    simulate.add_argument("--articles", type=int, default=100)
    simulate.add_argument("--events-per-user", type=int, default=15)

    mind = subcommands.add_parser(
        "import-mind", help="convert local MIND TSV files without downloading data"
    )
    mind.add_argument("--news", required=True, help="path to MIND news.tsv")
    mind.add_argument("--behaviors", required=True, help="path to MIND behaviors.tsv")
    mind.add_argument(
        "--catalog-published-at",
        required=True,
        help="declared ISO-8601 availability time because MIND omits publication times",
    )
    mind.add_argument(
        "--behavior-utc-offset",
        required=True,
        type=float,
        help="fixed UTC offset for MIND's timezone-naive behavior timestamps",
    )
    mind.add_argument("--directory", required=True)

    mind_evaluate = subcommands.add_parser(
        "evaluate-mind",
        help="evaluate scores against complete imported MIND candidate sets",
    )
    mind_evaluate.add_argument("--impressions", required=True)
    mind_evaluate.add_argument("--scores", required=True)
    mind_evaluate.add_argument("--cutoff", action="append", type=int, dest="cutoffs")
    mind_evaluate.add_argument("--output")

    train_click = subcommands.add_parser(
        "train-click-model",
        help="fit a leakage-safe pointwise logistic click model",
    )
    train_click.add_argument("--articles", required=True)
    train_click.add_argument("--events", required=True)
    train_click.add_argument("--as-of", required=True)
    train_click.add_argument("--config")
    train_click.add_argument("--output", required=True)
    train_click.add_argument("--epochs", type=int, default=20)
    train_click.add_argument("--learning-rate", type=float, default=0.05)
    train_click.add_argument("--l2", type=float, default=0.001)
    train_click.add_argument("--seed", type=int, default=17)

    rank_click = subcommands.add_parser(
        "rank-click-model",
        help="rank a point-in-time catalog with a saved click model",
    )
    rank_click.add_argument("--model", required=True)
    rank_click.add_argument("--articles", required=True)
    rank_click.add_argument("--events", required=True)
    rank_click.add_argument("--user", required=True)
    rank_click.add_argument("--as-of", required=True)
    rank_click.add_argument("--k", type=int, default=10)
    rank_click.add_argument("--output")
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
    if args.command == "benchmark":
        benchmark_report = run_policy_benchmark(
            load_articles(args.articles),
            load_events(args.events),
            as_of=_clock(args.as_of),
            config=_config(args.config),
            k=args.k,
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed,
        )
        benchmark_payload = benchmark_report.to_dict()
        if args.output:
            write_json(args.output, benchmark_payload)
        else:
            print(json.dumps(benchmark_payload, indent=2, sort_keys=True))
        if args.html:
            write_benchmark_html(args.html, benchmark_report)
        return 0
    if args.command == "simulate":
        synthetic = generate_synthetic(
            seed=args.seed,
            users=args.users,
            articles=args.articles,
            events_per_user=args.events_per_user,
        )
        directory = Path(args.directory)
        directory.mkdir(parents=True, exist_ok=True)
        article_records = [_article_record(article) for article in synthetic.articles]
        write_json(directory / "articles.json", article_records)
        write_json(
            directory / "events.json",
            [_event_record(event) for event in synthetic.events],
        )
        write_json(
            directory / "metadata.json",
            {
                "as_of": synthetic.as_of.isoformat(),
                "seed": args.seed,
                "users": list(synthetic.user_ids),
            },
        )
        print(
            f"wrote {len(synthetic.articles)} articles and "
            f"{len(synthetic.events)} events to {directory}"
        )
        return 0
    if args.command == "import-mind":
        catalog_published_at = parse_datetime(args.catalog_published_at, "catalog_published_at")
        behavior_timezone = fixed_offset(args.behavior_utc_offset)
        behavior_offset_hours = float(args.behavior_utc_offset)
        if behavior_offset_hours == 0.0:
            behavior_offset_hours = 0.0
        mind_dataset = load_mind(
            args.news,
            args.behaviors,
            catalog_published_at=catalog_published_at,
            behavior_timezone=behavior_timezone,
        )
        directory = Path(args.directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_json(
            directory / "articles.json",
            [_article_record(article) for article in mind_dataset.articles],
        )
        write_json(
            directory / "events.json",
            [_event_record(event) for event in mind_dataset.events],
        )
        write_mind_impressions(directory / "impressions.json", mind_dataset.impression_records)
        write_json(
            directory / "metadata.json",
            {
                "adapter": "mind-tsv-v2",
                "articles": len(mind_dataset.articles),
                "click_events": len(mind_dataset.events),
                "impressions": mind_dataset.impressions,
                "ignored_history_items": mind_dataset.ignored_history_items,
                "news_sha256": mind_dataset.news_sha256,
                "behaviors_sha256": mind_dataset.behaviors_sha256,
                "catalog_published_at": catalog_published_at.isoformat(),
                "behavior_utc_offset_hours": behavior_offset_hours,
                "limitations": [
                    "one caller-declared catalog availability time is used",
                    "news category is used as a source proxy",
                    "undated history is counted but is not converted into preference events",
                ],
            },
        )
        print(
            f"converted {len(mind_dataset.articles)} articles and "
            f"{len(mind_dataset.events)} clicks to {directory}"
        )
        return 0
    if args.command == "evaluate-mind":
        mind_report = evaluate_mind_impressions(
            load_mind_impressions(args.impressions),
            load_mind_scores(args.scores),
            cutoffs=(5, 10) if args.cutoffs is None else tuple(args.cutoffs),
        )
        payload = mind_report.to_dict()
        if args.output:
            write_json(args.output, payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "train-click-model":
        model = PointwiseLogisticRanker(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            seed=args.seed,
        ).fit(
            load_articles(args.articles),
            load_events(args.events),
            as_of=_clock(args.as_of),
            config=_config(args.config),
        )
        model.save(args.output)
        print(
            json.dumps(
                {
                    "model": str(args.output),
                    "training_examples": model.training_examples,
                    "training_sha256": model.training_sha256,
                    "weights": model.weights,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "rank-click-model":
        model = PointwiseLogisticRanker.load(args.model)
        predictions = model.rank_for_user(
            args.user,
            load_articles(args.articles),
            load_events(args.events),
            as_of=_clock(args.as_of),
            k=args.k,
        )
        payload = {
            "user_id": args.user,
            "as_of": _clock(args.as_of).isoformat(),
            "predictions": [prediction.to_dict() for prediction in predictions],
        }
        if args.output:
            write_json(args.output, payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable command")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(argv)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
