"""Command-line workflows for ranking, evaluation, and simulation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from mosaicfeed import __version__
from mosaicfeed.benchmark import render_benchmark_html, run_policy_benchmark
from mosaicfeed.config import FeedConfig
from mosaicfeed.datasets import fixed_offset, load_mind
from mosaicfeed.event_stream import (
    EventStreamLimits,
    ProfileEventStore,
    load_event_store,
    load_interaction_events,
    profile_to_dict,
)
from mosaicfeed.io import (
    atomic_write_texts,
    feed_to_dict,
    json_text,
    load_articles,
    load_events,
    load_json_text,
    parse_datetime,
    write_json,
)
from mosaicfeed.learning import PointwiseLogisticRanker
from mosaicfeed.metrics import evaluate_leave_last_out
from mosaicfeed.mind import (
    evaluate_mind_impressions,
    impression_to_dict,
    load_mind_impressions,
    load_mind_scores,
)
from mosaicfeed.pipeline import build_feed
from mosaicfeed.report import render_feed_html
from mosaicfeed.server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    ServingLimits,
    create_rank_server,
    load_click_rank_service,
    resolve_bearer_token,
    serve_rank_server,
)
from mosaicfeed.simulation import generate_synthetic
from mosaicfeed.text_features import TextFeatureConfig


def _clock(value: str | None) -> datetime:
    return datetime.now(UTC) if value is None else parse_datetime(value, "as_of")


def _config(path: str | None, *, maximum_bytes: int | None = None) -> FeedConfig:
    if path is None:
        return FeedConfig()
    if maximum_bytes is None:
        return FeedConfig.from_json(path)
    with Path(path).open("rb") as source:
        data = source.read(maximum_bytes + 1)
    if len(data) > maximum_bytes:
        raise ValueError("configuration exceeds max_config_bytes")
    try:
        payload = load_json_text(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("configuration must be strict UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("configuration must be a JSON object")
    return FeedConfig.from_mapping(cast(dict[str, object], payload))


def _article_record(article: object) -> dict[str, object]:
    from mosaicfeed.models import Article

    if not isinstance(article, Article):
        raise TypeError("article must be an Article")
    result = {
        "id": article.id,
        "title": "" if article.title_missing else article.title,
        "summary": article.summary,
        "topics": list(article.topics),
        "source": article.source,
        "published_at": article.published_at.isoformat(),
        "quality": article.quality,
        "popularity": article.popularity,
    }
    if article.title_missing:
        result["title_missing"] = True
    if article.category_missing:
        result["category_missing"] = True
    if article.subcategory_missing:
        result["subcategory_missing"] = True
    if article.mind_category is not None:
        result["mind_category"] = article.mind_category
        result["mind_subcategory"] = article.mind_subcategory
    return result


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
    if event.weight != 1.0:
        result["weight"] = event.weight
    return result


def _add_event_stream_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--articles", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--config")
    parser.add_argument("--allowed-lateness-seconds", type=float, default=300.0)
    parser.add_argument(
        "--late-event-policy",
        choices=("reject", "accept-rebuild"),
        default="reject",
    )
    parser.add_argument(
        "--recover-torn-tail",
        action="store_true",
        help="discard an incomplete final log record after validating every complete record",
    )
    parser.add_argument("--max-event-bytes", type=int, default=16 * 1024)
    parser.add_argument("--max-input-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-batch-events", type=int, default=10_000)
    parser.add_argument("--max-log-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--max-checkpoint-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--max-state-bytes", type=int, default=192 * 1024 * 1024)
    parser.add_argument("--max-events", type=int, default=1_000_000)
    parser.add_argument("--max-users", type=int, default=100_000)
    parser.add_argument("--max-events-per-user", type=int, default=100_000)
    parser.add_argument("--max-catalog-articles", type=int, default=100_000)
    parser.add_argument("--max-catalog-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-config-bytes", type=int, default=64 * 1024)
    parser.add_argument("--max-topics-per-article", type=int, default=1_024)
    parser.add_argument("--max-catalog-topic-cells", type=int, default=2_000_000)
    parser.add_argument("--max-profile-topic-cells", type=int, default=2_000_000)
    parser.add_argument("--max-identifier-chars", type=int, default=256)
    parser.add_argument("--max-weight", type=float, default=100.0)


def _event_stream_limits(args: argparse.Namespace) -> EventStreamLimits:
    return EventStreamLimits(
        max_event_bytes=args.max_event_bytes,
        max_input_bytes=args.max_input_bytes,
        max_batch_events=args.max_batch_events,
        max_log_bytes=args.max_log_bytes,
        max_checkpoint_bytes=args.max_checkpoint_bytes,
        max_state_bytes=args.max_state_bytes,
        max_events=args.max_events,
        max_users=args.max_users,
        max_events_per_user=args.max_events_per_user,
        max_catalog_articles=args.max_catalog_articles,
        max_catalog_bytes=args.max_catalog_bytes,
        max_config_bytes=args.max_config_bytes,
        max_topics_per_article=args.max_topics_per_article,
        max_catalog_topic_cells=args.max_catalog_topic_cells,
        max_profile_topic_cells=args.max_profile_topic_cells,
        max_identifier_chars=args.max_identifier_chars,
        max_weight=args.max_weight,
    )


def _open_event_store(
    args: argparse.Namespace,
    *,
    checkpoint: str | Path | None,
) -> ProfileEventStore:
    limits = _event_stream_limits(args)
    return load_event_store(
        args.log,
        args.articles,
        config=_config(args.config, maximum_bytes=limits.max_config_bytes),
        limits=limits,
        allowed_lateness_seconds=args.allowed_lateness_seconds,
        late_event_policy=args.late_event_policy,
        checkpoint_path=checkpoint,
        recover_torn_tail=args.recover_torn_tail,
    )


def _emit_json(value: object, output: str | None) -> None:
    if output:
        write_json(output, value)
    else:
        print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _require_distinct_paths(
    values: dict[str, str | None],
    *,
    allowed_equal: frozenset[frozenset[str]] = frozenset(),
) -> None:
    paths = {name: Path(value) for name, value in values.items() if value is not None}
    resolved = {name: path.resolve() for name, path in paths.items()}
    names = tuple(paths)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            pair = frozenset({left, right})
            if pair in allowed_equal:
                continue
            same_file = False
            with suppress(OSError):
                same_file = (
                    paths[left].exists()
                    and paths[right].exists()
                    and os.path.samefile(paths[left], paths[right])
                )
            if resolved[left] == resolved[right] or same_file:
                raise ValueError(f"--{left} and --{right} must refer to different paths")


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
    train_click.add_argument(
        "--text-features", action="store_true", help="fit a training-only news TF-IDF vocabulary"
    )
    train_click.add_argument("--text-max-vocabulary", type=int, default=8_192)
    train_click.add_argument("--text-min-document-frequency", type=int, default=1)
    train_click.add_argument(
        "--text-vocabulary-articles",
        help="declared training-news JSON snapshot visible before the first training event",
    )

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

    serve_click = subcommands.add_parser(
        "serve-click-model",
        help="serve a bounded local HTTP API from a saved click model",
    )
    serve_click.add_argument("--model", required=True)
    serve_click.add_argument("--articles", required=True)
    serve_click.add_argument("--events", required=True)
    serve_click.add_argument("--host", default=DEFAULT_HOST)
    serve_click.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve_click.add_argument(
        "--token-env",
        help="name of the environment variable containing the optional bearer token",
    )
    serve_click.add_argument(
        "--allow-nonloopback",
        action="store_true",
        help="explicitly allow a non-loopback host (also requires --token-env)",
    )
    serve_click.add_argument("--max-request-bytes", type=int, default=64 * 1024)
    serve_click.add_argument("--max-response-bytes", type=int, default=2 * 1024 * 1024)
    serve_click.add_argument("--max-k", type=int, default=100)
    serve_click.add_argument("--max-candidates", type=int, default=10_000)
    serve_click.add_argument("--max-catalog-articles", type=int, default=100_000)
    serve_click.add_argument("--max-history-events", type=int, default=1_000_000)
    serve_click.add_argument("--max-model-bytes", type=int, default=4 * 1024 * 1024)
    serve_click.add_argument("--max-catalog-bytes", type=int, default=64 * 1024 * 1024)
    serve_click.add_argument("--max-history-bytes", type=int, default=128 * 1024 * 1024)
    serve_click.add_argument("--max-topics-per-article", type=int, default=1_024)
    serve_click.add_argument("--max-catalog-topic-cells", type=int, default=2_000_000)
    serve_click.add_argument("--max-concurrency", type=int, default=16)
    serve_click.add_argument("--request-timeout-seconds", type=float, default=10.0)
    serve_click.add_argument("--max-identifier-chars", type=int, default=256)
    serve_click.add_argument("--max-event-weight", type=float, default=100.0)

    stream_ingest = subcommands.add_parser(
        "stream-ingest",
        help="validate and append one versioned interaction JSONL batch",
    )
    _add_event_stream_options(stream_ingest)
    stream_ingest.add_argument("--input", required=True)
    stream_ingest.add_argument(
        "--checkpoint",
        help="restore this checkpoint when present, then atomically refresh it",
    )
    stream_ingest.add_argument("--output")

    stream_checkpoint = subcommands.add_parser(
        "stream-checkpoint",
        help="validate/replay a log and atomically write a checksummed checkpoint",
    )
    _add_event_stream_options(stream_checkpoint)
    stream_checkpoint.add_argument("--checkpoint", help="optional existing checkpoint to resume")
    stream_checkpoint.add_argument("--output", required=True, help="destination checkpoint")

    stream_replay = subcommands.add_parser(
        "stream-replay",
        help="replay a consistent log prefix into point-in-time user profiles",
    )
    _add_event_stream_options(stream_replay)
    stream_replay.add_argument("--checkpoint", help="optional checkpoint to resume")
    stream_replay.add_argument("--as-of", required=True)
    stream_replay.add_argument("--user", action="append", dest="users")
    stream_replay.add_argument("--output")
    stream_replay.add_argument(
        "--events-output",
        help="write a frozen legacy history snapshot for training or HTTP serving",
    )
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
        _require_distinct_paths(
            {
                "articles": args.articles,
                "config": args.config,
                "events": args.events,
                "html": args.html,
                "output": args.output,
            }
        )
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
        outputs: dict[Path, str] = {}
        if args.output:
            outputs[Path(args.output)] = json_text(payload)
        if args.html:
            article_map = {article.id: article for article in articles}
            outputs[Path(args.html)] = render_feed_html(feed, article_map)
        atomic_write_texts(outputs)
        if not args.output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "evaluate":
        _require_distinct_paths(
            {
                "articles": args.articles,
                "config": args.config,
                "events": args.events,
                "output": args.output,
            }
        )
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
        _require_distinct_paths(
            {
                "articles": args.articles,
                "config": args.config,
                "events": args.events,
                "html": args.html,
                "output": args.output,
            }
        )
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
        outputs = {}
        if args.output:
            outputs[Path(args.output)] = json_text(benchmark_payload)
        if args.html:
            outputs[Path(args.html)] = render_benchmark_html(benchmark_report)
        atomic_write_texts(outputs)
        if not args.output:
            print(json.dumps(benchmark_payload, indent=2, sort_keys=True))
        return 0
    if args.command == "simulate":
        directory = Path(args.directory)
        _require_distinct_paths(
            {
                "articles-output": str(directory / "articles.json"),
                "events-output": str(directory / "events.json"),
                "metadata-output": str(directory / "metadata.json"),
            }
        )
        synthetic = generate_synthetic(
            seed=args.seed,
            users=args.users,
            articles=args.articles,
            events_per_user=args.events_per_user,
        )
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_texts(
            {
                directory / "articles.json": json_text(
                    [_article_record(article) for article in synthetic.articles]
                ),
                directory / "events.json": json_text(
                    [_event_record(event) for event in synthetic.events]
                ),
                directory / "metadata.json": json_text(
                    {
                        "as_of": synthetic.as_of.isoformat(),
                        "seed": args.seed,
                        "users": list(synthetic.user_ids),
                    }
                ),
            }
        )
        print(
            f"wrote {len(synthetic.articles)} articles and "
            f"{len(synthetic.events)} events to {directory}"
        )
        return 0
    if args.command == "import-mind":
        directory = Path(args.directory)
        _require_distinct_paths(
            {
                "articles-output": str(directory / "articles.json"),
                "behaviors": args.behaviors,
                "events-output": str(directory / "events.json"),
                "impressions-output": str(directory / "impressions.json"),
                "metadata-output": str(directory / "metadata.json"),
                "news": args.news,
            }
        )
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
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_texts(
            {
                directory / "articles.json": json_text(
                    [_article_record(article) for article in mind_dataset.articles]
                ),
                directory / "events.json": json_text(
                    [_event_record(event) for event in mind_dataset.events]
                ),
                directory / "impressions.json": json_text(
                    [impression_to_dict(item) for item in mind_dataset.impression_records]
                ),
                directory / "metadata.json": json_text(
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
                            "undated history is counted but is not converted into "
                            "preference events",
                        ],
                    }
                ),
            }
        )
        print(
            f"converted {len(mind_dataset.articles)} articles and "
            f"{len(mind_dataset.events)} clicks to {directory}"
        )
        return 0
    if args.command == "evaluate-mind":
        _require_distinct_paths(
            {
                "impressions": args.impressions,
                "output": args.output,
                "scores": args.scores,
            }
        )
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
        _require_distinct_paths(
            {
                "articles": args.articles,
                "config": args.config,
                "events": args.events,
                "text-vocabulary-articles": args.text_vocabulary_articles,
                "output": args.output,
            }
        )
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
            text_features=args.text_features,
            text_config=(
                TextFeatureConfig(
                    max_vocabulary=args.text_max_vocabulary,
                    min_document_frequency=args.text_min_document_frequency,
                )
                if args.text_features
                else None
            ),
            text_vocabulary_articles=(
                load_articles(args.text_vocabulary_articles)
                if args.text_vocabulary_articles is not None
                else None
            ),
        )
        model.save(args.output)
        result: dict[str, object] = {
            "model": str(args.output),
            "training_examples": model.training_examples,
            "training_sha256": model.training_sha256,
            "weights": model.weights,
        }
        if model.text_encoder is not None:
            result["text_vocabulary_size"] = len(model.text_encoder.vocabulary)
            result["text_training_articles_sha256"] = model.text_encoder.training_articles_sha256
            result["text_vocabulary_source"] = model.text_encoder.source_kind
        print(
            json.dumps(
                result,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "rank-click-model":
        _require_distinct_paths(
            {
                "articles": args.articles,
                "events": args.events,
                "model": args.model,
                "output": args.output,
            }
        )
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
    if args.command == "stream-ingest":
        _require_distinct_paths(
            {
                "articles": args.articles,
                "checkpoint": args.checkpoint,
                "config": args.config,
                "input": args.input,
                "log": args.log,
                "output": args.output,
            }
        )
        stream_limits = _event_stream_limits(args)
        incoming = load_interaction_events(args.input, limits=stream_limits)
        checkpoint_path = None
        if args.checkpoint is not None and Path(args.checkpoint).exists():
            checkpoint_path = args.checkpoint
        store = _open_event_store(args, checkpoint=checkpoint_path)
        stream_report = store.ingest(incoming)
        stream_payload: dict[str, object] = {
            "ingest": stream_report.to_dict(),
            "object": "mosaicfeed.interaction_ingest",
            "recovered_torn_tail_bytes": store.recovered_torn_tail_bytes,
        }
        if args.checkpoint is not None:
            stream_payload["checkpoint"] = store.checkpoint(args.checkpoint).to_dict()
        _emit_json(stream_payload, args.output)
        return 0
    if args.command == "stream-checkpoint":
        _require_distinct_paths(
            {
                "articles": args.articles,
                "checkpoint": args.checkpoint,
                "config": args.config,
                "log": args.log,
                "output": args.output,
            },
            allowed_equal=frozenset({frozenset({"checkpoint", "output"})}),
        )
        store = _open_event_store(args, checkpoint=args.checkpoint)
        checkpoint_payload = {
            "checkpoint": store.checkpoint(args.output).to_dict(),
            "object": "mosaicfeed.interaction_checkpoint",
            "recovered_torn_tail_bytes": store.recovered_torn_tail_bytes,
        }
        _emit_json(checkpoint_payload, None)
        return 0
    if args.command == "stream-replay":
        _require_distinct_paths(
            {
                "articles": args.articles,
                "checkpoint": args.checkpoint,
                "config": args.config,
                "events-output": args.events_output,
                "log": args.log,
                "output": args.output,
            }
        )
        store = _open_event_store(args, checkpoint=args.checkpoint)
        as_of = _clock(args.as_of)
        if args.users is None:
            profiles = store.profiles(as_of=as_of)
        else:
            if len(args.users) != len(set(args.users)):
                raise ValueError("--user values must be unique")
            profiles = tuple(store.profile(user_id, as_of=as_of) for user_id in args.users)
        history = store.history_snapshot()
        replay_payload = {
            "as_of": as_of.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "history_snapshot": history.metadata(),
            "object": "mosaicfeed.profile_replay",
            "profiles": [profile_to_dict(profile, as_of=as_of) for profile in profiles],
            "recovered_torn_tail_bytes": store.recovered_torn_tail_bytes,
            "torn_tail_bytes": store.torn_tail_bytes,
        }
        outputs = {}
        if args.events_output:
            outputs[Path(args.events_output)] = json_text(
                [_event_record(event) for event in history.events]
            )
        if args.output:
            outputs[Path(args.output)] = json_text(replay_payload)
        atomic_write_texts(outputs)
        if not args.output:
            print(json.dumps(replay_payload, indent=2, sort_keys=True))
        return 0
    if args.command == "serve-click-model":
        limits = ServingLimits(
            max_request_bytes=args.max_request_bytes,
            max_response_bytes=args.max_response_bytes,
            max_k=args.max_k,
            max_candidates=args.max_candidates,
            max_catalog_articles=args.max_catalog_articles,
            max_history_events=args.max_history_events,
            max_model_bytes=args.max_model_bytes,
            max_catalog_bytes=args.max_catalog_bytes,
            max_history_bytes=args.max_history_bytes,
            max_topics_per_article=args.max_topics_per_article,
            max_catalog_topic_cells=args.max_catalog_topic_cells,
            max_concurrency=args.max_concurrency,
            request_timeout_seconds=args.request_timeout_seconds,
            max_identifier_chars=args.max_identifier_chars,
            max_event_weight=args.max_event_weight,
        )
        service = load_click_rank_service(
            args.model,
            args.articles,
            args.events,
            limits=limits,
        )
        token = resolve_bearer_token(args.token_env)
        server = create_rank_server(
            service,
            host=args.host,
            port=args.port,
            bearer_token=token,
            allow_nonloopback=args.allow_nonloopback,
        )
        address = cast(tuple[str, int], server.server_address)
        print(
            f"serving frozen click model on http://{address[0]}:{address[1]} "
            f"(authentication {'enabled' if token is not None else 'disabled'})",
            file=sys.stderr,
        )
        serve_rank_server(server)
        return 0
    raise AssertionError("unreachable command")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(argv)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
