from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import mosaicfeed.event_stream as stream_module
from mosaicfeed.cli import main
from mosaicfeed.config import FeedConfig
from mosaicfeed.event_stream import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_SCHEMA_VERSION,
    EMPTY_EVENT_CHAIN_SHA256,
    EVENT_SCHEMA_VERSION,
    EventConflictError,
    EventHistorySnapshot,
    EventStreamLimits,
    InteractionEvent,
    LateEventError,
    LogIntegrityError,
    ProfileEventStore,
    load_event_store,
    load_interaction_events,
    profile_to_dict,
)
from mosaicfeed.io import load_events, write_json
from mosaicfeed.models import Article, EventKind
from mosaicfeed.profile import build_profile

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def catalog() -> tuple[Article, ...]:
    return (
        Article("a", "Alpha", "", ("x", "y"), "one", BASE, 0.8, 0.2),
        Article("b", "Beta", "", ("x",), "two", BASE, 0.7, 0.3),
        Article("c", "Gamma", "", ("z",), "three", BASE, 0.6, 0.4),
    )


def interaction(
    event_id: str,
    *,
    user: str = "u",
    item: str = "a",
    day: float = 1.0,
    kind: EventKind = EventKind.CLICK,
    weight: float = 1.0,
) -> InteractionEvent:
    return InteractionEvent(
        event_id=event_id,
        user_id=user,
        item_id=item,
        timestamp=BASE + timedelta(days=day),
        kind=kind,
        weight=weight,
    )


def event_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "event_id": "e1",
        "user_id": "u",
        "item_id": "a",
        "timestamp": "2026-01-02T00:00:00Z",
        "type": "click",
        "weight": 1.0,
    }
    payload.update(changes)
    return payload


def write_catalog(path: Path, values: tuple[Article, ...] | None = None) -> None:
    records = []
    for article in values or catalog():
        records.append(
            {
                "id": article.id,
                "title": article.title,
                "summary": article.summary,
                "topics": list(article.topics),
                "source": article.source,
                "published_at": article.published_at.isoformat(),
                "quality": article.quality,
                "popularity": article.popularity,
            }
        )
    write_json(path, records)


def write_batch(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in events),
        encoding="utf-8",
        newline="\n",
    )


def rewrite_checkpoint(path: Path, mutate: object) -> None:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(envelope, dict)
    callback = mutate
    assert callable(callback)
    callback(envelope)
    payload = envelope["payload"]
    envelope["checksum_sha256"] = hashlib.sha256(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    path.write_text(
        json.dumps(envelope, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def assert_profile_matches_rebuild(
    store: ProfileEventStore,
    user_id: str,
    *,
    as_of: datetime,
    config: FeedConfig,
) -> None:
    expected = build_profile(
        user_id,
        store.history_events(),
        {article.id: article for article in catalog()},
        as_of=as_of,
        config=config,
    )
    actual = store.profile(user_id, as_of=as_of)
    assert actual.user_id == expected.user_id
    assert actual.event_count == expected.event_count
    assert actual.seen_article_ids == expected.seen_article_ids
    assert actual.topic_weights == pytest.approx(expected.topic_weights)


def test_interaction_event_schema_is_strict_and_canonical() -> None:
    parsed = InteractionEvent.from_mapping(event_payload(timestamp="2026-01-01T18:00:00-06:00"))
    assert parsed.timestamp == datetime(2026, 1, 2, tzinfo=UTC)
    assert parsed.order_key == (parsed.timestamp, "e1")
    assert parsed.sha256 == hashlib.sha256(parsed.canonical_bytes()).hexdigest()
    assert parsed.log_line() == (
        b'{"event_id":"e1","item_id":"a","schema_version":1,'
        b'"timestamp":"2026-01-02T00:00:00Z","type":"click","user_id":"u",'
        b'"weight":1.0}\n'
    )
    legacy = parsed.to_event()
    assert (legacy.user_id, legacy.article_id, legacy.kind, legacy.weight) == (
        "u",
        "a",
        EventKind.CLICK,
        1.0,
    )
    assert EVENT_SCHEMA_VERSION == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": True}, "schema_version"),
        ({"schema_version": 2}, "schema_version"),
        ({"event_id": ""}, "event_id"),
        ({"event_id": " x"}, "event_id"),
        ({"event_id": "x\n"}, "event_id"),
        ({"event_id": "x\ny"}, "printing"),
        ({"user_id": 1}, "user_id"),
        ({"item_id": ""}, "item_id"),
        ({"timestamp": "2026-01-02"}, "timezone"),
        ({"timestamp": 1}, "ISO-8601"),
        ({"type": 1}, "string"),
        ({"type": "share"}, "unsupported interaction type"),
        ({"weight": True}, "finite number"),
        ({"weight": 0}, "positive"),
        ({"weight": -1}, "positive"),
        ({"weight": float("inf")}, "finite"),
    ],
)
def test_interaction_event_rejects_invalid_fields(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        InteractionEvent.from_mapping(event_payload(**changes))


def test_interaction_event_rejects_missing_unknown_and_direct_invalid_values() -> None:
    missing = event_payload()
    missing.pop("weight")
    with pytest.raises(ValueError, match="missing or unknown"):
        InteractionEvent.from_mapping(missing)
    with pytest.raises(ValueError, match="missing or unknown"):
        InteractionEvent.from_mapping(event_payload(extra=1))
    with pytest.raises(ValueError, match="JSON object"):
        InteractionEvent.from_mapping([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timezone-aware"):
        InteractionEvent("e", "u", "a", datetime(2026, 1, 2), EventKind.CLICK)
    with pytest.raises(ValueError, match="EventKind"):
        InteractionEvent("e", "u", "a", BASE, "click")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        InteractionEvent("e", "u", "a", BASE, EventKind.CLICK, 10**400)


def test_limits_reject_invalid_or_incoherent_values() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        EventStreamLimits(max_events=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        EventStreamLimits(max_events=0)
    with pytest.raises(ValueError, match="max_weight"):
        EventStreamLimits(max_weight=float("nan"))
    with pytest.raises(ValueError, match="max_event_bytes"):
        EventStreamLimits(max_event_bytes=3, max_input_bytes=2)
    with pytest.raises(ValueError, match="max_state_bytes"):
        EventStreamLimits(max_state_bytes=3, max_checkpoint_bytes=2)
    with pytest.raises(ValueError, match="positive integer"):
        EventStreamLimits(max_profile_topic_cells=0)
    with pytest.raises(ValueError, match="positive integer"):
        EventStreamLimits(max_catalog_topic_cells=0)
    with pytest.raises(ValueError, match="at least"):
        ProfileEventStore("ignored", catalog(), allowed_lateness_seconds=-1)


def test_manual_profile_oracle_and_point_in_time_reads(tmp_path: Path) -> None:
    config = FeedConfig(
        profile_half_life_days=1.0,
        click_signal=2.0,
        like_signal=4.0,
    )
    store = ProfileEventStore.open(tmp_path / "events.jsonl", catalog(), config=config)
    first = interaction("e1", day=1, item="a", kind=EventKind.CLICK)
    second = interaction("e2", day=2, item="b", kind=EventKind.LIKE, weight=0.5)

    report = store.ingest((first, second))

    assert report.accepted == 2
    assert report.duplicates == 0
    assert report.rebuilt_users == 1
    assert report.incremental_users == 0
    assert report.event_count == 2
    assert report.log_offset == (tmp_path / "events.jsonl").stat().st_size
    assert report.event_chain_sha256 == store.event_chain_sha256
    profile = store.profile("u", as_of=BASE + timedelta(days=3))
    assert profile.topic_weights == pytest.approx({"x": 1.0, "y": 0.2})
    assert profile.seen_article_ids == frozenset({"a", "b"})
    assert profile.event_count == 2
    early = store.profile("u", as_of=BASE + timedelta(days=1, hours=12))
    assert early.topic_weights == {"x": 1.0, "y": 1.0}
    assert early.event_count == 1
    assert store.profile("missing", as_of=BASE).event_count == 0
    assert [value.user_id for value in store.profiles(as_of=BASE + timedelta(days=3))] == ["u"]
    assert profile_to_dict(profile, as_of=BASE + timedelta(days=3)) == {
        "as_of": "2026-01-04T00:00:00Z",
        "event_count": 2,
        "seen_item_ids": ["a", "b"],
        "topic_weights": {"x": 1.0, "y": pytest.approx(0.2)},
        "user_id": "u",
    }
    with pytest.raises(ValueError, match="timezone-aware"):
        store.profile("u", as_of=datetime(2026, 1, 4))
    with pytest.raises(ValueError, match="timezone-aware"):
        store.profiles(as_of=datetime(2026, 1, 4))


def test_incremental_and_out_of_order_paths_equal_full_rebuild(tmp_path: Path) -> None:
    config = FeedConfig(profile_half_life_days=2.5)
    store = ProfileEventStore.open(
        tmp_path / "events.jsonl",
        catalog(),
        config=config,
        late_event_policy="accept-rebuild",
    )
    assert store.ingest((interaction("e1", day=1),)).rebuilt_users == 1
    incremental = store.ingest((interaction("e2", day=3, item="b"),))
    assert (incremental.incremental_users, incremental.rebuilt_users) == (1, 0)
    rebuilt = store.ingest((interaction("e0", day=2, item="c", kind=EventKind.HIDE),))
    assert (rebuilt.incremental_users, rebuilt.rebuilt_users) == (0, 1)
    assert [event.event_id for event in store.events()] == ["e1", "e2", "e0"]
    assert [event.article_id for event in store.history_events()] == ["a", "b", "c"]
    assert_profile_matches_rebuild(
        store,
        "u",
        as_of=BASE + timedelta(days=5),
        config=config,
    )


def test_randomized_replay_matches_full_profile_rebuild(tmp_path: Path) -> None:
    rng = random.Random(173)
    config = FeedConfig(profile_half_life_days=3.0)
    store = ProfileEventStore.open(
        tmp_path / "events.jsonl",
        catalog(),
        config=config,
        late_event_policy="accept-rebuild",
    )
    events = [
        interaction(
            f"e{index:03d}",
            user=f"u{index % 4}",
            item=("a", "b", "c")[index % 3],
            day=1.0 + (index % 11) / 4,
            kind=(EventKind.VIEW, EventKind.CLICK, EventKind.LIKE, EventKind.HIDE)[index % 4],
            weight=0.5 + (index % 5) / 4,
        )
        for index in range(80)
    ]
    rng.shuffle(events)
    for start in range(0, len(events), 7):
        store.ingest(events[start : start + 7])
    for user_id in ("u0", "u1", "u2", "u3"):
        for day in (1.5, 2.5, 5.0):
            assert_profile_matches_rebuild(
                store,
                user_id,
                as_of=BASE + timedelta(days=day),
                config=config,
            )


def test_duplicate_is_idempotent_and_conflict_is_rejected(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    store = ProfileEventStore.open(log, catalog())
    original = interaction("same")
    first = store.ingest((original,))
    contents = log.read_bytes()
    equivalent = InteractionEvent.from_mapping(
        event_payload(event_id="same", timestamp="2026-01-01T18:00:00-06:00", weight=1)
    )
    duplicate = store.ingest((original, original, equivalent))
    assert first.accepted == 1
    assert duplicate.to_dict()["duplicates"] == 3
    assert duplicate.accepted == 0
    assert log.read_bytes() == contents
    with pytest.raises(EventConflictError, match="conflicts"):
        store.ingest((interaction("same", item="b"),))
    assert log.read_bytes() == contents


def test_watermark_boundary_reject_policy_and_accept_rebuild(tmp_path: Path) -> None:
    store = ProfileEventStore.open(
        tmp_path / "reject.jsonl",
        catalog(),
        allowed_lateness_seconds=86_400,
    )
    newest = interaction("new", day=4)
    store.ingest((newest,))
    assert store.watermark("u") == BASE + timedelta(days=3)
    assert store.watermark("absent") is None
    boundary = interaction("boundary", day=3)
    assert store.ingest((boundary,)).accepted == 1
    with pytest.raises(LateEventError, match="watermark"):
        store.ingest((interaction("too-old", day=2.99),))
    assert store.ingest((boundary,)).duplicates == 1

    rebuilding = ProfileEventStore.open(
        tmp_path / "accept.jsonl",
        catalog(),
        allowed_lateness_seconds=0,
        late_event_policy="accept-rebuild",
    )
    rebuilding.ingest((newest,))
    report = rebuilding.ingest((interaction("old", day=2),))
    assert report.rebuilt_users == 1
    assert report.accepted == 1
    with pytest.raises(ValueError, match="reject or accept-rebuild"):
        ProfileEventStore(tmp_path / "bad", catalog(), late_event_policy="drop")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reject or accept-rebuild"):
        ProfileEventStore(tmp_path / "bad", catalog(), late_event_policy=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ten years"):
        ProfileEventStore(tmp_path / "bad", catalog(), allowed_lateness_seconds=315_576_001)


def test_store_validates_catalog_events_and_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="config"):
        ProfileEventStore(tmp_path / "log", catalog(), config=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="limits"):
        ProfileEventStore(tmp_path / "log", catalog(), limits=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        ProfileEventStore(tmp_path / "log", ())
    with pytest.raises(ValueError, match="Article"):
        ProfileEventStore(tmp_path / "log", (object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        ProfileEventStore(tmp_path / "log", (catalog()[0], catalog()[0]))
    with pytest.raises(ValueError, match="max_catalog_articles"):
        ProfileEventStore(
            tmp_path / "log",
            catalog(),
            limits=EventStreamLimits(max_catalog_articles=2),
        )
    with pytest.raises(ValueError, match="max_catalog_bytes"):
        ProfileEventStore(
            tmp_path / "large-catalog",
            catalog(),
            limits=EventStreamLimits(max_catalog_bytes=50),
        )
    with pytest.raises(ValueError, match="max_topics_per_article"):
        ProfileEventStore(
            tmp_path / "wide-topic-catalog",
            catalog(),
            limits=EventStreamLimits(max_topics_per_article=1),
        )
    with pytest.raises(ValueError, match="max_catalog_topic_cells"):
        ProfileEventStore(
            tmp_path / "wide-catalog",
            catalog(),
            limits=EventStreamLimits(max_catalog_topic_cells=3),
        )
    with pytest.raises(ValueError, match="identifier_chars"):
        ProfileEventStore(
            tmp_path / "long-topic",
            (Article("a", "A", "", ("long",), "source", BASE),),
            limits=EventStreamLimits(max_identifier_chars=2),
        )
    with pytest.raises(ValueError, match="identifier_chars"):
        ProfileEventStore(
            tmp_path / "long-catalog-id",
            (Article("long", "Long", "", ("x",), "source", BASE),),
            limits=EventStreamLimits(max_identifier_chars=2),
        )
    with pytest.raises(ValueError, match="max_catalog_articles"):
        ProfileEventStore(
            tmp_path / "infinite-catalog",
            itertools.repeat(catalog()[0]),
            limits=EventStreamLimits(max_catalog_articles=2),
        )
    store = ProfileEventStore.open(tmp_path / "events.jsonl", catalog())
    with pytest.raises(ValueError, match="InteractionEvent"):
        store.ingest((object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown item"):
        store.ingest((interaction("unknown", item="missing"),))
    with pytest.raises(ValueError, match="predates publication"):
        store.ingest((interaction("early", day=-1),))
    with pytest.raises(ValueError, match="identifier_chars"):
        tiny = ProfileEventStore.open(
            tmp_path / "tiny.jsonl",
            catalog(),
            limits=EventStreamLimits(max_identifier_chars=2),
        )
        tiny.ingest((interaction("long"),))
    with pytest.raises(ValueError, match="max_weight"):
        weighted = ProfileEventStore.open(
            tmp_path / "weighted.jsonl",
            catalog(),
            limits=EventStreamLimits(max_weight=1.0),
        )
        weighted.ingest((interaction("heavy", weight=2.0),))
    with pytest.raises(ValueError, match="must differ"):
        store.checkpoint(tmp_path / "events.jsonl")
    with pytest.raises(ValueError, match="boolean"):
        ProfileEventStore.open(
            tmp_path / "events.jsonl",
            catalog(),
            recover_torn_tail=1,  # type: ignore[arg-type]
        )


def test_catalog_profile_hash_is_streamed_and_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = stream_module._canonical_json
    observed_types: list[type[object]] = []

    def observe(value: object) -> bytes:
        observed_types.append(type(value))
        assert not isinstance(value, list)
        return original(value)

    monkeypatch.setattr(stream_module, "_canonical_json", observe)
    store = ProfileEventStore(tmp_path / "events.jsonl", catalog())
    records = [
        {
            "id": article.id,
            "published_at": article.published_at.isoformat().replace("+00:00", "Z"),
            "topics": list(article.topics),
        }
        for article in sorted(catalog(), key=lambda item: item.id)
    ]
    expected = hashlib.sha256(original(records)).hexdigest()
    assert list not in observed_types
    assert store._catalog_sha256 == expected


def test_resource_limits_reject_before_logical_commit(tmp_path: Path) -> None:
    cases: list[tuple[EventStreamLimits, tuple[InteractionEvent, ...], str]] = [
        (EventStreamLimits(max_batch_events=1), (interaction("1"), interaction("2")), "batch"),
        (EventStreamLimits(max_events=1), (interaction("1"), interaction("2")), "max_events"),
        (
            EventStreamLimits(max_users=1),
            (interaction("1", user="u1"), interaction("2", user="u2")),
            "max_users",
        ),
        (
            EventStreamLimits(max_events_per_user=1),
            (interaction("1"), interaction("2")),
            "max_events_per_user",
        ),
        (
            EventStreamLimits(max_event_bytes=128),
            (interaction("x" * 80),),
            "max_event_bytes",
        ),
        (
            EventStreamLimits(max_state_bytes=128),
            (interaction("x" * 80),),
            "state",
        ),
        (
            EventStreamLimits(max_log_bytes=128),
            (interaction("x" * 80),),
            "max_log_bytes",
        ),
        (
            EventStreamLimits(max_profile_topic_cells=1),
            (interaction("topic-wide"),),
            "max_profile_topic_cells",
        ),
    ]
    for index, (limits, values, message) in enumerate(cases):
        log = tmp_path / f"limited-{index}.jsonl"
        store = ProfileEventStore.open(log, catalog(), limits=limits)
        with pytest.raises(ValueError, match=message):
            store.ingest(values)
        assert store.event_count == 0
        assert not log.exists()
    bounded = ProfileEventStore.open(
        tmp_path / "infinite-batch.jsonl",
        catalog(),
        limits=EventStreamLimits(max_batch_events=2),
    )
    with pytest.raises(ValueError, match="batch"):
        bounded.ingest(itertools.repeat(interaction("same")))
    assert bounded.event_count == 0


def test_empty_ingest_has_stable_empty_chain(tmp_path: Path) -> None:
    store = ProfileEventStore.open(tmp_path / "events.jsonl", catalog())
    report = store.ingest(())
    assert report.accepted == report.duplicates == 0
    assert report.event_chain_sha256 == EMPTY_EVENT_CHAIN_SHA256
    assert report.log_offset == 0
    assert not (tmp_path / "events.jsonl").exists()


def test_checkpoint_round_trip_and_tail_replay(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    checkpoint = tmp_path / "checkpoint.json"
    config = FeedConfig(profile_half_life_days=2.0)
    store = ProfileEventStore.open(log, catalog(), config=config)
    store.ingest((interaction("e1", day=1), interaction("e2", day=2, item="b")))

    report = store.checkpoint(checkpoint)
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload = envelope["payload"]
    assert payload["format"] == CHECKPOINT_FORMAT
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["log"]["last_applied_offset"] == log.stat().st_size
    assert payload["log"]["last_event_sha256"] == interaction("e2", day=2, item="b").sha256
    canonical_payload = json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    assert envelope["checksum_sha256"] == hashlib.sha256(canonical_payload).hexdigest()
    assert report.checkpoint_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert report.to_dict()["event_count"] == 2

    restored = ProfileEventStore.open(
        log,
        catalog(),
        config=config,
        checkpoint_path=checkpoint,
    )
    assert restored.events() == store.events()
    assert restored.event_chain_sha256 == store.event_chain_sha256

    store.ingest((interaction("e3", day=3, item="c"),))
    with_tail = ProfileEventStore.open(
        log,
        catalog(),
        config=config,
        checkpoint_path=checkpoint,
    )
    full_replay = ProfileEventStore.open(log, catalog(), config=config)
    assert with_tail.events() == full_replay.events() == store.events()
    assert with_tail.log_offset == log.stat().st_size
    assert_profile_matches_rebuild(
        with_tail,
        "u",
        as_of=BASE + timedelta(days=4),
        config=config,
    )


def test_checkpoint_supports_empty_store(tmp_path: Path) -> None:
    store = ProfileEventStore.open(tmp_path / "missing-log.jsonl", catalog())
    checkpoint = tmp_path / "empty.json"
    report = store.checkpoint(checkpoint)
    assert report.event_count == report.last_applied_offset == 0
    restored = ProfileEventStore.open(
        tmp_path / "missing-log.jsonl",
        catalog(),
        checkpoint_path=checkpoint,
    )
    assert restored.events() == ()


def test_checkpoint_detects_checksum_derived_state_and_prefix_tampering(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    checkpoint = tmp_path / "checkpoint.json"
    store = ProfileEventStore.open(log, catalog())
    store.ingest((interaction("e1"),))
    store.checkpoint(checkpoint)
    original_checkpoint = checkpoint.read_bytes()

    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    envelope["checksum_sha256"] = "0" * 64
    checkpoint.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
    with pytest.raises(LogIntegrityError, match="checksum"):
        ProfileEventStore.open(log, catalog(), checkpoint_path=checkpoint)

    checkpoint.write_bytes(original_checkpoint)

    def change_state(value: dict[str, object]) -> None:
        payload = value["payload"]
        assert isinstance(payload, dict)
        users = payload["users"]
        assert isinstance(users, list)
        users[0]["event_count"] = 99

    rewrite_checkpoint(checkpoint, change_state)
    with pytest.raises(LogIntegrityError, match="derived user state"):
        ProfileEventStore.open(log, catalog(), checkpoint_path=checkpoint)

    checkpoint.write_bytes(original_checkpoint)

    def change_event(value: dict[str, object]) -> None:
        payload = value["payload"]
        assert isinstance(payload, dict)
        events = payload["events"]
        assert isinstance(events, list)
        events[0]["weight"] = 2.0

    rewrite_checkpoint(checkpoint, change_event)
    with pytest.raises(LogIntegrityError, match="event log prefix"):
        ProfileEventStore.open(log, catalog(), checkpoint_path=checkpoint)

    checkpoint.write_bytes(original_checkpoint)
    contents = bytearray(log.read_bytes())
    contents[contents.index(b"e1")] = ord("x")
    log.write_bytes(contents)
    with pytest.raises(LogIntegrityError, match="prefix"):
        ProfileEventStore.open(log, catalog(), checkpoint_path=checkpoint)


def test_checkpoint_detects_same_size_external_log_mutation(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    store = ProfileEventStore.open(log, catalog())
    store.ingest((interaction("e1"),))
    contents = bytearray(log.read_bytes())
    contents[contents.index(b"e1")] = ord("x")
    log.write_bytes(contents)
    with pytest.raises(LogIntegrityError, match="content differs"):
        store.checkpoint(tmp_path / "checkpoint.json")


def test_checkpoint_rejects_truncation_schema_and_setting_mismatch(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    checkpoint = tmp_path / "checkpoint.json"
    store = ProfileEventStore.open(log, catalog())
    store.ingest((interaction("e1"),))
    store.checkpoint(checkpoint)
    complete = checkpoint.read_bytes()

    checkpoint.write_bytes(complete[:-1])
    with pytest.raises(LogIntegrityError, match="truncated"):
        ProfileEventStore.open(log, catalog(), checkpoint_path=checkpoint)
    checkpoint.write_bytes(b"[]\n")
    with pytest.raises(LogIntegrityError, match="JSON object"):
        ProfileEventStore.open(log, catalog(), checkpoint_path=checkpoint)
    checkpoint.write_bytes(complete)
    with pytest.raises(LogIntegrityError, match="settings"):
        ProfileEventStore.open(
            log,
            catalog(),
            checkpoint_path=checkpoint,
            allowed_lateness_seconds=0,
        )
    with pytest.raises(LogIntegrityError, match="settings"):
        ProfileEventStore.open(
            log,
            (*catalog()[:-1], Article("d", "D", "", ("d",), "d", BASE)),
            checkpoint_path=checkpoint,
        )

    def make_schema_boolean(value: dict[str, object]) -> None:
        payload = value["payload"]
        assert isinstance(payload, dict)
        payload["schema_version"] = True

    rewrite_checkpoint(checkpoint, make_schema_boolean)
    with pytest.raises(LogIntegrityError, match="unsupported checkpoint"):
        ProfileEventStore.open(log, catalog(), checkpoint_path=checkpoint)


def test_checkpoint_bounds_and_atomic_replace_failure(tmp_path: Path, monkeypatch) -> None:
    log = tmp_path / "events.jsonl"
    checkpoint = tmp_path / "checkpoint.json"
    store = ProfileEventStore.open(log, catalog())
    store.ingest((interaction("e1"),))
    store.checkpoint(checkpoint)
    previous = checkpoint.read_bytes()

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(stream_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        store.checkpoint(checkpoint)
    assert checkpoint.read_bytes() == previous
    assert list(tmp_path.glob(".checkpoint.json.*.tmp")) == []

    bounded = ProfileEventStore.open(
        tmp_path / "bounded.jsonl",
        catalog(),
        limits=EventStreamLimits(max_checkpoint_bytes=300, max_state_bytes=300),
    )
    with pytest.raises(ValueError, match="checkpoint exceeds"):
        bounded.checkpoint(tmp_path / "too-large.json")


def test_checkpoint_rejects_short_or_missing_log_prefix(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    checkpoint = tmp_path / "checkpoint.json"
    store = ProfileEventStore.open(log, catalog())
    store.ingest((interaction("e1"),))
    store.checkpoint(checkpoint)
    log.unlink()
    with pytest.raises(LogIntegrityError, match="missing event log"):
        ProfileEventStore.open(log, catalog(), checkpoint_path=checkpoint)
    log.write_bytes(interaction("e1").log_line()[:-5])
    with pytest.raises(LogIntegrityError, match="exceeds the event log"):
        ProfileEventStore.open(log, catalog(), checkpoint_path=checkpoint)


def test_torn_tail_is_visible_and_explicitly_recoverable(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    valid = interaction("e1").log_line()
    torn = b'{"schema_version":1,"event_id":"partial"'
    log.write_bytes(valid + torn)

    read_only = ProfileEventStore.open(log, catalog())
    assert read_only.event_count == 1
    assert read_only.log_offset == len(valid)
    assert read_only.torn_tail_bytes == len(torn)
    assert read_only.recovered_torn_tail_bytes == 0
    with pytest.raises(LogIntegrityError, match="torn tail"):
        read_only.ingest((interaction("e2", day=2),))
    with pytest.raises(LogIntegrityError, match="torn tail"):
        read_only.checkpoint(tmp_path / "checkpoint.json")

    recovered = ProfileEventStore.open(log, catalog(), recover_torn_tail=True)
    assert recovered.torn_tail_bytes == 0
    assert recovered.recovered_torn_tail_bytes == len(torn)
    assert log.read_bytes() == valid
    assert recovered.ingest((interaction("e2", day=2),)).accepted == 1


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (b"\n", "blank line"),
        (b"not-json\n", "invalid strict JSON"),
        (b"[]\n", "JSON object"),
        (
            json.dumps(event_payload(), indent=2).encode() + b"\n",
            "invalid strict JSON",
        ),
        (
            json.dumps({**event_payload(), "unknown": 1}, separators=(",", ":")).encode() + b"\n",
            "invalid event",
        ),
        (
            json.dumps(event_payload(), sort_keys=False).encode() + b"\n",
            "canonical form",
        ),
    ],
)
def test_log_replay_rejects_malformed_complete_records(
    tmp_path: Path, contents: bytes, message: str
) -> None:
    log = tmp_path / "events.jsonl"
    log.write_bytes(contents)
    with pytest.raises(LogIntegrityError, match=message):
        ProfileEventStore.open(log, catalog())


def test_log_replay_rejects_duplicate_and_oversized_records(tmp_path: Path) -> None:
    line = interaction("e1").log_line()
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_bytes(line + line)
    with pytest.raises(LogIntegrityError, match="duplicate event_id"):
        ProfileEventStore.open(duplicate, catalog())

    oversized_line = tmp_path / "oversized-line.jsonl"
    oversized_line.write_bytes(line)
    with pytest.raises(ValueError, match="line exceeds"):
        ProfileEventStore.open(
            oversized_line,
            catalog(),
            limits=EventStreamLimits(max_event_bytes=len(line) - 1),
        )

    oversized_log = tmp_path / "oversized-log.jsonl"
    oversized_log.write_bytes(line)
    with pytest.raises(ValueError, match="log exceeds"):
        ProfileEventStore.open(
            oversized_log,
            catalog(),
            limits=EventStreamLimits(max_log_bytes=len(line) - 1),
        )

    bounded_count = tmp_path / "bounded-count.jsonl"
    bounded_count.write_bytes(line + interaction("e2", day=2).log_line())
    with pytest.raises(ValueError, match="max_events"):
        ProfileEventStore.open(
            bounded_count,
            catalog(),
            limits=EventStreamLimits(max_events=1),
        )
    with pytest.raises(ValueError, match="max_state_bytes"):
        ProfileEventStore.open(
            bounded_count,
            catalog(),
            limits=EventStreamLimits(max_state_bytes=len(line)),
        )


def test_external_log_growth_requires_reopen(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    store = ProfileEventStore.open(log, catalog())
    store.ingest((interaction("e1"),))
    with log.open("ab") as destination:
        destination.write(interaction("external", day=2).log_line())
    with pytest.raises(LogIntegrityError, match="size differs"):
        store.ingest((interaction("e2", day=3),))
    with pytest.raises(LogIntegrityError, match="size differs"):
        store.checkpoint(tmp_path / "checkpoint.json")
    reopened = ProfileEventStore.open(log, catalog())
    assert reopened.event_count == 2


def test_load_interaction_events_accepts_noncanonical_ingress_and_bounds_input(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "incoming.jsonl"
    write_batch(batch, [event_payload(), event_payload(event_id="e2", item_id="b")])
    loaded = load_interaction_events(batch)
    assert [event.event_id for event in loaded] == ["e1", "e2"]
    assert loaded[0].log_line() != batch.read_bytes().splitlines(keepends=True)[0]
    with pytest.raises(ValueError, match="limits"):
        load_interaction_events(batch, limits=object())  # type: ignore[arg-type]

    no_newline = tmp_path / "no-newline.jsonl"
    no_newline.write_text(json.dumps(event_payload()), encoding="utf-8")
    with pytest.raises(ValueError, match="end with a newline"):
        load_interaction_events(no_newline)
    blank = tmp_path / "blank.jsonl"
    blank.write_bytes(b"\n")
    with pytest.raises(ValueError, match="blank line"):
        load_interaction_events(blank)
    array = tmp_path / "array.jsonl"
    array.write_bytes(b"[]\n")
    with pytest.raises(ValueError, match="JSON objects"):
        load_interaction_events(array)
    duplicate_field = tmp_path / "duplicate-field.jsonl"
    duplicate_field.write_bytes(
        b'{"schema_version":1,"schema_version":1,"event_id":"e","user_id":"u",'
        b'"item_id":"a","timestamp":"2026-01-02T00:00:00Z","type":"click",'
        b'"weight":1}\n'
    )
    with pytest.raises(ValueError, match="strict JSON"):
        load_interaction_events(duplicate_field)
    with pytest.raises(ValueError, match="max_input_bytes"):
        load_interaction_events(
            batch,
            limits=EventStreamLimits(max_input_bytes=10, max_event_bytes=10),
        )
    with pytest.raises(ValueError, match="max_batch_events"):
        load_interaction_events(batch, limits=EventStreamLimits(max_batch_events=1))
    with pytest.raises(ValueError, match="max_event_bytes"):
        load_interaction_events(
            batch,
            limits=EventStreamLimits(max_event_bytes=100),
        )


def test_load_event_store_bounds_catalog_and_replays(tmp_path: Path) -> None:
    catalog_path = tmp_path / "articles.json"
    log = tmp_path / "events.jsonl"
    write_catalog(catalog_path)
    log.write_bytes(interaction("e1").log_line())
    loaded = load_event_store(log, catalog_path)
    assert loaded.event_count == 1
    with pytest.raises(ValueError, match="limits"):
        load_event_store(log, catalog_path, limits=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_catalog_bytes"):
        load_event_store(
            log,
            catalog_path,
            limits=EventStreamLimits(max_catalog_bytes=10),
        )


def test_state_and_checkpoint_reports_serialize_stably(tmp_path: Path) -> None:
    store = ProfileEventStore.open(tmp_path / "events.jsonl", catalog())
    report = store.ingest((interaction("e1"),))
    assert report.to_dict() == {
        "accepted": 1,
        "duplicates": 0,
        "event_chain_sha256": store.event_chain_sha256,
        "event_count": 1,
        "incremental_users": 0,
        "log_offset": store.log_offset,
        "rebuilt_users": 1,
    }


def test_history_snapshot_requires_explicit_refresh(tmp_path: Path) -> None:
    store = ProfileEventStore.open(tmp_path / "events.jsonl", catalog())
    store.ingest((interaction("e1"),))
    snapshot = store.history_snapshot()
    frozen_history = snapshot.events
    assert snapshot.metadata() == {
        "event_chain_sha256": store.event_chain_sha256,
        "event_count": 1,
        "last_applied_offset": store.log_offset,
    }
    assert store.limits == EventStreamLimits()
    store.ingest((interaction("e2", item="b", day=2),))
    assert len(frozen_history) == 1
    assert len(store.history_events()) == 2


def test_history_snapshot_binds_one_raw_log_prefix_and_validates_provenance() -> None:
    source_event = interaction("e1")
    event = source_event.to_event()
    source = source_event.log_line()
    snapshot = EventHistorySnapshot(source)
    assert type(snapshot.events) is tuple
    assert snapshot.events == (event,)
    assert snapshot.last_applied_offset == len(source_event.log_line())
    assert snapshot.verify_provenance()
    with pytest.raises(ValueError, match="bytes"):
        EventHistorySnapshot(bytearray(source))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        EventHistorySnapshot(  # type: ignore[call-arg]
            source,
            event_count=1,
            last_applied_offset=10,
            event_chain_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="max_events"):
        EventHistorySnapshot(source + source, max_events=1)
    with pytest.raises(ValueError, match="torn"):
        EventHistorySnapshot(source[:-1])


def test_event_log_rejects_hardlinks_before_append_or_torn_tail_recovery(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    store = ProfileEventStore.open(log, catalog())
    store.ingest((interaction("e1"),))
    with log.open("ab") as destination:
        destination.write(b'{"torn":')
    alias = tmp_path / "alias.jsonl"
    os.link(log, alias)
    before = alias.read_bytes()

    with pytest.raises(LogIntegrityError, match="hard link"):
        ProfileEventStore.open(log, catalog(), recover_torn_tail=True)
    with pytest.raises(LogIntegrityError, match="hard link"):
        store.ingest((interaction("e2", day=2),))
    assert alias.read_bytes() == before


def test_event_log_rejects_symlink_before_replay(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_bytes(interaction("e1").log_line())
    link = tmp_path / "events.jsonl"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(LogIntegrityError, match="symlink"):
        ProfileEventStore.open(link, catalog())
    assert target.read_bytes() == interaction("e1").log_line()


def test_event_log_create_race_fails_closed_without_touching_racer_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "events.jsonl"
    store = ProfileEventStore.open(log, catalog())
    original_open = stream_module.os.open
    racer = b"racer-owned\n"
    triggered = False

    def racing_open(path: object, flags: int, mode: int = 0o777) -> int:
        nonlocal triggered
        if not triggered and Path(path) == log and flags & os.O_EXCL:
            triggered = True
            log.write_bytes(racer)
        return original_open(path, flags, mode)

    monkeypatch.setattr(stream_module.os, "open", racing_open)
    with pytest.raises(LogIntegrityError, match="appeared concurrently"):
        store.ingest((interaction("e1"),))
    assert log.read_bytes() == racer
    assert store.event_count == 0


def test_event_log_identity_swap_between_check_and_open_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "events.jsonl"
    store = ProfileEventStore.open(log, catalog())
    store.ingest((interaction("e1"),))
    original_log = tmp_path / "original.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    replacement_bytes = interaction("replacement", item="b", day=2).log_line()
    replacement.write_bytes(replacement_bytes)
    original_open = stream_module.os.open
    triggered = False

    def racing_open(path: object, flags: int, mode: int = 0o777) -> int:
        nonlocal triggered
        if not triggered and Path(path) == log and flags & os.O_APPEND:
            triggered = True
            os.replace(log, original_log)
            os.replace(replacement, log)
        return original_open(path, flags, mode)

    monkeypatch.setattr(stream_module.os, "open", racing_open)
    with pytest.raises(LogIntegrityError, match="identity"):
        store.ingest((interaction("e2", item="b", day=2),))
    assert log.read_bytes() == replacement_bytes
    assert original_log.read_bytes() == interaction("e1").log_line()
    assert store.event_count == 1


def test_watermark_clamps_at_earliest_datetime(tmp_path: Path) -> None:
    earliest = datetime.min.replace(tzinfo=UTC)
    old_catalog = (Article("old", "Old", "", ("x",), "source", earliest),)
    store = ProfileEventStore.open(
        tmp_path / "events.jsonl",
        old_catalog,
        allowed_lateness_seconds=300,
    )
    store.ingest(
        (
            InteractionEvent(
                "e1",
                "u",
                "old",
                earliest + timedelta(seconds=1),
                EventKind.CLICK,
            ),
        )
    )
    assert store.watermark("u") == earliest


def test_weighted_profile_overflow_and_exact_cancellation_are_checked(tmp_path: Path) -> None:
    overflowing_signal = ProfileEventStore.open(
        tmp_path / "signal.jsonl",
        catalog(),
        config=FeedConfig(click_signal=1e308),
    )
    with pytest.raises(ValueError, match="weighted event signal"):
        overflowing_signal.ingest((interaction("e1", weight=100),))
    overflowing_total = ProfileEventStore.open(
        tmp_path / "total.jsonl",
        catalog(),
        config=FeedConfig(click_signal=1e308),
    )
    with pytest.raises(ValueError, match="topic total"):
        overflowing_total.ingest((interaction("e1", item="b"), interaction("e2", item="b")))

    cancelling = ProfileEventStore.open(
        tmp_path / "cancel.jsonl",
        catalog(),
        config=FeedConfig(click_signal=1, hide_signal=-1),
    )
    cancelling.ingest(
        (
            interaction("a", item="b", kind=EventKind.CLICK),
            interaction("b", item="b", kind=EventKind.HIDE),
        )
    )
    assert cancelling.profile("u", as_of=BASE + timedelta(days=2)).topic_weights == {}


def test_historical_all_user_read_obeys_the_global_topic_cell_limit(tmp_path: Path) -> None:
    config = FeedConfig(
        profile_half_life_days=1.0,
        click_signal=1.0,
        hide_signal=-0.5,
    )
    store = ProfileEventStore.open(
        tmp_path / "historical-cells.jsonl",
        catalog(),
        config=config,
        limits=EventStreamLimits(max_profile_topic_cells=1),
    )
    store.ingest(
        (
            interaction("u1-positive", user="u1", item="b", day=1),
            interaction(
                "u1-cancel",
                user="u1",
                item="b",
                day=2,
                kind=EventKind.HIDE,
            ),
            interaction("u2-positive", user="u2", item="b", day=1),
            interaction(
                "u2-cancel",
                user="u2",
                item="b",
                day=2,
                kind=EventKind.HIDE,
            ),
        )
    )
    assert store.profiles(as_of=BASE + timedelta(days=3))[0].topic_weights == {}
    assert store.profile("u1", as_of=BASE + timedelta(days=1, hours=12)).topic_weights == {"x": 1.0}
    with pytest.raises(ValueError, match="materialized profiles"):
        store.profiles(as_of=BASE + timedelta(days=1, hours=12))


def test_threaded_duplicate_ingestion_is_exactly_once(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    store = ProfileEventStore.open(log, catalog())
    event = interaction("one")
    with ThreadPoolExecutor(max_workers=8) as executor:
        reports = list(executor.map(lambda _: store.ingest((event,)), range(40)))
    assert sum(report.accepted for report in reports) == 1
    assert sum(report.duplicates for report in reports) == 39
    assert store.event_count == 1
    assert log.read_bytes() == event.log_line()


def test_threaded_unique_ingestion_replays_to_same_profile(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    config = FeedConfig(profile_half_life_days=4.0)
    store = ProfileEventStore.open(
        log,
        catalog(),
        config=config,
        late_event_policy="accept-rebuild",
    )
    values = tuple(
        interaction(
            f"e{index}",
            user=f"u{index % 3}",
            item=("a", "b", "c")[index % 3],
            day=1 + (index % 7),
        )
        for index in range(60)
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        reports = list(executor.map(lambda value: store.ingest((value,)), values))
    assert sum(report.accepted for report in reports) == 60
    replayed = ProfileEventStore.open(
        log,
        catalog(),
        config=config,
        late_event_policy="accept-rebuild",
    )
    assert replayed.events() == store.events()
    for user_id in ("u0", "u1", "u2"):
        assert replayed.profile(user_id, as_of=BASE + timedelta(days=10)) == store.profile(
            user_id,
            as_of=BASE + timedelta(days=10),
        )


def test_legacy_event_io_round_trips_optional_weight(tmp_path: Path) -> None:
    events_path = tmp_path / "events.json"
    write_json(
        events_path,
        [
            {
                "user_id": "u",
                "article_id": "a",
                "kind": "click",
                "occurred_at": "2026-01-02T00:00:00Z",
                "weight": 2.5,
            }
        ],
    )
    assert load_events(events_path)[0].weight == 2.5


def test_event_stream_cli_ingest_checkpoint_and_replay(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    articles = tmp_path / "articles.json"
    incoming = tmp_path / "incoming.jsonl"
    log = tmp_path / "events.jsonl"
    checkpoint = tmp_path / "checkpoint.json"
    replay_output = tmp_path / "replay.json"
    events_output = tmp_path / "history.json"
    second_checkpoint = tmp_path / "checkpoint-2.json"
    write_catalog(articles)
    write_batch(
        incoming,
        [
            event_payload(weight=2.0),
            event_payload(event_id="e2", user_id="u2", item_id="b", weight=1.0),
        ],
    )

    assert (
        main(
            [
                "stream-ingest",
                "--articles",
                str(articles),
                "--log",
                str(log),
                "--input",
                str(incoming),
                "--checkpoint",
                str(checkpoint),
            ]
        )
        == 0
    )
    ingested = json.loads(capsys.readouterr().out)
    assert ingested["object"] == "mosaicfeed.interaction_ingest"
    assert ingested["ingest"]["accepted"] == 2
    assert ingested["checkpoint"]["event_count"] == 2

    assert (
        main(
            [
                "stream-replay",
                "--articles",
                str(articles),
                "--log",
                str(log),
                "--checkpoint",
                str(checkpoint),
                "--as-of",
                "2026-01-03T00:00:00Z",
                "--user",
                "u",
                "--events-output",
                str(events_output),
                "--output",
                str(replay_output),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == ""
    replayed = json.loads(replay_output.read_text(encoding="utf-8"))
    assert replayed["object"] == "mosaicfeed.profile_replay"
    assert replayed["history_snapshot"]["event_count"] == 2
    assert replayed["profiles"][0]["user_id"] == "u"
    assert json.loads(events_output.read_text(encoding="utf-8"))[0]["weight"] == 2.0

    assert (
        main(
            [
                "stream-checkpoint",
                "--articles",
                str(articles),
                "--log",
                str(log),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(second_checkpoint),
            ]
        )
        == 0
    )
    checkpoint_report = json.loads(capsys.readouterr().out)
    assert checkpoint_report["object"] == "mosaicfeed.interaction_checkpoint"
    assert checkpoint_report["checkpoint"]["event_count"] == 2
    assert second_checkpoint.exists()
    assert (
        main(
            [
                "stream-checkpoint",
                "--articles",
                str(articles),
                "--log",
                str(log),
                "--checkpoint",
                str(second_checkpoint),
                "--output",
                str(second_checkpoint),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["checkpoint"]["event_count"] == 2


def test_event_stream_cli_resumes_checkpoint_and_reports_usage_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    articles = tmp_path / "articles.json"
    first_input = tmp_path / "first.jsonl"
    second_input = tmp_path / "second.jsonl"
    log = tmp_path / "events.jsonl"
    checkpoint = tmp_path / "checkpoint.json"
    result_file = tmp_path / "result.json"
    write_catalog(articles)
    write_batch(first_input, [event_payload()])
    write_batch(
        second_input,
        [event_payload(), event_payload(event_id="e2", item_id="b", weight=1.5)],
    )
    common = ["--articles", str(articles), "--log", str(log)]
    assert (
        main(
            [
                "stream-ingest",
                *common,
                "--input",
                str(first_input),
                "--checkpoint",
                str(checkpoint),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "stream-ingest",
                *common,
                "--input",
                str(second_input),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(result_file),
            ]
        )
        == 0
    )
    resumed = json.loads(result_file.read_text(encoding="utf-8"))
    assert resumed["ingest"]["accepted"] == 1
    assert resumed["ingest"]["duplicates"] == 1

    assert (
        main(
            [
                "stream-replay",
                *common,
                "--checkpoint",
                str(checkpoint),
                "--as-of",
                "2026-01-04T00:00:00Z",
            ]
        )
        == 0
    )
    all_profiles = json.loads(capsys.readouterr().out)
    assert [profile["user_id"] for profile in all_profiles["profiles"]] == ["u"]

    duplicate_users = main(
        [
            "stream-replay",
            *common,
            "--as-of",
            "2026-01-04T00:00:00Z",
            "--user",
            "u",
            "--user",
            "u",
        ]
    )
    assert duplicate_users == 2
    assert "must be unique" in capsys.readouterr().err

    invalid_limit = main(
        [
            "stream-replay",
            *common,
            "--as-of",
            "2026-01-04T00:00:00Z",
            "--max-events",
            "0",
        ]
    )
    assert invalid_limit == 2
    assert "positive integer" in capsys.readouterr().err

    config = tmp_path / "config.json"
    config.write_text('{"size": 10}\n', encoding="utf-8")
    oversized_config = main(
        [
            "stream-replay",
            *common,
            "--as-of",
            "2026-01-04T00:00:00Z",
            "--config",
            str(config),
            "--max-config-bytes",
            "5",
        ]
    )
    assert oversized_config == 2
    assert "max_config_bytes" in capsys.readouterr().err


def test_event_stream_cli_requires_explicit_torn_tail_recovery(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    articles = tmp_path / "articles.json"
    log = tmp_path / "events.jsonl"
    incoming = tmp_path / "incoming.jsonl"
    write_catalog(articles)
    torn = b'{"event_id":"torn"'
    log.write_bytes(interaction("e1").log_line() + torn)
    write_batch(incoming, [event_payload(event_id="e2", item_id="b")])
    args = [
        "stream-ingest",
        "--articles",
        str(articles),
        "--log",
        str(log),
        "--input",
        str(incoming),
    ]
    assert main(args) == 2
    assert "torn tail" in capsys.readouterr().err
    assert main([*args, "--recover-torn-tail"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recovered_torn_tail_bytes"] == len(torn)
    assert ProfileEventStore.open(log, catalog()).event_count == 2


def test_event_stream_cli_refuses_path_collisions_before_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    articles = tmp_path / "articles.json"
    incoming = tmp_path / "incoming.jsonl"
    log = tmp_path / "events.jsonl"
    write_catalog(articles)
    write_batch(incoming, [event_payload()])
    assert (
        main(
            [
                "stream-ingest",
                "--articles",
                str(articles),
                "--input",
                str(incoming),
                "--log",
                str(log),
                "--output",
                str(log),
            ]
        )
        == 2
    )
    assert "different paths" in capsys.readouterr().err
    assert not log.exists()

    log.write_bytes(interaction("e1").log_line())
    original_log = log.read_bytes()
    assert (
        main(
            [
                "stream-replay",
                "--articles",
                str(articles),
                "--log",
                str(log),
                "--as-of",
                "2026-01-03T00:00:00Z",
                "--events-output",
                str(log),
            ]
        )
        == 2
    )
    assert "different paths" in capsys.readouterr().err
    assert log.read_bytes() == original_log

    original_catalog = articles.read_bytes()
    assert (
        main(
            [
                "stream-checkpoint",
                "--articles",
                str(articles),
                "--log",
                str(log),
                "--output",
                str(articles),
            ]
        )
        == 2
    )
    assert "different paths" in capsys.readouterr().err
    assert articles.read_bytes() == original_catalog


def test_event_stream_cli_refuses_hard_link_alias_before_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    articles = tmp_path / "articles.json"
    incoming = tmp_path / "incoming.jsonl"
    output_alias = tmp_path / "output.json"
    log = tmp_path / "events.jsonl"
    write_catalog(articles)
    write_batch(incoming, [event_payload()])
    os.link(incoming, output_alias)
    original = incoming.read_bytes()

    assert (
        main(
            [
                "stream-ingest",
                "--articles",
                str(articles),
                "--input",
                str(incoming),
                "--log",
                str(log),
                "--output",
                str(output_alias),
            ]
        )
        == 2
    )
    assert "different paths" in capsys.readouterr().err
    assert incoming.read_bytes() == original
    assert output_alias.read_bytes() == original
    assert not log.exists()
