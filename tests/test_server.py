from __future__ import annotations

import http.client
import itertools
import json
import socket
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from mosaicfeed import cli
from mosaicfeed.config import FeedConfig
from mosaicfeed.io import write_json
from mosaicfeed.learning import ClickPrediction, PointwiseLogisticRanker
from mosaicfeed.models import Article, Event, EventKind
from mosaicfeed.server import (
    ClickRankService,
    RankHTTPServer,
    RankResult,
    ServingLimits,
    create_rank_server,
    is_loopback_host,
    load_click_rank_service,
    resolve_bearer_token,
    serve_rank_server,
)

ORIGIN = datetime(2026, 1, 1, tzinfo=UTC)
AS_OF = ORIGIN + timedelta(days=3)


def catalog() -> list[Article]:
    return [
        Article("positive", "Positive", "", ("useful",), "A", ORIGIN, 0.95, 0.4),
        Article("negative", "Negative", "", ("noise",), "B", ORIGIN, 0.05, 0.4),
        Article("candidate-a", "Candidate A", "", ("useful",), "C", ORIGIN, 0.85, 0.3),
        Article("candidate-b", "Candidate B", "", ("noise",), "D", ORIGIN, 0.15, 0.3),
    ]


def history() -> list[Event]:
    return [
        Event("u1", "positive", EventKind.CLICK, ORIGIN + timedelta(days=1)),
        Event("u1", "negative", EventKind.VIEW, ORIGIN + timedelta(days=2)),
        Event("u2", "positive", EventKind.LIKE, ORIGIN + timedelta(days=1, hours=1)),
        Event("u2", "negative", EventKind.HIDE, ORIGIN + timedelta(days=2, hours=1)),
    ]


def fitted_model() -> PointwiseLogisticRanker:
    return PointwiseLogisticRanker(epochs=40, learning_rate=0.1, seed=7).fit(
        catalog(),
        history(),
        as_of=AS_OF,
        config=FeedConfig(exploration_weight=0.0),
    )


def service(
    *,
    articles: list[Article] | None = None,
    events: list[Event] | None = None,
    limits: ServingLimits | None = None,
) -> ClickRankService:
    return ClickRankService(
        fitted_model(),
        catalog() if articles is None else articles,
        history() if events is None else events,
        limits=limits,
    )


@contextmanager
def running_server(
    rank_service: ClickRankService,
    *,
    token: str | None = None,
) -> Iterator[RankHTTPServer]:
    server = create_rank_server(rank_service, host="127.0.0.1", port=0, bearer_token=token)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def request(
    server: RankHTTPServer,
    method: str,
    path: str,
    payload: object | bytes | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[int, http.client.HTTPMessage, bytes]:
    body = payload
    active_headers = dict(headers or {})
    if payload is not None and not isinstance(payload, bytes):
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        active_headers.setdefault("Content-Type", "application/json")
    connection = http.client.HTTPConnection(*server.server_address, timeout=2)
    try:
        connection.request(method, path, body=body, headers=active_headers)
        response = connection.getresponse()
        return response.status, response.headers, response.read()
    finally:
        connection.close()


def raw_request(server: RankHTTPServer, message: bytes) -> tuple[int, dict[str, object]]:
    connection = socket.create_connection(server.server_address, timeout=2)
    try:
        connection.sendall(message)
        connection.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while chunk := connection.recv(64 * 1024):
            chunks.append(chunk)
    finally:
        connection.close()
    head, body = b"".join(chunks).split(b"\r\n\r\n", maxsplit=1)
    status = int(head.split(b"\r\n", maxsplit=1)[0].split()[1])
    return status, json.loads(body)


def rank_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "user_id": "new-user",
        "as_of": "2026-01-04T00:00:00Z",
        "k": 2,
        "candidate_ids": ["candidate-b", "candidate-a"],
    }
    payload.update(updates)
    return payload


def test_health_metadata_and_rank_use_real_loopback_http() -> None:
    with running_server(service()) as server:
        status, headers, body = request(server, "GET", "/health")
        assert status == 200
        assert json.loads(body) == {"status": "ok"}
        assert headers["Cache-Control"] == "no-store"
        assert headers["X-Content-Type-Options"] == "nosniff"

        status, _, metadata_body = request(server, "GET", "/metadata")
        metadata = json.loads(metadata_body)
        assert status == 200
        assert metadata["catalog_articles"] == 4
        assert metadata["history_events"] == 4
        assert metadata["model"]["format"] == "mosaicfeed.pointwise-logistic"
        assert len(metadata["model"]["training_examples_sha256"]) == 64
        assert "at or before" in metadata["time_semantics"]

        first = request(server, "POST", "/v1/rank", rank_payload())
        second = request(
            server,
            "POST",
            "/v1/rank",
            rank_payload(candidate_ids=["candidate-a", "candidate-b"]),
        )
        assert first[0] == second[0] == 200
        assert first[2] == second[2]
        ranked = json.loads(first[2])
        assert ranked["object"] == "mosaicfeed.click_ranking"
        assert ranked["as_of"] == "2026-01-04T00:00:00Z"
        assert ranked["candidate_count"] == 2
        assert ranked["requested_k"] == 2
        assert {item["article_id"] for item in ranked["predictions"]} == {
            "candidate-a",
            "candidate-b",
        }

        status, _, _ = request(
            server,
            "POST",
            "/v1/rank",
            json.dumps(rank_payload()).encode(),
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        assert status == 200


def test_equal_concurrent_requests_are_thread_safe_and_byte_stable() -> None:
    with running_server(service()) as server, ThreadPoolExecutor(max_workers=8) as executor:
        responses = tuple(
            executor.map(
                lambda _: request(server, "POST", "/v1/rank", rank_payload()),
                range(16),
            )
        )
    assert {response[0] for response in responses} == {200}
    assert len({response[2] for response in responses}) == 1


def test_rank_result_enforces_immutable_response_invariants() -> None:
    first = ClickPrediction("a", 0.5, 1)
    valid = RankResult("u", AS_OF, 1, 1, (first,))
    assert valid.to_dict()["predictions"] == [first.to_dict()]
    with pytest.raises(ValueError, match="timezone"):
        RankResult("u", AS_OF.replace(tzinfo=None), 1, 1, (first,))
    with pytest.raises(ValueError, match="requested_k"):
        RankResult("u", AS_OF, 0, 1, ())
    with pytest.raises(ValueError, match="candidate_count"):
        RankResult("u", AS_OF, 1, -1, ())
    with pytest.raises(ValueError, match="tuple"):
        RankResult("u", AS_OF, 1, 1, [first])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exceed"):
        RankResult("u", AS_OF, 1, 0, (first,))
    with pytest.raises(ValueError, match="contiguous"):
        RankResult("u", AS_OF, 2, 2, (ClickPrediction("a", 0.5, 2),))
    with pytest.raises(ValueError, match="unique"):
        RankResult(
            "u",
            AS_OF,
            2,
            2,
            (ClickPrediction("a", 0.5, 1), ClickPrediction("a", 0.4, 2)),
        )

    class SwitchingPredictions(tuple[ClickPrediction, ...]):
        switched = False

        def __iter__(self):
            if self.switched:
                return iter((ClickPrediction("forged", 1.0, 1),))
            return tuple.__iter__(self)

    source = SwitchingPredictions((first,))
    frozen = RankResult("u", AS_OF, 1, 1, source)
    source.switched = True
    assert type(frozen.predictions) is tuple
    assert frozen.predictions == (first,)


def test_configured_identifier_limit_is_used_end_to_end() -> None:
    long_user = "u" * 257
    limits = replace(ServingLimits(), max_identifier_chars=300)
    with running_server(service(limits=limits)) as server:
        status, _, body = request(
            server,
            "POST",
            "/v1/rank",
            rank_payload(user_id=long_user),
        )
    assert status == 200
    assert json.loads(body)["user_id"] == long_user


def test_candidate_subset_keeps_full_history_and_point_in_time_semantics() -> None:
    exact = Event("target", "positive", EventKind.CLICK, AS_OF)
    future = Event("target", "negative", EventKind.HIDE, AS_OF + timedelta(microseconds=1))
    exact_service = service(events=[*history(), exact])
    future_service = service(events=[*history(), exact, future])

    exact_result = exact_service.rank("target", as_of=AS_OF, k=1, candidate_ids=("candidate-a",))
    future_result = future_service.rank("target", as_of=AS_OF, k=1, candidate_ids=("candidate-a",))
    cold_result = exact_service.rank("cold", as_of=AS_OF, k=1, candidate_ids=("candidate-a",))
    assert exact_result == future_result
    assert exact_result.predictions[0].probability != cold_result.predictions[0].probability


@pytest.mark.parametrize(
    ("payload", "status", "code"),
    [
        (["not-an-object"], 400, "invalid_request"),
        ({"user_id": "u"}, 400, "invalid_request"),
        (rank_payload(extra=True), 400, "invalid_request"),
        (rank_payload(user_id=" spaced "), 400, "invalid_request"),
        (rank_payload(as_of="2026-01-04T00:00:00"), 400, "invalid_request"),
        (rank_payload(k=True), 400, "invalid_request"),
        (rank_payload(k=0), 400, "invalid_request"),
        (rank_payload(k=101), 400, "invalid_request"),
        (rank_payload(candidate_ids="candidate-a"), 400, "invalid_request"),
        (rank_payload(candidate_ids=None), 400, "invalid_request"),
        (rank_payload(candidate_ids=["candidate-a", "candidate-a"]), 400, "invalid_request"),
        (rank_payload(candidate_ids=[" bad "]), 400, "invalid_request"),
        (rank_payload(candidate_ids=["private-unknown-id"]), 422, "invalid_ranking_request"),
    ],
)
def test_rank_schema_and_semantic_errors_are_bounded_and_redacted(
    payload: object, status: int, code: str
) -> None:
    with running_server(service()) as server:
        actual_status, _, body = request(server, "POST", "/v1/rank", payload)
    response = json.loads(body)
    assert actual_status == status
    assert response["error"]["code"] == code
    assert "private-unknown-id" not in body.decode()


def test_request_candidate_and_catalog_limits() -> None:
    limits = replace(ServingLimits(), max_candidates=1)
    with running_server(service(limits=limits)) as server:
        status, _, body = request(server, "POST", "/v1/rank", rank_payload())
        assert status == 400
        assert json.loads(body)["error"]["code"] == "invalid_request"

        all_catalog = rank_payload()
        del all_catalog["candidate_ids"]
        status, _, body = request(server, "POST", "/v1/rank", all_catalog)
        assert status == 422
        assert json.loads(body)["error"]["code"] == "invalid_ranking_request"


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (b'{"user_id":"u","user_id":"v","as_of":"2026-01-04T00:00:00Z"}', "application/json"),
        (b'{"user_id":"u","as_of":"2026-01-04T00:00:00Z","k":NaN}', "application/json"),
        (b"\xff", "application/json"),
        (b"{}", "text/plain"),
        (b"{}", "application/json; charset=latin1"),
    ],
)
def test_strict_json_and_media_type(body: bytes, content_type: str) -> None:
    with running_server(service()) as server:
        status, _, response = request(
            server,
            "POST",
            "/v1/rank",
            body,
            headers={"Content-Type": content_type},
        )
    assert status in {400, 415}
    assert json.loads(response)["error"]["code"] in {"invalid_json", "unsupported_media_type"}


def test_request_body_and_content_encoding_limits() -> None:
    limits = replace(ServingLimits(), max_request_bytes=8)
    with running_server(service(limits=limits)) as server:
        status, _, body = request(
            server,
            "POST",
            "/v1/rank",
            b"{}" * 5,
            headers={"Content-Type": "application/json"},
        )
        assert status == 413
        assert json.loads(body)["error"]["code"] == "request_too_large"

    with running_server(service()) as server:
        status, _, body = request(
            server,
            "POST",
            "/v1/rank",
            b"{}",
            headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
        )
        assert status == 415
        assert json.loads(body)["error"]["code"] == "unsupported_media_type"


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (b"", 411),
        (b"Content-Length: x\r\n", 400),
        (b"Content-Length: " + b"9" * 5_000 + b"\r\n", 400),
        (b"Content-Length: 2\r\nContent-Length: 2\r\n", 400),
        (b"Transfer-Encoding: chunked\r\n", 400),
    ],
)
def test_raw_framing_is_strict(headers: bytes, expected: int) -> None:
    with running_server(service()) as server:
        status, response = raw_request(
            server,
            b"POST /v1/rank HTTP/1.1\r\nHost: localhost\r\n"
            + b"Content-Type: application/json\r\n"
            + headers
            + b"Connection: close\r\n\r\n{}",
        )
    assert status == expected
    assert "error" in response


def test_error_body_survives_oversized_decimal_content_length() -> None:
    with running_server(service()) as server:
        status, response = raw_request(
            server,
            b"GET /unknown HTTP/1.1\r\nHost: localhost\r\nContent-Length: "
            + b"9" * 5_000
            + b"\r\nConnection: close\r\n\r\n",
        )
    assert status == 404
    assert response["error"]["code"] == "not_found"


def test_short_body_is_rejected() -> None:
    with running_server(service()) as server:
        status, body = raw_request(
            server,
            b"POST /v1/rank HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Type: application/json\r\nContent-Length: 10\r\n"
            b"Connection: close\r\n\r\n{}",
        )
    assert status == 400
    assert body["error"]["code"] == "invalid_json"


def test_optional_bearer_auth_protects_metadata_and_ranking_but_not_health() -> None:
    with running_server(service(), token="correct-secret") as server:
        assert request(server, "GET", "/health")[0] == 200
        status, headers, body = request(server, "GET", "/metadata")
        assert status == 401
        assert headers["WWW-Authenticate"] == 'Bearer realm="mosaicfeed"'
        assert b"correct-secret" not in body
        assert (
            request(
                server,
                "POST",
                "/v1/rank",
                rank_payload(),
                headers={"Authorization": "Bearer wrong"},
            )[0]
            == 401
        )
        assert (
            request(
                server,
                "POST",
                "/v1/rank",
                rank_payload(),
                headers={"Authorization": "Bearer correct-secret"},
            )[0]
            == 200
        )


def test_unknown_paths_and_methods_are_explicit() -> None:
    with running_server(service()) as server:
        assert request(server, "GET", "/health?details=true")[0] == 404
        assert request(server, "POST", "/unknown", {})[0] == 404
        status, headers, body = request(server, "PUT", "/v1/rank", {})
        assert status == 405
        assert headers["Allow"] == "GET, POST"
        assert json.loads(body)["error"]["code"] == "method_not_allowed"


def test_excess_http_headers_use_the_bounded_json_error_shape() -> None:
    excessive_headers = b"".join(f"X-Test-{index}: x\r\n".encode() for index in range(101))
    with running_server(service()) as server:
        status, body = raw_request(
            server,
            b"GET /health HTTP/1.1\r\nHost: localhost\r\n" + excessive_headers + b"\r\n",
        )
    assert status == 431
    assert body["error"]["code"] == "invalid_http_request"


def test_concurrency_is_rejected_before_a_worker_is_started() -> None:
    limits = replace(ServingLimits(), max_concurrency=1)
    with running_server(service(limits=limits)) as server:
        assert server._capacity.acquire(blocking=False)  # exercise the accept-side hard cap
        try:
            status, headers, body = request(server, "GET", "/health")
        finally:
            server._capacity.release()
    assert status == 503
    assert headers["Retry-After"] == "1"
    assert json.loads(body)["error"]["code"] == "server_busy"


class ExplodingService(ClickRankService):
    def rank(
        self,
        user_id: str,
        *,
        as_of: datetime,
        k: int,
        candidate_ids: tuple[str, ...] | None,
        deadline_check: Callable[[], None] | None = None,
    ) -> RankResult:
        del user_id, as_of, k, candidate_ids, deadline_check
        raise RuntimeError("sensitive internal model failure")


def test_internal_errors_are_redacted() -> None:
    exploding = ExplodingService(fitted_model(), catalog(), history())
    with running_server(exploding) as server:
        status, _, body = request(server, "POST", "/v1/rank", rank_payload())
    assert status == 500
    assert json.loads(body)["error"]["code"] == "internal_error"
    assert b"sensitive" not in body


def test_completed_but_late_ranking_returns_gateway_timeout() -> None:
    with running_server(service()) as server:
        readings = iter((10.0, 21.0))
        server.monotonic = lambda: next(readings)
        status, _, body = request(server, "POST", "/v1/rank", rank_payload())
    assert status == 504
    assert json.loads(body)["error"]["code"] == "request_timeout"


def test_large_valid_history_is_cooperatively_cancelled_and_releases_worker() -> None:
    limits = replace(
        ServingLimits(),
        max_history_events=100_000,
        max_concurrency=1,
        request_timeout_seconds=0.001,
    )
    repeated_event = Event("bulk-user", "positive", EventKind.CLICK, AS_OF)
    large_service = service(events=[repeated_event] * 100_000, limits=limits)

    class ExpiringClock:
        calls = 0

        def __call__(self) -> float:
            self.calls += 1
            return 0.0 if self.calls <= 10 else 0.002

    clock = ExpiringClock()
    with running_server(large_service) as server:
        server.monotonic = clock
        status, _, body = request(server, "POST", "/v1/rank", rank_payload())

    assert status == 504
    assert json.loads(body)["error"]["code"] == "request_timeout"
    assert clock.calls == 11
    assert server._capacity.acquire(blocking=False)
    server._capacity.release()


def test_stalled_request_body_hits_socket_timeout() -> None:
    limits = replace(ServingLimits(), request_timeout_seconds=0.05)
    with running_server(service(limits=limits)) as server:
        connection = socket.create_connection(server.server_address, timeout=2)
        try:
            connection.sendall(
                b"POST /v1/rank HTTP/1.1\r\nHost: localhost\r\n"
                b"Content-Type: application/json\r\nContent-Length: 100\r\n"
                b"Connection: close\r\n\r\n{}"
            )
            response = b""
            while chunk := connection.recv(64 * 1024):
                response += chunk
        finally:
            connection.close()
    head, body = response.split(b"\r\n\r\n", maxsplit=1)
    assert b" 408 " in head.split(b"\r\n", maxsplit=1)[0]
    assert json.loads(body)["error"]["code"] == "request_timeout"


def test_response_size_has_a_final_hard_limit() -> None:
    articles = catalog()
    for index in range(20):
        articles.append(
            Article(
                f"extra-{index:02}",
                f"Extra {index}",
                "",
                ("useful",),
                "X",
                ORIGIN,
                0.5,
                0.5,
            )
        )
    limits = replace(
        ServingLimits(),
        max_response_bytes=512,
        max_k=24,
        max_candidates=24,
    )
    with running_server(service(articles=articles, limits=limits)) as server:
        status, _, body = request(
            server,
            "POST",
            "/v1/rank",
            rank_payload(k=24, candidate_ids=[article.id for article in articles]),
        )
    assert status == 500
    assert json.loads(body)["error"]["code"] == "response_too_large"


@pytest.mark.parametrize(
    "updates",
    [
        {"max_request_bytes": 0},
        {"max_response_bytes": 511},
        {"max_k": True},
        {"max_catalog_topic_cells": 0},
        {"max_concurrency": 0},
        {"max_event_weight": float("nan")},
        {"request_timeout_seconds": 0},
        {"request_timeout_seconds": float("inf")},
    ],
)
def test_serving_limits_validate_every_resource_class(updates: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ServingLimits(**updates)  # type: ignore[arg-type]


def test_snapshot_validates_catalog_history_and_isolated_model() -> None:
    model = fitted_model()
    snapshot = ClickRankService(model, catalog(), history())
    before = snapshot.rank("new", as_of=AS_OF, k=2, candidate_ids=None)
    model.fit(catalog(), list(reversed(history())), as_of=AS_OF)
    assert snapshot.rank("new", as_of=AS_OF, k=2, candidate_ids=None) == before
    with pytest.raises(FrozenInstanceError):
        snapshot._events = ()  # type: ignore[misc]

    with pytest.raises(ValueError, match="fitted"):
        ClickRankService(PointwiseLogisticRanker(), catalog(), history())
    with pytest.raises(ValueError, match="at least one"):
        ClickRankService(fitted_model(), [], history())
    with pytest.raises(ValueError, match="unique"):
        ClickRankService(fitted_model(), [catalog()[0], catalog()[0]], history())
    with pytest.raises(ValueError, match="unknown"):
        ClickRankService(
            fitted_model(),
            catalog(),
            [Event("u", "missing", EventKind.VIEW, AS_OF)],
        )
    with pytest.raises(ValueError, match="predates"):
        ClickRankService(
            fitted_model(),
            catalog(),
            [Event("u", "positive", EventKind.VIEW, ORIGIN - timedelta(seconds=1))],
        )
    with pytest.raises(ValueError, match="max_catalog"):
        ClickRankService(
            fitted_model(),
            catalog(),
            history(),
            limits=replace(ServingLimits(), max_catalog_articles=1),
        )
    with pytest.raises(ValueError, match="max_history"):
        ClickRankService(
            fitted_model(),
            catalog(),
            history(),
            limits=replace(ServingLimits(), max_history_events=1),
        )
    with pytest.raises(ValueError, match="max_topics_per_article"):
        ClickRankService(
            fitted_model(),
            [Article("a", "A", "", ("x", "y"), "source", ORIGIN)],
            (),
            limits=replace(ServingLimits(), max_topics_per_article=1),
        )
    with pytest.raises(ValueError, match="max_catalog_topic_cells"):
        ClickRankService(
            fitted_model(),
            [
                Article("a", "A", "", ("x",), "source", ORIGIN),
                Article("b", "B", "", ("y",), "source", ORIGIN),
            ],
            (),
            limits=replace(ServingLimits(), max_catalog_topic_cells=1),
        )
    with pytest.raises(ValueError, match="character"):
        ClickRankService(
            fitted_model(),
            [Article("a", "A", "", ("long",), "source", ORIGIN)],
            (),
            limits=replace(ServingLimits(), max_identifier_chars=2),
        )
    with pytest.raises(ValueError, match="max_event_weight"):
        ClickRankService(
            fitted_model(),
            [Article("a", "A", "", ("x",), "source", ORIGIN)],
            [Event("u", "a", EventKind.CLICK, ORIGIN, weight=2.0)],
            limits=replace(ServingLimits(), max_event_weight=1.0),
        )
    with pytest.raises(ValueError, match="ServingLimits"):
        ClickRankService(fitted_model(), catalog(), history(), limits=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_catalog"):
        ClickRankService(
            fitted_model(),
            itertools.repeat(catalog()[0]),
            (),
            limits=replace(ServingLimits(), max_catalog_articles=1),
        )
    with pytest.raises(ValueError, match="max_history"):
        ClickRankService(
            fitted_model(),
            catalog(),
            itertools.repeat(history()[0]),
            limits=replace(ServingLimits(), max_history_events=1),
        )


def article_record(article: Article) -> dict[str, object]:
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


def event_record(event: Event) -> dict[str, object]:
    return {
        "user_id": event.user_id,
        "article_id": event.article_id,
        "kind": event.kind.value,
        "occurred_at": event.occurred_at.isoformat(),
    }


def write_snapshot(directory: Path) -> tuple[Path, Path, Path]:
    model_path = directory / "model.json"
    article_path = directory / "articles.json"
    event_path = directory / "events.json"
    fitted_model().save(model_path)
    write_json(article_path, [article_record(article) for article in catalog()])
    write_json(event_path, [event_record(event) for event in history()])
    return model_path, article_path, event_path


def test_bounded_snapshot_loader(tmp_path: Path) -> None:
    model_path, article_path, event_path = write_snapshot(tmp_path)
    loaded = load_click_rank_service(model_path, article_path, event_path)
    assert loaded.metadata()["catalog_articles"] == 4
    with pytest.raises(ValueError, match="ServingLimits"):
        load_click_rank_service(
            model_path,
            article_path,
            event_path,
            limits=object(),  # type: ignore[arg-type]
        )

    for field, path in (
        ("max_model_bytes", model_path),
        ("max_catalog_bytes", article_path),
        ("max_history_bytes", event_path),
    ):
        with pytest.raises(ValueError, match="byte limit"):
            load_click_rank_service(
                model_path,
                article_path,
                event_path,
                limits=replace(ServingLimits(), **{field: path.stat().st_size - 1}),
            )


def test_token_resolution_and_binding_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOSAIC_TEST_TOKEN", "from-environment")
    assert resolve_bearer_token(None) is None
    assert resolve_bearer_token("MOSAIC_TEST_TOKEN") == "from-environment"
    with pytest.raises(ValueError, match="unset"):
        resolve_bearer_token("MISSING_TOKEN")
    monkeypatch.setenv("BAD_TOKEN", "line\nbreak")
    with pytest.raises(ValueError, match="invalid"):
        resolve_bearer_token("BAD_TOKEN")
    with pytest.raises(ValueError, match="visible ASCII"):
        create_rank_server(service(), port=0, bearer_token="has space")
    with pytest.raises(ValueError, match="ClickRankService"):
        create_rank_server(object(), port=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="boolean"):
        create_rank_server(service(), port=0, allow_nonloopback=1)  # type: ignore[arg-type]

    assert is_loopback_host("localhost")
    assert is_loopback_host("127.0.0.2")
    assert is_loopback_host("::1")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("example.invalid")
    with monkeypatch.context() as resolver:
        resolver.setattr(
            socket,
            "getaddrinfo",
            lambda *args, **kwargs: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.7", 0))
            ],
        )
        assert not is_loopback_host("localhost")
        with pytest.raises(ValueError, match="non-loopback"):
            create_rank_server(service(), host="localhost", port=0)
    with pytest.raises(ValueError, match="non-loopback"):
        create_rank_server(service(), host="0.0.0.0", port=0)
    with pytest.raises(ValueError, match="authentication"):
        create_rank_server(service(), host="0.0.0.0", port=0, allow_nonloopback=True)
    public_server = create_rank_server(
        service(),
        host="0.0.0.0",
        port=0,
        bearer_token="configured",
        allow_nonloopback=True,
    )
    public_server.server_close()
    with pytest.raises(ValueError, match="host"):
        create_rank_server(service(), host=" bad ", port=0)
    with pytest.raises(ValueError, match="port"):
        create_rank_server(service(), port=65_536)
    with pytest.raises(ValueError, match="bearer"):
        create_rank_server(service(), port=0, bearer_token="bad\nvalue")


def test_graceful_runner_closes_after_keyboard_interrupt() -> None:
    class InterruptingServer:
        def __init__(self) -> None:
            self.closed = False

        def serve_forever(self, *, poll_interval: float) -> None:
            assert poll_interval == 0.2
            raise KeyboardInterrupt

        def server_close(self) -> None:
            self.closed = True

    fake = InterruptingServer()
    serve_rank_server(fake)  # type: ignore[arg-type]
    assert fake.closed


def test_serve_cli_loads_snapshot_without_exposing_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_path, article_path, event_path = write_snapshot(tmp_path)
    observed: dict[str, object] = {}

    class FakeServer:
        server_address = ("127.0.0.1", 43210)

    def fake_create(
        rank_service: ClickRankService,
        *,
        host: str,
        port: int,
        bearer_token: str | None,
        allow_nonloopback: bool,
    ) -> Any:
        observed.update(
            service=rank_service,
            host=host,
            port=port,
            bearer_token=bearer_token,
            allow_nonloopback=allow_nonloopback,
        )
        return FakeServer()

    monkeypatch.setenv("MOSAIC_TEST_TOKEN", "do-not-print-this")
    monkeypatch.setattr(cli, "create_rank_server", fake_create)
    monkeypatch.setattr(cli, "serve_rank_server", lambda server: observed.update(server=server))
    assert (
        cli.main(
            [
                "serve-click-model",
                "--model",
                str(model_path),
                "--articles",
                str(article_path),
                "--events",
                str(event_path),
                "--port",
                "0",
                "--token-env",
                "MOSAIC_TEST_TOKEN",
                "--max-k",
                "5",
            ]
        )
        == 0
    )
    stderr = capsys.readouterr().err
    assert "127.0.0.1:43210" in stderr
    assert "authentication enabled" in stderr
    assert "do-not-print-this" not in stderr
    assert observed["bearer_token"] == "do-not-print-this"
    assert isinstance(observed["service"], ClickRankService)


def test_serve_cli_reports_startup_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_path, article_path, event_path = write_snapshot(tmp_path)
    result = cli.main(
        [
            "serve-click-model",
            "--model",
            str(model_path),
            "--articles",
            str(article_path),
            "--events",
            str(event_path),
            "--host",
            "0.0.0.0",
        ]
    )
    assert result == 2
    assert "non-loopback" in capsys.readouterr().err
