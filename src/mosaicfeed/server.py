"""Bounded local HTTP inference for a frozen learned click-ranker snapshot."""

from __future__ import annotations

import hmac
import ipaddress
import json
import math
import os
import select
import socket
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import islice
from pathlib import Path
from typing import Any, cast

from mosaicfeed.io import (
    load_articles_bytes,
    load_events_bytes,
    load_json_text,
    parse_datetime,
)
from mosaicfeed.learning import (
    MODEL_FORMAT,
    MODEL_SCHEMA_VERSION,
    TEXT_MODEL_SCHEMA_VERSION,
    ClickPrediction,
    PointwiseLogisticRanker,
)
from mosaicfeed.models import Article, Event

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
MAX_CONTENT_LENGTH_DIGITS = 20


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a positive finite number") from error
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


@dataclass(frozen=True, slots=True)
class ServingLimits:
    """Resource ceilings and cooperative model-inference deadline."""

    max_request_bytes: int = 64 * 1024
    max_response_bytes: int = 2 * 1024 * 1024
    max_k: int = 100
    max_candidates: int = 10_000
    max_catalog_articles: int = 100_000
    max_history_events: int = 1_000_000
    max_model_bytes: int = 4 * 1024 * 1024
    max_catalog_bytes: int = 64 * 1024 * 1024
    max_history_bytes: int = 128 * 1024 * 1024
    max_topics_per_article: int = 1_024
    max_catalog_topic_cells: int = 2_000_000
    max_concurrency: int = 16
    request_timeout_seconds: float = 10.0
    max_identifier_chars: int = 256
    max_event_weight: float = 100.0

    def __post_init__(self) -> None:
        for name in (
            "max_request_bytes",
            "max_k",
            "max_candidates",
            "max_catalog_articles",
            "max_history_events",
            "max_model_bytes",
            "max_catalog_bytes",
            "max_history_bytes",
            "max_topics_per_article",
            "max_catalog_topic_cells",
            "max_concurrency",
            "max_identifier_chars",
        ):
            _positive_int(getattr(self, name), name)
        if _positive_int(self.max_response_bytes, "max_response_bytes") < 512:
            raise ValueError("max_response_bytes must be at least 512")
        _positive_number(self.request_timeout_seconds, "request_timeout_seconds")
        _positive_number(self.max_event_weight, "max_event_weight")


@dataclass(frozen=True, slots=True)
class RankResult:
    """A deterministic response produced from a frozen serving snapshot."""

    user_id: str
    as_of: datetime
    requested_k: int
    candidate_count: int
    predictions: tuple[ClickPrediction, ...]

    def __post_init__(self) -> None:
        _identifier(self.user_id, "user_id", maximum=None)
        if (
            not isinstance(self.as_of, datetime)
            or self.as_of.tzinfo is None
            or self.as_of.utcoffset() is None
        ):
            raise ValueError("as_of must be timezone-aware")
        _positive_int(self.requested_k, "requested_k")
        _positive_int(self.candidate_count, "candidate_count", allow_zero=True)
        if not isinstance(self.predictions, tuple):
            raise ValueError("predictions must be a tuple of ClickPrediction values")
        predictions = tuple(self.predictions)
        if any(not isinstance(prediction, ClickPrediction) for prediction in predictions):
            raise ValueError("predictions must be a tuple of ClickPrediction values")
        object.__setattr__(self, "predictions", predictions)
        if len(predictions) > min(self.requested_k, self.candidate_count):
            raise ValueError("predictions exceed requested_k or candidate_count")
        if [prediction.rank for prediction in predictions] != list(range(1, len(predictions) + 1)):
            raise ValueError("prediction ranks must be contiguous and start at one")
        ids = [prediction.article_id for prediction in predictions]
        if len(ids) != len(set(ids)):
            raise ValueError("predictions must have unique article ids")

    def to_dict(self) -> dict[str, object]:
        normalized_time = self.as_of.astimezone(UTC).isoformat().replace("+00:00", "Z")
        return {
            "as_of": normalized_time,
            "candidate_count": self.candidate_count,
            "object": "mosaicfeed.click_ranking",
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "requested_k": self.requested_k,
            "user_id": self.user_id,
        }


@dataclass(frozen=True, slots=True, init=False)
class ClickRankService:
    """Read-only model, catalog, and history snapshot used by all HTTP workers."""

    _article_ids: frozenset[str]
    _articles: tuple[Article, ...]
    _events: tuple[Event, ...]
    _limits: ServingLimits
    _model: PointwiseLogisticRanker = field(repr=False)

    def __init__(
        self,
        model: PointwiseLogisticRanker,
        articles: Iterable[Article],
        events: Iterable[Event],
        *,
        limits: ServingLimits | None = None,
    ) -> None:
        if limits is not None and not isinstance(limits, ServingLimits):
            raise ValueError("limits must be ServingLimits")
        active_limits = limits or ServingLimits()
        if not isinstance(model, PointwiseLogisticRanker) or not model.is_fitted:
            raise ValueError("model must be a fitted PointwiseLogisticRanker")
        article_values = tuple(islice(articles, active_limits.max_catalog_articles + 1))
        event_values = tuple(islice(events, active_limits.max_history_events + 1))
        if not article_values or any(
            not isinstance(article, Article) for article in article_values
        ):
            raise ValueError("catalog must contain at least one Article")
        if any(not isinstance(event, Event) for event in event_values):
            raise ValueError("history must contain only Event values")
        if len(article_values) > active_limits.max_catalog_articles:
            raise ValueError("catalog exceeds max_catalog_articles")
        if len(event_values) > active_limits.max_history_events:
            raise ValueError("history exceeds max_history_events")
        article_map = {article.id: article for article in article_values}
        if len(article_map) != len(article_values):
            raise ValueError("catalog article ids must be unique")
        catalog_topic_cells = 0
        for article in article_values:
            _identifier(
                article.id,
                "catalog article id",
                maximum=active_limits.max_identifier_chars,
            )
            if len(article.topics) > active_limits.max_topics_per_article:
                raise ValueError("catalog article exceeds max_topics_per_article")
            catalog_topic_cells += len(article.topics)
            if catalog_topic_cells > active_limits.max_catalog_topic_cells:
                raise ValueError("catalog exceeds max_catalog_topic_cells")
            for topic in article.topics:
                _identifier(topic, "catalog topic", maximum=active_limits.max_identifier_chars)
        for event in event_values:
            _identifier(
                event.user_id, "history user id", maximum=active_limits.max_identifier_chars
            )
            _identifier(
                event.article_id,
                "history article id",
                maximum=active_limits.max_identifier_chars,
            )
            if event.weight > active_limits.max_event_weight:
                raise ValueError("history event weight exceeds max_event_weight")
            referenced_article = article_map.get(event.article_id)
            if referenced_article is None:
                raise ValueError(f"history references unknown article: {event.article_id}")
            if event.occurred_at < referenced_article.published_at:
                raise ValueError(
                    f"history event for {event.article_id} predates article publication"
                )

        # Rehydrate a private copy. Callers can refit their original object without
        # changing the snapshot currently serving requests.
        object.__setattr__(self, "_model", PointwiseLogisticRanker.from_state(model.to_state()))
        object.__setattr__(self, "_articles", article_values)
        object.__setattr__(self, "_events", event_values)
        object.__setattr__(self, "_limits", active_limits)
        object.__setattr__(self, "_article_ids", frozenset(article_map))

    @property
    def limits(self) -> ServingLimits:
        return self._limits

    def metadata(self) -> dict[str, object]:
        """Return provenance and ceilings without exposing weights, content, or user ids."""

        return {
            "catalog_articles": len(self._articles),
            "history_events": len(self._events),
            "limits": {
                "max_candidates": self._limits.max_candidates,
                "max_catalog_articles": self._limits.max_catalog_articles,
                "max_concurrency": self._limits.max_concurrency,
                "max_history_events": self._limits.max_history_events,
                "max_identifier_chars": self._limits.max_identifier_chars,
                "max_k": self._limits.max_k,
                "max_request_bytes": self._limits.max_request_bytes,
                "max_response_bytes": self._limits.max_response_bytes,
                "max_topics_per_article": self._limits.max_topics_per_article,
                "max_catalog_topic_cells": self._limits.max_catalog_topic_cells,
                "max_event_weight": self._limits.max_event_weight,
                "request_timeout_seconds": self._limits.request_timeout_seconds,
            },
            "model": {
                "format": MODEL_FORMAT,
                "schema_version": (
                    TEXT_MODEL_SCHEMA_VERSION
                    if self._model.text_encoder is not None
                    else MODEL_SCHEMA_VERSION
                ),
                "training_examples_sha256": self._model.training_sha256,
            },
            "object": "mosaicfeed.inference_metadata",
            "time_semantics": "history at or before as_of is eligible; future items are excluded",
        }

    def rank(
        self,
        user_id: str,
        *,
        as_of: datetime,
        k: int,
        candidate_ids: tuple[str, ...] | None,
        deadline_check: Callable[[], None] | None = None,
    ) -> RankResult:
        """Rank one bounded request without mutating the serving snapshot."""

        if deadline_check is not None:
            deadline_check()
        normalized_user = _identifier(
            user_id,
            "user_id",
            maximum=self._limits.max_identifier_chars,
        )
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= self._limits.max_k:
            raise ValueError("k is outside the configured serving range")
        clock = parse_datetime(as_of.isoformat(), "as_of") if isinstance(as_of, datetime) else None
        if clock is None:
            raise ValueError("as_of must be a timezone-aware datetime")
        if candidate_ids is None:
            count = len(self._articles)
            if count > self._limits.max_candidates:
                raise ValueError(
                    "catalog exceeds per-request candidate limit; provide candidate_ids"
                )
            selected_ids = None
        else:
            if len(candidate_ids) > self._limits.max_candidates:
                raise ValueError("candidate_ids exceeds max_candidates")
            normalized_values: list[str] = []
            for index, article_id in enumerate(candidate_ids):
                if deadline_check is not None and index % 64 == 0:
                    deadline_check()
                normalized_values.append(
                    _identifier(
                        article_id,
                        "candidate_id",
                        maximum=self._limits.max_identifier_chars,
                    )
                )
            normalized_ids = tuple(normalized_values)
            if len(normalized_ids) != len(set(normalized_ids)):
                raise ValueError("candidate_ids must be unique")
            if not set(normalized_ids) <= self._article_ids:
                raise ValueError("candidate_ids contain unknown articles")
            selected_ids = normalized_ids
            count = len(normalized_ids)
        predictions = self._model.rank_for_user(
            normalized_user,
            self._articles,
            self._events,
            as_of=clock,
            k=k,
            candidate_ids=selected_ids,
            deadline_check=deadline_check,
        )
        return RankResult(
            user_id=normalized_user,
            as_of=clock,
            requested_k=k,
            candidate_count=count,
            predictions=predictions,
        )


def _identifier(value: object, name: str, *, maximum: int | None) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace")
    if (maximum is not None and len(value) > maximum) or any(
        not character.isprintable() for character in value
    ):
        raise ValueError(f"{name} exceeds its character or control-character limit")
    return value


def _bounded_file_bytes(path: str | Path, maximum: int, name: str) -> tuple[Path, bytes]:
    source = Path(path)
    with source.open("rb") as handle:
        data = handle.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f"{name} exceeds its configured byte limit")
    return source, data


def load_click_rank_service(
    model_path: str | Path,
    articles_path: str | Path,
    events_path: str | Path,
    *,
    limits: ServingLimits | None = None,
) -> ClickRankService:
    """Load and validate one immutable serving snapshot from bounded local files."""

    if limits is not None and not isinstance(limits, ServingLimits):
        raise ValueError("limits must be ServingLimits")
    active_limits = limits or ServingLimits()
    _, model_bytes = _bounded_file_bytes(model_path, active_limits.max_model_bytes, "model file")
    catalog_file, catalog_bytes = _bounded_file_bytes(
        articles_path, active_limits.max_catalog_bytes, "catalog file"
    )
    history_file, history_bytes = _bounded_file_bytes(
        events_path, active_limits.max_history_bytes, "history file"
    )
    try:
        model_payload = load_json_text(model_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("model file must be strict UTF-8 JSON") from error
    if not isinstance(model_payload, Mapping):
        raise ValueError("pointwise model file must contain a JSON object")
    service = ClickRankService(
        PointwiseLogisticRanker.from_state(model_payload),
        load_articles_bytes(catalog_bytes, source_name=str(catalog_file)),
        load_events_bytes(history_bytes, source_name=str(history_file)),
        limits=active_limits,
    )
    return service


def resolve_bearer_token(environment_variable: str | None) -> str | None:
    """Read authentication only from a caller-named environment variable."""

    if environment_variable is None:
        return None
    name = _identifier(environment_variable, "token environment variable", maximum=256)
    value = os.environ.get(name)
    if value is None or not value:
        raise ValueError(f"token environment variable is unset or empty: {name}")
    try:
        _validate_bearer_token(value)
    except ValueError as error:
        raise ValueError(f"token environment variable contains an invalid value: {name}") from error
    return value


def _validate_bearer_token(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 4096
        or any(not 33 <= ord(character) <= 126 for character in value)
    ):
        raise ValueError("bearer token must be 1-4096 visible ASCII characters")
    return value


def _resolved_loopback(host: str) -> str | None:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if host.casefold() != "localhost":
            return None
        try:
            resolved = {
                cast(str, item[4][0]).split("%", maxsplit=1)[0]
                for item in socket.getaddrinfo(host, 0, type=socket.SOCK_STREAM)
            }
        except OSError:
            return None
        if not resolved or any(not ipaddress.ip_address(value).is_loopback for value in resolved):
            return None
        return sorted(resolved, key=lambda value: (":" in value, value))[0]
    return str(address) if address.is_loopback else None


def is_loopback_host(host: str) -> bool:
    """Require an IP literal or a currently all-loopback localhost resolution."""

    return _resolved_loopback(host) is not None


class _ClientError(Exception):
    def __init__(self, status: HTTPStatus, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class _RankingDeadlineExceeded(Exception):
    """Internal cooperative-cancellation signal for ranking work."""


class RankHTTPServer(ThreadingHTTPServer):
    """Thread-capped HTTP server whose workers share one immutable snapshot."""

    allow_reuse_address = True
    daemon_threads = False
    block_on_close = True

    def __init__(
        self,
        address: tuple[str, int],
        service: ClickRankService,
        *,
        bearer_token: str | None = None,
    ) -> None:
        self.rank_service = service
        self.bearer_token = None if bearer_token is None else bearer_token.encode("utf-8")
        self.monotonic: Callable[[], float] = time.monotonic
        self._capacity = threading.BoundedSemaphore(service.limits.max_concurrency)
        super().__init__(address, RankRequestHandler)

    def get_request(self) -> tuple[socket.socket, Any]:
        request, address = super().get_request()
        request.settimeout(self.rank_service.limits.request_timeout_seconds)
        return request, address

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._capacity.acquire(blocking=False):
            try:
                _write_busy_response(
                    cast(socket.socket, request),
                    drain_limit=self.rank_service.limits.max_request_bytes + 64 * 1024,
                )
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._capacity.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._capacity.release()


class _IPv6RankHTTPServer(RankHTTPServer):
    address_family = socket.AF_INET6


def _write_busy_response(request: socket.socket, *, drain_limit: int) -> None:
    # A Windows TCP close can discard the response when unread request bytes remain.
    # Give the just-accepted connection one short bounded window to deliver headers,
    # then drain only bytes already in the kernel buffer.
    remaining = drain_limit
    wait_seconds = 0.05
    while remaining > 0 and select.select([request], [], [], wait_seconds)[0]:
        wait_seconds = 0.0
        try:
            chunk = request.recv(min(remaining, 64 * 1024))
        except OSError:
            break
        if not chunk:
            break
        remaining -= len(chunk)
    body = b'{"error":{"code":"server_busy","message":"request concurrency limit reached"}}'
    headers = (
        b"HTTP/1.1 503 Service Unavailable\r\n"
        b"Content-Type: application/json; charset=utf-8\r\n"
        b"Cache-Control: no-store\r\n"
        b"X-Content-Type-Options: nosniff\r\n"
        b"Connection: close\r\n"
        b"Retry-After: 1\r\n" + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    )
    with suppress(OSError):
        request.sendall(headers + body)
        request.shutdown(socket.SHUT_WR)
        # Let a well-behaved client consume the short response and close its read
        # side before the server releases the socket. This prevents a Windows RST
        # without turning overload handling into an unbounded wait.
        if select.select([request], [], [], 0.05)[0]:
            request.recv(64 * 1024)


class RankRequestHandler(BaseHTTPRequestHandler):
    """Strict JSON protocol adapter; request content is never logged."""

    protocol_version = "HTTP/1.1"
    server_version = "MosaicFeed"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self._body_bytes_read = 0

    @property
    def rank_server(self) -> RankHTTPServer:
        return cast(RankHTTPServer, self.server)

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def do_GET(self) -> None:
        try:
            if self.path == "/health":
                self._send_json({"status": "ok"})
                return
            if self.path == "/metadata":
                self._authorize()
                self._send_json(self.rank_server.rank_service.metadata())
                return
            raise _ClientError(HTTPStatus.NOT_FOUND, "not_found", "endpoint not found")
        except _ClientError as error:
            self._send_client_error(error)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            self._send_internal_error()

    def do_POST(self) -> None:
        started = self.rank_server.monotonic()
        expires_at = started + self.rank_server.rank_service.limits.request_timeout_seconds

        def check_ranking_deadline() -> None:
            if self.rank_server.monotonic() > expires_at:
                raise _RankingDeadlineExceeded

        try:
            if self.path != "/v1/rank":
                raise _ClientError(HTTPStatus.NOT_FOUND, "not_found", "endpoint not found")
            self._authorize()
            request = self._read_json_object()
            user_id, as_of, k, candidate_ids = self._parse_rank_request(request)
            try:
                result = self.rank_server.rank_service.rank(
                    user_id,
                    as_of=as_of,
                    k=k,
                    candidate_ids=candidate_ids,
                    deadline_check=check_ranking_deadline,
                )
            except _RankingDeadlineExceeded as error:
                raise _ClientError(
                    HTTPStatus.GATEWAY_TIMEOUT,
                    "request_timeout",
                    "ranking exceeded the request time limit",
                ) from error
            except ValueError as error:
                raise _ClientError(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "invalid_ranking_request",
                    "request does not satisfy the catalog or ranking constraints",
                ) from error
            try:
                check_ranking_deadline()
            except _RankingDeadlineExceeded as error:
                raise _ClientError(
                    HTTPStatus.GATEWAY_TIMEOUT,
                    "request_timeout",
                    "ranking exceeded the request time limit",
                ) from error
            self._send_json(result.to_dict())
        except _ClientError as error:
            self._send_client_error(error)
        except TimeoutError:
            self._send_client_error(
                _ClientError(
                    HTTPStatus.REQUEST_TIMEOUT,
                    "request_timeout",
                    "request body was not received within the time limit",
                )
            )
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            self._send_internal_error()

    def do_PUT(self) -> None:
        self._method_not_allowed()

    do_PATCH = do_PUT
    do_DELETE = do_PUT
    do_OPTIONS = do_PUT
    do_HEAD = do_PUT

    def _method_not_allowed(self) -> None:
        self._send_client_error(
            _ClientError(
                HTTPStatus.METHOD_NOT_ALLOWED,
                "method_not_allowed",
                "method not allowed",
            ),
            extra_headers={"Allow": "GET, POST"},
        )

    def _authorize(self) -> None:
        expected = self.rank_server.bearer_token
        if expected is None:
            return
        values = self.headers.get_all("Authorization")
        if values is None or len(values) != 1:
            raise _ClientError(HTTPStatus.UNAUTHORIZED, "unauthorized", "bearer token required")
        supplied = values[0].encode("utf-8", errors="replace")
        if not hmac.compare_digest(supplied, b"Bearer " + expected):
            raise _ClientError(HTTPStatus.UNAUTHORIZED, "unauthorized", "bearer token required")

    def _read_json_object(self) -> Mapping[str, object]:
        if self.headers.get("Transfer-Encoding") is not None:
            raise _ClientError(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "transfer encoding is not supported",
            )
        if self.headers.get("Content-Encoding", "identity").casefold() != "identity":
            raise _ClientError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                "content encoding is not supported",
            )
        content_types = self.headers.get_all("Content-Type")
        if content_types is None or len(content_types) != 1:
            raise _ClientError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                "Content-Type must be application/json",
            )
        content_type_parts = [part.strip().casefold() for part in content_types[0].split(";")]
        if content_type_parts[0] != "application/json" or content_type_parts[1:] not in (
            [],
            ["charset=utf-8"],
        ):
            raise _ClientError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                "Content-Type must be application/json with optional charset=utf-8",
            )
        lengths = self.headers.get_all("Content-Length")
        if lengths is None:
            raise _ClientError(
                HTTPStatus.LENGTH_REQUIRED,
                "length_required",
                "Content-Length is required",
            )
        if (
            len(lengths) != 1
            or not lengths[0].isascii()
            or not lengths[0].isdigit()
            or len(lengths[0]) > MAX_CONTENT_LENGTH_DIGITS
        ):
            raise _ClientError(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "Content-Length must be one non-negative decimal integer",
            )
        length = int(lengths[0])
        if length > self.rank_server.rank_service.limits.max_request_bytes:
            raise _ClientError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request_too_large",
                "request body exceeds max_request_bytes",
            )
        body = self.rfile.read(length)
        self._body_bytes_read += len(body)
        if len(body) != length:
            raise _ClientError(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "request body ended before Content-Length",
            )
        try:
            payload = load_json_text(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise _ClientError(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "request body must be strict UTF-8 JSON",
            ) from error
        if not isinstance(payload, dict):
            raise _ClientError(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "request body must be a JSON object",
            )
        return cast(Mapping[str, object], payload)

    def _parse_rank_request(
        self, payload: Mapping[str, object]
    ) -> tuple[str, datetime, int, tuple[str, ...] | None]:
        allowed = {"user_id", "as_of", "k", "candidate_ids"}
        required = {"user_id", "as_of"}
        if set(payload) - allowed or required - set(payload):
            raise _ClientError(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "request has missing or unknown fields",
            )
        limits = self.rank_server.rank_service.limits
        try:
            user_id = _identifier(
                payload["user_id"],
                "user_id",
                maximum=limits.max_identifier_chars,
            )
            as_of = parse_datetime(payload["as_of"], "as_of")
        except ValueError as error:
            raise _ClientError(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "user_id or as_of is invalid",
            ) from error
        raw_k = payload.get("k", 10)
        if isinstance(raw_k, bool) or not isinstance(raw_k, int) or not 1 <= raw_k <= limits.max_k:
            raise _ClientError(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "k must be an integer within the configured limit",
            )
        if "candidate_ids" not in payload:
            candidate_ids = None
        else:
            raw_candidates = payload["candidate_ids"]
            if not isinstance(raw_candidates, list):
                raise _ClientError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_request",
                    "candidate_ids must be a JSON array",
                )
            if len(raw_candidates) > limits.max_candidates:
                raise _ClientError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_request",
                    "candidate_ids exceeds the configured limit",
                )
            try:
                candidate_ids = tuple(
                    _identifier(value, "candidate_id", maximum=limits.max_identifier_chars)
                    for value in raw_candidates
                )
            except ValueError as error:
                raise _ClientError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_request",
                    "candidate_ids contains an invalid identifier",
                ) from error
            if len(candidate_ids) != len(set(candidate_ids)):
                raise _ClientError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_request",
                    "candidate_ids must not contain duplicates",
                )
        return user_id, as_of, raw_k, candidate_ids

    def _send_client_error(
        self,
        error: _ClientError,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        headers = dict(extra_headers or {})
        if error.status == HTTPStatus.UNAUTHORIZED:
            headers["WWW-Authenticate"] = 'Bearer realm="mosaicfeed"'
        self._discard_bounded_request_body()
        try:
            self._send_json(
                {"error": {"code": error.code, "message": error.message}},
                status=error.status,
                extra_headers=headers,
            )
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def send_error(
        self,
        code: int,
        message: str | None = None,
        explain: str | None = None,
    ) -> None:
        del message, explain
        try:
            status = HTTPStatus(code)
        except ValueError:
            status = HTTPStatus.BAD_REQUEST
        self._send_client_error(
            _ClientError(status, "invalid_http_request", "HTTP request is malformed")
        )

    def _discard_bounded_request_body(self) -> None:
        headers = getattr(self, "headers", None)
        if headers is None:
            return
        lengths = headers.get_all("Content-Length")
        if (
            lengths is None
            or len(lengths) != 1
            or not lengths[0].isascii()
            or not lengths[0].isdigit()
            or len(lengths[0]) > MAX_CONTENT_LENGTH_DIGITS
        ):
            return
        declared = int(lengths[0])
        remaining = declared - self._body_bytes_read
        if remaining <= 0:
            return
        # This is response-delivery hygiene for already-buffered small bodies,
        # especially on Windows. It is not permission to consume an attacker-sized
        # declaration: both wait and bytes are capped independently.
        drain_limit = self.rank_server.rank_service.limits.max_request_bytes + 64 * 1024
        original_timeout = self.connection.gettimeout()
        try:
            self.connection.settimeout(0.05)
            discarded = self.rfile.read(min(remaining, drain_limit))
            self._body_bytes_read += len(discarded)
        except (OSError, ValueError):
            return
        finally:
            with suppress(OSError):
                self.connection.settimeout(original_timeout)

    def _send_internal_error(self) -> None:
        try:
            self._send_json(
                {
                    "error": {
                        "code": "internal_error",
                        "message": "the ranking request could not be completed",
                    }
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _send_json(
        self,
        payload: Mapping[str, object],
        *,
        status: HTTPStatus = HTTPStatus.OK,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        body = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(body) > self.rank_server.rank_service.limits.max_response_bytes:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            body = (
                b'{"error":{"code":"response_too_large",'
                b'"message":"response exceeds max_response_bytes"}}'
            )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True


def create_rank_server(
    service: ClickRankService,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    bearer_token: str | None = None,
    allow_nonloopback: bool = False,
) -> RankHTTPServer:
    """Validate binding policy, then create (but do not start) the HTTP server."""

    if not isinstance(service, ClickRankService):
        raise ValueError("service must be ClickRankService")
    if not isinstance(allow_nonloopback, bool):
        raise ValueError("allow_nonloopback must be a boolean")
    if not isinstance(host, str) or not host or host != host.strip():
        raise ValueError("host must be a non-empty string without surrounding whitespace")
    _positive_int(port, "port", allow_zero=True)
    if port > 65_535:
        raise ValueError("port must be at most 65535")
    if bearer_token is not None:
        _validate_bearer_token(bearer_token)
    loopback_address = _resolved_loopback(host)
    if loopback_address is None:
        if not allow_nonloopback:
            raise ValueError("non-loopback binding requires --allow-nonloopback")
        if bearer_token is None:
            raise ValueError("non-loopback binding requires bearer-token authentication")
    bind_host = host if loopback_address is None else loopback_address
    try:
        address = ipaddress.ip_address(bind_host)
    except ValueError:
        address = None
    server_class = (
        _IPv6RankHTTPServer if address is not None and address.version == 6 else RankHTTPServer
    )
    return server_class((bind_host, port), service, bearer_token=bearer_token)


def serve_rank_server(server: RankHTTPServer) -> None:
    """Run until interrupted, always closing the listening socket and workers."""

    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
