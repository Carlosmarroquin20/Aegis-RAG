"""
Unit tests for BodySizeLimitMiddleware.

The middleware exists so that the amount of data the server is willing to accept
does not depend on a reverse proxy being present. The cases that matter are the
ones a proxy-less attacker would try: a huge declared length, and no declared
length at all.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from aegis.interface.api.middleware.body_limit import (
    BodySizeLimitMiddleware,
    _BodyTooLargeError,
)

_LIMIT = 64


def _app() -> tuple[FastAPI, list[int]]:
    """An app with one route that reads the whole body and records its size."""
    sizes: list[int] = []
    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=_LIMIT)

    @app.post("/echo")
    async def echo(request: Request) -> dict[str, int]:
        body = await request.body()
        sizes.append(len(body))
        return {"received": len(body)}

    @app.get("/ping")
    async def ping() -> dict[str, bool]:
        return {"pong": True}

    return app, sizes


def _chunks(total: int, size: int = 16) -> Iterator[bytes]:
    """A body with no Content-Length: httpx sends it with chunked encoding."""
    for _ in range(total // size):
        yield b"x" * size


class TestDeclaredLength:
    def test_body_within_the_limit_is_accepted(self) -> None:
        app, sizes = _app()
        response = TestClient(app).post("/echo", content=b"x" * _LIMIT)
        assert response.status_code == 200
        assert sizes == [_LIMIT]

    def test_body_over_the_limit_is_refused(self) -> None:
        app, sizes = _app()
        response = TestClient(app).post("/echo", content=b"x" * (_LIMIT + 1))
        assert response.status_code == 413
        assert "limit" in response.json()["detail"]
        assert sizes == []  # the route never ran

    def test_refusal_closes_the_connection(self) -> None:
        app, _ = _app()
        response = TestClient(app).post("/echo", content=b"x" * (_LIMIT + 1))
        assert response.headers["connection"] == "close"

    async def test_oversized_declaration_is_refused_without_reading_the_body(self) -> None:
        """The point of checking the header: no body byte should be pulled at all."""
        app, _ = _app()
        receive_calls = 0
        sent: list[Message] = []

        async def receive() -> Message:
            nonlocal receive_calls
            receive_calls += 1
            return {"type": "http.request", "body": b"x" * 1024, "more_body": True}

        async def send(message: Message) -> None:
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/echo",
            "raw_path": b"/echo",
            "query_string": b"",
            "headers": [(b"content-length", str(10**12).encode())],
        }
        await BodySizeLimitMiddleware(app, max_bytes=_LIMIT)(scope, receive, send)

        assert receive_calls == 0
        assert sent[0]["status"] == 413

    def test_requests_without_a_body_are_unaffected(self) -> None:
        app, _ = _app()
        assert TestClient(app).get("/ping").status_code == 200


class TestStreamedBodyWithoutLength:
    """Chunked uploads carry no Content-Length, so only counting can stop them."""

    def test_streamed_body_within_the_limit_is_accepted(self) -> None:
        app, sizes = _app()
        response = TestClient(app).post("/echo", content=_chunks(_LIMIT))
        assert response.status_code == 200
        assert sizes == [_LIMIT]

    def test_streamed_body_over_the_limit_is_cut_off(self) -> None:
        app, sizes = _app()
        response = TestClient(app).post("/echo", content=_chunks(_LIMIT * 8))
        assert response.status_code == 413
        assert sizes == []

    async def test_reading_stops_as_soon_as_the_limit_is_crossed(self) -> None:
        """An endless stream must be abandoned, not drained to find out how big it is."""
        app, _ = _app()
        receive_calls = 0
        sent: list[Message] = []

        async def receive() -> Message:
            nonlocal receive_calls
            receive_calls += 1
            return {"type": "http.request", "body": b"x" * 16, "more_body": True}

        async def send(message: Message) -> None:
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/echo",
            "raw_path": b"/echo",
            "query_string": b"",
            "headers": [(b"transfer-encoding", b"chunked")],
        }
        await BodySizeLimitMiddleware(app, max_bytes=_LIMIT)(scope, receive, send)

        # 64-byte limit, 16-byte chunks: the fifth chunk crosses it.
        assert receive_calls == 5
        assert sent[0]["status"] == 413


class TestEdgeCases:
    def test_malformed_content_length_falls_back_to_counting(self) -> None:
        app, sizes = _app()
        client = TestClient(app)
        response = client.post(
            "/echo", content=_chunks(_LIMIT * 4), headers={"Content-Length": "not-a-number"}
        )
        assert response.status_code == 413
        assert sizes == []

    async def test_non_http_scopes_pass_through_untouched(self) -> None:
        seen: list[str] = []

        async def inner(scope: Scope, receive: Receive, send: Send) -> None:
            seen.append(scope["type"])

        async def receive() -> Message:
            return {"type": "lifespan.startup"}

        async def send(_message: Message) -> None:
            return None

        await BodySizeLimitMiddleware(inner, max_bytes=1)({"type": "lifespan"}, receive, send)
        assert seen == ["lifespan"]

    async def test_overflow_after_the_response_started_is_not_swallowed(self) -> None:
        """Once headers are sent a 413 is impossible; the error must surface instead."""

        async def inner(scope: Scope, receive: Receive, send: Send) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await receive()

        async def receive() -> Message:
            return {"type": "http.request", "body": b"x" * 10, "more_body": False}

        async def send(_message: Message) -> None:
            return None

        middleware = BodySizeLimitMiddleware(inner, max_bytes=1)
        with pytest.raises(_BodyTooLargeError):
            await middleware({"type": "http", "headers": []}, receive, send)
