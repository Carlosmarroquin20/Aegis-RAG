"""
Request body size limit, enforced while the body is being received.

The ingest route refuses files over its ceiling, but it can only do so after the
multipart parser has consumed the whole request and spooled it to disk — by then
the bytes have already been accepted. A reverse proxy normally caps this
(``client_max_body_size``), yet the API can also be reached without one (the
Terraform deployment publishes it directly), and a limit that exists only in a
neighbouring component is not a property of this service.

This middleware closes the gap at the ASGI layer:

  1. A declared ``Content-Length`` over the limit is refused immediately, before
     a single body byte is read.
  2. Bodies without a usable length (chunked transfer encoding) are counted as
     they stream in, and the request is aborted the moment the count crosses the
     limit — so a client cannot get around step 1 by simply not declaring a size.

It is written as a plain ASGI middleware rather than a ``BaseHTTPMiddleware``
because it has to wrap ``receive``; the latter buffers and re-exposes the body
and would defeat the point.

Placement: innermost. Authentication runs first, so an unauthenticated caller is
rejected without the server reading anything, and the 413 produced here still
travels back out through the logging and security-header layers.
"""

from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_HTTP_413_PAYLOAD_TOO_LARGE = 413


class _BodyTooLargeError(Exception):
    """Raised from the wrapped ``receive`` to unwind the application."""


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        if self._declared_length(scope) > self._max_bytes:
            await self._reject(scope, receive, send)
            return

        received = 0
        response_started = False

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_bytes:
                    raise _BodyTooLargeError
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self._app(scope, counting_receive, tracking_send)
        except _BodyTooLargeError:
            if response_started:
                # Headers are already on the wire; a second response is impossible.
                raise
            await self._reject(scope, receive, send)

    @staticmethod
    def _declared_length(scope: Scope) -> int:
        """Returns the Content-Length header as an int, or 0 if absent or malformed."""
        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    return int(value)
                except ValueError:
                    # Not this layer's job to police; the server or the streaming
                    # count below deals with a body that does not match.
                    return 0
        return 0

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        limit_mb = self._max_bytes / (1024 * 1024)
        response = JSONResponse(
            status_code=_HTTP_413_PAYLOAD_TOO_LARGE,
            content={"detail": f"Request body exceeds the {limit_mb:g} MB limit."},
            # Tell the client (and any proxy reusing the connection) not to keep
            # sending the remainder of a body we have decided not to read.
            headers={"Connection": "close"},
        )
        await response(scope, receive, send)
