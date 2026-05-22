"""Streaming HTTP pass-through to an sglang worker.

The gateway does not rewrite request or response bodies (architecture doc
§A.3) — its job is auth + routing + streaming. This module is the routing
half: an ``httpx.AsyncClient`` opens a streaming request against the worker
and a FastAPI ``StreamingResponse`` relays bytes back to the client.

Client disconnect propagates as ``asyncio.CancelledError``, which closes the
upstream connection; sglang handles client-disconnect natively and cancels
generation (architecture doc §A.4).
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from src.serving import errors

logger = logging.getLogger(__name__)

# Hop-by-hop headers per RFC 7230 §6.1 — we never forward these in either direction.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        # We re-write these ourselves on the response side.
        "content-length",
        "content-encoding",
    }
)

# Request headers from the client that should NOT be forwarded upstream.
_DROP_REQUEST_HEADERS = frozenset({"host", "authorization", "content-length", "connection"})

PROXY_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)


def _filter_request_headers(headers: dict[str, str], request_id: str, client_name: str) -> dict[str, str]:
    """Strip auth + hop-by-hop headers, inject internal trace headers."""
    out = {k: v for k, v in headers.items() if k.lower() not in _DROP_REQUEST_HEADERS}
    out["x-request-id"] = request_id
    out["x-verkos-client"] = client_name
    return out


def _filter_response_headers(headers: httpx.Headers, request_id: str) -> dict[str, str]:
    """Drop hop-by-hop response headers, attach our request id."""
    out = {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}
    out["x-request-id"] = request_id
    return out


async def proxy_request(
    request: Request,
    *,
    upstream_url: str,
    upstream_path: str,
    body: bytes,
    streaming: bool,
    client: httpx.AsyncClient,
    request_id: str,
    client_name: str,
) -> Response:
    """Forward ``request`` to ``upstream_url + upstream_path`` and return the response.

    Args:
        request: The incoming FastAPI request (used for header forwarding only).
        upstream_url: Base URL of the worker, e.g. ``http://127.0.0.1:30001``.
        upstream_path: Path on the worker, e.g. ``/v1/chat/completions``.
        body: The already-read request body. The gateway buffers it because we may
            need to forward it as bytes to httpx; chat/embedding request bodies are
            small (no file uploads on v1).
        streaming: True iff the worker is expected to reply with SSE.
        client: A shared httpx.AsyncClient managed by the FastAPI lifespan.
        request_id: The gateway-assigned ``x-request-id`` for this request.
        client_name: The bearer-key owner, forwarded as ``x-verkos-client``.

    Returns:
        A FastAPI response — ``StreamingResponse`` for SSE, plain ``Response``
        for JSON. Worker errors are wrapped via :func:`_wrap_upstream_error` so
        the OpenAI envelope is preserved.
    """
    method = request.method
    upstream_headers = _filter_request_headers(dict(request.headers), request_id, client_name)
    target = upstream_url.rstrip("/") + upstream_path

    if streaming:
        return await _proxy_streaming(client, method, target, upstream_headers, body, request_id)
    return await _proxy_buffered(client, method, target, upstream_headers, body, request_id)


async def _proxy_buffered(
    client: httpx.AsyncClient,
    method: str,
    target: str,
    headers: dict[str, str],
    body: bytes,
    request_id: str,
) -> Response:
    try:
        upstream = await client.request(method, target, headers=headers, content=body, timeout=PROXY_TIMEOUT)
    except httpx.TimeoutException as exc:
        logger.warning("upstream timeout", extra={"target": target, "error": str(exc)})
        raise errors.timeout_error() from exc
    except httpx.HTTPError as exc:
        logger.warning("upstream connect failed", extra={"target": target, "error": str(exc)})
        raise errors.worker_error(f"upstream error: {exc}") from exc

    return _build_buffered_response(upstream, request_id)


def _build_buffered_response(upstream: httpx.Response, request_id: str) -> Response:
    headers = _filter_response_headers(upstream.headers, request_id)
    if upstream.status_code >= 400:
        return Response(
            content=_normalize_error_body(upstream.content, upstream.status_code),
            status_code=upstream.status_code,
            headers=headers,
            media_type="application/json",
        )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=headers,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


async def _proxy_streaming(
    client: httpx.AsyncClient,
    method: str,
    target: str,
    headers: dict[str, str],
    body: bytes,
    request_id: str,
) -> Response:
    # Open the upstream stream eagerly so we can detect non-200 / non-SSE before returning.
    try:
        req = client.build_request(method, target, headers=headers, content=body, timeout=PROXY_TIMEOUT)
        upstream = await client.send(req, stream=True)
    except httpx.TimeoutException as exc:
        logger.warning("upstream stream timeout", extra={"target": target, "error": str(exc)})
        raise errors.timeout_error() from exc
    except httpx.HTTPError as exc:
        logger.warning("upstream stream failed", extra={"target": target, "error": str(exc)})
        raise errors.worker_error(f"upstream error: {exc}") from exc

    if upstream.status_code >= 400:
        # Worker rejected the request before streaming started — buffer it and pass through.
        body_bytes = await upstream.aread()
        await upstream.aclose()
        return Response(
            content=_normalize_error_body(body_bytes, upstream.status_code),
            status_code=upstream.status_code,
            headers=_filter_response_headers(upstream.headers, request_id),
            media_type="application/json",
        )

    async def relay() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        relay(),
        status_code=upstream.status_code,
        headers=_filter_response_headers(upstream.headers, request_id),
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )


def _normalize_error_body(body: bytes, status_code: int) -> bytes:
    """If the worker didn't return the OpenAI envelope, wrap it ourselves."""
    try:
        parsed: Any = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
        return body
    message = body.decode("utf-8", errors="replace") if body else f"upstream returned HTTP {status_code}"
    return json.dumps(errors.envelope(message=message, error_type="worker_error")).encode("utf-8")
