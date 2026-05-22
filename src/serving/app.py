"""FastAPI application factory for the Verkos inference gateway.

Wires together :mod:`.auth`, :mod:`.manager`, :mod:`.proxy`, :mod:`.metrics`
and :mod:`.errors` into one ASGI app. Endpoint surface is defined in
architecture doc §A.2; per-route behavior is documented inline below.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import signal
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from src.serving import errors
from src.serving.auth import KeyStore, parse_bearer
from src.serving.config import GatewayConfig, load_gateway_config
from src.serving.manager import (
    LoadTimeoutError,
    ModelDescriptor,
    ModelManager,
    SlotState,
    SpawnFailedError,
)
from src.serving.metrics import Metrics, run_gpu_poller
from src.serving.proxy import proxy_request

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "x-request-id"
MAX_REQUEST_BYTES = 50 * 1024 * 1024  # 50 MiB — multimodal payloads can be large.


def create_app(config: GatewayConfig, config_root: Path) -> FastAPI:
    """Build the FastAPI app from a parsed gateway config.

    Args:
        config: Validated ``gateway.yaml``.
        config_root: Directory the YAML was loaded from; relative ``config_ref``
            paths in :attr:`~GatewayConfig.models` are resolved against this.

    Returns:
        A FastAPI app ready to hand to uvicorn. The lifespan starts the GPU
        metrics poll and installs a SIGHUP handler that reloads the key store.
        The model manager owns the single sglang child process (if any).
    """
    key_store = KeyStore(config.auth.keys_file)
    key_store.reload()  # Fail-fast at startup if keys.json is missing or malformed.

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        async with httpx.AsyncClient() as client:
            metrics = Metrics()
            manager = ModelManager.from_config(config, client, config_root, metrics=metrics)
            stop_event = asyncio.Event()

            app.state.manager = manager
            app.state.metrics = metrics
            app.state.client = client
            app.state.key_store = key_store

            _install_sighup_reload(key_store)
            gpu_task = asyncio.create_task(
                run_gpu_poller(metrics, config.gpu.poll_interval_s, stop_event),
                name="gpu-poller",
            )
            try:
                yield
            finally:
                stop_event.set()
                gpu_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await gpu_task
                await manager.shutdown()

    app = FastAPI(title="verkos-inference-gateway", lifespan=lifespan)

    _register_middleware(app)
    _register_exception_handlers(app)
    _register_routes(app)
    return app


def _register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id_and_log(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or _new_request_id()
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "unhandled exception",
                extra={"request_id": request_id, "path": request.url.path, "method": request.method},
            )
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(elapsed_ms, 2),
                "client_key_name": getattr(request.state, "client_name", None),
                "model": getattr(request.state, "model", None),
            },
        )
        return response


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def on_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None) or _new_request_id()
        if isinstance(exc.detail, dict) and "type" in exc.detail:
            body = {"error": exc.detail}
        else:
            body = errors.envelope(
                message=str(exc.detail),
                error_type="invalid_request_error" if exc.status_code < 500 else "internal_error",
            )
        headers = dict(exc.headers or {})
        headers[REQUEST_ID_HEADER] = request_id
        return JSONResponse(status_code=exc.status_code, content=body, headers=headers)

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception) -> JSONResponse:
        return errors.error_response(
            request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="internal server error",
            error_type="internal_error",
        )


def _register_routes(app: FastAPI) -> None:
    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> Response:
        manager: ModelManager = request.app.state.manager
        snapshot = manager.current()
        if snapshot.state is SlotState.READY:
            return JSONResponse({"status": "ready", "model": snapshot.model_id})
        return JSONResponse(
            {"status": "not_ready", "slot_state": snapshot.state.value, "model": snapshot.model_id},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @app.get("/metrics")
    async def metrics_endpoint(request: Request) -> Response:
        metrics: Metrics = request.app.state.metrics
        body, content_type = metrics.render()
        return Response(content=body, media_type=content_type)

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict[str, Any]:
        _require_auth(request)
        manager: ModelManager = request.app.state.manager
        return {"object": "list", "data": [_model_payload(desc, manager) for desc in manager.catalog()]}

    @app.get("/v1/models/{model_id}")
    async def get_model(request: Request, model_id: str) -> dict[str, Any]:
        _require_auth(request)
        manager: ModelManager = request.app.state.manager
        desc = manager.get(model_id)
        if desc is None:
            raise errors.model_not_found(model_id)
        return _model_payload(desc, manager)

    @app.post("/v1/models/{model_id}/load")
    async def load_model(request: Request, model_id: str) -> dict[str, Any]:
        _require_auth(request)
        manager: ModelManager = request.app.state.manager
        if not manager.has(model_id):
            raise errors.model_not_found(model_id)
        request.state.model = model_id
        host, port = await _ensure_loaded(manager, model_id)
        return {"status": "ready", "model": model_id, "host": host, "port": port}

    @app.post("/v1/models/{model_id}/unload")
    async def unload_model(request: Request, model_id: str) -> dict[str, Any]:
        _require_auth(request)
        manager: ModelManager = request.app.state.manager
        if not manager.has(model_id):
            raise errors.model_not_found(model_id)
        request.state.model = model_id
        killed = await manager.unload_if(model_id)
        return {"status": "unloaded" if killed else "noop", "model": model_id}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _proxy_v1(request, "/v1/chat/completions")

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> Response:
        return await _proxy_v1(request, "/v1/embeddings")


async def _proxy_v1(request: Request, path: str) -> Response:
    """Common path for streaming/non-streaming OpenAI-style endpoints.

    1. Auth.
    2. Decode JSON; pull out ``model``.
    3. 404 if the model is not in the catalog.
    4. ``manager.ensure_loaded`` — blocks across a cold-start or implicit swap.
    5. Streaming pass-through to ``http://host:port`` for the loaded model.
    """
    _require_auth(request)
    body = await _read_body(request)
    payload = _decode_json(body)
    model_id = payload.get("model")
    if not isinstance(model_id, str) or not model_id:
        raise errors.invalid_request(message="`model` is required and must be a string", param="model")
    request.state.model = model_id

    manager: ModelManager = request.app.state.manager
    if not manager.has(model_id):
        raise errors.model_not_found(model_id)

    host, port = await _ensure_loaded(manager, model_id)
    upstream_url = f"http://{host}:{port}"
    streaming = bool(payload.get("stream", False))
    metrics: Metrics = request.app.state.metrics
    client_name = getattr(request.state, "client_name", "anonymous")
    inflight = metrics.inflight_requests.labels(route=path, model=model_id)
    inflight.inc()
    start = time.perf_counter()
    try:
        response = await proxy_request(
            request,
            upstream_url=upstream_url,
            upstream_path=path,
            body=body,
            streaming=streaming,
            client=request.app.state.client,
            request_id=request.state.request_id,
            client_name=client_name,
        )
    finally:
        inflight.dec()
        metrics.request_duration.labels(route=path, model=model_id).observe(time.perf_counter() - start)
        metrics.requests_total.labels(route=path, model=model_id, status="dispatched").inc()
    return response


async def _ensure_loaded(manager: ModelManager, model_id: str) -> tuple[str, int]:
    """Adapter: translate manager exceptions into HTTP error envelopes."""
    try:
        return await manager.ensure_loaded(model_id)
    except KeyError as exc:
        # Caller is expected to have called manager.has(); this is belt-and-braces.
        raise errors.model_not_found(model_id) from exc
    except LoadTimeoutError as exc:
        raise errors.timeout_error(str(exc)) from exc
    except SpawnFailedError as exc:
        raise errors.worker_error(str(exc)) from exc


def _require_auth(request: Request) -> None:
    """Resolve the bearer token to a client name and stash it on request state."""
    token = parse_bearer(request.headers.get("authorization"))
    if token is None:
        raise errors.authentication_error()
    store: KeyStore = request.app.state.key_store
    name = store.lookup(token)
    if name is None:
        raise errors.authentication_error()
    request.state.client_name = name


async def _read_body(request: Request) -> bytes:
    body = await request.body()
    if len(body) > MAX_REQUEST_BYTES:
        raise errors.GatewayError(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"request body exceeds {MAX_REQUEST_BYTES} bytes",
            "request_too_large",
        )
    return body


def _decode_json(body: bytes) -> dict[str, Any]:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise errors.invalid_request(message=f"invalid JSON body: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise errors.invalid_request(message="request body must be a JSON object")
    return parsed


def _model_payload(desc: ModelDescriptor, manager: ModelManager) -> dict[str, Any]:
    """Render one ``/v1/models`` row: OpenAI shape + capabilities + slot state."""
    return {
        "id": desc.id,
        "object": "model",
        "state": manager.state_for(desc.id),
        "capabilities": desc.capabilities.model_dump(),
        "last_error": manager.last_error_for(desc.id),
    }


def _new_request_id() -> str:
    return f"req_{uuid.uuid4().hex}"


def _install_sighup_reload(store: KeyStore) -> None:
    """Reload the key store on SIGHUP. No-op on platforms without SIGHUP."""
    if not hasattr(signal, "SIGHUP"):
        return
    loop = asyncio.get_running_loop()

    def _on_sighup() -> None:
        try:
            count = store.reload()
            logger.info("SIGHUP key reload complete", extra={"active_keys": count})
        except Exception:
            logger.exception("SIGHUP key reload failed")

    loop.add_signal_handler(signal.SIGHUP, _on_sighup)


def app_from_env() -> FastAPI:
    """Build the app using the gateway config path in ``VERKOS_GATEWAY_CONFIG``."""
    import os

    path = Path(os.environ.get("VERKOS_GATEWAY_CONFIG", "gateway.yaml")).resolve()
    return create_app(load_gateway_config(path), path.parent)
