"""Single-slot sglang subprocess supervisor.

The gateway owns at most one sglang process at a time. This module implements
the state machine that decides when to spawn, swap, or kill that process:

    EMPTY ──ensure_loaded(X)──► LOADING(X) ──/health 200──► READY(X)
                                    │                           │
                                    │ spawn/timeout failed      │ ensure_loaded(Y)
                                    ▼                           ▼
                                  EMPTY                     UNLOADING(X) ──► EMPTY ──► LOADING(Y) ──► READY(Y)

A single :class:`asyncio.Lock` guards all transitions, so concurrent inference
requests serialize behind a swap and same-model callers coalesce on the same
spawn. See architecture doc §B.5.
"""

import asyncio
import contextlib
import logging
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import httpx
import yaml

from src.serving.config import (
    GatewayConfig,
    ManagerConfig,
    WorkerCapabilities,
    WorkerFileConfig,
)
from src.serving.launcher import build_command
from src.serving.metrics import Metrics

logger = logging.getLogger(__name__)


class SlotState(StrEnum):
    """The four states the single sglang slot can occupy."""

    EMPTY = "empty"
    LOADING = "loading"
    READY = "ready"
    UNLOADING = "unloading"


@dataclass(frozen=True)
class ModelDescriptor:
    """Static catalog entry — everything we need to spawn a model on demand."""

    id: str
    host: str
    port: int
    capabilities: WorkerCapabilities
    raw_config: dict[str, Any]
    config_path: Path


@dataclass(frozen=True)
class SlotSnapshot:
    """Read-only view of the slot for ``/v1/models``, ``/readyz``, and metrics."""

    state: SlotState
    model_id: str | None
    port: int | None
    pid: int | None
    last_error: str | None


class ManagerError(Exception):
    """Base class for manager-side failures (spawn, timeout, etc.)."""


class SpawnFailedError(ManagerError):
    """sglang exited non-zero during cold-start, or argv composition failed."""


class LoadTimeoutError(ManagerError):
    """``/health`` did not return 200 within ``manager.load_timeout_s``."""


class ModelManager:
    """Owns the single sglang subprocess and its lifecycle transitions."""

    def __init__(
        self,
        catalog: dict[str, ModelDescriptor],
        client: httpx.AsyncClient,
        manager_config: ManagerConfig,
        metrics: Metrics | None = None,
    ) -> None:
        """Construct an already-resolved manager; prefer :meth:`from_config`."""
        self._catalog = catalog
        self._client = client
        self._config = manager_config
        self._metrics = metrics
        self._lock = asyncio.Lock()
        self._state: SlotState = SlotState.EMPTY
        self._loaded_id: str | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._last_error: dict[str, str] = {}
        # Pre-seed all catalog gauges to EMPTY so dashboards have a stable surface.
        if metrics is not None:
            for model_id in catalog:
                metrics.set_model_state(model_id, SlotState.EMPTY.value)

    # ------------------------------------------------------------------
    # Construction

    @classmethod
    def from_config(
        cls,
        config: GatewayConfig,
        client: httpx.AsyncClient,
        config_root: Path,
        metrics: Metrics | None = None,
    ) -> Self:
        """Build a manager by resolving every catalog entry's worker YAML.

        Args:
            config: Parsed ``gateway.yaml``.
            client: Long-lived httpx client (used for ``/health`` polling).
            config_root: Directory the gateway config came from; ``config_ref``
                paths are resolved relative to it.
            metrics: Optional Prometheus registry that gets per-model gauges.
        """
        catalog: dict[str, ModelDescriptor] = {}
        for entry in config.models:
            path = entry.config_ref if entry.config_ref.is_absolute() else (config_root / entry.config_ref).resolve()
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"{path}: top-level must be a mapping")
            worker_file = WorkerFileConfig.model_validate(raw)
            catalog[entry.id] = ModelDescriptor(
                id=entry.id,
                host=worker_file.worker.host,
                port=worker_file.worker.port,
                capabilities=worker_file.capabilities,
                raw_config=raw,
                config_path=path,
            )
        return cls(catalog, client, config.manager, metrics)

    # ------------------------------------------------------------------
    # Read-only introspection

    def catalog(self) -> list[ModelDescriptor]:
        """Return every catalog entry; order matches ``gateway.yaml::models``."""
        return list(self._catalog.values())

    def has(self, model_id: str) -> bool:
        """True iff ``model_id`` is in the catalog."""
        return model_id in self._catalog

    def get(self, model_id: str) -> ModelDescriptor | None:
        """Look up a catalog entry by id, or ``None`` if absent."""
        return self._catalog.get(model_id)

    def current(self) -> SlotSnapshot:
        """Snapshot the slot's current state (cheap; no lock acquisition)."""
        port: int | None = None
        if self._loaded_id is not None:
            desc = self._catalog.get(self._loaded_id)
            port = desc.port if desc else None
        pid = self._process.pid if self._process is not None else None
        last_error = self._last_error.get(self._loaded_id) if self._loaded_id else None
        return SlotSnapshot(
            state=self._state,
            model_id=self._loaded_id,
            port=port,
            pid=pid,
            last_error=last_error,
        )

    def state_for(self, model_id: str) -> str:
        """Return the gauge-friendly state label for one catalog entry."""
        if self._loaded_id == model_id:
            return self._state.value
        return SlotState.EMPTY.value

    def last_error_for(self, model_id: str) -> str | None:
        """The last spawn error recorded for ``model_id`` (or ``None``)."""
        return self._last_error.get(model_id)

    # ------------------------------------------------------------------
    # Transitions

    async def ensure_loaded(self, model_id: str) -> tuple[str, int]:
        """Block until the slot is ``READY(model_id)`` and return ``(host, port)``.

        Behavior:
            * Fast path — slot is already READY for ``model_id``: return immediately.
            * Cold start — slot is EMPTY: spawn.
            * Swap — slot is READY for a different model: hard-kill the loaded
              model, then spawn ``model_id``. **Any in-flight request to the
              previous model has its TCP connection severed.**

        Raises:
            KeyError: ``model_id`` is not in the catalog. Callers should check
                :meth:`has` first and map to 404.
            SpawnFailedError: sglang exited during cold start.
            LoadTimeoutError: ``/health`` never returned 200 within the timeout.
        """
        desc = self._catalog.get(model_id)
        if desc is None:
            raise KeyError(model_id)

        async with self._lock:
            if (
                self._state is SlotState.READY
                and self._loaded_id == model_id
                and self._process is not None
                and self._process.returncode is None
            ):
                return desc.host, desc.port

            # READY but process died externally — clear before reloading.
            if self._process is not None and self._process.returncode is not None:
                logger.warning(
                    "loaded process exited externally",
                    extra={"model": self._loaded_id, "returncode": self._process.returncode},
                )
                self._process = None
                self._state = SlotState.EMPTY
                self._loaded_id = None

            # Implicit swap: a different model is loaded.
            if self._state is SlotState.READY and self._loaded_id != model_id:
                await self._kill_current()

            await self._spawn(desc)
            return desc.host, desc.port

    async def unload_if(self, model_id: str) -> bool:
        """SIGKILL the slot if it currently holds ``model_id``; no-op otherwise.

        Returns:
            True iff a process was killed. False if the slot was EMPTY or held
            a different model.
        """
        async with self._lock:
            if self._state is not SlotState.READY or self._loaded_id != model_id:
                return False
            await self._kill_current()
            return True

    async def shutdown(self) -> None:
        """Kill any live child and release the slot. Called from the lifespan teardown."""
        async with self._lock:
            if self._process is not None:
                await self._kill_current()

    # ------------------------------------------------------------------
    # Internals — must only be called while holding ``self._lock``

    async def _spawn(self, desc: ModelDescriptor) -> None:
        """Transition EMPTY → LOADING(desc.id) → READY(desc.id), or unwind on failure."""
        prev_id = self._loaded_id
        self._state = SlotState.LOADING
        self._loaded_id = desc.id
        if self._metrics is not None:
            if prev_id is not None and prev_id != desc.id:
                self._metrics.set_model_state(prev_id, SlotState.EMPTY.value)
            self._metrics.set_model_state(desc.id, SlotState.LOADING.value)

        python_bin = self._config.spawn_python or sys.executable
        start = time.perf_counter()
        logger.info("load_started", extra={"model": desc.id, "port": desc.port})

        try:
            argv = build_command(desc.raw_config, desc.config_path, python_bin)
        except ValueError as exc:
            self._fail_spawn(desc.id, f"argv composition failed: {exc}", start)
            raise SpawnFailedError(str(exc)) from exc

        try:
            process = await asyncio.create_subprocess_exec(*argv)
        except OSError as exc:
            self._fail_spawn(desc.id, f"create_subprocess_exec failed: {exc}", start)
            raise SpawnFailedError(f"failed to spawn sglang for {desc.id!r}: {exc}") from exc

        self._process = process
        try:
            await self._wait_until_healthy(desc, process, start)
        except (SpawnFailedError, LoadTimeoutError):
            # _wait_until_healthy already killed the process and reset state.
            raise

        duration = time.perf_counter() - start
        self._state = SlotState.READY
        self._last_error.pop(desc.id, None)
        logger.info(
            "load_complete",
            extra={"model": desc.id, "pid": process.pid, "duration_ms": round(duration * 1000.0, 2)},
        )
        if self._metrics is not None:
            self._metrics.set_model_state(desc.id, SlotState.READY.value)
            self._metrics.record_load(desc.id, "ok", duration)

    async def _wait_until_healthy(
        self,
        desc: ModelDescriptor,
        process: asyncio.subprocess.Process,
        start: float,
    ) -> None:
        deadline = start + self._config.load_timeout_s
        url = f"http://{desc.host}:{desc.port}/health"
        while True:
            if process.returncode is not None:
                msg = f"sglang exited with code {process.returncode} during load"
                self._fail_spawn(desc.id, msg, start, process_already_gone=True)
                raise SpawnFailedError(msg)
            if time.perf_counter() >= deadline:
                msg = f"sglang for {desc.id!r} did not become ready in {self._config.load_timeout_s}s"
                await self._kill_process(process)
                self._fail_spawn(desc.id, msg, start, process_already_gone=True)
                raise LoadTimeoutError(msg)
            try:
                resp = await self._client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(self._config.health_poll_interval_s)

    def _fail_spawn(
        self,
        model_id: str,
        reason: str,
        start: float,
        *,
        process_already_gone: bool = False,
    ) -> None:
        """Reset slot to EMPTY after a failed load and emit metrics."""
        logger.warning("load_failed", extra={"model": model_id, "error": reason})
        self._last_error[model_id] = reason
        if not process_already_gone:
            self._process = None
        else:
            self._process = None
        self._state = SlotState.EMPTY
        self._loaded_id = None
        if self._metrics is not None:
            self._metrics.set_model_state(model_id, SlotState.EMPTY.value)
            self._metrics.record_load(model_id, "error", time.perf_counter() - start)

    async def _kill_current(self) -> None:
        """Transition READY(id) → UNLOADING(id) → EMPTY. Idempotent."""
        if self._process is None:
            self._state = SlotState.EMPTY
            self._loaded_id = None
            return
        unloading_id = self._loaded_id
        self._state = SlotState.UNLOADING
        if self._metrics is not None and unloading_id is not None:
            self._metrics.set_model_state(unloading_id, SlotState.EMPTY.value)
        logger.info("unload_started", extra={"model": unloading_id, "pid": self._process.pid})
        await self._kill_process(self._process)
        logger.info("unload_complete", extra={"model": unloading_id})
        self._process = None
        self._state = SlotState.EMPTY
        self._loaded_id = None

    @staticmethod
    async def _kill_process(process: asyncio.subprocess.Process) -> None:
        """SIGKILL ``process`` and await its exit. Tolerates already-dead processes."""
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        try:
            await process.wait()
        except Exception:  # noqa: BLE001 — never let teardown raise
            logger.exception("process.wait failed during kill")
