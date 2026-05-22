"""Prometheus metrics + NVML GPU gauges.

The gateway maintains one global :class:`~prometheus_client.CollectorRegistry`
and a small background task that samples NVML every ``gpu.poll_interval_s``
seconds (architecture doc §A.8). The bench harness's reader, ``src/gpu_poller.py``,
is reused via :func:`read_samples`.
"""

import asyncio
import logging

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from src.gpu_poller import read_samples

logger = logging.getLogger(__name__)


class Metrics:
    """Prometheus collectors owned by the gateway.

    Held on ``app.state.metrics`` so handlers and the GPU poller can mutate it
    without touching module-level globals.
    """

    def __init__(self) -> None:
        """Build a fresh registry and register every gateway collector."""
        self.registry = CollectorRegistry()
        self.requests_total = Counter(
            "verkos_requests_total",
            "Total /v1/* requests handled by the gateway, labelled by route and status.",
            labelnames=("route", "model", "status"),
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "verkos_request_duration_seconds",
            "End-to-end gateway latency for /v1/* requests.",
            labelnames=("route", "model"),
            registry=self.registry,
        )
        self.tokens_total = Counter(
            "verkos_tokens_total",
            "Prompt and completion tokens observed in upstream responses.",
            labelnames=("model", "kind"),
            registry=self.registry,
        )
        self.inflight_requests = Gauge(
            "verkos_inflight_requests",
            "In-flight /v1/* requests currently being proxied.",
            labelnames=("route", "model"),
            registry=self.registry,
        )
        self.gpu_memory_used_bytes = Gauge(
            "verkos_gpu_memory_used_bytes",
            "NVML reported used GPU memory.",
            labelnames=("gpu",),
            registry=self.registry,
        )
        self.gpu_memory_total_bytes = Gauge(
            "verkos_gpu_memory_total_bytes",
            "NVML reported total GPU memory.",
            labelnames=("gpu",),
            registry=self.registry,
        )
        self.gpu_utilization_ratio = Gauge(
            "verkos_gpu_utilization_ratio",
            "NVML reported GPU utilization (0.0 - 1.0).",
            labelnames=("gpu",),
            registry=self.registry,
        )
        self.model_state = Gauge(
            "verkos_model_state",
            "Current state of each catalog model (1.0 for the active state, 0.0 for the others).",
            labelnames=("model", "state"),
            registry=self.registry,
        )
        self.model_loads_total = Counter(
            "verkos_model_loads_total",
            "Cold-start attempts per model, labelled by outcome.",
            labelnames=("model", "result"),
            registry=self.registry,
        )
        self.model_load_duration_seconds = Histogram(
            "verkos_model_load_duration_seconds",
            "Wall-clock time for a successful sglang cold start.",
            labelnames=("model",),
            registry=self.registry,
        )

    # ------------------------------------------------------------------
    # Helpers used by the model manager

    _SLOT_STATES = ("empty", "loading", "ready")

    def set_model_state(self, model_id: str, state: str) -> None:
        """Pin ``model_id`` to exactly one slot state in the gauge surface."""
        for candidate in self._SLOT_STATES:
            self.model_state.labels(model=model_id, state=candidate).set(1.0 if candidate == state else 0.0)

    def record_load(self, model_id: str, result: str, duration_s: float) -> None:
        """Record one cold-start outcome plus its duration."""
        self.model_loads_total.labels(model=model_id, result=result).inc()
        if result == "ok":
            self.model_load_duration_seconds.labels(model=model_id).observe(duration_s)

    def render(self) -> tuple[bytes, str]:
        """Return ``(body, content-type)`` for the ``/metrics`` endpoint."""
        return generate_latest(self.registry), CONTENT_TYPE_LATEST

    def sample_gpus(self) -> None:
        """Read NVML once and update the GPU gauges."""
        for sample in read_samples():
            label = str(sample.gpu_index)
            self.gpu_memory_used_bytes.labels(gpu=label).set(sample.memory_used_bytes)
            self.gpu_memory_total_bytes.labels(gpu=label).set(sample.memory_total_bytes)
            self.gpu_utilization_ratio.labels(gpu=label).set(sample.gpu_util_pct / 100.0)


async def run_gpu_poller(metrics: Metrics, interval_s: float, stop_event: asyncio.Event) -> None:
    """Background task: refresh GPU gauges every ``interval_s`` seconds."""
    while not stop_event.is_set():
        try:
            metrics.sample_gpus()
        except Exception:  # noqa: BLE001 — never let the loop die
            logger.exception("gpu sample failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except TimeoutError:
            continue
