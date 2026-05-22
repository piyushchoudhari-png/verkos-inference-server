"""NVML-based GPU sampling.

Kept from the original bench harness for reuse by the gateway metrics layer
(see ``src/serving/metrics.py`` once Phase 4 lands). ``read_samples`` is the
synchronous single-shot helper the Prometheus exporter will call; ``poll_gpu``
is the legacy async-queue producer, preserved verbatim aside from the dropped
``run_id``/torch dependencies.
"""

import asyncio
import time
import warnings
from dataclasses import dataclass
from typing import Any

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import pynvml


@dataclass(frozen=True)
class GpuSample:
    """One NVML reading for a single GPU at a point in time."""

    t_epoch: float
    gpu_index: int
    memory_used_bytes: int
    memory_total_bytes: int
    gpu_util_pct: int
    power_w: float | None
    temperature_c: int | None


_nvml_initialized: bool = False


def _ensure_nvml() -> bool:
    global _nvml_initialized
    if _nvml_initialized:
        return True
    try:
        pynvml.nvmlInit()
        _nvml_initialized = True
        return True
    except Exception:
        return False


def _power_w(handle: Any) -> float | None:
    try:
        return float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
    except Exception:
        return None


def _temperature_c(handle: Any) -> int | None:
    try:
        return int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        return None


def read_samples() -> list[GpuSample]:
    """Read one sample per visible GPU. Empty list if NVML is unavailable."""
    if not _ensure_nvml():
        return []
    try:
        device_count = pynvml.nvmlDeviceGetCount()
    except Exception:
        return []

    now = time.time()
    samples: list[GpuSample] = []
    for i in range(device_count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            samples.append(
                GpuSample(
                    t_epoch=now,
                    gpu_index=i,
                    memory_used_bytes=mem.used,
                    memory_total_bytes=mem.total,
                    gpu_util_pct=util.gpu,
                    power_w=_power_w(handle),
                    temperature_c=_temperature_c(handle),
                )
            )
        except Exception:
            continue
    return samples


async def poll_gpu(
    queue: asyncio.Queue[GpuSample],
    interval_s: float,
    stop_event: asyncio.Event,
) -> None:
    """Push samples to ``queue`` every ``interval_s`` until ``stop_event`` is set."""
    while not stop_event.is_set():
        for sample in read_samples():
            await queue.put(sample)
        await asyncio.sleep(interval_s)
