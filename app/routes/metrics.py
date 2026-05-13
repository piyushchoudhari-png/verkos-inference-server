from __future__ import annotations

import time
from collections import defaultdict

from fastapi import APIRouter

from app.schemas.metrics import GpuInfo, MetricsResponse

try:
    import pynvml

    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False

router = APIRouter()

_start_time = time.time()
_nvml_initialized: bool = False
_request_counts: dict[str, int] = defaultdict(int)
_in_flight: int = 0
_latency_sum: float = 0.0
_latency_count: int = 0
_model_name: str | None = None


def record_request(status: str, latency: float) -> None:
    global _latency_sum, _latency_count
    _request_counts[status] += 1
    _latency_sum += latency
    _latency_count += 1


def inc_in_flight() -> None:
    global _in_flight
    _in_flight += 1


def dec_in_flight() -> None:
    global _in_flight
    _in_flight -= 1


def set_model(name: str | None) -> None:
    global _model_name
    _model_name = name


def _ensure_nvml() -> bool:
    global _nvml_initialized
    if _nvml_initialized:
        return True
    if not _NVML_AVAILABLE:
        return False
    try:
        pynvml.nvmlInit()
        _nvml_initialized = True
        return True
    except Exception:
        return False


def _gpu_info() -> list[GpuInfo]:
    if not _ensure_nvml():
        return []
    try:
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(GpuInfo(index=i, memory_used_bytes=mem.used, memory_total_bytes=mem.total))
        return gpus
    except Exception:
        return []


@router.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    avg = (_latency_sum / _latency_count) if _latency_count else None
    return MetricsResponse(
        uptime_seconds=round(time.time() - _start_time, 1),
        model=_model_name,
        requests=dict(_request_counts),
        in_flight=_in_flight,
        avg_latency_seconds=round(avg, 4) if avg is not None else None,
        gpus=_gpu_info(),
    )
