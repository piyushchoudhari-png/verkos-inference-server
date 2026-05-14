import asyncio
import time
import warnings
from typing import Any

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import pynvml

from simulator.metrics import GpuSample

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


def _torch_allocated(gpu_index: int) -> int:
    if not _TORCH_AVAILABLE:
        return 0
    try:
        return int(torch.cuda.memory_allocated(device=gpu_index))
    except Exception:
        return 0


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


async def poll_gpu(
    run_id: str,
    queue: asyncio.Queue[GpuSample],
    interval_s: float,
    stop_event: asyncio.Event,
) -> None:
    if not _ensure_nvml():
        return

    try:
        device_count = pynvml.nvmlDeviceGetCount()
    except Exception:
        return

    while not stop_event.is_set():
        t = time.time()
        try:
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                await queue.put(
                    GpuSample(
                        run_id=run_id,
                        t_epoch=t,
                        gpu_index=i,
                        memory_used_bytes=mem.used,
                        memory_total_bytes=mem.total,
                        torch_allocated_bytes=_torch_allocated(i),
                        gpu_util_pct=util.gpu,
                        power_w=_power_w(handle),
                        temperature_c=_temperature_c(handle),
                    )
                )
        except Exception:
            pass

        await asyncio.sleep(interval_s)
