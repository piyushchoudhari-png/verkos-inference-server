import asyncio
import time
import warnings
from pathlib import Path
from typing import Any

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams  # noqa: F401 — re-exported for feed_runner

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import pynvml

from src.config import SimConfig
from src.metrics import IdleBaseline

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


def create_engine(config: SimConfig, stat_logger: Any | None = None) -> Any:
    if not _VLLM_AVAILABLE:
        raise RuntimeError("vLLM is not installed — cannot run simulator without vLLM")
    if not Path(config.model_path).exists():
        raise FileNotFoundError(f"Model path not found: {config.model_path}")

    engine_kwargs: dict[str, Any] = {
        "model": config.model_path,
        "dtype": config.dtype,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "tensor_parallel_size": config.tensor_parallel_size,
    }
    if config.max_model_len is not None:
        engine_kwargs["max_model_len"] = config.max_model_len
    engine_args = AsyncEngineArgs(**engine_kwargs)

    if stat_logger is not None:
        # vLLM has changed this signature across versions (dict vs list). Try both;
        # if neither is accepted, fall through to the unmonitored default.
        for kwargs in ({"stat_loggers": {"sim": stat_logger}}, {"stat_loggers": [stat_logger]}):
            try:
                return AsyncLLMEngine.from_engine_args(engine_args, **kwargs)  # type: ignore[arg-type, unused-ignore]
            except TypeError:
                continue
    return AsyncLLMEngine.from_engine_args(engine_args)


async def get_tokenizer(engine: Any) -> Any:
    result = engine.get_tokenizer()
    if asyncio.iscoroutine(result):
        return await result
    return result


def measure_idle_vram(run_id: str) -> list[IdleBaseline]:
    if not _ensure_nvml():
        return []
    baselines: list[IdleBaseline] = []
    try:
        t = time.time()
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            baselines.append(
                IdleBaseline(
                    run_id=run_id,
                    t_epoch=t,
                    gpu_index=i,
                    memory_used_bytes=mem.used,
                    memory_total_bytes=mem.total,
                    torch_allocated_bytes=_torch_allocated(i),
                )
            )
    except Exception:
        pass
    return baselines
