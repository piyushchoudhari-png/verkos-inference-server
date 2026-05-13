from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings

try:
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams  # noqa: F401 — re-exported for routes

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


async def create_engine(settings: Settings) -> Any | None:
    if not _VLLM_AVAILABLE:
        return None
    if not Path(settings.model.path).exists():
        return None

    engine_kwargs: dict[str, Any] = {
        "model": settings.model.path,
        "dtype": settings.model.dtype,
        "gpu_memory_utilization": settings.engine.gpu_memory_utilization,
        "tensor_parallel_size": settings.engine.tensor_parallel_size,
        "max_num_seqs": settings.model.max_num_seqs,
        **settings.engine.extra_kwargs(),
    }

    if settings.model.quantization != "none":
        engine_kwargs["quantization"] = settings.model.quantization
    if settings.model.max_model_len is not None:
        engine_kwargs["max_model_len"] = settings.model.max_model_len

    engine_args = AsyncEngineArgs(**engine_kwargs)
    return AsyncLLMEngine.from_engine_args(engine_args)
