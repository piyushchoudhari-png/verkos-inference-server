"""HuggingFace transformers inference backend.

Replaces the vLLM `AsyncLLMEngine` while preserving the call surface used by
`src.feed_runner`:

    async for output in engine.generate(inputs, sampling_params, request_id):
        output.outputs[0].text
        output.outputs[0].token_ids
        output.prompt_token_ids

Concurrency: HF `model.generate()` is not safe to call concurrently on a single
GPU. All `generate()` calls are serialized behind a single asyncio.Lock owned
by the engine instance — multiple feeds will queue up.
"""

import asyncio
import logging
import time
import warnings
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import Any

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import pynvml

from src.config import SimConfig
from src.metrics import IdleBaseline

log = logging.getLogger("src.engine")

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


@dataclass
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = 256
    n: int = 1


@dataclass
class CompletionOutput:
    text: str
    token_ids: list[int] = field(default_factory=list)


@dataclass
class GenerateOutput:
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]


_DTYPE_MAP: dict[str, Any] = {}


def _resolve_dtype(name: str) -> Any:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch not available")
    if not _DTYPE_MAP:
        _DTYPE_MAP.update(
            {
                "auto": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
        )
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {name!r}")
    return _DTYPE_MAP[name]


class HFEngine:
    """Async wrapper around HF transformers for VLMs.

    `generate()` matches the vLLM AsyncLLMEngine surface enough that
    `feed_runner._infer_frame_local` works unchanged.
    """

    def __init__(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor
        self._lock = asyncio.Lock()

    async def generate(
        self,
        inputs: dict[str, Any],
        sampling_params: SamplingParams,
        request_id: str,  # noqa: ARG002 — kept for vLLM API parity
    ) -> AsyncIterator[GenerateOutput]:
        if sampling_params.n != 1:
            raise NotImplementedError("HFEngine only supports n=1")

        async with self._lock:
            async for out in self._run(inputs, sampling_params):
                yield out

    async def _run(
        self,
        inputs: dict[str, Any],
        sampling_params: SamplingParams,
    ) -> AsyncIterator[GenerateOutput]:
        messages = inputs["messages"]

        proc_inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_ids = proc_inputs["input_ids"]
        prompt_token_ids: list[int] = input_ids[0].tolist()
        prompt_len = int(input_ids.shape[1])

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs: dict[str, Any] = dict(proc_inputs)
        gen_kwargs["streamer"] = streamer
        gen_kwargs["max_new_tokens"] = sampling_params.max_tokens
        if sampling_params.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = sampling_params.temperature
        else:
            gen_kwargs["do_sample"] = False

        collected_ids_holder: dict[str, Any] = {"out": None}

        def _run_generate() -> None:
            with torch.inference_mode():
                collected_ids_holder["out"] = self.model.generate(**gen_kwargs)

        thread = Thread(target=_run_generate, daemon=True)
        thread.start()

        text_parts: list[str] = []
        first_yielded = False
        loop = asyncio.get_running_loop()

        while True:
            chunk = await loop.run_in_executor(None, _next_chunk, streamer)
            if chunk is _SENTINEL_DONE:
                break
            text_parts.append(chunk)
            if not first_yielded and chunk:
                first_yielded = True
                yield GenerateOutput(
                    prompt_token_ids=prompt_token_ids,
                    outputs=[CompletionOutput(text="".join(text_parts), token_ids=[])],
                )

        await loop.run_in_executor(None, thread.join)

        full_ids_tensor = collected_ids_holder["out"]
        completion_token_ids: list[int] = []
        if full_ids_tensor is not None:
            completion_token_ids = full_ids_tensor[0, prompt_len:].tolist()

        final_text = "".join(text_parts)
        yield GenerateOutput(
            prompt_token_ids=prompt_token_ids,
            outputs=[CompletionOutput(text=final_text, token_ids=completion_token_ids)],
        )


_SENTINEL_DONE = object()


def _next_chunk(streamer: Any) -> Any:
    try:
        return next(streamer)
    except StopIteration:
        return _SENTINEL_DONE


def create_engine(config: SimConfig, stat_logger: Any | None = None) -> HFEngine:  # noqa: ARG001
    if not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers is not installed — cannot run simulator")
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is not installed — cannot run simulator")
    assert config.model_path is not None
    if not Path(config.model_path).exists():
        raise FileNotFoundError(f"Model path not found: {config.model_path}")

    _warn_vllm_only_fields(config)

    dtype = _resolve_dtype(config.dtype)
    log.info("loading processor from %s", config.model_path)
    processor = AutoProcessor.from_pretrained(config.model_path, trust_remote_code=True)  # type: ignore[no-untyped-call]
    log.info("loading model from %s (dtype=%s)", config.model_path, dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_path,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()  # type: ignore[no-untyped-call]
    log.info("model loaded on device=%s", model.device)
    return HFEngine(model=model, processor=processor)


def _warn_vllm_only_fields(config: SimConfig) -> None:
    noops: list[str] = []
    if config.gpu_memory_utilization != 0.90:
        noops.append(f"gpu_memory_utilization={config.gpu_memory_utilization}")
    if config.tensor_parallel_size != 1:
        noops.append(f"tensor_parallel_size={config.tensor_parallel_size}")
    if config.mm_processor_kwargs is not None:
        noops.append(f"mm_processor_kwargs={config.mm_processor_kwargs}")
    if config.limit_mm_per_prompt is not None:
        noops.append(f"limit_mm_per_prompt={config.limit_mm_per_prompt}")
    if config.max_model_len is not None:
        noops.append(f"max_model_len={config.max_model_len}")
    if noops:
        log.warning("vLLM-only config fields ignored under transformers backend: %s", ", ".join(noops))


async def get_tokenizer(engine: HFEngine, model_path: str) -> Any:  # noqa: ARG001
    return engine.processor


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
