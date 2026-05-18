import asyncio
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("src.feed_runner")

from src.config import PromptConfig, SimConfig
from src.frame_extractor import VideoFrameSource
from src.metrics import MetricSample

try:
    from vllm.sampling_params import SamplingParams

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


def _build_inputs(
    tokenizer: Any,
    prompt: PromptConfig,
    frame_index: int,
    image: Any,
    model_name: str,
) -> Any:
    user_text = prompt.user.format(frame_index=frame_index)
    messages = [
        {"role": "system", "content": prompt.system},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    prompt_text: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    has_vision = "<|vision_start|>" in prompt_text or "<image>" in prompt_text
    log.debug("vision tokens present: %s | prompt length: %d chars", has_vision, len(prompt_text))
    if not has_vision:
        log.warning("no vision tokens in prompt — image may not be embedded correctly for this model")
    return {
        "prompt": prompt_text,
        "multi_modal_data": {"image": image},
    }


async def _infer_frame_local(
    feed_id: int,
    frame_index: int,
    image: Any,
    engine: Any,
    tokenizer: Any,
    prompt: PromptConfig,
    config: SimConfig,
    metric_sink: asyncio.Queue[MetricSample],
    run_id: str,
    model_name: str,
) -> None:
    if not _VLLM_AVAILABLE:
        raise RuntimeError("vLLM not available")

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        n=1,
    )
    inputs = _build_inputs(tokenizer, prompt, frame_index, image, model_name)
    request_id = f"{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    resolution_w: int = getattr(image, "width", 0)
    resolution_h: int = getattr(image, "height", 0)

    t0_epoch = time.time()
    t0_mono = time.monotonic()
    ttft_s: float | None = None
    prompt_tokens = 0
    completion_tokens = 0
    status = "ok"
    error_msg: str | None = None
    output_text: str | None = None

    try:
        final_output: Any = None
        async for output in engine.generate(inputs, sampling_params, request_id):
            if ttft_s is None and output.outputs and output.outputs[0].text:
                ttft_s = time.monotonic() - t0_mono
            final_output = output

        if final_output is None:
            raise RuntimeError("Engine returned no output")

        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = sum(len(c.token_ids) for c in final_output.outputs)
        output_text = final_output.outputs[0].text if final_output.outputs else None
    except Exception as exc:
        status = "error"
        error_msg = str(exc)

    latency_s = time.monotonic() - t0_mono
    await metric_sink.put(
        MetricSample(
            run_id=run_id,
            feed_id=feed_id,
            frame_index=frame_index,
            t0_epoch=t0_epoch,
            latency_s=latency_s,
            ttft_s=ttft_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            resolution_w=resolution_w,
            resolution_h=resolution_h,
            model=model_name,
            inference_mode=config.inference_mode,
            status=status,
            error_msg=error_msg,
            output_text=output_text,
        )
    )


async def _infer_frame_openrouter(
    feed_id: int,
    frame_index: int,
    image: Any,
    prompt: PromptConfig,
    config: SimConfig,
    metric_sink: asyncio.Queue[MetricSample],
    run_id: str,
) -> None:
    from src.openrouter import infer_frame_openrouter

    assert config.openrouter is not None
    model_name = config.openrouter.model

    resolution_w: int = getattr(image, "width", 0)
    resolution_h: int = getattr(image, "height", 0)

    t0_epoch = time.time()
    t0_mono = time.monotonic()

    result = await infer_frame_openrouter(image, prompt, frame_index, config)

    latency_s = time.monotonic() - t0_mono
    await metric_sink.put(
        MetricSample(
            run_id=run_id,
            feed_id=feed_id,
            frame_index=frame_index,
            t0_epoch=t0_epoch,
            latency_s=latency_s,
            ttft_s=result.ttft_s,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            resolution_w=resolution_w,
            resolution_h=resolution_h,
            model=model_name,
            inference_mode=config.inference_mode,
            status=result.status,
            error_msg=result.error_msg,
            output_text=result.output_text,
        )
    )


async def run_feed(
    feed_id: int,
    video_path: str,
    engine: Any,
    tokenizer: Any,
    prompt: PromptConfig,
    config: SimConfig,
    metric_sink: asyncio.Queue[MetricSample],
    run_id: str,
    model_name: str,
) -> None:
    frames_dir = Path(config.output_dir) / run_id / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    source = VideoFrameSource(
        path=video_path,
        interval_s=config.frame_interval_s,
        resolution=config.resolution,
        downscale_factor=config.downscale_factor,
        max_frames=config.max_frames_per_feed,
    )

    inference_tasks: list[asyncio.Task[None]] = []

    async for frame_index, image in source.frames():
        frame_path = frames_dir / f"feed{feed_id:02d}_frame{frame_index:04d}.jpg"
        image.save(frame_path, format="JPEG", quality=90)

        if config.inference_mode == "openrouter":
            task: asyncio.Task[None] = asyncio.create_task(
                _infer_frame_openrouter(
                    feed_id=feed_id,
                    frame_index=frame_index,
                    image=image,
                    prompt=prompt,
                    config=config,
                    metric_sink=metric_sink,
                    run_id=run_id,
                )
            )
        else:
            task = asyncio.create_task(
                _infer_frame_local(
                    feed_id=feed_id,
                    frame_index=frame_index,
                    image=image,
                    engine=engine,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    config=config,
                    metric_sink=metric_sink,
                    run_id=run_id,
                    model_name=model_name,
                )
            )
        inference_tasks.append(task)

    if inference_tasks:
        await asyncio.gather(*inference_tasks, return_exceptions=True)
