"""One-shot run orchestration shared by the single-run CLI (`__main__`) and the sweep runner."""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simulator.config import PromptConfig, SimConfig
from simulator.engine import measure_idle_vram
from simulator.engine_poller import SchedulerStatLogger, drain_loop
from simulator.feed_runner import run_feed
from simulator.gpu_poller import poll_gpu
from simulator.metrics import EngineSample, GpuSample, IdleBaseline, MetricSample
from simulator.reporter import write_results

log = logging.getLogger("simulator.runner")


@dataclass
class RunArtifacts:
    run_id: str
    samples: list[MetricSample] = field(default_factory=list)
    gpu_samples: list[GpuSample] = field(default_factory=list)
    engine_samples: list[EngineSample] = field(default_factory=list)
    baselines: list[IdleBaseline] = field(default_factory=list)
    paths: dict[str, Path] = field(default_factory=dict)


async def execute_run(
    config: SimConfig,
    run_id: str,
    engine: Any,
    tokenizer: Any,
    prompt: PromptConfig,
    stat_logger: SchedulerStatLogger | None,
) -> RunArtifacts:
    """Execute one simulation pass against an already-loaded engine."""
    model_name = Path(config.model_path).name

    baselines = measure_idle_vram(run_id)
    for b in baselines:
        log.info(
            "idle VRAM  gpu=%d  used=%.2f GB / %.2f GB",
            b.gpu_index,
            b.memory_used_bytes / 1024**3,
            b.memory_total_bytes / 1024**3,
        )

    metric_queue: asyncio.Queue[MetricSample] = asyncio.Queue()
    gpu_queue: asyncio.Queue[GpuSample] = asyncio.Queue()
    engine_queue: asyncio.Queue[EngineSample] = asyncio.Queue()
    stop_event = asyncio.Event()

    gpu_task = asyncio.create_task(poll_gpu(run_id, gpu_queue, config.gpu_poll_interval_s, stop_event))
    engine_drain_task: asyncio.Task[None] | None = None
    if stat_logger is not None:
        stat_logger._run_id = run_id  # noqa: SLF001 — internal field set by author
        engine_drain_task = asyncio.create_task(drain_loop(stat_logger, engine_queue, stop_event))

    feed_tasks = [
        asyncio.create_task(
            run_feed(
                feed_id=i,
                video_path=config.video_paths[i % len(config.video_paths)],
                engine=engine,
                tokenizer=tokenizer,
                prompt=prompt,
                config=config,
                metric_sink=metric_queue,
                run_id=run_id,
                model_name=model_name,
            )
        )
        for i in range(config.num_feeds)
    ]

    log.info("started %d feed(s)", config.num_feeds)
    await asyncio.gather(*feed_tasks)
    log.info("all feeds complete")

    stop_event.set()
    await gpu_task
    if engine_drain_task is not None:
        await engine_drain_task

    arts = RunArtifacts(run_id=run_id, baselines=baselines)
    while not metric_queue.empty():
        arts.samples.append(metric_queue.get_nowait())
    while not gpu_queue.empty():
        arts.gpu_samples.append(gpu_queue.get_nowait())
    while not engine_queue.empty():
        arts.engine_samples.append(engine_queue.get_nowait())

    log.info(
        "collected %d metric, %d GPU, %d engine samples",
        len(arts.samples),
        len(arts.gpu_samples),
        len(arts.engine_samples),
    )

    arts.paths = write_results(
        metric_samples=arts.samples,
        gpu_samples=arts.gpu_samples,
        engine_samples=arts.engine_samples,
        baselines=arts.baselines,
        config=config,
        run_id=run_id,
    )
    return arts
