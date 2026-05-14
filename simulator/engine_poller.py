"""Custom vLLM StatLogger that buffers EngineSample objects for the simulator.

Hooks into vLLM's supported StatLoggerBase API rather than poking at scheduler
internals — version-proof across vLLM releases.
"""

import asyncio
import threading
import time
from typing import Any

from simulator.metrics import EngineSample

try:
    from vllm.engine.metrics import StatLoggerBase

    _VLLM_STATS_AVAILABLE = True
except ImportError:
    _VLLM_STATS_AVAILABLE = False
    StatLoggerBase = object  # type: ignore[misc, assignment, unused-ignore]


class SchedulerStatLogger(StatLoggerBase):  # type: ignore[misc, valid-type, unused-ignore]
    """Buffers scheduler + KV cache stats emitted by vLLM each engine step."""

    def __init__(self, run_id: str, local_interval: float = 1.0) -> None:
        if _VLLM_STATS_AVAILABLE:
            super().__init__(local_interval=local_interval)
        self._run_id = run_id
        self._buf: list[EngineSample] = []
        self._lock = threading.Lock()
        self._cum_preemption: int = 0

    def log(self, stats: Any) -> None:  # noqa: A003 — name fixed by vLLM API
        try:
            preempt_iter = int(getattr(stats, "num_preemption_iter", 0) or 0)
            self._cum_preemption += preempt_iter
            sample = EngineSample(
                run_id=self._run_id,
                t_epoch=float(getattr(stats, "now", time.time())),
                num_running=int(getattr(stats, "num_running_sys", 0) or 0),
                num_waiting=int(getattr(stats, "num_waiting_sys", 0) or 0),
                num_swapped=int(getattr(stats, "num_swapped_sys", 0) or 0),
                gpu_kv_cache_usage_pct=float(getattr(stats, "gpu_cache_usage_sys", 0.0) or 0.0) * 100.0,
                cpu_kv_cache_usage_pct=float(getattr(stats, "cpu_cache_usage_sys", 0.0) or 0.0) * 100.0,
                num_preemption_total=self._cum_preemption,
            )
            with self._lock:
                self._buf.append(sample)
        except Exception:
            # Never let stats collection fail the engine
            pass

    def info(self, type_: str, obj: Any) -> None:
        # Required abstract method on newer vLLM StatLoggerBase; we don't use it.
        return None

    def drain(self) -> list[EngineSample]:
        with self._lock:
            out = list(self._buf)
            self._buf.clear()
            return out


async def drain_loop(
    logger: SchedulerStatLogger,
    queue: asyncio.Queue[EngineSample],
    stop_event: asyncio.Event,
    interval_s: float = 1.0,
) -> None:
    """Move samples from the logger buffer into the asyncio queue, until stopped."""
    while not stop_event.is_set():
        for sample in logger.drain():
            await queue.put(sample)
        await asyncio.sleep(interval_s)
    for sample in logger.drain():
        await queue.put(sample)
