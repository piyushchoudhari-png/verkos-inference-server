"""No-op stat logger stub.

vLLM exposed scheduler state (KV cache %, running/waiting queues) via
StatLoggerBase. The transformers backend has no equivalent scheduler, so this
module preserves the class/function signatures used by `src.runner` and
`src.__main__` while always producing zero samples.
"""

import asyncio
import threading
from typing import Any

from src.metrics import EngineSample


class SchedulerStatLogger:
    """No-op stub; preserves the interface used by the runner."""

    def __init__(self, run_id: str, local_interval: float = 1.0) -> None:  # noqa: ARG002
        self._run_id = run_id
        self._buf: list[EngineSample] = []
        self._lock = threading.Lock()

    def log(self, stats: Any) -> None:  # noqa: A003, ARG002
        return None

    def info(self, type_: str, obj: Any) -> None:  # noqa: ARG002
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
    while not stop_event.is_set():
        for sample in logger.drain():
            await queue.put(sample)
        await asyncio.sleep(interval_s)
    for sample in logger.drain():
        await queue.put(sample)
