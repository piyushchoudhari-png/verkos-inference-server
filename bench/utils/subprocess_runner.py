from __future__ import annotations

import subprocess
from collections.abc import Generator
from pathlib import Path


class RunProcess:
    """Launch a profiling run subprocess and iterate its stdout lines."""

    returncode: int | None

    def __init__(self, config_path: str, run_id: str) -> None:
        self.returncode = None
        root = Path(config_path).parent.parent.parent  # bench/tmp → bench → project root
        self._proc = subprocess.Popen(
            ["python", "-m", "src", "--config", config_path, "--run-id", run_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(root),
        )

    def __iter__(self) -> Generator[str, None, None]:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            yield line.rstrip()
        self._proc.wait()
        self.returncode = self._proc.returncode

    def terminate(self) -> None:
        self._proc.terminate()
