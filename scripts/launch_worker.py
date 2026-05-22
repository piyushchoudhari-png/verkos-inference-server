#!/usr/bin/env python3
"""Debug-only helper: launch an sglang worker process from a worker YAML.

Production launches sglang as a child of the gateway (see
:mod:`src.serving.manager`). This script exists so an operator can boot sglang
standalone — to verify weights load or debug a model — without bringing up the
gateway.

The YAML→argv translation logic is shared with the gateway via
:mod:`src.serving.launcher`; this file is a thin CLI wrapper.

Run it via the installed console script or as a module so the package context
resolves correctly:

    uv run python -m scripts.launch_worker --config workers/chat-vlm.yaml
    uv run python -m scripts.launch_worker --config workers/chat-vlm.yaml --dry-run
    uv run verkos-launch-worker --config workers/chat-vlm.yaml --python /opt/venv/bin/python
"""

import argparse
import os
import shlex
import sys
from collections.abc import Iterable
from pathlib import Path

import yaml

from src.serving.launcher import build_command


def _format_argv(argv: Iterable[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="launch_worker.py",
        description="Translate a worker YAML config into an sglang.launch_server invocation and exec it.",
    )
    p.add_argument("--config", type=Path, required=True, help="Path to workers/<name>.yaml.")
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to exec. Default: the interpreter running this script.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and exit instead of execing.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _build_parser().parse_args(argv)
    path: Path = args.config
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        print(f"error: {path}: top-level must be a mapping", file=sys.stderr)
        return 2
    try:
        command = build_command(config, path, args.python)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(_format_argv(command))
        return 0

    # Replace this process so signals (SIGTERM from systemd, Ctrl-C) hit sglang directly.
    os.execvp(command[0], command)
    return 0  # unreachable; for mypy


if __name__ == "__main__":
    raise SystemExit(main())
