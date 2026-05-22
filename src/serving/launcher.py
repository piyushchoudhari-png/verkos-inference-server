"""Shared YAML→argv translator for ``sglang.launch_server`` invocations.

Both the production gateway (:mod:`src.serving.manager`) and the debug helper
(``scripts/launch_worker.py``) need to translate a worker YAML config into the
exact argv that boots ``sglang.launch_server``. Keeping that translation in one
place ensures the debug helper and the gateway agree byte-for-byte on how a
given YAML is launched.
"""

from pathlib import Path
from typing import Any

REQUIRED_WORKER_KEYS = ("name", "host", "port")
REQUIRED_SGLANG_KEYS = ("model_path", "served_model_name")
SGLANG_MODULE = "sglang.launch_server"


def _flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def _sglang_args(sglang_block: dict[str, Any]) -> list[str]:
    """Translate the ``sglang`` block into ``launch_server`` CLI args."""
    args: list[str] = []
    for key, value in sglang_block.items():
        flag = _flag_name(key)
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, list | tuple):
            for item in value:
                args.extend([flag, str(item)])
            continue
        args.extend([flag, str(value)])
    return args


def _validate(config: dict[str, Any], path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    worker = config.get("worker")
    sglang_block = config.get("sglang")
    if not isinstance(worker, dict):
        raise ValueError(f"{path}: missing or non-dict `worker` block")
    if not isinstance(sglang_block, dict):
        raise ValueError(f"{path}: missing or non-dict `sglang` block")
    for key in REQUIRED_WORKER_KEYS:
        if key not in worker:
            raise ValueError(f"{path}: worker.{key} is required")
    for key in REQUIRED_SGLANG_KEYS:
        if key not in sglang_block:
            raise ValueError(f"{path}: sglang.{key} is required")
    if worker["host"] not in ("127.0.0.1", "localhost"):
        # Workers trust the gateway; binding off-loopback exposes an unauthenticated endpoint.
        raise ValueError(
            f"{path}: worker.host must be 127.0.0.1 (got {worker['host']!r}). "
            "Workers MUST stay on loopback — the gateway is the only ingress."
        )
    return worker, sglang_block


def build_command(config: dict[str, Any], path: Path, python_bin: str) -> list[str]:
    """Compose the full ``python -m sglang.launch_server ...`` argv from a worker YAML.

    Args:
        config: Parsed worker YAML (top-level mapping with ``worker`` and ``sglang`` blocks).
        path: Source path of the YAML; used in error messages.
        python_bin: Path to the Python interpreter that should host sglang.

    Returns:
        ``[python_bin, "-m", "sglang.launch_server", ...flags...]``. ``--host`` and
        ``--port`` are pulled up from the ``worker`` block (they are sglang flags
        in the YAML but live outside the ``sglang`` block for readability).
    """
    worker, sglang_block = _validate(config, path)
    forwarded: dict[str, Any] = {"host": worker["host"], "port": worker["port"], **sglang_block}
    return [python_bin, "-m", SGLANG_MODULE, *_sglang_args(forwarded)]
