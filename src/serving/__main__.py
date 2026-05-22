"""``python -m src.serving --config gateway.yaml`` entrypoint.

Loads the gateway config, configures structured logging, and hands the FastAPI
app to uvicorn. See architecture doc §A.9.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import uvicorn
from pythonjsonlogger.json import JsonFormatter

from src.serving.app import create_app
from src.serving.config import load_gateway_config

DEFAULT_CONFIG = Path("gateway.yaml")


def _configure_logging(log_level: str) -> None:
    """Install a single JSON formatter on the root logger."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter(
            "{asctime} {name} {levelname} {message}",
            style="{",
            rename_fields={"asctime": "time", "levelname": "level"},
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(log_level.upper())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verkos-gateway",
        description="Run the Verkos OpenAI-compatible inference gateway.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to gateway YAML config. Default: {DEFAULT_CONFIG}",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _build_parser().parse_args(argv)
    config_path: Path = args.config.resolve()
    config = load_gateway_config(config_path)

    _configure_logging(config.server.log_level)
    app = create_app(config, config_path.parent)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
        # Disable uvicorn's own access log; we emit our own structured line in middleware.
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
