"""Pydantic models for the gateway and worker YAML configs.

The gateway boots from ``gateway.yaml`` (top-level :class:`GatewayConfig`) and,
for each catalog entry, opens the referenced ``workers/<name>.yaml``
(:class:`WorkerFileConfig`) so the capability + sglang block is available to
the model manager. The ``sglang`` block is consumed by the manager (shared
helper :mod:`src.serving.launcher`); the gateway forwards it verbatim.

See architecture doc §B.3 (per-model YAML) and §B.4 (gateway YAML).
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class WorkerCapabilities(BaseModel):
    """Capability metadata advertised under ``capabilities`` in a worker YAML."""

    model_config = ConfigDict(extra="allow")

    chat: bool = False
    vision: bool = False
    embeddings: bool = False
    max_context_tokens: int | None = None
    max_image_pixels: int | None = None


class WorkerNetwork(BaseModel):
    """The ``worker`` block of a worker YAML — name and bind address."""

    name: str
    host: str
    port: int


class WorkerFileConfig(BaseModel):
    """Top-level schema of ``workers/<name>.yaml``.

    The ``sglang`` block is intentionally untyped: any flag sglang accepts is
    forwarded verbatim by the manager, so we don't want this schema to drift
    as sglang grows new flags.
    """

    model_config = ConfigDict(extra="allow")

    worker: WorkerNetwork
    sglang: dict[str, Any]
    capabilities: WorkerCapabilities = Field(default_factory=WorkerCapabilities)


class AuthConfig(BaseModel):
    """Where the gateway looks up bearer keys."""

    keys_file: Path


class ModelEntry(BaseModel):
    """One bootable model in the catalog.

    ``id`` is the value clients put in the OpenAI ``model`` field. ``config_ref``
    points at the per-model YAML (:class:`WorkerFileConfig`) that contains the
    sglang flags and the loopback port.
    """

    id: str
    config_ref: Path


class ServerConfig(BaseModel):
    """Uvicorn-facing settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


class GpuConfig(BaseModel):
    """How often the metrics layer samples NVML."""

    poll_interval_s: float = 5.0


class ManagerConfig(BaseModel):
    """How the single-slot model manager supervises sglang subprocesses.

    Defaults match architecture doc §B.4.
    """

    load_timeout_s: float = 120.0
    health_poll_interval_s: float = 0.5
    spawn_python: str | None = None  # None → use the gateway's interpreter.


class GatewayConfig(BaseModel):
    """Top-level schema of ``gateway.yaml``."""

    auth: AuthConfig
    models: list[ModelEntry]
    server: ServerConfig = Field(default_factory=ServerConfig)
    gpu: GpuConfig = Field(default_factory=GpuConfig)
    manager: ManagerConfig = Field(default_factory=ManagerConfig)


def load_gateway_config(path: Path) -> GatewayConfig:
    """Parse and validate a gateway YAML config."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return GatewayConfig.model_validate(data)


def load_worker_file(path: Path) -> WorkerFileConfig:
    """Parse and validate a worker YAML config."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return WorkerFileConfig.model_validate(data)
