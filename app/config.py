import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "/etc/inference-server/config.yaml"))

_ENV_MAP: dict[str, tuple[str, str]] = {
    "MODEL_PATH": ("model", "path"),
    "MODEL_DTYPE": ("model", "dtype"),
    "QUANTIZATION": ("model", "quantization"),
    "GPU_MEMORY_UTILIZATION": ("engine", "gpu_memory_utilization"),
    "MAX_MODEL_LEN": ("model", "max_model_len"),
    "MAX_NUM_SEQS": ("model", "max_num_seqs"),
    "TENSOR_PARALLEL_SIZE": ("engine", "tensor_parallel_size"),
    "PORT": ("server", "port"),
    "LOG_LEVEL": ("server", "log_level"),
}


class ModelSource(BaseModel):
    type: str = "huggingface"
    repo_id: str | None = None
    revision: str = "main"


class ModelConfig(BaseModel):
    name: str
    path: str
    source: ModelSource | None = None
    dtype: str = "auto"
    quantization: str = "none"
    max_model_len: int | None = None
    max_num_seqs: int = 256


class EngineConfig(BaseModel):
    model_config = {"extra": "allow"}

    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1

    def extra_kwargs(self) -> dict[str, Any]:
        return self.model_extra or {}


class ServerConfig(BaseModel):
    port: int = 8000
    log_level: str = "info"


class Settings(BaseModel):
    model: ModelConfig
    engine: EngineConfig = EngineConfig()
    server: ServerConfig = ServerConfig()


def load_settings(
    config_path: Path = CONFIG_PATH,
) -> tuple[Settings, dict[str, dict[str, Any]]]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Mount a config.yaml to /etc/inference-server/config.yaml"
        )

    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    conflicts: dict[str, dict[str, Any]] = {}
    for env_var, (section, key) in _ENV_MAP.items():
        env_val = os.environ.get(env_var)
        if env_val is None:
            continue
        file_val = raw.get(section, {}).get(key)
        if file_val is not None and str(file_val) != env_val:
            conflicts[env_var] = {"file": file_val, "env": env_val}
        raw.setdefault(section, {})[key] = env_val

    return Settings(**raw), conflicts
