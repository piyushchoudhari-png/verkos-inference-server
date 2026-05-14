import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, model_validator


class PromptConfig(BaseModel):
    system: str
    user: str  # supports {frame_index} placeholder


class OpenRouterConfig(BaseModel):
    api_key: str
    model: str = "qwen/qwen3-vl-32b-instruct"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_s: float = 60.0


class SimConfig(BaseModel):
    model_path: str | None = None
    video_paths: list[str]
    prompts_file: str
    num_feeds: int = 4
    frame_interval_s: float = 10.0
    max_frames_per_feed: int | None = None
    resolution: tuple[int, int] | None = None
    downscale_factor: float | None = None
    gpu_poll_interval_s: float = 0.5
    max_tokens: int = 256
    temperature: float = 0.0
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    output_dir: str = "./sim_results"
    run_id: str | None = None
    inference_mode: Literal["local", "openrouter"] = "local"
    openrouter: OpenRouterConfig | None = None

    @model_validator(mode="after")
    def check_resolution_conflict(self) -> "SimConfig":
        if self.resolution is not None and self.downscale_factor is not None:
            raise ValueError("Set resolution or downscale_factor, not both")
        return self

    @model_validator(mode="after")
    def check_local_model_path(self) -> "SimConfig":
        if self.inference_mode == "local" and self.model_path is None:
            raise ValueError("model_path is required when inference_mode is 'local'")
        if self.inference_mode == "openrouter" and self.openrouter is None:
            raise ValueError("openrouter config block is required when inference_mode is 'openrouter'")
        return self


def load_sim_config(path: str) -> SimConfig:
    load_dotenv()
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    if isinstance(raw.get("openrouter"), dict) and not raw["openrouter"].get("api_key"):
        key = os.environ.get("OPENROUTER_API_KEY")
        if key:
            raw["openrouter"]["api_key"] = key
    return SimConfig.model_validate(raw)


def load_prompt_config(path: str) -> PromptConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    with open(p) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return PromptConfig.model_validate(raw)
