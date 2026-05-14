"""Run a parameter sweep over the simulator.

Cross-products the axes from a sweep YAML, executes each run sequentially, and emits
a sweep index CSV + a heatmap-driven HTML report. Engine is reloaded only when
`model_path` changes between runs.

Usage:
    uv run python -m simulator.sweep --config sweep.yaml
"""

import argparse
import asyncio
import csv
import itertools
import logging
import uuid
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, model_validator

from simulator.config import SimConfig, load_prompt_config, load_sim_config
from simulator.engine import create_engine, get_tokenizer
from simulator.engine_poller import SchedulerStatLogger
from simulator.metrics import aggregate
from simulator.report_html import write_sweep_report
from simulator.runner import execute_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("simulator.sweep")


class SweepConfig(BaseModel):
    base_config: str
    output_dir: str = "./sweep_results"
    sweep_id: str | None = None
    axes: dict[str, list[Any]]

    @model_validator(mode="after")
    def check_axes_nonempty(self) -> "SweepConfig":
        if not self.axes:
            raise ValueError("sweep config must define at least one axis")
        for name, values in self.axes.items():
            if not values:
                raise ValueError(f"axis '{name}' has no values")
        return self


def load_sweep_config(path: str) -> SweepConfig:
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return SweepConfig.model_validate(raw)


def _slug(value: Any) -> str:
    s = str(value).replace("/", "-").replace(".", "p").replace(" ", "")
    return "".join(c for c in s if c.isalnum() or c in "_-")


def _make_run_id(sweep_id: str, idx: int, axis_values: dict[str, Any]) -> str:
    suffix = "_".join(f"{k}-{_slug(v)}" for k, v in axis_values.items())[:80]
    return f"{sweep_id}_{idx:03d}_{suffix}" if suffix else f"{sweep_id}_{idx:03d}"


async def run_sweep(sweep_cfg: SweepConfig) -> Path:
    base_cfg: SimConfig = load_sim_config(sweep_cfg.base_config)
    prompt = load_prompt_config(base_cfg.prompts_file)

    sweep_id = sweep_cfg.sweep_id or uuid.uuid4().hex[:12]
    sweep_dir = Path(sweep_cfg.output_dir) / sweep_id
    runs_dir = sweep_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    axis_names = list(sweep_cfg.axes.keys())
    axis_value_lists = [sweep_cfg.axes[a] for a in axis_names]
    combos = list(itertools.product(*axis_value_lists))

    log.info("sweep_id=%s  %d runs across axes=%s", sweep_id, len(combos), axis_names)

    engine: Any = None
    tokenizer: Any = None
    stat_logger: SchedulerStatLogger | None = None
    current_model: str | None = None

    run_records: list[dict[str, Any]] = []

    for idx, combo in enumerate(combos):
        axis_values = dict(zip(axis_names, combo, strict=True))
        run_config = base_cfg.model_copy(update={**axis_values, "output_dir": str(runs_dir)})
        run_id = _make_run_id(sweep_id, idx, axis_values)

        log.info("[%d/%d] run %s  axes=%s", idx + 1, len(combos), run_id, axis_values)

        if run_config.model_path != current_model:
            if engine is not None:
                log.info("model_path changed — replacing engine (prior engine left to GC)")
            log.info("loading engine for %s", run_config.model_path)
            stat_logger = SchedulerStatLogger(run_id=run_id)
            engine = create_engine(run_config, stat_logger=stat_logger)
            tokenizer = await get_tokenizer(engine)
            current_model = run_config.model_path

        arts = await execute_run(run_config, run_id, engine, tokenizer, prompt, stat_logger)
        stats = aggregate(arts.samples)
        peak_vram_gb = round(
            max((s.memory_used_bytes for s in arts.gpu_samples), default=0) / 1024**3,
            2,
        )

        run_records.append(
            {
                **axis_values,
                "run_id": run_id,
                "latency_p50_ms": stats.get("latency_p50_ms", 0),
                "latency_p95_ms": stats.get("latency_p95_ms", 0),
                "latency_p99_ms": stats.get("latency_p99_ms", 0),
                "throughput_fps": stats.get("throughput_fps", 0),
                "peak_vram_gb": peak_vram_gb,
                "error_rate_pct": stats.get("error_rate_pct", 0),
                "report_path": f"runs/{arts.paths['report'].name}",
            }
        )

    index_path = sweep_dir / f"{sweep_id}_index.csv"
    if run_records:
        with index_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(run_records[0].keys()))
            writer.writeheader()
            writer.writerows(run_records)

    report_path = sweep_dir / f"{sweep_id}_report.html"
    write_sweep_report(
        sweep_id=sweep_id,
        axes=dict(zip(axis_names, axis_value_lists, strict=True)),
        runs=run_records,
        output_path=report_path,
    )

    log.info("sweep complete  index=%s  report=%s", index_path, report_path)
    return report_path


def main() -> None:
    p = argparse.ArgumentParser(description="Parameter sweep for the VLM profiling simulator")
    p.add_argument("--config", required=True, help="Path to sweep.yaml")
    p.add_argument("--sweep-id", default=None, help="Override sweep_id (default: auto UUID)")
    p.add_argument("--output-dir", default=None, help="Override output_dir from sweep config")
    args = p.parse_args()

    sweep_cfg: SweepConfig = load_sweep_config(args.config)
    if args.sweep_id:
        sweep_cfg = sweep_cfg.model_copy(update={"sweep_id": args.sweep_id})
    if args.output_dir:
        sweep_cfg = sweep_cfg.model_copy(update={"output_dir": args.output_dir})

    asyncio.run(run_sweep(sweep_cfg))


if __name__ == "__main__":
    main()
