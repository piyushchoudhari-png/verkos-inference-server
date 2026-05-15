import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("src.reporter")

from src.config import SimConfig
from src.metrics import EngineSample, GpuSample, IdleBaseline, MetricSample, aggregate


def _mask_config(config: SimConfig) -> dict[str, Any]:
    d = config.model_dump()
    if isinstance(d.get("openrouter"), dict):
        d["openrouter"]["api_key"] = "***"
    return d


def _model_label(config: SimConfig) -> str:
    if config.model_path:
        return Path(config.model_path).name
    if config.openrouter:
        return config.openrouter.model
    return "unknown"


def write_run_meta_stub(config: SimConfig, run_id: str) -> None:
    """Write run_meta.json (status=running) and config_snapshot.json before feeds start."""
    out = Path(config.output_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "run_id": run_id,
        "started_at_iso": datetime.now(timezone.utc).isoformat(),
        "finished_at_iso": None,
        "inference_mode": config.inference_mode,
        "model": _model_label(config),
        "num_feeds": config.num_feeds,
        "video_paths": config.video_paths,
        "status": "running",
        "total_frames": None,
        "successful_frames": None,
        "error_count": None,
        "latency_p50_ms": None,
        "throughput_fps": None,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    (out / "config_snapshot.json").write_text(json.dumps(_mask_config(config), indent=2))


def _write_final_run_meta(
    metric_samples: list[MetricSample],
    config: SimConfig,
    run_id: str,
    out: Path,
) -> None:
    stats = aggregate(metric_samples)
    if metric_samples:
        start_ts = min(s.t0_epoch for s in metric_samples)
        end_ts = max(s.t0_epoch + s.latency_s for s in metric_samples)
        started_iso = datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat()
        finished_iso = datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat()
    else:
        now = datetime.now(timezone.utc).isoformat()
        started_iso = now
        finished_iso = now

    # Preserve started_at_iso from stub if it exists
    stub_path = out / "run_meta.json"
    if stub_path.exists():
        try:
            existing = json.loads(stub_path.read_text())
            if existing.get("started_at_iso"):
                started_iso = existing["started_at_iso"]
        except Exception:
            pass

    meta: dict[str, Any] = {
        "run_id": run_id,
        "started_at_iso": started_iso,
        "finished_at_iso": finished_iso,
        "inference_mode": config.inference_mode,
        "model": _model_label(config),
        "num_feeds": config.num_feeds,
        "video_paths": config.video_paths,
        "status": "completed",
        "total_frames": stats.get("total_frames", 0),
        "successful_frames": stats.get("successful_frames", 0),
        "error_count": stats.get("error_count", 0),
        "latency_p50_ms": stats.get("latency_p50_ms", 0.0),
        "throughput_fps": stats.get("throughput_fps", 0.0),
    }
    stub_path.write_text(json.dumps(meta, indent=2))
    # Write config_snapshot if not already written by stub
    snap_path = out / "config_snapshot.json"
    if not snap_path.exists():
        snap_path.write_text(json.dumps(_mask_config(config), indent=2))


def _gb(b: int) -> str:
    return f"{b / 1024**3:.2f} GB"


def _ms(s: float) -> str:
    return f"{s * 1000:.0f} ms"


def _render_summary(
    stats: dict[str, Any],
    baselines: list[IdleBaseline],
    gpu_samples: list[GpuSample],
    config: SimConfig,
    run_id: str,
) -> str:
    model_label = config.model_path or (config.openrouter.model if config.openrouter else "unknown")
    lines: list[str] = [
        f"Sim Run: {run_id}",
        "",
        "Config",
        f"  inference_mode: {config.inference_mode}",
        f"  model: {model_label}",
        f"  num_feeds: {config.num_feeds}",
        f"  frame_interval_s: {config.frame_interval_s}",
        f"  max_frames_per_feed: {config.max_frames_per_feed or 'unlimited'}",
        f"  max_tokens: {config.max_tokens}",
        f"  temperature: {config.temperature}",
        f"  resolution: {config.resolution or 'native'}",
        f"  downscale_factor: {config.downscale_factor or 'none'}",
        "",
        "VRAM",
    ]

    for b in baselines:
        peak_used = max(
            (s.memory_used_bytes for s in gpu_samples if s.gpu_index == b.gpu_index),
            default=b.memory_used_bytes,
        )
        peak_torch = max(
            (s.torch_allocated_bytes for s in gpu_samples if s.gpu_index == b.gpu_index),
            default=b.torch_allocated_bytes,
        )
        delta = peak_used - b.memory_used_bytes
        min_spec = int(peak_used * 1.20)
        lines += [
            f"  GPU {b.gpu_index}",
            f"    idle (after model load): {_gb(b.memory_used_bytes)} of {_gb(b.memory_total_bytes)}",
            f"    peak active: {_gb(peak_used)}",
            f"    delta (active - idle): {_gb(delta)}",
            f"    peak torch allocated: {_gb(peak_torch)}",
            f"    suggested min VRAM spec (peak + 20% headroom): {_gb(min_spec)}",
        ]

    if not baselines:
        lines.append("  (not measured in openrouter mode)")

    lines += [
        "",
        "Latency",
        f"  P50: {_ms(stats.get('latency_p50_ms', 0) / 1000)}",
        f"  P95: {_ms(stats.get('latency_p95_ms', 0) / 1000)}",
        f"  P99: {_ms(stats.get('latency_p99_ms', 0) / 1000)}",
        f"  mean: {_ms(stats.get('latency_mean_ms', 0) / 1000)}",
        "",
        "Token rates (tok/s)",
        f"  prefill P50 / P95: {stats.get('prefill_tps_p50', 0)} / {stats.get('prefill_tps_p95', 0)}",
        f"  decode  P50 / P95: {stats.get('decode_tps_p50', 0)} / {stats.get('decode_tps_p95', 0)}",
        "",
        "Throughput",
        f"  frames: {stats.get('successful_frames', 0)} ok / {stats.get('total_frames', 0)} total",
        f"  throughput: {stats.get('throughput_fps', 0):.4f} fps",
        f"  stability (CoV, 10s windows): {stats.get('throughput_cov_10s', 0)}",
        f"  run duration: {stats.get('run_duration_s', 0):.1f} s",
        f"  errors: {stats.get('error_count', 0)} ({stats.get('error_rate_pct', 0)}%)",
    ]

    error_classes: dict[str, int] = stats.get("error_classes", {}) or {}
    if error_classes:
        lines.append("  error classes:")
        for cls, cnt in sorted(error_classes.items(), key=lambda x: -x[1]):
            lines.append(f"    {cls}: {cnt}")

    if gpu_samples:
        utils = [s.gpu_util_pct for s in gpu_samples]
        lines += [
            "",
            "GPU Utilization",
            f"  min: {min(utils)}%",
            f"  mean: {sum(utils) / len(utils):.1f}%",
            f"  max: {max(utils)}%",
        ]

    per_feed: dict[int, Any] = stats.get("per_feed", {})
    if per_feed:
        lines += ["", "Per-Feed Breakdown"]
        header = f"  {'feed':>4}  {'frames':>6}  {'P50 (ms)':>8}  {'P95 (ms)':>8}"
        lines.append(header)
        for fid, fd in sorted(per_feed.items()):
            lines.append(f"  {fid:>4}  {fd['frames']:>6}  {fd['latency_p50_ms']:>8}  {fd['latency_p95_ms']:>8}")

    return "\n".join(lines) + "\n"


def _write_json(records: list[Any], path: Path) -> None:
    path.write_text(json.dumps([r.model_dump() for r in records], indent=2))


def write_results(
    metric_samples: list[MetricSample],
    gpu_samples: list[GpuSample],
    engine_samples: list[EngineSample],
    baselines: list[IdleBaseline],
    config: SimConfig,
    run_id: str,
) -> dict[str, Path]:
    out = Path(config.output_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    gpu_path = out / "gpu.json"
    _write_json(gpu_samples, gpu_path)
    paths["gpu"] = gpu_path

    baselines_path = out / "baselines.json"
    _write_json(baselines, baselines_path)
    paths["baselines"] = baselines_path

    engine_path = out / "engine.json"
    _write_json(engine_samples, engine_path)
    paths["engine"] = engine_path

    outputs_path = out / "outputs.json"
    outputs = [s.model_dump() for s in metric_samples]
    outputs_path.write_text(json.dumps(outputs, indent=2))
    paths["outputs"] = outputs_path

    stats = aggregate(metric_samples)
    summary = _render_summary(stats, baselines, gpu_samples, config, run_id)
    summary_path = out / "summary.txt"
    summary_path.write_text(summary)
    paths["summary"] = summary_path

    from src.bbox_renderer import annotate_run
    n = annotate_run(out / "frames", metric_samples)
    if n:
        log.info("annotated %d frames with bounding boxes", n)

    _write_final_run_meta(metric_samples, config, run_id, out)

    return paths
