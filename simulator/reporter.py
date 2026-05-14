import csv
import json
from pathlib import Path
from typing import Any

from simulator.config import SimConfig
from simulator.metrics import EngineSample, GpuSample, IdleBaseline, MetricSample, aggregate
from simulator.report_html import write_run_report


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
    lines: list[str] = [
        f"Sim Run: {run_id}",
        "",
        "Config",
        f"  model: {config.model_path}",
        f"  num_feeds: {config.num_feeds}",
        f"  frame_interval_s: {config.frame_interval_s}",
        f"  run_duration_s: {config.run_duration_s}",
        f"  max_frames_per_feed: {config.max_frames_per_feed}",
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


def _write_csv(records: list[Any], path: Path) -> None:
    if not records:
        path.write_text("")
        return
    rows = [r.model_dump() for r in records]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_results(
    metric_samples: list[MetricSample],
    gpu_samples: list[GpuSample],
    engine_samples: list[EngineSample],
    baselines: list[IdleBaseline],
    config: SimConfig,
    run_id: str,
) -> dict[str, Path]:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    samples_path = out / f"{run_id}_samples.csv"
    _write_csv(metric_samples, samples_path)
    paths["samples"] = samples_path

    gpu_path = out / f"{run_id}_gpu.csv"
    _write_csv(gpu_samples, gpu_path)
    paths["gpu"] = gpu_path

    engine_path = out / f"{run_id}_engine.csv"
    _write_csv(engine_samples, engine_path)
    paths["engine"] = engine_path

    outputs_path = out / f"{run_id}_outputs.json"
    outputs: list[dict[str, Any]] = [
        {
            "run_id": s.run_id,
            "feed_id": s.feed_id,
            "frame_index": s.frame_index,
            "t0_epoch": s.t0_epoch,
            "model": s.model,
            "status": s.status,
            "output_text": s.output_text,
            "error_msg": s.error_msg,
        }
        for s in metric_samples
    ]
    outputs_path.write_text(json.dumps(outputs, indent=2))
    paths["outputs"] = outputs_path

    stats = aggregate(metric_samples)
    summary = _render_summary(stats, baselines, gpu_samples, config, run_id)
    summary_path = out / f"{run_id}_summary.txt"
    summary_path.write_text(summary)
    paths["summary"] = summary_path

    report_path = out / f"{run_id}_report.html"
    write_run_report(
        metric_samples=metric_samples,
        gpu_samples=gpu_samples,
        engine_samples=engine_samples,
        baselines=baselines,
        config=config,
        run_id=run_id,
        output_path=report_path,
    )
    paths["report"] = report_path

    return paths
