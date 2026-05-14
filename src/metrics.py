import statistics
from typing import Any

from pydantic import BaseModel


class MetricSample(BaseModel):
    run_id: str
    feed_id: int
    frame_index: int
    t0_epoch: float  # wall clock at request submission, for timeline alignment
    latency_s: float  # monotonic: submit → final token
    ttft_s: float | None  # submit → first non-empty token
    prompt_tokens: int
    completion_tokens: int
    resolution_w: int
    resolution_h: int
    model: str
    inference_mode: str  # "local" | "openrouter"
    status: str  # "ok" | "error"
    error_msg: str | None
    output_text: str | None


class GpuSample(BaseModel):
    run_id: str
    t_epoch: float
    gpu_index: int
    memory_used_bytes: int
    memory_total_bytes: int
    torch_allocated_bytes: int  # torch.cuda.memory_allocated(device)
    gpu_util_pct: int  # 0–100 from nvmlDeviceGetUtilizationRates
    power_w: float | None  # nvmlDeviceGetPowerUsage / 1000; None if unsupported
    temperature_c: int | None  # nvmlDeviceGetTemperature; None if unsupported


class EngineSample(BaseModel):
    """One snapshot of vLLM scheduler + KV cache state."""

    run_id: str
    t_epoch: float
    num_running: int
    num_waiting: int
    num_swapped: int
    gpu_kv_cache_usage_pct: float  # 0–100
    cpu_kv_cache_usage_pct: float  # 0–100; 0 if cpu offload disabled
    num_preemption_total: int  # cumulative since engine start


class IdleBaseline(BaseModel):
    """Single snapshot after model load, before first request."""

    run_id: str
    t_epoch: float
    gpu_index: int
    memory_used_bytes: int
    memory_total_bytes: int
    torch_allocated_bytes: int


def _percentile(sorted_data: list[float], p: float) -> float:
    n = len(sorted_data)
    if n == 0:
        return 0.0
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac


def classify_error(msg: str | None) -> str:
    if not msg:
        return "none"
    m = msg.lower()
    if "out of memory" in m or "oom" in m:
        return "oom"
    if "timeout" in m or "timed out" in m:
        return "timeout"
    if "cuda" in m or "nccl" in m:
        return "cuda"
    if "tokeniz" in m or "vocab" in m or "model" in m:
        return "model"
    return "other"


def _rate(numer: int, denom: float | None) -> float | None:
    if denom is None or denom <= 0:
        return None
    return numer / denom


def _throughput_cov(samples: list["MetricSample"], window_s: float = 10.0) -> float:
    """Coefficient of variation of throughput across rolling windows. Lower = more stable."""
    ok = [s for s in samples if s.status == "ok"]
    if len(ok) < 4:
        return 0.0
    t0 = min(s.t0_epoch for s in ok)
    t_end = max(s.t0_epoch + s.latency_s for s in ok)
    if t_end - t0 < window_s * 2:
        return 0.0
    buckets: dict[int, int] = {}
    for s in ok:
        bucket = int((s.t0_epoch - t0) // window_s)
        buckets[bucket] = buckets.get(bucket, 0) + 1
    counts = list(buckets.values())
    if len(counts) < 2:
        return 0.0
    mean = statistics.mean(counts)
    if mean == 0:
        return 0.0
    return round(statistics.stdev(counts) / mean, 3)


def aggregate(samples: list[MetricSample]) -> dict[str, Any]:
    if not samples:
        return {"total_frames": 0}

    ok = [s for s in samples if s.status == "ok"]
    latencies_ms = sorted(s.latency_s * 1000.0 for s in ok)

    if ok:
        run_start = min(s.t0_epoch for s in ok)
        run_end = max(s.t0_epoch + s.latency_s for s in ok)
        run_duration_s = run_end - run_start
        throughput_fps = len(ok) / run_duration_s if run_duration_s > 0 else 0.0
    else:
        run_duration_s = 0.0
        throughput_fps = 0.0

    per_feed: dict[int, dict[str, Any]] = {}
    for fid in sorted({s.feed_id for s in ok}):
        feed_lats = sorted(s.latency_s * 1000.0 for s in ok if s.feed_id == fid)
        per_feed[fid] = {
            "frames": len(feed_lats),
            "latency_p50_ms": round(_percentile(feed_lats, 50), 1),
            "latency_p95_ms": round(_percentile(feed_lats, 95), 1),
        }

    prefill_rates: list[float] = []
    decode_rates: list[float] = []
    for s in ok:
        pr = _rate(s.prompt_tokens, s.ttft_s)
        if pr is not None:
            prefill_rates.append(pr)
        if s.ttft_s is not None and s.latency_s > s.ttft_s:
            dr = _rate(s.completion_tokens, s.latency_s - s.ttft_s)
            if dr is not None:
                decode_rates.append(dr)
    prefill_sorted = sorted(prefill_rates)
    decode_sorted = sorted(decode_rates)

    errors = [s for s in samples if s.status != "ok"]
    error_classes: dict[str, int] = {}
    for s in errors:
        cls = classify_error(s.error_msg)
        error_classes[cls] = error_classes.get(cls, 0) + 1

    return {
        "total_frames": len(samples),
        "successful_frames": len(ok),
        "error_count": len(errors),
        "error_rate_pct": round(100.0 * len(errors) / len(samples), 1),
        "error_classes": error_classes,
        "latency_p50_ms": round(_percentile(latencies_ms, 50), 1),
        "latency_p95_ms": round(_percentile(latencies_ms, 95), 1),
        "latency_p99_ms": round(_percentile(latencies_ms, 99), 1),
        "latency_mean_ms": round(statistics.mean(latencies_ms), 1) if latencies_ms else 0.0,
        "latency_ms_sorted": latencies_ms,  # CDF input
        "throughput_fps": round(throughput_fps, 4),
        "throughput_cov_10s": _throughput_cov(samples, window_s=10.0),
        "run_duration_s": round(run_duration_s, 1),
        "prefill_tps_p50": round(_percentile(prefill_sorted, 50), 1),
        "prefill_tps_p95": round(_percentile(prefill_sorted, 95), 1),
        "decode_tps_p50": round(_percentile(decode_sorted, 50), 1),
        "decode_tps_p95": round(_percentile(decode_sorted, 95), 1),
        "per_feed": per_feed,
    }
