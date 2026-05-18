from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def list_benchmarks(benchmarks_dir: Path) -> list[dict[str, Any]]:
    if not benchmarks_dir.exists():
        return []
    results: list[dict[str, Any]] = []
    for meta_path in benchmarks_dir.glob("*/benchmark_meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
            meta["_benchmark_dir"] = str(meta_path.parent)
            results.append(meta)
        except Exception:
            continue
    results.sort(key=lambda r: r.get("started_at_iso", ""), reverse=True)
    return results


def save_benchmark_meta(benchmark_dir: Path, meta: dict[str, Any]) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "benchmark_meta.json").write_text(json.dumps(meta, indent=2))


def merge_outputs(
    local_outputs: list[dict[str, Any]],
    or_outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge local and OR outputs by (feed_id, frame_index), outer-join style."""
    local_map = {(o["feed_id"], o["frame_index"]): o for o in local_outputs}
    or_map = {(o["feed_id"], o["frame_index"]): o for o in or_outputs}
    all_keys = sorted(local_map.keys() | or_map.keys())
    merged: list[dict[str, Any]] = []
    for key in all_keys:
        merged.append({"key": key, "local": local_map.get(key), "or": or_map.get(key)})
    return merged


def build_export_rows(
    benchmark_id: str,
    merged: list[dict[str, Any]],
    local_model: str,
    or_model: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in merged:
        feed_id, frame_index = entry["key"]
        loc = entry["local"] or {}
        orr = entry["or"] or {}
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "feed_id": feed_id,
                "frame_index": frame_index,
                "local_model": local_model,
                "or_model": or_model,
                "local_status": loc.get("status", ""),
                "or_status": orr.get("status", ""),
                "local_latency_ms": round((loc.get("latency_s") or 0) * 1000, 1),
                "or_latency_ms": round((orr.get("latency_s") or 0) * 1000, 1),
                "local_ttft_ms": round((loc.get("ttft_s") or 0) * 1000, 1),
                "or_ttft_ms": round((orr.get("ttft_s") or 0) * 1000, 1),
                "local_prompt_tokens": loc.get("prompt_tokens", ""),
                "or_prompt_tokens": orr.get("prompt_tokens", ""),
                "local_completion_tokens": loc.get("completion_tokens", ""),
                "or_completion_tokens": orr.get("completion_tokens", ""),
                "local_output_json": loc.get("output_text", ""),
                "or_output_json": orr.get("output_text", ""),
                "local_error_msg": loc.get("error_msg", ""),
                "or_error_msg": orr.get("error_msg", ""),
            }
        )
    return rows
