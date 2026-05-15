from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.utils.run_loader import (
    frames_for_feed,
    list_runs,
    load_baselines,
    load_config_snapshot,
    load_engine,
    load_gpu,
    load_outputs,
)

_STATUS_ICON = {"completed": "🟢", "running": "🟡", "failed": "🔴"}
_FRAMES_PER_PAGE = 5


def _latency_pill(ms: float) -> str:
    color = "#22c55e" if ms < 1000 else "#f59e0b" if ms < 3000 else "#ef4444"
    return (
        f'<span style="background:{color}22;color:{color};padding:2px 10px;'
        f'border-radius:12px;font-size:0.82rem;font-weight:600;white-space:nowrap">'
        f"⏱ {ms:.0f} ms</span>"
    )


def _token_pill(prompt: Any, completion: Any) -> str:
    return (
        f'<span style="background:#0ea5e922;color:#0ea5e9;padding:2px 10px;'
        f'border-radius:12px;font-size:0.82rem;white-space:nowrap">'
        f"🔤 {prompt}→{completion}</span>"
    )


def _ttft_pill(ms: float) -> str:
    return (
        f'<span style="background:#6366f122;color:#818cf8;padding:2px 10px;'
        f'border-radius:12px;font-size:0.82rem;white-space:nowrap">'
        f"TTFT {ms:.0f} ms</span>"
    )


def _detection_count(parsed: Any) -> int | None:
    if not isinstance(parsed, dict):
        return None
    for key in ("detections", "objects", "results", "items"):
        val = parsed.get(key)
        if isinstance(val, list):
            return len(val)
    return None


def _run_label(meta: dict[str, Any]) -> str:
    ts = meta.get("started_at_iso", "")[:16].replace("T", " ")
    mode = meta.get("inference_mode", "?")
    model = meta.get("model", "?")
    total = meta.get("total_frames")
    status = meta.get("status", "?")
    icon = _STATUS_ICON.get(status, "⚪")
    frames_str = f"{total} frames" if total is not None else "in progress"
    return f"{icon}  {meta.get('run_id', '?')}   [{mode} · {model}]   {ts}   {frames_str}"



def _parse_output_text(raw: str | None) -> tuple[Any, bool]:
    """Return (parsed_json_or_raw_str, is_json)."""
    if not raw:
        return "", False
    cleaned = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE | re.MULTILINE)
    try:
        return json.loads(cleaned), True
    except Exception:
        return raw, False


def _frame_index_from_path(p: Path) -> int:
    # stem = "feed00_frame0004"
    try:
        return int(p.stem.split("_frame")[1])
    except (IndexError, ValueError):
        return -1


def _section_frames(run_dir: Path, outputs: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    st.subheader("① Frames & Detections")
    if not outputs:
        st.warning("No frame outputs found for this run.")
        return

    feed_ids = sorted({o["feed_id"] for o in outputs})
    run_key = meta.get("run_id", "run")

    head_col, feed_col = st.columns([3, 2])
    with feed_col:
        if len(feed_ids) > 1:
            feed_id: int = st.radio(
                "Feed",
                feed_ids,
                horizontal=True,
                format_func=lambda x: f"Feed {x}",
                key=f"feed_sel_{run_key}",
            )
        else:
            feed_id = feed_ids[0]
            st.caption(f"Feed {feed_id}")

    feed_outputs = sorted(
        [o for o in outputs if o["feed_id"] == feed_id],
        key=lambda o: o["frame_index"],
    )

    frame_map: dict[int, Path] = {
        _frame_index_from_path(p): p
        for p in frames_for_feed(run_dir, feed_id)
    }

    total_frames = len(feed_outputs)
    total_pages = max(1, (total_frames + _FRAMES_PER_PAGE - 1) // _FRAMES_PER_PAGE)

    pg_col1, pg_col2, pg_col3 = st.columns([3, 1, 1])
    with pg_col2:
        page: int = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key=f"page_{run_key}_{feed_id}",
        )
    with pg_col3:
        st.markdown(f"<div style='padding-top:32px;color:#888'>/ {total_pages}</div>", unsafe_allow_html=True)
    with pg_col1:
        start_n = (_FRAMES_PER_PAGE * (page - 1)) + 1
        end_n = min(_FRAMES_PER_PAGE * page, total_frames)
        st.markdown(
            f"<div style='padding-top:32px;color:#888;font-size:0.85rem'>"
            f"Frames {start_n}–{end_n} of {total_frames}</div>",
            unsafe_allow_html=True,
        )

    start_idx = (page - 1) * _FRAMES_PER_PAGE
    page_outputs = feed_outputs[start_idx : start_idx + _FRAMES_PER_PAGE]

    for o in page_outputs:
        frame_idx: int = o["frame_index"]
        frame_path = frame_map.get(frame_idx)
        latency_ms = (o.get("latency_s") or 0) * 1000
        ttft_ms = (o.get("ttft_s") or 0) * 1000
        ptok = o.get("prompt_tokens", "?")
        ctok = o.get("completion_tokens", "?")
        is_error = o.get("status") == "error"

        with st.container(border=True):
            # ── Card header ──
            h1, h2, h3, h4 = st.columns([3, 2, 2, 2])
            status_icon = "❌" if is_error else "✅"
            h1.markdown(f"**{status_icon} Frame {frame_idx}**")
            h2.markdown(_latency_pill(latency_ms), unsafe_allow_html=True)
            h3.markdown(_ttft_pill(ttft_ms), unsafe_allow_html=True)
            h4.markdown(_token_pill(ptok, ctok), unsafe_allow_html=True)

            # ── Image + output ──
            col_img, col_out = st.columns([1, 1], gap="small")

            with col_img:
                if frame_path and frame_path.exists():
                    st.image(str(frame_path), use_container_width=True)
                else:
                    st.markdown(
                        f'<div style="background:#1e1e2e;height:220px;display:flex;'
                        f'align-items:center;justify-content:center;border-radius:6px;'
                        f'color:#4a4a6a;font-size:0.85rem;border:1px dashed #2a2a3e">'
                        f"Frame {frame_idx} — not saved</div>",
                        unsafe_allow_html=True,
                    )

            with col_out:
                if is_error:
                    st.error(o.get("error_msg") or "Unknown error")
                else:
                    parsed, is_json = _parse_output_text(o.get("output_text"))
                    if is_json:
                        count = _detection_count(parsed)
                        if count is not None:
                            st.markdown(
                                f'<span style="background:#10b98122;color:#10b981;padding:2px 10px;'
                                f'border-radius:12px;font-size:0.8rem;font-weight:600">'
                                f"🎯 {count} detection(s)</span>",
                                unsafe_allow_html=True,
                            )
                        st.json(parsed, expanded=2)
                    else:
                        st.code(str(parsed) if parsed else "(no output)", language="text")


def _section_summary(outputs: list[dict[str, Any]]) -> None:
    st.subheader("② Summary Statistics")
    try:
        from src.metrics import MetricSample, aggregate
        samples = [MetricSample(**o) for o in outputs]
        stats = aggregate(samples)
    except Exception as exc:
        st.error(f"Could not compute stats: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P50 Latency", f"{stats.get('latency_p50_ms', 0):.0f} ms")
    c2.metric("P95 Latency", f"{stats.get('latency_p95_ms', 0):.0f} ms")
    c3.metric("Throughput", f"{stats.get('throughput_fps', 0):.3f} fps")
    c4.metric("Error Rate", f"{stats.get('error_rate_pct', 0):.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("P99 Latency", f"{stats.get('latency_p99_ms', 0):.0f} ms")
    c6.metric("Prefill tok/s (P50)", f"{stats.get('prefill_tps_p50', 0):.0f}")
    c7.metric("Decode tok/s (P50)", f"{stats.get('decode_tps_p50', 0):.0f}")
    c8.metric("Run Duration", f"{stats.get('run_duration_s', 0):.1f} s")

    per_feed: dict[int, Any] = stats.get("per_feed") or {}
    if per_feed:
        with st.expander("Per-feed breakdown", expanded=True):
            rows = [
                {
                    "Feed": f"Feed {fid}",
                    "Frames": fd["frames"],
                    "P50 ms": fd["latency_p50_ms"],
                    "P95 ms": fd["latency_p95_ms"],
                }
                for fid, fd in sorted(per_feed.items())
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)

    error_classes: dict[str, int] = stats.get("error_classes") or {}
    if error_classes:
        with st.expander("Error breakdown"):
            for cls, cnt in sorted(error_classes.items(), key=lambda x: -x[1]):
                st.write(f"**{cls}**: {cnt}")


def _section_gpu(run_dir: Path, meta: dict[str, Any]) -> None:
    from bench.utils.gpu_charts import engine_chart, gpu_util_chart, power_chart, temp_chart, vram_chart

    st.subheader("③ GPU Metrics")
    gpu_data = load_gpu(run_dir)
    engine_data = load_engine(run_dir)

    if not gpu_data and not engine_data:
        st.info("No GPU samples recorded for this run.")
        return

    baselines_raw = load_baselines(run_dir)
    idle_baseline: dict[int, int] = {b["gpu_index"]: b["memory_used_bytes"] for b in baselines_raw}

    with st.expander("GPU Metrics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(vram_chart(gpu_data, idle_baseline or None), use_container_width=True)
        with col2:
            st.plotly_chart(gpu_util_chart(gpu_data), use_container_width=True)

        p_fig = power_chart(gpu_data)
        t_fig = temp_chart(gpu_data)
        col3, col4 = st.columns(2)
        with col3:
            if p_fig:
                st.plotly_chart(p_fig, use_container_width=True)
            else:
                st.caption("Power draw data not available.")
        with col4:
            if t_fig:
                st.plotly_chart(t_fig, use_container_width=True)
            else:
                st.caption("Temperature data not available.")

    if engine_data:
        with st.expander("vLLM Scheduler Stats", expanded=True):
            st.plotly_chart(engine_chart(engine_data), use_container_width=True)

    if gpu_data and idle_baseline:
        st.subheader("VRAM Summary")
        rows = []
        for gi in sorted({s["gpu_index"] for s in gpu_data}):
            subs = [s for s in gpu_data if s["gpu_index"] == gi]
            peak = max(s["memory_used_bytes"] for s in subs)
            idle = idle_baseline.get(gi, 0)
            total = subs[0]["memory_total_bytes"]
            rows.append({
                "GPU": gi,
                "Idle (GB)": round(idle / 1e9, 2),
                "Peak Active (GB)": round(peak / 1e9, 2),
                "Delta (GB)": round((peak - idle) / 1e9, 2),
                "Total (GB)": round(total / 1e9, 2),
                "+20% Spec (GB)": round(peak * 1.20 / 1e9, 2),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)


def render_results(runs_dir: Path) -> None:
    runs = list_runs(runs_dir)

    if not runs:
        st.markdown(
            """
            <div style="text-align:center;padding:4rem 2rem;color:#666">
                <h3 style="margin-bottom:0.5rem">No runs yet</h3>
                <p>Go to <b>⚙️ Configure & Run</b> to start your first profiling run.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    run_ids = [r["run_id"] for r in runs]
    default_idx = 0
    last = st.session_state.get("last_run_id")
    if last and last in run_ids:
        default_idx = run_ids.index(last)

    selected_idx: int = st.selectbox(  # type: ignore[assignment]
        "Select run",
        range(len(runs)),
        format_func=lambda i: _run_label(runs[i]),
        index=default_idx,
        key="run_selector",
    )

    meta = runs[selected_idx]
    run_dir = Path(meta["_run_dir"])

    status = meta.get("status", "unknown")
    icon = _STATUS_ICON.get(status, "⚪")

    meta_col, vid_col = st.columns([3, 1])
    with meta_col:
        st.markdown(
            f"{icon} **{status}**&nbsp;&nbsp;·&nbsp;&nbsp;"
            f"Mode: **{meta.get('inference_mode', '?')}**&nbsp;&nbsp;·&nbsp;&nbsp;"
            f"Model: **{meta.get('model', '?')}**&nbsp;&nbsp;·&nbsp;&nbsp;"
            f"Feeds: **{meta.get('num_feeds', '?')}**",
            unsafe_allow_html=True,
        )
    with vid_col:
        video_paths: list[str] = meta.get("video_paths") or []
        for vp in video_paths[:1]:
            p = Path(vp)
            if p.exists():
                st.video(str(p))
            else:
                st.caption(f"Video not found")

    st.divider()

    outputs = load_outputs(run_dir)
    _section_frames(run_dir, outputs, meta)
    st.divider()

    if outputs:
        _section_summary(outputs)
    else:
        if meta.get("status") == "running":
            st.info("Run is still in progress — refresh to see results.")
        else:
            st.warning("No outputs.json found for this run.")

    if meta.get("inference_mode") == "local":
        st.divider()
        _section_gpu(run_dir, meta)
