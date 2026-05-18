from __future__ import annotations

import csv
import io
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
import yaml
from pydantic import ValidationError

_FRAMES_PER_PAGE = 5
_STATUS_ICON = {"completed": "🟢", "running": "🟡", "failed": "🔴"}


# ── Session state ──────────────────────────────────────────────────────────────

def _init_state() -> None:
    if "bm_vid_path_ids" not in st.session_state:
        st.session_state["bm_vid_path_ids"] = [uuid.uuid4().hex[:8]]


# ── Video source widgets ───────────────────────────────────────────────────────

def _video_enter_path() -> list[str]:
    path_ids: list[str] = st.session_state["bm_vid_path_ids"]
    to_remove: str | None = None
    for pid in path_ids:
        c1, c2 = st.columns([10, 1])
        with c1:
            st.text_input("Path", key=f"bm_vpath_{pid}", label_visibility="collapsed", placeholder="/path/to/video.mp4")
        with c2:
            st.write("")
            if len(path_ids) > 1 and st.button("✕", key=f"bm_rm_{pid}"):
                to_remove = pid
    if to_remove:
        path_ids.remove(to_remove)
        st.rerun()
    if st.button("+ Add video", key="bm_add_video"):
        path_ids.append(uuid.uuid4().hex[:8])
        st.rerun()
    return [v for pid in path_ids if (v := st.session_state.get(f"bm_vpath_{pid}", "").strip())]


def _video_upload(root: Path) -> list[str]:
    uploaded = st.file_uploader("Upload video files", type=["mp4", "avi", "mov", "mkv"], accept_multiple_files=True, key="bm_vid_upload")
    paths: list[str] = []
    tmp_dir = root / "bench" / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    for uf in uploaded or []:
        dest = tmp_dir / uf.name
        dest.write_bytes(uf.read())
        paths.append(str(dest.resolve()))
    return paths


# ── Config section ─────────────────────────────────────────────────────────────

def _render_config(root: Path) -> dict[str, Any] | None:
    """Render config widgets. Returns config dict with 'local' and 'or' keys, or None on error."""
    _init_state()

    # ── Local model ───────────────────────────────────────────────────────────
    st.subheader("Local Model")
    models_dir = root / "models"
    model_dirs: list[str] = sorted(d.name for d in models_dir.iterdir() if d.is_dir()) if models_dir.exists() and models_dir.is_dir() else []
    if not model_dirs:
        st.warning("No models found in `models/`. Run `scripts/download_model.py` to add one.")
        local_model_path = None
    else:
        sel: str = st.selectbox("Model", model_dirs, key="bm_local_model")  # type: ignore[assignment]
        local_model_path = str(models_dir / sel)
        st.caption(f"`{local_model_path}`")

    lc1, lc2, lc3 = st.columns(3)
    local_dtype: str = lc1.selectbox("dtype", ["auto", "float16", "bfloat16", "float32"], key="bm_local_dtype")  # type: ignore[assignment]
    local_tp: int = lc2.number_input("Tensor parallel", min_value=1, max_value=8, value=1, step=1, key="bm_local_tp")  # type: ignore[assignment]
    local_gpu_mem: float = lc3.slider("GPU mem util", min_value=0.5, max_value=1.0, value=0.90, step=0.01, key="bm_local_gpu_mem")  # type: ignore[assignment]

    mml_raw: str = st.text_input("Max model len (blank = model default)", value="", key="bm_local_mml")
    local_max_model_len: int | None = int(mml_raw) if mml_raw.strip().isdigit() else None

    local_gpu_poll: float = st.number_input("GPU poll interval (s)", min_value=0.1, max_value=10.0, value=0.5, step=0.1, key="bm_local_gpu_poll")  # type: ignore[assignment]

    st.divider()

    # ── OpenRouter ────────────────────────────────────────────────────────────
    st.subheader("OpenRouter")
    oc1, oc2 = st.columns(2)
    or_model: str = oc1.text_input("Model", value="qwen/qwen3-vl-32b-instruct", key="bm_or_model")
    or_timeout: float = oc2.number_input("Timeout (s)", min_value=10.0, max_value=300.0, value=60.0, step=5.0, key="bm_or_timeout")  # type: ignore[assignment]
    or_api_key: str = st.text_input("API Key (or set OPENROUTER_API_KEY env var)", type="password", key="bm_or_api_key")

    st.divider()

    # ── Video sources ─────────────────────────────────────────────────────────
    st.subheader("Video Sources")
    vid_method: str = st.radio("Input method", ["Enter path", "Upload file"], horizontal=True, key="bm_vid_method", label_visibility="collapsed")  # type: ignore[assignment]
    video_paths = _video_upload(root) if vid_method == "Upload file" else _video_enter_path()

    st.divider()

    # ── Frame sampling ────────────────────────────────────────────────────────
    st.subheader("Frame Sampling")
    fs1, fs2, fs3 = st.columns(3)
    num_feeds: int = fs1.number_input("Num feeds", min_value=1, max_value=64, value=1, step=1, key="bm_num_feeds")  # type: ignore[assignment]
    frame_interval: float = fs2.number_input("Frame interval (s)", min_value=0.1, max_value=300.0, value=10.0, step=0.5, key="bm_frame_interval")  # type: ignore[assignment]
    max_frames_raw: str = fs3.text_input("Max frames/feed (blank = ∞)", value="", key="bm_max_frames")
    max_frames_per_feed: int | None = int(max_frames_raw) if max_frames_raw.strip().isdigit() else None

    re1, re2 = st.columns(2)
    resolution_raw: str = re1.text_input("Resolution WxH (blank = native)", value="", key="bm_resolution")
    downscale_raw: str = re2.text_input("Downscale factor (blank = none)", value="", key="bm_downscale")

    resolution: list[int] | None = None
    if resolution_raw.strip():
        try:
            parts = resolution_raw.lower().replace("x", " ").split()
            resolution = [int(parts[0]), int(parts[1])]
        except (ValueError, IndexError):
            st.error(f"Invalid resolution `{resolution_raw}`. Use WxH format, e.g. 640x480.")

    downscale_factor: float | None = None
    if downscale_raw.strip():
        try:
            downscale_factor = float(downscale_raw)
        except ValueError:
            st.error(f"Invalid downscale factor `{downscale_raw}`.")

    st.divider()

    # ── Generation ────────────────────────────────────────────────────────────
    st.subheader("Generation")
    g1, g2 = st.columns(2)
    max_tokens: int = g1.number_input("Max tokens", min_value=1, max_value=8192, value=256, step=1, key="bm_max_tokens")  # type: ignore[assignment]
    temperature: float = g2.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="bm_temperature")  # type: ignore[assignment]

    st.divider()

    # ── Prompts ───────────────────────────────────────────────────────────────
    st.subheader("Prompts")
    default_sys, default_usr = "", ""
    prompts_yaml_path = root / "src" / "config" / "prompts.yaml"
    if prompts_yaml_path.exists():
        try:
            pd = yaml.safe_load(prompts_yaml_path.read_text()) or {}
            default_sys, default_usr = pd.get("system", ""), pd.get("user", "")
        except Exception:
            pass
    system_prompt: str = st.text_area("System prompt", value=default_sys, height=180, key="bm_sys_prompt")
    user_prompt: str = st.text_area("User prompt", value=default_usr, height=80, key="bm_usr_prompt")

    st.divider()

    # ── Run settings ──────────────────────────────────────────────────────────
    st.subheader("Run Settings")
    run_id_input: str = st.text_input("Benchmark ID (blank = auto)", value="", key="bm_run_id")

    return {
        "video_paths": video_paths,
        "num_feeds": int(num_feeds),
        "frame_interval_s": float(frame_interval),
        "max_frames_per_feed": max_frames_per_feed,
        "resolution": resolution,
        "downscale_factor": downscale_factor,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "run_id_input": run_id_input,
        "local_model_path": local_model_path,
        "local_dtype": local_dtype,
        "local_tp": int(local_tp),
        "local_gpu_mem": float(local_gpu_mem),
        "local_max_model_len": local_max_model_len,
        "local_gpu_poll": float(local_gpu_poll),
        "or_model": or_model,
        "or_timeout": float(or_timeout),
        "or_api_key": or_api_key,
    }


# ── Run benchmark ──────────────────────────────────────────────────────────────

def _run_benchmark(root: Path, runs_dir: Path, benchmarks_dir: Path, cfg: dict[str, Any]) -> None:
    from bench.utils.subprocess_runner import RunProcess
    from src.config import SimConfig

    if not cfg["video_paths"]:
        st.error("Add at least one video path before launching.")
        return

    benchmark_id = cfg["run_id_input"].strip() or f"bm-{uuid.uuid4().hex[:8]}-{time.strftime('%Y%m%d%H%M%S')}"
    ts = int(time.time())
    tmp_dir = root / "bench" / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    prompts_path = tmp_dir / f"bm_prompts_{ts}.yaml"
    prompts_path.write_text(yaml.dump({"system": cfg["system_prompt"], "user": cfg["user_prompt"]}, allow_unicode=True))

    shared: dict[str, Any] = {
        "video_paths": cfg["video_paths"],
        "num_feeds": cfg["num_feeds"],
        "frame_interval_s": cfg["frame_interval_s"],
        "max_frames_per_feed": cfg["max_frames_per_feed"],
        "resolution": cfg["resolution"],
        "downscale_factor": cfg["downscale_factor"],
        "max_tokens": cfg["max_tokens"],
        "temperature": cfg["temperature"],
        "output_dir": str(runs_dir),
        "prompts_file": str(prompts_path.resolve()),
    }

    local_run_id = f"{benchmark_id}-local"
    or_run_id = f"{benchmark_id}-openrouter"

    local_cfg: dict[str, Any] = {
        **shared,
        "inference_mode": "local",
        "model_path": cfg["local_model_path"],
        "dtype": cfg["local_dtype"],
        "tensor_parallel_size": cfg["local_tp"],
        "gpu_memory_utilization": cfg["local_gpu_mem"],
        "max_model_len": cfg["local_max_model_len"],
        "gpu_poll_interval_s": cfg["local_gpu_poll"],
    }

    effective_key = cfg["or_api_key"] or os.environ.get("OPENROUTER_API_KEY", "")
    or_cfg: dict[str, Any] = {
        **shared,
        "inference_mode": "openrouter",
        "openrouter": {
            "api_key": effective_key,
            "model": cfg["or_model"],
            "timeout_s": cfg["or_timeout"],
        },
    }

    # Validate both configs before launching
    try:
        SimConfig.model_validate(local_cfg)
    except ValidationError as exc:
        st.error(f"Local config error:\n```\n{exc}\n```")
        return
    try:
        SimConfig.model_validate(or_cfg)
    except ValidationError as exc:
        st.error(f"OpenRouter config error:\n```\n{exc}\n```")
        return

    local_cfg_path = tmp_dir / f"bm_local_{ts}.yaml"
    or_cfg_path = tmp_dir / f"bm_or_{ts}.yaml"
    local_cfg_path.write_text(yaml.dump(local_cfg, allow_unicode=True))
    or_cfg_path.write_text(yaml.dump(or_cfg, allow_unicode=True))

    st.session_state["bm_run_in_progress"] = True

    phase_area = st.empty()
    log_container = st.container(height=400, border=True)
    log_area = log_container.empty()
    log_lines: list[str] = []

    local_ok = or_ok = False

    try:
        phase_area.info(f"Phase 1 / 2 — Local model ({cfg['local_model_path'] or 'none'})…")
        runner = RunProcess(str(local_cfg_path.resolve()), local_run_id)
        for line in runner:
            log_lines.append(f"[local] {line}")
            log_area.code("\n".join(log_lines[-200:]), language="text")
        local_ok = runner.returncode == 0

        phase_area.info(f"Phase 2 / 2 — OpenRouter ({cfg['or_model']})…")
        runner = RunProcess(str(or_cfg_path.resolve()), or_run_id)
        for line in runner:
            log_lines.append(f"[openrouter] {line}")
            log_area.code("\n".join(log_lines[-200:]), language="text")
        or_ok = runner.returncode == 0

    finally:
        st.session_state["bm_run_in_progress"] = False

    # Save benchmark meta
    from bench.utils.benchmark_store import save_benchmark_meta
    from datetime import datetime

    bm_dir = benchmarks_dir / benchmark_id
    save_benchmark_meta(
        bm_dir,
        {
            "benchmark_id": benchmark_id,
            "started_at_iso": datetime.now().astimezone().isoformat(),
            "local_run_id": local_run_id,
            "or_run_id": or_run_id,
            "local_model": cfg["local_model_path"] or "",
            "or_model": cfg["or_model"],
            "local_status": "completed" if local_ok else "failed",
            "or_status": "completed" if or_ok else "failed",
            "status": "completed" if (local_ok and or_ok) else "failed",
        },
    )

    if local_ok and or_ok:
        phase_area.success(f"Benchmark `{benchmark_id}` complete — switch to **Compare Results** to inspect.")
        st.session_state["bm_last_benchmark_id"] = benchmark_id
    else:
        parts = []
        if not local_ok:
            parts.append("local")
        if not or_ok:
            parts.append("OpenRouter")
        phase_area.error(f"{' and '.join(parts)} run(s) failed — check log above.")


# ── Compare results ────────────────────────────────────────────────────────────

def _parse_json(raw: str | None) -> tuple[Any, bool]:
    if not raw:
        return "", False
    cleaned = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE | re.MULTILINE)
    try:
        return json.loads(cleaned), True
    except Exception:
        return raw, False


def _latency_pill(ms: float, color: str) -> str:
    return (
        f'<span style="background:{color}22;color:{color};padding:2px 8px;'
        f'border-radius:10px;font-size:0.78rem;font-weight:600">⏱ {ms:.0f} ms</span>'
    )


def _frame_index_from_path(p: Path) -> int:
    try:
        return int(p.stem.split("_frame")[1])
    except (IndexError, ValueError):
        return -1


def _output_block(entry_side: dict[str, Any] | None) -> None:
    """Render one model's output for a single frame."""
    if not entry_side:
        st.caption("No output")
        return
    if entry_side.get("status") == "error":
        st.error(entry_side.get("error_msg") or "error")
        return
    parsed, is_json = _parse_json(entry_side.get("output_text"))
    if is_json:
        st.json(parsed, expanded=2)
    else:
        st.code(str(parsed) or "(no output)", language="text")


def _frame_placeholder(frame_idx: int) -> None:
    st.markdown(
        f'<div style="background:#1e1e2e;height:180px;display:flex;align-items:center;'
        f'justify-content:center;border-radius:6px;color:#4a4a6a;font-size:0.8rem;'
        f'border:1px dashed #2a2a3e">frame {frame_idx} — not saved</div>',
        unsafe_allow_html=True,
    )


def _build_frame_map(run_dir: Path, feed_id: int) -> dict[int, Path]:
    frames_dir = run_dir / "frames"
    if not frames_dir.exists():
        return {}
    return {_frame_index_from_path(p): p for p in sorted(frames_dir.glob(f"feed{feed_id:02d}_frame*.jpg"))}


def _section_compare_frames(
    merged: list[dict[str, Any]],
    local_run_dir: Path,
    or_run_dir: Path,
    bm_id: str,
    local_model_label: str,
    or_model_label: str,
) -> None:
    st.subheader("① Frame-by-Frame Comparison")
    if not merged:
        st.warning("No outputs to compare.")
        return

    feed_ids = sorted({e["key"][0] for e in merged})
    feed_id: int
    if len(feed_ids) > 1:
        feed_id = st.radio("Feed", feed_ids, horizontal=True, format_func=lambda x: f"Feed {x}", key=f"bm_feed_{bm_id}")  # type: ignore[assignment]
    else:
        feed_id = feed_ids[0]

    feed_entries = [e for e in merged if e["key"][0] == feed_id]
    feed_entries.sort(key=lambda e: e["key"][1])

    local_frame_map = _build_frame_map(local_run_dir, feed_id)
    or_frame_map = _build_frame_map(or_run_dir, feed_id)

    total = len(feed_entries)
    total_pages = max(1, (total + _FRAMES_PER_PAGE - 1) // _FRAMES_PER_PAGE)

    pg1, pg2, pg3 = st.columns([3, 1, 1])
    with pg2:
        page: int = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key=f"bm_page_{bm_id}_{feed_id}")  # type: ignore[assignment]
    with pg3:
        st.markdown(f"<div style='padding-top:32px;color:#888'>/ {total_pages}</div>", unsafe_allow_html=True)
    with pg1:
        s = (_FRAMES_PER_PAGE * (page - 1)) + 1
        e_end = min(_FRAMES_PER_PAGE * page, total)
        st.markdown(f"<div style='padding-top:32px;color:#888;font-size:0.85rem'>Frames {s}–{e_end} of {total}</div>", unsafe_allow_html=True)

    # Persistent column headers
    lhdr, rhdr = st.columns(2)
    lhdr.markdown(
        f'<div style="background:#14532d22;border:1px solid #22c55e44;border-radius:6px;'
        f'padding:6px 12px;text-align:center;font-weight:700;color:#22c55e">🖥️ Local — {local_model_label}</div>',
        unsafe_allow_html=True,
    )
    rhdr.markdown(
        f'<div style="background:#1e1b4b22;border:1px solid #6366f144;border-radius:6px;'
        f'padding:6px 12px;text-align:center;font-weight:700;color:#818cf8">🌐 OpenRouter — {or_model_label}</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    start = (page - 1) * _FRAMES_PER_PAGE
    for entry in feed_entries[start : start + _FRAMES_PER_PAGE]:
        frame_idx = entry["key"][1]
        loc = entry["local"] or {}
        orr = entry["or"] or {}
        loc_ms = (loc.get("latency_s") or 0) * 1000
        or_ms = (orr.get("latency_s") or 0) * 1000
        loc_ptok = loc.get("prompt_tokens", "?")
        loc_ctok = loc.get("completion_tokens", "?")
        or_ptok = orr.get("prompt_tokens", "?")
        or_ctok = orr.get("completion_tokens", "?")

        st.markdown(f"**Frame {frame_idx}**  ·  Feed {feed_id}")
        loc_col, or_col = st.columns(2, gap="medium")

        with loc_col:
            with st.container(border=True):
                local_fp = local_frame_map.get(frame_idx)
                if local_fp and local_fp.exists():
                    st.image(str(local_fp), use_container_width=True)
                else:
                    _frame_placeholder(frame_idx)
                st.markdown(
                    f'<div style="background:#14532d22;border-left:3px solid #22c55e;padding:4px 10px;'
                    f'border-radius:0 4px 4px 0;margin:6px 0">'
                    f'{_latency_pill(loc_ms, "#22c55e" if loc_ms < 5000 else "#ef4444")}'
                    f'&nbsp;&nbsp;<span style="color:#888;font-size:0.78rem">🔤 {loc_ptok}→{loc_ctok}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                _output_block(entry["local"])

        with or_col:
            with st.container(border=True):
                or_fp = or_frame_map.get(frame_idx)
                if or_fp and or_fp.exists():
                    st.image(str(or_fp), use_container_width=True)
                else:
                    _frame_placeholder(frame_idx)
                st.markdown(
                    f'<div style="background:#1e1b4b22;border-left:3px solid #6366f1;padding:4px 10px;'
                    f'border-radius:0 4px 4px 0;margin:6px 0">'
                    f'{_latency_pill(or_ms, "#818cf8" if or_ms < 15000 else "#ef4444")}'
                    f'&nbsp;&nbsp;<span style="color:#888;font-size:0.78rem">🔤 {or_ptok}→{or_ctok}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                _output_block(entry["or"])
        st.write("")


def _section_compare_stats(merged: list[dict[str, Any]]) -> None:
    st.subheader("② Latency Comparison")
    local_latencies = [(e["local"]["latency_s"] or 0) * 1000 for e in merged if e["local"] and e["local"].get("status") == "ok"]
    or_latencies = [(e["or"]["latency_s"] or 0) * 1000 for e in merged if e["or"] and e["or"].get("status") == "ok"]

    def _p(vals: list[float], pct: int) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        idx = min(int(len(s) * pct / 100), len(s) - 1)
        return s[idx]

    metrics = [
        ("Frames", len(local_latencies), len(or_latencies)),
        ("P50 ms", _p(local_latencies, 50), _p(or_latencies, 50)),
        ("P95 ms", _p(local_latencies, 95), _p(or_latencies, 95)),
        ("P99 ms", _p(local_latencies, 99), _p(or_latencies, 99)),
    ]

    cols = st.columns(len(metrics))
    for col, (label, lv, ov) in zip(cols, metrics):
        with col:
            if label == "Frames":
                st.metric(f"Local {label}", int(lv))
                st.metric(f"OR {label}", int(ov))
            else:
                delta = round(ov - lv, 1)
                st.metric(f"Local {label}", f"{lv:.0f}")
                st.metric(f"OR {label}", f"{ov:.0f}", delta=f"{delta:+.0f} ms", delta_color="inverse")


# ── Export helpers ─────────────────────────────────────────────────────────────

# Columns that appear once per row (frame metadata)
_META_COLS = ["frame_index", "feed_id"]

# Columns per model side (in this order)
_SIDE_COLS = ["status", "latency_ms", "ttft_ms", "prompt_tokens", "completion_tokens", "output_json", "error_msg"]


def _export_csv(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    buf = io.StringIO()
    # Build flat header: meta cols, then local_ prefixed, then or_ prefixed
    fieldnames = _META_COLS + [f"local_{c}" for c in _SIDE_COLS] + [f"or_{c}" for c in _SIDE_COLS]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _export_xlsx(rows: list[dict[str, Any]], local_label: str, or_label: str) -> bytes:
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Benchmark"  # type: ignore[union-attr]

    if not rows:
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()

    # Row 1: group headers (meta | LOCAL (merged) | OPENROUTER (merged))
    # Row 2: field sub-headers
    # Row 3+: data

    meta_count = len(_META_COLS)
    side_count = len(_SIDE_COLS)
    local_start = meta_count + 1          # 1-based col where local block starts
    or_start = local_start + side_count   # 1-based col where OR block starts

    dark_fill = PatternFill(start_color="0f172a", end_color="0f172a", fill_type="solid")
    local_fill = PatternFill(start_color="14532d", end_color="14532d", fill_type="solid")
    or_fill = PatternFill(start_color="1e1b4b", end_color="1e1b4b", fill_type="solid")
    sub_fill = PatternFill(start_color="1e293b", end_color="1e293b", fill_type="solid")
    bold_white = Font(bold=True, color="f8fafc")
    local_font = Font(bold=True, color="86efac")
    or_font = Font(bold=True, color="a5b4fc")
    sub_font = Font(bold=True, color="cbd5e1")
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # Row 1: group headers
    for col_idx in range(1, meta_count + 1):
        cell = ws.cell(row=1, column=col_idx, value="")  # type: ignore[union-attr]
        cell.fill = dark_fill
    # Local group header (merge across side_count cols)
    local_end = local_start + side_count - 1
    or_end = or_start + side_count - 1
    ws.cell(row=1, column=local_start, value=f"🖥️ LOCAL — {local_label}").fill = local_fill  # type: ignore[union-attr]
    ws.cell(row=1, column=local_start).font = local_font  # type: ignore[union-attr]
    ws.cell(row=1, column=local_start).alignment = center  # type: ignore[union-attr]
    ws.merge_cells(start_row=1, start_column=local_start, end_row=1, end_column=local_end)  # type: ignore[union-attr]
    ws.cell(row=1, column=or_start, value=f"🌐 OPENROUTER — {or_label}").fill = or_fill  # type: ignore[union-attr]
    ws.cell(row=1, column=or_start).font = or_font  # type: ignore[union-attr]
    ws.cell(row=1, column=or_start).alignment = center  # type: ignore[union-attr]
    ws.merge_cells(start_row=1, start_column=or_start, end_row=1, end_column=or_end)  # type: ignore[union-attr]

    # Row 2: sub-headers
    for col_idx, name in enumerate(_META_COLS, 1):
        cell = ws.cell(row=2, column=col_idx, value=name)  # type: ignore[union-attr]
        cell.fill = sub_fill
        cell.font = bold_white
        cell.alignment = center
    for offset, name in enumerate(_SIDE_COLS):
        for col_base, fnt in [(local_start, local_font), (or_start, or_font)]:
            cell = ws.cell(row=2, column=col_base + offset, value=name)  # type: ignore[union-attr]
            cell.fill = sub_fill
            cell.font = fnt
            cell.alignment = center

    # Data rows
    alt_fill = PatternFill(start_color="0f172a", end_color="0f172a", fill_type="solid")
    for row_idx, row in enumerate(rows, 3):
        fill = alt_fill if row_idx % 2 == 1 else None
        for col_idx, key in enumerate(_META_COLS, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=row.get(key, ""))  # type: ignore[union-attr]
            if fill:
                cell.fill = fill
        for offset, name in enumerate(_SIDE_COLS):
            lval = row.get(f"local_{name}", "")
            oval = row.get(f"or_{name}", "")
            lc = ws.cell(row=row_idx, column=local_start + offset, value=lval)  # type: ignore[union-attr]
            oc = ws.cell(row=row_idx, column=or_start + offset, value=oval)  # type: ignore[union-attr]
            if fill:
                lc.fill = fill
                oc.fill = fill
            # Wrap long JSON strings
            if name == "output_json":
                lc.alignment = Alignment(wrap_text=True, vertical="top")
                oc.alignment = Alignment(wrap_text=True, vertical="top")

    # Column widths
    col_widths = {col_idx: 14 for col_idx in range(1, or_end + 1)}
    for col_idx in range(1, meta_count + 1):
        col_widths[col_idx] = 12
    # output_json cols get wider
    json_local_col = local_start + _SIDE_COLS.index("output_json")
    json_or_col = or_start + _SIDE_COLS.index("output_json")
    col_widths[json_local_col] = 60
    col_widths[json_or_col] = 60
    from openpyxl.utils import get_column_letter
    for col_idx, width in col_widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = width  # type: ignore[union-attr]

    ws.freeze_panes = "A3"  # type: ignore[union-attr]
    ws.row_dimensions[1].height = 28  # type: ignore[union-attr]
    ws.row_dimensions[2].height = 22  # type: ignore[union-attr]

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ── Results tab ────────────────────────────────────────────────────────────────

def _render_compare(runs_dir: Path, benchmarks_dir: Path) -> None:
    from bench.utils.benchmark_store import build_export_rows, list_benchmarks, merge_outputs
    from bench.utils.run_loader import load_outputs

    benchmarks = list_benchmarks(benchmarks_dir)
    if not benchmarks:
        st.markdown(
            '<div style="text-align:center;padding:4rem 2rem;color:#666">'
            "<h3>No benchmarks yet</h3>"
            "<p>Go to <b>⚙️ Configure &amp; Run</b> to start one.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    bm_ids = [b["benchmark_id"] for b in benchmarks]
    default_idx = 0
    last = st.session_state.get("bm_last_benchmark_id")
    if last and last in bm_ids:
        default_idx = bm_ids.index(last)

    def _bm_label(b: dict[str, Any]) -> str:
        ts = b.get("started_at_iso", "")[:16].replace("T", " ")
        icon = _STATUS_ICON.get(b.get("status", ""), "⚪")
        return f"{icon}  {b['benchmark_id']}   {ts}   local:{b.get('local_status','?')} / or:{b.get('or_status','?')}"

    sel_idx: int = st.selectbox("Select benchmark", range(len(benchmarks)), format_func=lambda i: _bm_label(benchmarks[i]), index=default_idx, key="bm_selector")  # type: ignore[assignment]

    bm = benchmarks[sel_idx]
    local_run_id = bm.get("local_run_id", "")
    or_run_id = bm.get("or_run_id", "")
    local_model = bm.get("local_model", "local")
    or_model = bm.get("or_model", "openrouter")

    mc1, mc2 = st.columns(2)
    mc1.markdown(f"**Local run:** `{local_run_id}`  ·  `{local_model}`")
    mc2.markdown(f"**OpenRouter run:** `{or_run_id}`  ·  `{or_model}`")
    st.divider()

    local_run_dir = runs_dir / local_run_id
    or_run_dir = runs_dir / or_run_id
    local_outputs = load_outputs(local_run_dir)
    or_outputs = load_outputs(or_run_dir)

    merged = merge_outputs(local_outputs, or_outputs)

    local_label = Path(local_model).name if local_model else "local"
    or_label = or_model

    _section_compare_frames(merged, local_run_dir, or_run_dir, bm["benchmark_id"], local_label, or_label)
    st.divider()
    _section_compare_stats(merged)
    st.divider()

    # ── Export ────────────────────────────────────────────────────────────────
    st.subheader("③ Export")
    rows = build_export_rows(bm["benchmark_id"], merged, local_label, or_label)
    if not rows:
        st.info("No data to export.")
        return

    ex1, ex2 = st.columns(2)
    with ex1:
        st.download_button(
            "⬇ Download CSV",
            data=_export_csv(rows),
            file_name=f"{bm['benchmark_id']}.csv",
            mime="text/csv",
            width="stretch",
        )
    with ex2:
        st.download_button(
            "⬇ Download XLSX",
            data=_export_xlsx(rows, local_label, or_label),
            file_name=f"{bm['benchmark_id']}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )


# ── Entry point ────────────────────────────────────────────────────────────────

def render_benchmark(root: Path, runs_dir: Path, benchmarks_dir: Path) -> None:
    tab_cfg, tab_compare = st.tabs(["⚙️ Configure & Run", "📊 Compare Results"])

    with tab_cfg:
        is_running = bool(st.session_state.get("bm_run_in_progress", False))
        cfg = _render_config(root)

        st.divider()
        btn_col, status_col = st.columns([1, 4])
        with btn_col:
            launch = st.button("▶ Run Benchmark", disabled=is_running, type="primary", key="bm_launch_btn")
        if is_running:
            status_col.info("A benchmark is in progress…")

        if launch and not is_running and cfg is not None:
            _run_benchmark(root, runs_dir, benchmarks_dir, cfg)

    with tab_compare:
        _render_compare(runs_dir, benchmarks_dir)
