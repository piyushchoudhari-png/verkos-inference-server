from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
import yaml
from pydantic import ValidationError


def _init_state() -> None:
    if "cfg_video_path_ids" not in st.session_state:
        st.session_state["cfg_video_path_ids"] = [uuid.uuid4().hex[:8]]


def _video_sources_enter_path() -> list[str]:
    path_ids: list[str] = st.session_state["cfg_video_path_ids"]
    to_remove: str | None = None

    for pid in path_ids:
        c1, c2 = st.columns([10, 1])
        with c1:
            st.text_input(
                "Path",
                key=f"cfg_vpath_{pid}",
                label_visibility="collapsed",
                placeholder="/path/to/video.mp4",
            )
        with c2:
            st.write("")
            if len(path_ids) > 1 and st.button("✕", key=f"cfg_rm_{pid}"):
                to_remove = pid

    if to_remove:
        path_ids.remove(to_remove)
        st.rerun()

    if st.button("+ Add video", key="cfg_add_video"):
        path_ids.append(uuid.uuid4().hex[:8])
        st.rerun()

    return [
        v
        for pid in path_ids
        if (v := st.session_state.get(f"cfg_vpath_{pid}", "").strip())
    ]


def _video_sources_upload(root: Path) -> list[str]:
    uploaded = st.file_uploader(
        "Upload video files",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
        key="cfg_vid_upload",
    )
    paths: list[str] = []
    tmp_dir = root / "bench" / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    for uf in uploaded or []:
        dest = tmp_dir / uf.name
        dest.write_bytes(uf.read())
        paths.append(str(dest.resolve()))
    return paths


def render_configure(root: Path, runs_dir: Path) -> None:  # noqa: C901
    _init_state()

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode: str = st.radio(
        "Inference mode",
        ["local", "openrouter"],
        horizontal=True,
        key="cfg_mode",
    )  # type: ignore[assignment]

    st.divider()

    # ── Mode-specific config ───────────────────────────────────────────────────
    model_path: str | None = None
    dtype: str = "auto"
    tensor_parallel: int = 1
    gpu_mem: float = 0.90
    max_model_len: int | None = None
    gpu_poll_s: float = 0.5

    or_model: str = "qwen/qwen3-vl-32b-instruct"
    or_timeout: float = 60.0
    or_api_key: str = ""

    if mode == "local":
        st.subheader("Local GPU Config")
        models_dir = root / "models"
        model_dirs: list[str] = (
            sorted(d.name for d in models_dir.iterdir() if d.is_dir())
            if models_dir.exists() and models_dir.is_dir()
            else []
        )
        if not model_dirs:
            st.warning("No models found in `models/`. Run `scripts/download_model.py` to add one.")
        else:
            sel_model: str = st.selectbox("Model", model_dirs, key="cfg_model")  # type: ignore[assignment]
            model_path = str(models_dir / sel_model)
            st.caption(f"`{model_path}`")

        lc1, lc2, lc3 = st.columns(3)
        dtype = lc1.selectbox("dtype", ["auto", "float16", "bfloat16", "float32"], key="cfg_dtype")  # type: ignore[assignment]
        tensor_parallel = lc2.number_input("Tensor parallel", min_value=1, max_value=8, value=1, step=1, key="cfg_tp")  # type: ignore[assignment]
        gpu_mem = lc3.slider("GPU memory util", min_value=0.5, max_value=1.0, value=0.90, step=0.01, key="cfg_gpu_mem")  # type: ignore[assignment]

        mml_raw: str = st.text_input("Max model len (blank = model default)", value="", key="cfg_max_model_len")
        max_model_len = int(mml_raw) if mml_raw.strip().isdigit() else None

    else:
        st.subheader("OpenRouter Config")
        oc1, oc2 = st.columns(2)
        or_model = oc1.text_input("Model", value="qwen/qwen3-vl-32b-instruct", key="cfg_or_model")
        or_timeout = oc2.number_input("Timeout (s)", min_value=10.0, max_value=300.0, value=60.0, step=5.0, key="cfg_or_timeout")  # type: ignore[assignment]
        or_api_key = st.text_input(
            "API Key (or set OPENROUTER_API_KEY env var)",
            type="password",
            key="cfg_or_api_key",
        )

    st.divider()

    # ── Video sources ──────────────────────────────────────────────────────────
    st.subheader("Video Sources")
    vid_method: str = st.radio(
        "Input method",
        ["Enter path", "Upload file"],
        horizontal=True,
        key="cfg_vid_method",
        label_visibility="collapsed",
    )  # type: ignore[assignment]

    video_paths = _video_sources_upload(root) if vid_method == "Upload file" else _video_sources_enter_path()

    st.divider()

    # ── Frame sampling ─────────────────────────────────────────────────────────
    st.subheader("Frame Sampling")
    fs1, fs2, fs3 = st.columns(3)
    num_feeds: int = fs1.number_input("Num feeds", min_value=1, max_value=64, value=1, step=1, key="cfg_num_feeds")  # type: ignore[assignment]
    frame_interval: float = fs2.number_input("Frame interval (s)", min_value=0.1, max_value=300.0, value=10.0, step=0.5, key="cfg_frame_interval")  # type: ignore[assignment]
    max_frames_raw: str = fs3.text_input("Max frames/feed (blank = ∞)", value="", key="cfg_max_frames")
    max_frames_per_feed: int | None = int(max_frames_raw) if max_frames_raw.strip().isdigit() else None

    re1, re2 = st.columns(2)
    resolution_raw: str = re1.text_input("Resolution WxH (blank = native)", value="", key="cfg_resolution")
    downscale_raw: str = re2.text_input("Downscale factor (blank = none)", value="", key="cfg_downscale")

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
    g1, g2, g3 = st.columns(3)
    max_tokens: int = g1.number_input("Max tokens", min_value=1, max_value=8192, value=256, step=1, key="cfg_max_tokens")  # type: ignore[assignment]
    temperature: float = g2.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="cfg_temperature")  # type: ignore[assignment]
    if mode == "local":
        gpu_poll_s = g3.number_input("GPU poll interval (s)", min_value=0.1, max_value=10.0, value=0.5, step=0.1, key="cfg_gpu_poll")  # type: ignore[assignment]

    st.divider()

    # ── Run settings ──────────────────────────────────────────────────────────
    st.subheader("Run Settings")
    rs1, rs2 = st.columns(2)
    output_dir: str = rs1.text_input("Output dir", value=str(runs_dir), key="cfg_output_dir")
    run_id_input: str = rs2.text_input("Run ID (blank = auto)", value="", key="cfg_run_id")

    st.divider()

    # ── Prompts ───────────────────────────────────────────────────────────────
    st.subheader("Prompts")
    prompt_tab, _ = st.tabs(["Prompts", "Advanced"])

    default_sys = ""
    default_usr = ""
    prompts_yaml_path = root / "src" / "config" / "prompts.yaml"
    if prompts_yaml_path.exists():
        try:
            with open(prompts_yaml_path) as _f:
                _pd = yaml.safe_load(_f) or {}
            default_sys = _pd.get("system", "")
            default_usr = _pd.get("user", "")
        except Exception:
            pass

    with prompt_tab:
        system_prompt: str = st.text_area("System prompt", value=default_sys, height=200, key="cfg_sys_prompt")
        user_prompt: str = st.text_area("User prompt", value=default_usr, height=100, key="cfg_usr_prompt")

    st.divider()

    # ── Launch ────────────────────────────────────────────────────────────────
    is_running = bool(st.session_state.get("cfg_run_in_progress", False))
    btn_col, status_col = st.columns([1, 4])
    with btn_col:
        launch = st.button("▶ Launch Run", disabled=is_running, type="primary", key="cfg_launch_btn")
    if is_running:
        status_col.info("A run is in progress…")

    if not launch or is_running:
        return

    if not video_paths:
        st.error("Add at least one video path before launching.")
        return

    # Build and validate config
    config_dict: dict[str, Any] = {
        "inference_mode": mode,
        "video_paths": video_paths,
        "num_feeds": int(num_feeds),
        "frame_interval_s": float(frame_interval),
        "max_frames_per_feed": max_frames_per_feed,
        "resolution": resolution,
        "downscale_factor": downscale_factor,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "output_dir": output_dir,
    }

    if mode == "local":
        config_dict["model_path"] = model_path
        config_dict["dtype"] = dtype
        config_dict["tensor_parallel_size"] = int(tensor_parallel)
        config_dict["gpu_memory_utilization"] = float(gpu_mem)
        config_dict["max_model_len"] = max_model_len
        config_dict["gpu_poll_interval_s"] = float(gpu_poll_s)
    else:
        effective_key = or_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        config_dict["openrouter"] = {
            "api_key": effective_key,
            "model": or_model,
            "timeout_s": float(or_timeout),
        }

    run_id = run_id_input.strip() or f"{uuid.uuid4().hex[:8]}-{time.strftime('%Y%m%d%H%M%S')}"
    ts = int(time.time())
    tmp_dir = root / "bench" / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    prompts_path = tmp_dir / f"prompts_{ts}.yaml"
    with open(prompts_path, "w") as pf:
        yaml.dump({"system": system_prompt, "user": user_prompt}, pf, allow_unicode=True)
    config_dict["prompts_file"] = str(prompts_path.resolve())

    try:
        from src.config import SimConfig
        SimConfig.model_validate(config_dict)
    except ValidationError as exc:
        st.error(f"Configuration error:\n```\n{exc}\n```")
        return

    config_path = tmp_dir / f"config_{ts}.yaml"
    with open(config_path, "w") as cf:
        yaml.dump(config_dict, cf, allow_unicode=True)

    # ── Subprocess ────────────────────────────────────────────────────────────
    from bench.utils.subprocess_runner import RunProcess

    st.session_state["cfg_run_in_progress"] = True

    log_lines: list[str] = []
    st.caption("Run log")
    log_container = st.container(height=350, border=True)
    log_area = log_container.empty()

    runner = RunProcess(str(config_path.resolve()), run_id)
    st.session_state["active_subprocess"] = runner

    try:
        for line in runner:
            log_lines.append(line)
            log_area.code("\n".join(log_lines[-200:]), language="text")
    finally:
        st.session_state["cfg_run_in_progress"] = False
        st.session_state["active_subprocess"] = None

    if runner.returncode == 0:
        st.toast("Run complete!", icon="✅")
        st.session_state["last_run_id"] = run_id
        st.success(f"Run `{run_id}` complete — switch to 🔬 Results to inspect it.")
    else:
        st.error("Run failed — check log above.")
