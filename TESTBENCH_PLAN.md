# Verkos Inference Server — Streamlit Test Bench Plan

## Goal

Replace the ad-hoc CLI workflow with a self-contained Streamlit dashboard that lets a
developer configure, launch, and inspect profiling runs against either a local GPU
(vLLM) or OpenRouter — without touching any YAML files directly.

---

## Directory Layout After Full Build

```
verkos-inference-server/
├── src/                          # existing — minor additions only
│   ├── reporter.py               # + write run_meta.json, config_snapshot.json
│   └── config/
│       └── sim_config.yaml       # kept for CLI fallback; UI generates its own
├── runs/                         # new canonical output root (replaces ./results)
│   └── <run_id>/
│       ├── run_meta.json         # NEW — metadata index for the UI
│       ├── config_snapshot.json  # NEW — full SimConfig used for this run
│       ├── outputs.json          # existing MetricSample array
│       ├── gpu.json              # existing GpuSample array
│       ├── engine.json           # existing EngineSample array
│       ├── summary.txt           # existing human-readable summary
│       └── frames/               # existing annotated JPEGs
│           ├── feed0_frame0000.jpg
│           └── ...
├── models/                       # existing; UI scans this folder for local model picker
│   └── smolvlm-256m/
├── bench/                        # NEW — Streamlit app root
│   ├── app.py                    # single entry point: `streamlit run bench/app.py`
│   ├── tabs/
│   │   ├── results.py            # Results tab
│   │   └── configure.py          # Configure & Run tab
│   └── utils/
│       ├── run_loader.py         # scan runs/, parse JSON artifacts
│       ├── gpu_charts.py         # Plotly helpers for GPU time-series
│       └── subprocess_runner.py  # launch + stream subprocess logs
└── TESTBENCH_PLAN.md             # this file
```

---

## Data Contract: `run_meta.json`

Every run directory gains a `run_meta.json` written by `reporter.py` at the end of the
run. This is the index entry the UI reads to populate the run selector without loading
all JSON artifacts upfront.

```json
{
  "run_id": "abc123-20260515143200",
  "started_at_iso": "2026-05-15T14:32:00Z",
  "finished_at_iso": "2026-05-15T14:35:47Z",
  "inference_mode": "local",
  "model": "smolvlm-256m",
  "num_feeds": 2,
  "total_frames": 40,
  "successful_frames": 38,
  "error_count": 2,
  "latency_p50_ms": 312.4,
  "throughput_fps": 0.218,
  "video_paths": ["/abs/path/to/input4.mp4"],
  "status": "completed"
}
```

A `config_snapshot.json` (full `SimConfig.model_dump()`) is also written so the UI can
restore every parameter from a past run.

---

## Phase 1 — Backend: Run Metadata & Output Folder

**Files changed:** `src/reporter.py`, `src/config.py`

### Tasks

1. **Change default `output_dir`** in `SimConfig` from `"./sim_results"` to `"./runs"`.

2. **Write `run_meta.json`** at the end of `write_results()`:
   - Includes `run_id`, ISO timestamps (derive from `t0_epoch` of first sample and
     `t0_epoch + latency_s` of last sample), `inference_mode`, model label, feed count,
     aggregate stats (total_frames, successful_frames, error_count, latency_p50_ms,
     throughput_fps), `video_paths`, `status: "completed"`.

3. **Write `config_snapshot.json`** — `config.model_dump()` serialised to JSON in the
   run directory.  Mask `openrouter.api_key` → `"***"` before writing.

4. **Write a `run_meta.json` stub at run start** (in `runner.py` before feed tasks are
   launched) with `status: "running"`. Overwrite it with the final version on completion.
   This lets the UI show in-progress runs in the selector.

5. **No other changes** to `runner.py`, `feed_runner.py`, `engine.py`, etc.

### Deliverable
`runs/<run_id>/` contains all five artifacts plus annotated frames.  Old `results/`
behaviour unchanged for anyone using the CLI directly.

---

## Phase 2 — Streamlit Scaffold & Run Selector

**New files:** `bench/app.py`, `bench/utils/run_loader.py`

### Tasks

1. **`bench/app.py`** — top-level entry point.
   - `st.set_page_config(layout="wide", page_title="Verkos Test Bench")`
   - Two tabs: `[🔬 Results, ⚙️ Configure & Run]`
   - Import and render each tab module.

2. **`bench/utils/run_loader.py`** — stateless helpers:
   - `list_runs(runs_dir) → list[dict]` — scan `runs/*/run_meta.json`, return sorted
     by `started_at_iso` descending.
   - `load_outputs(run_dir) → list[dict]` — parse `outputs.json`.
   - `load_gpu(run_dir) → list[dict]` — parse `gpu.json`.
   - `load_engine(run_dir) → list[dict]` — parse `engine.json`.
   - `load_config_snapshot(run_dir) → dict` — parse `config_snapshot.json`.
   - `frames_for_feed(run_dir, feed_id) → list[Path]` — glob `frames/feed{feed_id}_*.jpg`
     sorted by frame index.

3. **Run selector widget** (in `bench/tabs/results.py`):
   - `st.selectbox` populated from `list_runs()`.
   - Label format: `{run_id}  [{inference_mode}]  {started_at_iso}  —  {total_frames} frames`.
   - Show a "no runs yet" empty-state message if directory is empty.

### Deliverable
`streamlit run bench/app.py` opens a two-tab app.  Results tab shows a run dropdown.
Configure tab is a blank placeholder.

---

## Phase 3 — Results Tab: Video, Frames & JSON Output

**New/changed files:** `bench/tabs/results.py`

This is the primary display phase.  The Results tab renders in this exact order:

```
┌─────────────────────────────────────────────────┐
│  Run selector dropdown                           │
│  Run status badge  [completed / running / error] │
├─────────────────────────────────────────────────┤
│  ① INPUT VIDEO                                   │
│     st.video(video_path) if path exists          │
│     else: show absolute path as info text        │
├─────────────────────────────────────────────────┤
│  ② FRAMES & DETECTIONS                           │
│     Feed selector (if num_feeds > 1)             │
│     For each frame (paginated, 5 per page):      │
│     ┌──────────────────┬───────────────────────┐ │
│     │ Annotated frame  │ JSON output           │ │
│     │ (st.image)       │ (st.json / st.code)   │ │
│     │                  │ Latency, TTFT, tokens │ │
│     └──────────────────┴───────────────────────┘ │
├─────────────────────────────────────────────────┤
│  ③ SUMMARY STATS (expander, open by default)     │
│     Metric cards: p50/p95/p99 latency, fps,      │
│     error rate, prefill/decode tok/s             │
│     Per-feed latency breakdown table             │
├─────────────────────────────────────────────────┤
│  ④ GPU METRICS  (hidden for openrouter runs)     │
│     (Phase 4)                                    │
└─────────────────────────────────────────────────┘
```

### Tasks

1. **Input video section** — read `video_paths` from `run_meta.json`.
   - If `inference_mode == "openrouter"` and local path exists → `st.video(path)`.
   - If path doesn't exist (container/remote run) → show `st.info("Video: {path}")`.
   - Support multiple video_paths: one expander per video.

2. **Frame gallery**:
   - Group `outputs.json` records by `feed_id`.
   - Feed selector: `st.radio` (horizontal) if `num_feeds > 1`.
   - Page through frames 5 at a time (`st.number_input` page selector).
   - Each frame row: `st.columns([1, 1])`.
     - Left: `st.image(frame_jpeg_path, caption=f"Frame {frame_index}")`.
       Fall back to grey placeholder if JPEG not found (openrouter or run still in
       progress).
     - Right:
       - `st.json(json.loads(output_text))` if `output_text` is valid JSON, else
         `st.code(output_text)`.
       - Below JSON: small metadata row —
         `Latency: {latency_s*1000:.0f} ms | TTFT: {ttft_s*1000:.0f} ms | Tokens: {prompt_tokens}→{completion_tokens}`
       - If `status == "error"`: red `st.error(error_msg)` instead of JSON.

3. **Summary stats section**:
   - Pull `aggregate()` from `src.metrics` on loaded `outputs.json` records.
   - Show metric cards using `st.columns(4)`:
     - P50 latency, P95 latency, Throughput fps, Error rate %.
   - Second row: P99 latency, Prefill tok/s P50, Decode tok/s P50, Run duration.
   - `st.dataframe` for per-feed breakdown table.

### Deliverable
Full frame-by-frame inspection of any completed run with input video, annotated frames,
detection JSON, and summary stats.

---

## Phase 4 — GPU Metrics Section (local mode only)

**New/changed files:** `bench/tabs/results.py`, `bench/utils/gpu_charts.py`

### Tasks

1. **Guard**: render this entire section only when `run_meta.json["inference_mode"] == "local"`.

2. **`bench/utils/gpu_charts.py`** — Plotly figure builders (each returns a `go.Figure`):
   - `vram_chart(gpu_samples)` — line chart of `memory_used_bytes / 1e9` (GB) over
     `t_epoch - t_start` (seconds).  One trace per `gpu_index`.  Add horizontal dashed
     line for idle baseline from `IdleBaseline` if present.
   - `gpu_util_chart(gpu_samples)` — line chart of `gpu_util_pct` over time.
   - `power_chart(gpu_samples)` — line chart of `power_w` over time (skip if all None).
   - `temp_chart(gpu_samples)` — line chart of `temperature_c` over time (skip if all None).
   - `engine_chart(engine_samples)` — multi-line chart:
     - `num_running`, `num_waiting`, `num_swapped` as stacked area or separate lines.
     - Secondary Y-axis: `gpu_kv_cache_usage_pct`.

3. **GPU section layout** in Results tab:
   ```
   ┌─────────────────────────────────────────────────────────┐
   │  GPU METRICS  ▼ (expander)                              │
   │  ┌──────────────────┬────────────────────────────────┐  │
   │  │ VRAM Usage (GB)  │ GPU Utilization (%)            │  │
   │  │ [Plotly chart]   │ [Plotly chart]                 │  │
   │  ├──────────────────┼────────────────────────────────┤  │
   │  │ Power Draw (W)   │ Temperature (°C)               │  │
   │  │ [Plotly chart]   │ [Plotly chart]                 │  │
   │  └──────────────────┴────────────────────────────────┘  │
   │                                                         │
   │  vLLM SCHEDULER STATS  ▼ (expander)                     │
   │  [engine_chart — running / waiting / swapped / KV%]     │
   │                                                         │
   │  VRAM SUMMARY TABLE                                     │
   │  Idle (after load) | Peak Active | Delta | +20% spec   │
   └─────────────────────────────────────────────────────────┘
   ```

4. **VRAM summary table** — built from `gpu.json` + `IdleBaseline` (stored in
   `config_snapshot.json` for now; if needed, add `baselines.json` write to reporter).

### Deliverable
Time-series GPU charts appear at the bottom of every local-mode run.  OpenRouter runs
show nothing in this section.

---

## Phase 5 — Configure & Run Tab

**New/changed files:** `bench/tabs/configure.py`, `bench/utils/subprocess_runner.py`

### Layout

```
Configure & Run tab
├── Mode selector          [Local GPU | OpenRouter]  (st.radio, horizontal)
│
├── IF LOCAL:
│   ├── Model selector     scan models/ → st.selectbox with subfolder names
│   ├── dtype              st.selectbox ["auto","float16","bfloat16","float32"]
│   ├── gpu_memory_util    st.slider 0.5–1.0
│   ├── tensor_parallel    st.number_input 1–8
│   └── max_model_len      st.number_input (optional, leave blank = None)
│
├── IF OPENROUTER:
│   ├── model              st.text_input (default: qwen/qwen3-vl-32b-instruct)
│   ├── api_key            st.text_input(type="password")  — also accepts OPENROUTER_API_KEY env
│   └── timeout_s          st.number_input
│
├── VIDEO SOURCES
│   ├── Input method       [Upload file | Enter path]  (st.radio)
│   ├── Upload:            st.file_uploader(accept_multiple_files=True) → saves to tmp/
│   ├── Path:              dynamic list of st.text_input rows + [+ Add path] button
│   └── num_feeds          st.number_input (how many concurrent feeds to simulate)
│
├── FRAME SAMPLING
│   ├── frame_interval_s   st.number_input
│   ├── max_frames/feed    st.number_input (0 = unlimited)
│   ├── resolution         st.text_input "WxH" or blank (mutually exclusive with downscale)
│   └── downscale_factor   st.number_input (blank = None)
│
├── GENERATION
│   ├── max_tokens         st.number_input
│   ├── temperature        st.slider 0.0–1.0
│   └── gpu_poll_interval  st.number_input (hidden for openrouter)
│
├── RUN SETTINGS
│   ├── output_dir         st.text_input (default: ./runs)
│   └── run_id             st.text_input (blank = auto-generate)
│
├── [Prompts tab inside this section — st.tabs(["Prompts", "Advanced"])]
│   ├── system_prompt      st.text_area (prepopulated from prompts.yaml)
│   └── user_prompt        st.text_area
│
└── [▶ Launch Run]  button
    └── On click: validate → build SimConfig → write tmp YAML → spawn subprocess
        Log pane: st.empty() updates with stdout lines in real time
        Completion: auto-select new run in Results tab + success toast
```

### Tasks

1. **Mode switch** — `st.radio("Inference mode", ["local", "openrouter"], horizontal=True)`.
   Conditionally render local vs openrouter config blocks.

2. **Model picker (local)** — scan `models/` directory for subdirectories, present as
   `st.selectbox`.  Show path preview below.  "No models found" warning with link to
   `scripts/download_model.py` instructions.

3. **Video input** — Two modes:
   - *Upload*: `st.file_uploader` saves files to `bench/tmp/` (gitignored), returns
     absolute paths for config.
   - *Path*: Maintain a `st.session_state["video_paths"]` list.  Render one text input
     per path.  "+ Add video" appends empty string.  "✕" button removes.

4. **Prompts sub-tab** — pre-load `src/config/prompts.yaml` as default text.  User edits
   inline.  Write edited prompts to a temp YAML in `bench/tmp/prompts_{ts}.yaml` before
   launch.

5. **Config validation** — before launch, instantiate `SimConfig(**kwargs)` inside a
   `try/except ValidationError`.  Show `st.error` with the Pydantic message on failure.
   Do not launch if validation fails.

6. **Config serialisation** — on successful validation, write a temp YAML to
   `bench/tmp/config_{ts}.yaml`.  This file is passed to `python -m src --config`.

7. **`bench/utils/subprocess_runner.py`**:
   ```python
   def launch_run(config_path: str, run_id: str) -> Generator[str, None, None]:
       """Yield stdout lines from the run subprocess."""
   ```
   Uses `subprocess.Popen` with `stdout=PIPE, stderr=STDOUT, text=True`.
   Caller iterates lines and pushes into `st.empty()`.

8. **Log pane** in Streamlit:
   - `st.expander("Run log", expanded=True)` with a `st.code` block.
   - Update via `st.empty().code(accumulated_log)` on each line.
   - On process exit code 0: `st.toast("Run complete!", icon="✅")` + store run_id in
     `st.session_state["last_run_id"]` so Results tab auto-selects it.
   - On non-zero exit: `st.error("Run failed — check log above")`.

9. **Concurrent run guard** — store subprocess handle in `st.session_state`.
   Disable Launch button and show spinner if a run is already in progress.

### Deliverable
Full no-YAML configuration UI.  User fills in a form, hits Launch, watches live logs,
and is immediately taken to the Results tab showing the new run.

---

## Phase 6 — Polish & Packaging

**Files:** `bench/app.py`, `pyproject.toml`, `.gitignore`

### Tasks

1. **`bench/tmp/` directory** — create `.gitkeep` + add `bench/tmp/*.yaml`,
   `bench/tmp/*.mp4` to `.gitignore`.

2. **`pyproject.toml` additions** (via `uv add`):
   - `streamlit`
   - `plotly`

3. **`requirements.txt` regeneration** — `uv export --no-dev -o requirements.txt`.

4. **Launch shortcut** — add to `pyproject.toml` `[project.scripts]`:
   ```toml
   bench = "streamlit run bench/app.py"
   ```
   Or document: `uv run streamlit run bench/app.py`.

5. **`st.session_state` initialisation** in `app.py` for keys:
   `selected_run_id`, `last_run_id`, `active_subprocess`, `video_paths`.

6. **Empty-state screens**:
   - Results tab with no runs: friendly illustration + "→ Go to Configure & Run to start
     your first profiling run".
   - Local mode with no models: warning card pointing to `scripts/download_model.py`.

7. **Runs folder auto-creation** — `Path("runs").mkdir(exist_ok=True)` at app startup.

8. **`bench/tmp/` auto-cleanup** — on app startup, delete temp YAMLs older than 24 h.

---

## Implementation Order

| Phase | Focus                              | New files                                                      | Changed files            |
|-------|------------------------------------|----------------------------------------------------------------|--------------------------|
| 1     | Backend metadata                   | —                                                              | `src/reporter.py`, `src/config.py` |
| 2     | Streamlit scaffold + run selector  | `bench/app.py`, `bench/utils/run_loader.py`                   | —                        |
| 3     | Results: video + frames + JSON     | `bench/tabs/results.py`                                        | —                        |
| 4     | GPU metrics charts                 | `bench/utils/gpu_charts.py`                                    | `bench/tabs/results.py`  |
| 5     | Configure & Run tab                | `bench/tabs/configure.py`, `bench/utils/subprocess_runner.py`  | —                        |
| 6     | Polish & packaging                 | `bench/tmp/.gitkeep`                                           | `pyproject.toml`, `.gitignore` |

Each phase is independently shippable.  After Phase 3 the dashboard is already useful
for browsing existing runs; the launch capability lands in Phase 5.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| No YAML in UI | Generate temp YAML internally | CLI stays unchanged; UI builds `SimConfig` via Pydantic first |
| Subprocess vs direct import | Subprocess | vLLM cannot be imported in the Streamlit process; also isolates crashes |
| runs/ not results/ | New `runs/` default | Cleaner name; old behaviour preserved for CLI users with explicit `--output-dir` |
| run_meta.json | Written at start (status=running) + end | UI can show in-progress runs |
| GPU section gated | `inference_mode == "local"` check on run_meta | No NVML data exists for openrouter; avoids empty/confusing charts |
| Frame pagination | 5 frames per page | Loading all annotated JPEGs at once is slow for long runs |
| Prompts inline | Text areas, not file picker | Prompts change frequently; inline edit is faster than a file round-trip |
| Model download | Script only, models/ selectable | Download is a one-time heavy operation; UI just picks from existing weights |

---

## Out of Scope (this plan)

- Run comparison / A-B view across multiple runs
- Live frame streaming during an active run (polling approach acceptable as future work)
- Authentication or multi-user access
- Remote model serving (non-local GPU)
- Docker-ised dashboard
