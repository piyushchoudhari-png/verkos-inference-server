# Verkos VLM GPU Profiling Simulator

Simulates N concurrent drone video feeds sending frames to a locally-loaded VLM, and measures GPU resource consumption. Used to determine minimum viable GPU spec (VRAM, throughput, latency) for the AI-R V2 edge device before hardware procurement.

The simulator drives `AsyncLLMEngine` directly — no FastAPI, no OpenRouter — so profiling results reflect the same batching behaviour the production Verkos detection pipeline will exhibit.

---

## Quick Start

```bash
# 1. Copy and edit the example config
cp simulator/examples/sim_config.yaml my_run.yaml
cp simulator/examples/prompts.yaml my_prompts.yaml
# Edit my_run.yaml: set model_path, video_paths, num_feeds, run_duration_s

# 2. Run
uv run python -m simulator --config my_run.yaml
```

Results land in `./sim_results/` by default.

---

## Prerequisites

- NVIDIA GPU with CUDA 12.4+
- Model weights on disk (see [Model Configuration](#model-configuration))
- At least one MP4 video file (see [Video Sources](#video-sources))
- All Python deps installed via `uv sync`

---

## Configuration

All configuration lives in a single YAML file. Pass it with `--config`.

### Full reference

```yaml
# --- Model ---
model_path: "models/qwen2-vl-7b"   # path to model weights on disk (required)
dtype: "auto"                        # float16 | bfloat16 | auto
gpu_memory_utilization: 0.90         # fraction of GPU VRAM vLLM may use (0.0–1.0)
tensor_parallel_size: 1              # number of GPUs for tensor parallelism

# --- Video sources ---
video_paths:                         # list of MP4 files; cycled across feeds if
  - "data/feed_a.mp4"                # len(video_paths) < num_feeds
  - "data/feed_b.mp4"

# --- Prompts ---
prompts_file: "my_prompts.yaml"      # path to prompts YAML (see Prompts section)

# --- Simulation shape ---
num_feeds: 4                         # concurrent drone feeds
frame_interval_s: 10.0              # seconds between sampled frames per feed
                                     # (5–30 s matches production; lower = more load)

# --- Run termination (at least one required) ---
run_duration_s: 120.0               # stop after this many wall-clock seconds
max_frames_per_feed: null            # hard cap on frames per feed (alternative / additional)

# --- Frame resolution ---
downscale_factor: null               # e.g. 0.5 = half native resolution
resolution: null                     # explicit [width, height] override, e.g. [640, 360]
                                     # mutually exclusive with downscale_factor

# --- Inference ---
max_tokens: 256                      # max tokens per VLM response
temperature: 0.0                     # 0.0 = deterministic (recommended for profiling)

# --- Telemetry ---
gpu_poll_interval_s: 0.5             # pynvml + torch VRAM poll cadence (seconds)

# --- Output ---
output_dir: "./sim_results"          # Parquet + markdown land here
run_id: null                         # auto-generated 12-char hex if not set
```

### Termination conditions

One of `run_duration_s` or `max_frames_per_feed` **must** be set. Both can be set simultaneously — the run stops when the first condition is reached.

| Goal | Setting |
|------|---------|
| Time-bounded run | `run_duration_s: 120` |
| Fixed frame count | `max_frames_per_feed: 20` |
| Both | set both — first to trigger wins |

---

## Prompts

Prompts are loaded from a separate YAML file so they can be versioned and swapped without touching the sim config.

```yaml
# my_prompts.yaml

system: |
  You are an aerial surveillance AI. Detect security-relevant events and return
  a JSON response with "events" (label, confidence, bbox) and "scene_description".

user: |
  Frame {frame_index}. Detect if present: person, vehicle, fire, intrusion.
  Return bounding boxes normalised to [0, 1].
```

`{frame_index}` is replaced at runtime with the sequential frame number within that feed.

See `simulator/examples/prompts.yaml` for a full production-style example.

---

## Model Configuration

### Downloading weights

Use the project's existing download script:

```bash
# Edit config.yaml to set model.source (repo_id, revision)
uv run python scripts/download_model.py --config config.yaml
```

Or download manually via `huggingface-cli`:

```bash
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir models/qwen2-vl-7b
```

### Recommended models for profiling

| Model | VRAM (fp16) | Notes |
|-------|------------|-------|
| `Qwen/Qwen2-VL-7B-Instruct` | ~16 GB | Strong VLM baseline; good JSON output |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | ~22 GB | Higher ceiling test |
| `HuggingFaceTB/SmolVLM-256M-Instruct` | ~1 GB | Fast smoke-test; not production-representative |

Set `model_path` to the local directory containing the weights.

### Quantisation

vLLM supports AWQ and GPTQ. To test a quantised model, ensure the weights include the quantisation config and vLLM picks it up automatically via `dtype: auto`. The sim config has no explicit quantisation knob — unknown vLLM engine args pass through if you need to add them (not currently exposed; edit `engine.py` kwargs if needed).

---

## Video Sources

Any MP4 works. The simulator seeks to the correct timestamp for each frame rather than decoding the entire video, so file size does not affect RAM usage.

```
video_paths:
  - data/sample_drone.mp4
```

**One video, multiple feeds:** the same file is cycled across all feeds:

```yaml
video_paths:
  - data/sample.mp4
num_feeds: 4              # all 4 feeds use sample.mp4
```

**Different video per feed:** provide one path per feed:

```yaml
video_paths:
  - data/feed_0.mp4
  - data/feed_1.mp4
  - data/feed_2.mp4
  - data/feed_3.mp4
num_feeds: 4
```

Short videos loop automatically — a 30-second clip can feed a 2-minute run.

---

## Running

```bash
# Basic
uv run python -m simulator --config my_run.yaml

# Override output directory
uv run python -m simulator --config my_run.yaml --output-dir /data/results

# Set a fixed run ID for reproducible filenames
uv run python -m simulator --config my_run.yaml --run-id baseline_7b_4feeds
```

Progress is logged to stdout. The markdown summary is printed at the end of every run.

---

## Output

Per run, five files are written under `output_dir`:

| File | Purpose |
|------|---------|
| `{run_id}_summary.txt` | Headline numbers, printed to stdout at run end |
| `{run_id}_samples.csv` | One row per inference request |
| `{run_id}_gpu.csv` | One row per NVML poll tick (~2 Hz) |
| `{run_id}_engine.csv` | One row per vLLM stats tick — scheduler + KV cache |
| `{run_id}_outputs.json` | Raw model text outputs (for spot-checking quality) |
| `{run_id}_report.html` | Standalone interactive Plotly report (open in browser) |


### `{run_id}_summary.md`

Human-readable headline numbers. Printed to stdout at run end.

```
# Sim Run: a3f9c1d2e5b6

## VRAM
### GPU 0
- idle (after model load): 14.21 GB of 24.00 GB
- peak active: 17.83 GB
- delta (active − idle): 3.62 GB
- peak torch allocated: 15.10 GB
- suggested min VRAM spec (peak + 20% headroom): 21.40 GB

## Latency
- P50: 2 340 ms
- P95: 4 180 ms
- P99: 5 210 ms

## Throughput
- frames: 96 ok / 96 total
- throughput: 0.4000 fps
- run duration: 120.0 s

## Per-Feed Breakdown
| feed | frames | P50 (ms) | P95 (ms) |
...
```

### `{run_id}_samples.csv`

One row per inference request. Columns:

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | str | Identifies the run |
| `feed_id` | int | Which feed (0 … num_feeds−1) |
| `frame_index` | int | Frame sequence number within the feed |
| `t0_epoch` | float | Wall-clock time at request submission |
| `latency_s` | float | Submit → final token (monotonic) |
| `ttft_s` | float \| null | Submit → first token |
| `prompt_tokens` | int | |
| `completion_tokens` | int | |
| `resolution_w` | int | Pixel width after downscale |
| `resolution_h` | int | Pixel height after downscale |
| `model` | str | Model directory name |
| `status` | str | `"ok"` or `"error"` |
| `error_msg` | str \| null | Exception message on error |

### `{run_id}_gpu.csv`

One row per GPU poll tick (~2 Hz by default). Columns:

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | str | |
| `t_epoch` | float | Wall-clock time of sample |
| `gpu_index` | int | Device index |
| `memory_used_bytes` | int | Driver-level VRAM used (pynvml) |
| `memory_total_bytes` | int | Total VRAM (pynvml) |
| `torch_allocated_bytes` | int | Tensor memory allocated by PyTorch |
| `gpu_util_pct` | int | 0–100 GPU compute utilisation |
| `power_w` | float \| null | Power draw, watts (null if NVML unsupported) |
| `temperature_c` | int \| null | GPU temperature, °C (null if NVML unsupported) |

**Note on VRAM columns:** `memory_used_bytes` (pynvml) is what the GPU driver sees — it includes KV cache, CUDA graphs, and framework overhead. `torch_allocated_bytes` covers tensor allocations only. The gap between them is framework overhead; both are needed to calculate the minimum VRAM spec.

### `{run_id}_engine.csv`

One row per vLLM scheduler tick. Exposes saturation signals that NVML alone can't see.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | str | |
| `t_epoch` | float | Wall-clock time of sample |
| `num_running` | int | Requests currently in the GPU batch |
| `num_waiting` | int | Requests queued, not yet started |
| `num_swapped` | int | Requests swapped to CPU (preempted) |
| `gpu_kv_cache_usage_pct` | float | 0–100 — the single best saturation signal |
| `cpu_kv_cache_usage_pct` | float | 0–100, only non-zero if CPU offload enabled |
| `num_preemption_total` | int | Cumulative preemption count since engine start |

### `{run_id}_report.html`

Standalone interactive report. Open in any browser — Plotly is bundled inline (~5 MB), so the file works offline and can be emailed as-is.

Includes: headline metrics table, VRAM/KV-cache/queue-depth timeseries, GPU util/power/temp timeseries, latency-over-time scatter with rolling P95, latency CDF, per-feed boxplot, error class breakdown, and the first 20 model outputs.

---

## Ad-hoc Analysis

The CSV files load directly into pandas or polars:

```python
import pandas as pd

samples = pd.read_csv("sim_results/abc123_samples.csv")
gpu     = pd.read_csv("sim_results/abc123_gpu.csv")
engine  = pd.read_csv("sim_results/abc123_engine.csv")

# Latency percentiles per feed
print(samples.groupby("feed_id")["latency_s"].quantile([0.5, 0.95, 0.99]) * 1000)

# KV cache vs latency — best saturation diagnostic
import matplotlib.pyplot as plt
plt.plot(engine["t_epoch"] - engine["t_epoch"].min(), engine["gpu_kv_cache_usage_pct"])
plt.xlabel("elapsed (s)"); plt.ylabel("KV cache %"); plt.show()
```

For most cases though, just open `{run_id}_report.html` — it has all the standard charts.

---

## Parameter Sweeps

For triangulating GPU specs you'll want to run a grid, not single configs. The sweep runner
cross-products axes from a sweep YAML, executes each combination, and emits a heatmap-driven
HTML report.

```bash
uv run python -m simulator.sweep --config simulator/config/sweep.yaml
```

```yaml
# sweep.yaml
base_config: "simulator/config/sim_config.yaml"
output_dir: "./sweep_results"
axes:
  num_feeds: [1, 2, 4, 8]
  downscale_factor: [null, 0.5, 0.75]
```

Each axis value overrides the matching field on the base config. The engine is **reloaded
only when `model_path` changes** between runs — order your axes so `model_path` is the first
key (slowest-changing in `itertools.product`) to minimise reloads.

Output structure:

```
sweep_results/{sweep_id}/
  {sweep_id}_index.csv     one row per child run with headline metrics + axis values
  {sweep_id}_report.html   heatmaps across the first 2 axes; per-run links
  runs/                    {run_id}_* files from each child run
```

The sweep report shows heatmaps for **P95 latency, throughput, peak VRAM, and error rate**
across the two primary axes — read off the procurement floor from peak VRAM at your target
P95 budget.

---

## Profiling Strategy

Recommended sequence to triangulate the minimum GPU spec:

1. **Baseline — 1 feed, native resolution**
   ```yaml
   num_feeds: 1
   frame_interval_s: 5.0
   run_duration_s: 60
   ```
   Establishes idle VRAM and single-feed latency.

2. **Scale feeds — 2, 4, 8**
   Keep `frame_interval_s` fixed, increment `num_feeds`. Watch P95 latency and VRAM delta.

3. **Resolution sweep**
   Repeat step 2 with `downscale_factor: 0.5` and `downscale_factor: 0.75`. Lower resolution reduces prompt token count and speeds up prefill.

4. **Model swap**
   Replace `model_path` with a smaller or quantised variant and repeat step 2.

The `suggested min VRAM spec` in each summary is `peak_active × 1.20`. Use the maximum across all runs as your procurement floor.

---

## Architecture

```
__main__.py / sweep.py     CLI entry points
  │
  └── runner.execute_run    one simulation pass against a loaded engine
       │
       ├── engine.py            AsyncLLMEngine init; installs the StatLogger
       ├── gpu_poller.py        asyncio.Task: pynvml → GpuSample queue
       ├── engine_poller.py     custom StatLoggerBase: vLLM stats → EngineSample queue
       │
       └── feed_runner.py (×N feeds, concurrent)
             │
             ├── frame_extractor.py   VideoFrameSource: cv2 seek → PIL
             └── engine.generate()    vLLM AsyncLLMEngine
                   │
                   └── MetricSample → queue
                                       │
                                       └── reporter.py → CSVs + summary + HTML report
                                                          ↑
                                                          └── report_html.py (Plotly)
```

Each feed is an independent `asyncio.Task`. Frame extraction sleeps `frame_interval_s` between frames, yielding the event loop back to other feeds and the GPU poller. Inference tasks are fire-and-forget within each feed — a new frame is submitted on schedule even if the previous inference is still running, which matches production behaviour.
