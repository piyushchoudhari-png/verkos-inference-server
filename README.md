# Verkos VLM Simulator

Simulates N concurrent drone video feeds sending frames to a vision-language model and measures latency, throughput, token rates, and GPU resource consumption. Supports two inference backends:

- **openrouter** — sends frames to OpenRouter API (Qwen 3 VL 32B or any compatible model). No local GPU required.
- **local** — drives `AsyncLLMEngine` (vLLM) directly on a local GPU. Profiling results reflect the same batching behaviour the production Verkos detection pipeline will exhibit.

---

## Quick Start

```bash
# 1. Set your OpenRouter API key
cp .env.example .env
# edit .env: set OPENROUTER_API_KEY=sk-or-...

# 2. Edit the config
#    Set video_paths, num_feeds, run_duration_s as needed
nano src/config/sim_config.yaml

# 3. Run
uv run python -m src --config src/config/sim_config.yaml
```

Results land in `./results/` by default.

---

## Prerequisites

- Python 3.12+, all deps via `uv sync`
- At least one MP4 video file
- For `openrouter` mode: an OpenRouter API key in `.env`
- For `local` mode: NVIDIA GPU (CUDA 12.4+), model weights on disk

---

## Configuration

All configuration lives in `src/config/sim_config.yaml`. Pass it with `--config`.

### Inference mode

```yaml
inference_mode: openrouter   # "openrouter" | "local"
```

**OpenRouter** (no local GPU needed):

```yaml
inference_mode: openrouter
openrouter:
  model: "qwen/qwen3-vl-32b-instruct"
  base_url: "https://openrouter.ai/api/v1"
  timeout_s: 60.0
# api_key is read from OPENROUTER_API_KEY in .env — never put it here
```

**Local vLLM**:

```yaml
inference_mode: local
model_path: "/path/to/model/weights"
dtype: "auto"
gpu_memory_utilization: 0.90
tensor_parallel_size: 1
```

### Full reference

```yaml
# --- Inference ---
inference_mode: openrouter        # "openrouter" | "local"

# openrouter block (required when inference_mode = openrouter)
openrouter:
  model: "qwen/qwen3-vl-32b-instruct"
  base_url: "https://openrouter.ai/api/v1"
  timeout_s: 60.0

# local vLLM block (required when inference_mode = local)
# model_path: "/path/to/weights"
# dtype: "auto"
# gpu_memory_utilization: 0.90
# tensor_parallel_size: 1

# --- Video sources ---
video_paths:                      # list of MP4s; cycled if len < num_feeds
  - "src/data/video.mp4"

# --- Prompts ---
prompts_file: "src/config/prompts.yaml"

# --- Simulation shape ---
num_feeds: 1                      # concurrent drone feeds
frame_interval_s: 10.0           # seconds between frames per feed

# --- Run termination (at least one required) ---
run_duration_s: 120.0
# max_frames_per_feed: 20

# --- Frame resolution ---
downscale_factor: null            # e.g. 0.5 = half native resolution
# resolution: [640, 360]          # mutually exclusive with downscale_factor

# --- Inference knobs ---
max_tokens: 812
temperature: 0.7

# --- Telemetry ---
gpu_poll_interval_s: 0.5

# --- Output ---
output_dir: "./results"
# run_id: "my_run_001"
```

### Termination conditions

One of `run_duration_s` or `max_frames_per_feed` **must** be set. Both can be set — the run stops at whichever triggers first.

---

## Prompts

Prompts live in `src/config/prompts.yaml` and are versioned separately from the sim config.

```yaml
system: |
  You are a multilingual drone surveillance AI assistant...

user: |
  Frame {frame_index}. Detect the following events if present: ...
```

`{frame_index}` is replaced at runtime with the sequential frame number within that feed.

---

## Running

```bash
# Basic
uv run python -m src --config src/config/sim_config.yaml

# Override output directory
uv run python -m src --config src/config/sim_config.yaml --output-dir /data/results

# Set a fixed run ID for reproducible filenames
uv run python -m src --config src/config/sim_config.yaml --run-id baseline_4feeds
```

Progress is logged to stdout. The text summary is printed at run end.

---

## Output

Per run, four files are written under `output_dir`:

| File | Contents |
|------|----------|
| `{run_id}_summary.txt` | Headline numbers — printed to stdout at run end |
| `{run_id}_samples.csv` | One row per inference request (all `MetricSample` fields) |
| `{run_id}_gpu.csv` | One row per NVML poll tick |
| `{run_id}_engine.csv` | vLLM scheduler + KV cache stats (empty in openrouter mode) |
| `{run_id}_outputs.json` | Full record per frame: text output + all metadata |

### `{run_id}_outputs.json`

Each element is the complete record for one frame:

```json
[
  {
    "run_id": "abc123",
    "feed_id": 0,
    "frame_index": 4,
    "t0_epoch": 1747123456.789,
    "latency_s": 2.41,
    "ttft_s": 0.38,
    "prompt_tokens": 512,
    "completion_tokens": 128,
    "resolution_w": 1280,
    "resolution_h": 720,
    "model": "qwen/qwen3-vl-32b-instruct",
    "inference_mode": "openrouter",
    "status": "ok",
    "error_msg": null,
    "output_text": "{ \"detections\": [...] }"
  }
]
```

### `{run_id}_samples.csv`

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | str | Identifies the run |
| `feed_id` | int | Which feed (0 … num_feeds−1) |
| `frame_index` | int | Frame sequence number within the feed |
| `t0_epoch` | float | Wall-clock time at request submission |
| `latency_s` | float | Submit → final token (monotonic) |
| `ttft_s` | float\|null | Submit → first token |
| `prompt_tokens` | int | |
| `completion_tokens` | int | |
| `resolution_w` | int | Pixel width after any downscale |
| `resolution_h` | int | Pixel height after any downscale |
| `model` | str | Model name |
| `inference_mode` | str | `"openrouter"` or `"local"` |
| `status` | str | `"ok"` or `"error"` |
| `error_msg` | str\|null | Exception message on error |
| `output_text` | str\|null | Raw model output text |

### `{run_id}_gpu.csv`

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | str | |
| `t_epoch` | float | Wall-clock time of sample |
| `gpu_index` | int | Device index |
| `memory_used_bytes` | int | Driver-level VRAM used (pynvml) |
| `memory_total_bytes` | int | Total VRAM |
| `torch_allocated_bytes` | int | Tensor memory allocated by PyTorch |
| `gpu_util_pct` | int | 0–100 GPU compute utilisation |
| `power_w` | float\|null | Power draw in watts |
| `temperature_c` | int\|null | GPU temperature °C |

### `{run_id}_engine.csv`

Populated in `local` mode only. One row per vLLM scheduler tick.

| Column | Type | Description |
|--------|------|-------------|
| `num_running` | int | Requests in the GPU batch |
| `num_waiting` | int | Requests queued, not yet started |
| `num_swapped` | int | Requests preempted to CPU |
| `gpu_kv_cache_usage_pct` | float | 0–100 — best saturation signal |
| `cpu_kv_cache_usage_pct` | float | 0–100 |
| `num_preemption_total` | int | Cumulative preemptions since start |

---

## Dashboard

Open `dashboard/index.html` directly in a browser (or via `python -m http.server` in the project root). No build step required.

Drop in the output files using the three file pickers:

| Input | What renders |
|-------|-------------|
| `outputs.json` | Paginated table: frame, feed, status, latency, output text (expandable) |
| `samples.csv` | Summary cards: total frames, error rate, p50/p95/p99/mean latency, TTFT p50 |
| `gpu.csv` | VRAM over time line chart (one line per GPU) |

---

## Architecture

```
src/__main__.py          CLI entry point
  │
  └── runner.execute_run    one simulation pass
       │
       ├── engine.py            AsyncLLMEngine init (local mode only)
       ├── gpu_poller.py        asyncio.Task: pynvml → GpuSample queue
       ├── engine_poller.py     vLLM StatLoggerBase → EngineSample queue (local only)
       │
       └── feed_runner.py (×N feeds, concurrent)
             │
             ├── frame_extractor.py      VideoFrameSource: cv2 seek → PIL
             ├── openrouter.py           OpenRouter streaming inference (openrouter mode)
             └── engine.generate()       vLLM AsyncLLMEngine (local mode)
                   │
                   └── MetricSample → queue → reporter.py → CSVs + outputs.json + summary.txt

dashboard/
  index.html + script.js   static local dashboard, no build step
```
