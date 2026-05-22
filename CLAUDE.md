# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo state vs. original PRD

The repo is named `verkos-inference-server` and the original plan (still reflected in `Dockerfile`, root `config.yaml`/`config.yaml.example`, and `scripts/download_model.py`) is a FastAPI + vLLM wrapper exposing an OpenAI-compatible API for a single VLM. **That server is not implemented on this branch.**

What currently exists in `feat/test-bench`:

- `src/` — an async **VLM profiling simulator** that drives N concurrent simulated drone video feeds against an inference backend and writes CSV/JSON artifacts.
- `bench/` — a Streamlit "Verkos Test Bench" UI that wraps the simulator (Results / Configure & Run / Benchmark tabs).
- `scripts/download_model.py` — host-side weight fetcher (HF / S3 / local-copy), keyed off the root `config.yaml`.
- `Dockerfile` — still references a non-existent `app/` directory; do not assume it builds.

When working in this branch, prefer editing the simulator + bench. Treat the root `config.yaml`, `config.yaml.example`, and the FastAPI-shaped rules below as the *destination*, not the current truth.

## Common commands

```bash
# Sync deps (uv only — no pip/poetry/conda)
uv sync

# Run the simulator from CLI
uv run python -m src --config src/config/sim_config.yaml
uv run python -m src --config src/config/sim_config.yaml --run-id baseline_4feeds --output-dir /data/results

# Launch the Streamlit test bench (entry: bench/__main__.py → streamlit run bench/app.py)
uv run bench
# or: uv run streamlit run bench/app.py

# Lint / format / type-check
uv run ruff check .
uv run ruff format .
uv run mypy src bench

# Freeze pinned deps for Docker (don't hand-edit requirements.txt)
uv export --no-dev -o requirements.txt

# Pre-deployment weight fetch (uses root config.yaml, not sim_config.yaml)
uv run python scripts/download_model.py --config config.yaml
```

`OPENROUTER_API_KEY` is loaded from `.env` via `python-dotenv` inside `load_sim_config` — never put the key in YAML.

## Simulator architecture (`src/`)

Entry: `src/__main__.py` → `asyncio.run(run(...))` → `src.runner.execute_run`.

```
__main__.py        CLI; loads SimConfig + PromptConfig; constructs engine in local mode
  └── runner.execute_run        one simulation pass
        ├── engine.create_engine        HF transformers (AutoModelForImageTextToText) — local mode only
        ├── gpu_poller.poll_gpu          asyncio.Task: pynvml → GpuSample queue
        ├── engine_poller.drain_loop     SchedulerStatLogger → EngineSample queue (local only, vLLM-era stub)
        └── feed_runner.run_feed (×N)
              ├── frame_extractor.VideoFrameSource    cv2 seek → PIL.Image at frame_interval_s
              ├── _infer_frame_local       → engine.generate(...) (HFEngine)
              └── _infer_frame_openrouter  → src/openrouter.py (streaming chat completions)
                    └── MetricSample → queue → reporter.write_results → {run_id}_{samples,gpu,engine,outputs,summary}
```

Key facts:

- **Inference backend was migrated from vLLM `AsyncLLMEngine` to HF `transformers`** (commit `f031e18`). `HFEngine` in `src/engine.py` reimplements the vLLM call surface (`async generate()` yielding objects with `.outputs[0].text/.token_ids` and `.prompt_token_ids`) so `feed_runner._infer_frame_local` works unchanged. `model.generate()` is not thread-safe, so all local calls serialize through a single `asyncio.Lock` — feeds queue up rather than truly run in parallel on the GPU.
- `engine_poller.SchedulerStatLogger` was designed against vLLM's `StatLoggerBase`; under the transformers backend it emits no engine samples (the `*_engine.csv` will be empty). Don't delete it — the vLLM revival is still on the roadmap.
- vLLM-only config fields (`gpu_memory_utilization`, `tensor_parallel_size`, `mm_processor_kwargs`, `limit_mm_per_prompt`, `max_model_len`) are accepted by `SimConfig` but logged as no-ops by `_warn_vllm_only_fields` under the HF backend. Keep them on the model so re-enabling vLLM is a one-file change.
- Run termination: at least one of `run_duration_s` or `max_frames_per_feed` must be set; `VideoFrameSource` enforces both.
- Output artifacts land under `output_dir/{run_id}/` plus a frames dump at `output_dir/{run_id}/frames/feedNN_frameNNNN.jpg`. The Streamlit Results tab reads these directly.

## Test bench architecture (`bench/`)

- Entry point `bench` (via `[project.scripts]`) shells out to `streamlit run bench/app.py`. Don't try to `import bench.app` from non-Streamlit contexts.
- Three tabs in `bench/tabs/`: `results.py` (loads `RUNS_DIR`), `configure.py` (writes a tmp config and spawns the simulator), `benchmark.py` (matrix sweeps written to `BENCHMARKS_DIR`).
- `bench/utils/subprocess_runner.py` is how the UI launches the simulator — it does *not* import `src.runner` in-process. Long-running runs are subprocesses tracked in `st.session_state["active_subprocess"]`.
- `bench/tmp/` holds generated configs and uploaded videos; `bench/app.py` garbage-collects YAMLs older than 24 h on each page load. Treat `bench/tmp/` as ephemeral.

## Config

Two distinct config schemas live in this repo — don't conflate them:

| File | Schema | Used by |
|---|---|---|
| `config.yaml` (root) | `model:` / `engine:` / `server:` — pydantic schema TBD | `scripts/download_model.py`; future FastAPI server |
| `src/config/sim_config.yaml` | `src.config.SimConfig` (`pydantic.BaseModel`) | the simulator; loaded via `load_sim_config` |
| `src/config/prompts.yaml` | `src.config.PromptConfig` (`system`, `user` with `{frame_index}` placeholder) | the simulator |

`SimConfig.inference_mode` switches between `local` (requires `model_path`) and `openrouter` (requires `openrouter:` block; api_key auto-filled from `OPENROUTER_API_KEY` env).

## Rules for the future FastAPI server

Preserved from the original PRD — apply when (re)building the server surface, not to the simulator/bench:

- Python 3.11+, FastAPI + uvicorn. CUDA base: `nvidia/cuda:12.4.1-runtime-ubuntu24.04`. Linux x86_64 + NVIDIA only.
- Type-annotate everything. Use `pydantic.BaseModel` for every request/response shape — no bare `dict` returns. Define explicit response models per route.
- Use FastAPI `lifespan` (not `on_event`) to init/teardown vLLM. Hold the single `AsyncLLMEngine` on `app.state.engine`. One `APIRouter` per route file, mounted in `main.py`.
- Engine kwargs pass through verbatim: `AsyncLLMEngine.from_engine_args(EngineArgs(**engine_kwargs))`. Let vLLM reject bad keys at startup, not per-request.
- 501 stubs: `{"detail": "not implemented"}` with `status_code=501` — define routes early to reserve URLs.
- Stream with `StreamingResponse` + SSE-style chunks when `stream=True`. Never block the event loop.
- Image inputs: base64 inline only — air-gapped, no URL fetching from inside the container.
- Config: one `pydantic_settings.BaseSettings` subclass, YAML first then env-var overrides; log both on conflict. Server refuses to start if `config.yaml` is missing or invalid. `GET /admin/config` returns the merged effective config.
- Observability: structured JSON to stdout per request (`timestamp`, `latency_ms`, `prompt_tokens`, `completion_tokens`, `image_count`, `model`, `status`) via `python-json-logger` — no ad-hoc `json.dumps`. `/metrics` returns JSON (uptime, counts, in-flight, avg latency, GPU mem). Counters live at module level; mutate them from helpers, not from route handlers.
- Docker: pin the CUDA tag; freeze Python deps with `uv export`. Non-root `appuser`. Weights mounted RO at `/models`, config RO at `/etc/inference-server/config.yaml`. No model flags in `ENTRYPOINT`/`CMD`. `--shm-size=16g` is a `docker run` concern.

## Python style

- No global mutable state outside the simulator engine module / future server's `app.state`. Route layer raises `HTTPException`; engine layer raises domain exceptions.
- Keep functions short — if a function needs a comment to explain what it does, split it.
- Ruff config: `line-length = 120`, `target-version = "py312"`, selects `E,W,F,I,B,C4,UP,SIM`, ignores `E501`. mypy is `strict = true` with `vllm.*`, `cv2.*`, `pynvml`, `yaml` set to `ignore_missing_imports`.
- **Never** pin versions when adding deps: `uv add fastapi pydantic-settings`, not `uv add fastapi==x.y.z`. Let uv resolve.
- vLLM is GPU-only and not in the local dev venv; gate imports behind try/except so the rest of the app imports cleanly without it.

## Out of scope (don't add)

- Auth, rate limiting, multi-model loading, hot-swap, model router, K3s, BYOM UI.
- Outbound network calls from inside the container during inference.
- Platform support beyond Linux x86_64 + NVIDIA.
