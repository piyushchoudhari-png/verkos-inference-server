# verkos-inference-server

FastAPI wrapper around vLLM, serving a single vision-language model behind an OpenAI-compatible API. Config-driven, no model identity baked into the image.

## Stack

- Python 3.11+, FastAPI + uvicorn, vLLM (pinned)
- Base image: `nvidia/cuda:12.4.1-runtime-ubuntu24.04`
- Linux x86_64 + NVIDIA GPU only

## Python

- Type-annotate everything. 
- Prefer `pydantic.BaseModel` for all data shapes — request bodies, response bodies, config.
- No global mutable state outside of `engine.py`'s single engine instance.
- Raise `HTTPException` at the route layer; engine layer raises domain exceptions.
- Keep functions short. If a function needs a comment to explain what it does, split it.

## FastAPI

- Use the `lifespan` context manager (not deprecated `on_event`) to init/teardown the vLLM engine.
- Define explicit Pydantic response models for every route — no bare `dict` returns.
- Use `APIRouter` per route file; mount in `main.py`.
- 501 stubs return `{"detail": "not implemented"}` with `status_code=501` — define them now so URLs are reserved.
- Stream completions with `StreamingResponse` + `EventSourceResponse`-style chunks when `stream=True`.

## vLLM

- Instantiate `AsyncLLMEngine` once at startup via `lifespan`; hold it on `app.state.engine`.
- Pass engine kwargs from config directly: `AsyncLLMEngine.from_engine_args(EngineArgs(**engine_kwargs))`.
- Unknown keys under `engine:` in config pass through to `EngineArgs` verbatim — let vLLM reject bad ones at startup, not at request time.
- Use `engine.generate()` with `RequestOutput` for async inference; never block the event loop.
- Image inputs: accept base64-encoded inline only (no URL fetching — air-gapped).

## Configuration

- Single `pydantic_settings.BaseSettings` subclass reads YAML first, then env var overrides.
- Env vars listed in the PRD override their YAML counterparts; log both values at startup when a conflict is detected.
- Server refuses to start if `config.yaml` is absent or fails Pydantic validation — fail fast, clear error.
- `GET /admin/config` returns the effective merged config as JSON. Never expose secrets (there are none in v0.1, but enforce the pattern).

## Observability

- Structured JSON to stdout on every request: `timestamp`, `latency_ms`, `prompt_tokens`, `completion_tokens`, `image_count`, `model`, `status`.
- Use `python-json-logger` or equivalent — no ad-hoc `json.dumps` in route handlers.
- `/metrics` returns a JSON response with uptime, request counts, in-flight count, avg latency, and GPU memory info. Update module-level counters from helper functions, not inside request handlers.
- Log effective config (model name, path, all engine args) as a single JSON line at startup.

## Dependencies

- Do **not** pin version numbers when adding packages — install latest and let uv resolve. Example: `uv add fastapi pydantic-settings`, not `uv add fastapi==0.x.y`.
- vLLM is GPU-only and is **not** installed in the local dev environment. It is imported with a try/except in `engine.py`; the rest of the app runs without it.

## Docker

- Pin the CUDA base tag. Python packages in `requirements.txt` are pinned via `uv export`.
- Run as non-root: `RUN useradd -m appuser && USER appuser`.
- Model weights are never in the image — volume-mounted read-only at `/models`.
- Config file volume-mounted read-only at `/etc/inference-server/config.yaml`.
- `ENTRYPOINT` starts uvicorn; `CMD` provides overridable defaults. No model flags in either.
- `--shm-size=16g` is a `docker run` concern, not a Dockerfile concern.

## What's out of scope (don't add it)

- Auth, rate limiting, multi-model loading, hot-swap, model router, K3s, BYOM UI.
- Any outbound network call from inside the container during inference.
- Platform support beyond Linux x86_64 + NVIDIA.
