# Context Handoff — Verkos Inference Server

You are picking up an in-progress implementation. This document is the
complete state of play. **Read `docs/INFERENCE_API_ARCHITECTURE.md` first**
— it is the source of truth for the design. This file tells you what's been
built against that doc and what's left.

---

## TL;DR

- The repo was a CLI bench harness (Streamlit + vLLM benchmark dashboard).
  It has been stripped down to "API server only" per the user's request.
- The product is an **OpenAI-compatible HTTP gateway** in front of one or
  more `sglang` model workers (`python -m sglang.launch_server`).
- The gateway design pivoted mid-implementation from "N always-on workers
  behind a thin gateway" to **"single-slot, on-demand model manager"** —
  the gateway spawns/swaps/kills sglang as a child process on demand.
  **The doc is up to date; ~60% of the Phase 4 code is stale and needs a
  rewrite.** See "What's stale" below.

---

## What's been done

### Cleanup (complete)
- Deleted the bench harness: `bench/`, `runs/`, `benchmarks/`, the old
  modules under `src/*` except `gpu_poller.py` and `__init__.py`.
- Deleted obsolete vLLM-era files: `config.yaml`, `config.yaml.example`,
  `Dockerfile`, `requirements.txt`, `TESTBENCH_PLAN.md`, `temp.md`.
- `models/` is preserved (operator drops weights here).
- `src/gpu_poller.py` rewritten to drop the dead `src.metrics` import;
  inlined a small `GpuSample` dataclass and added a synchronous
  `read_samples()` helper for the Prometheus exporter.
- `pyproject.toml` trimmed: dropped streamlit/plotly/openpyxl/vllm/opencv/
  pillow/openai/huggingface-hub/python-dotenv; added `httpx`,
  `prometheus-client`. Console scripts: `verkos-gateway`,
  `verkos-generate-api-key`, `verkos-launch-worker`.
- `.gitignore` updated (`keys.json`, `gateway.local.yaml`,
  `workers/*.local.yaml`).
- `README.md`, `.env.example` rewritten for the new scope.

### Phase 2 — API key tooling (complete)
- `scripts/generate_api_key.py` — stdlib-only CLI. Subcommands: default
  `--name X` (mint key), `--list`, `--revoke X`. Names are append-only
  (refuses duplicates; revoke instead). `keys.json` written with mode 0600.
  Key format: `sk-flytbase-airr-<43 url-safe chars>`.
- **Verified** end-to-end: mint, duplicate-rejects, list redacts to prefix,
  revoke is idempotent, revoke-unknown rc=2, mutex on flags, file mode 0600.

### Phase 3 — sglang launch translation (complete, but role changed)
- `workers/chat-vlm.yaml` — per-model YAML schema (worker network + sglang
  flags + capabilities block).
- `scripts/launch_worker.py` — translates a worker YAML into
  `python -m sglang.launch_server` argv and execs it. Enforces loopback
  bind, validates required keys, supports `--dry-run`.
- **Verified** dry-run: snake→kebab, bool true→bare flag, bool false→omitted,
  lists→repeated flags, off-loopback rejected, missing keys rejected.
- **Role change:** when the design pivoted to gateway-as-supervisor, this
  script was downgraded to **debug-only**. The same translation logic lives
  inside `manager.py` (to be written). The script is kept so an operator can
  boot sglang standalone (without the gateway) to verify weights load.

### Phase 4 — gateway scaffolding (PARTIALLY DONE — STALE, needs rewrite)
Built per the **original** "always-on workers" design. Lint+mypy clean,
import smoke passes. **Do not waste time fixing what's there — about 30% is
reusable, 70% needs to be replaced.** See next section.

Files under `src/serving/`:
- `__init__.py` — stub.
- `config.py` — Pydantic schemas. **Needs rewrite.** Currently models
  `gateway.yaml::workers` (list of always-on workers); the new doc says
  `gateway.yaml::models` (catalog of bootable models). Per-model YAML schema
  (`workers/<name>.yaml`) is unchanged.
- `errors.py` — OpenAI-shaped envelope helpers (`{error: {message, type,
  code, param}}`) and `GatewayError(HTTPException)`. **Keep as-is.**
- `auth.py` — `KeyStore` with atomic snapshot swap on SIGHUP reload;
  `parse_bearer` helper. **Keep as-is.**
- `registry.py` — Worker discovery + background `/health` polling against
  a static list of always-on workers. **Delete.** Replace with `manager.py`.
- `proxy.py` — `httpx.AsyncClient` streaming pass-through; SSE for
  `stream=true`, buffered for JSON; hop-by-hop headers stripped;
  `x-request-id` + `x-verkos-client` injected; upstream errors normalized to
  envelope. **Keep mostly as-is** — only thing that changes is the caller
  resolves the upstream URL dynamically via the manager instead of looking
  it up in the static registry.
- `metrics.py` — Prometheus registry: `verkos_requests_total`,
  `verkos_request_duration_seconds`, `verkos_tokens_total`,
  `verkos_inflight_requests`, GPU gauges. Background poller reuses
  `src/gpu_poller.py::read_samples`. **Keep, extend** with the new
  manager-related metrics from doc §A.8:
  `verkos_model_state{model,state=empty|loading|ready}`,
  `verkos_model_loads_total{model,result=ok|error}`,
  `verkos_model_load_duration_seconds{model}`.
- `app.py` — FastAPI factory. Middleware (request-id + structured log),
  exception handlers, SIGHUP key reload, lifespan owns the httpx client +
  registry + GPU poller, endpoints. **Mostly rewrite** the lifespan
  (own a manager, not a registry) and the inference paths (call
  `manager.ensure_loaded` before proxy). Auth dependency, middleware,
  exception handlers, request-id flow all stay.
- `__main__.py` — `python -m src.serving --config gateway.yaml`,
  structured JSON logging via `python-json-logger`. **Keep as-is.**
- `gateway.yaml.example` at repo root — uses old `workers:` schema.
  **Rewrite** to new `models:` schema with `manager:` block.

### What's verified vs. what's not

| Check | Status |
|---|---|
| `uv sync` | ✅ |
| `uv run ruff check src/ scripts/` | ✅ |
| `uv run ruff format --check src/ scripts/` | ✅ |
| `uv run mypy --strict` (12 files) | ✅ |
| `from src.serving.app import create_app` | ✅ |
| Bringing up the gateway against a real sglang worker | ❌ (no model on box) |
| Implicit load / swap behavior | ❌ (not implemented — needs rewrite) |
| OpenAI SDK drop-in test | ❌ |

---

## What's stale (and exactly what to replace)

The design pivoted partway through Phase 4. The doc (`docs/
INFERENCE_API_ARCHITECTURE.md`) is current; the code is not. Here is the
exact delta.

### Old (in current code) vs. new (in current doc)

| Aspect | Old (still in code) | New (in doc) |
|---|---|---|
| Worker lifecycle | systemd-launched, always-on | Gateway spawns sglang as child subprocess, on demand |
| Concurrent models in VRAM | N (one per worker) | 1 (single slot in v1) |
| Routing | Static registry, polled `/health` | `ModelManager` with state machine `EMPTY/LOADING/READY/UNLOADING` |
| `gateway.yaml` top-level | `workers: [{name, url, config_ref}]` | `models: [{id, config_ref}]` + `manager: {load_timeout_s, health_poll_interval_s}` |
| `/v1/models` semantics | Union of healthy workers' advertised models | Full catalog with per-entry `state: empty\|loading\|ready` |
| Endpoints | (no lifecycle endpoints) | `POST /v1/models/{id}/load`, `POST /v1/models/{id}/unload` (new) |
| Inference path | Resolve model in registry → 404 if absent → proxy | `manager.ensure_loaded(model)` → spawn/swap if needed → proxy |
| Cross-model request behavior | Each model has its own always-on worker | Implicit swap: hard-kill loaded model, spawn requested one |
| `/unload` semantics | (no such endpoint) | SIGKILL the child, `await process.wait()`. No drain. |
| Drain on in-flight requests during swap | n/a | **Hard kill** in v1 (any in-flight stream is dropped). Drain-on-implicit-swap is deferred to v2. |
| Deployment | One systemd unit per worker + one for gateway | One systemd unit total (`verkos-gateway.service`); sglang is the gateway's child |
| `scripts/launch_worker.py` | Production launcher | Debug-only helper for booting sglang standalone |

### The locked-in v1 contract (the user committed to these via explicit Q&A)

1. **Cross-model request → implicit swap.** "Magic": any inference for an
   unloaded model triggers hard-kill of the current model + spawn of the
   requested one. The user accepted the gotcha (mid-stream clients get cut
   off).
2. **Single slot** in v1. Multi-slot with VRAM budget + priority algorithm
   is v2.
3. **Hard kill on explicit `/unload`.** No draining of in-flight requests.

### Open design question (the user did not pick)

When the user picks the answer, update doc §A.4 and §B.5 accordingly:

> Should an *implicit* swap (triggered by another client's inference)
> drain in-flight requests with a timeout, while *explicit* `/unload`
> continues to hard-kill?

The user was offered this nuance and said nothing — they may want the
distinction (kinder to multi-tenant), or they may want hard-kill everywhere
(simpler). **Ask before implementing.** Currently the doc lists
"Drain-on-implicit-swap" in "Deferred to v2", meaning hard-kill everywhere
in v1 — but flag it for the user when you start the rewrite.

---

## What to build (in order)

### Phase 4 rewrite (next thing to do)

Follow doc §B.5 for the state machine, §A.7 for routing, §A.2 for endpoints,
§B.4 for `gateway.yaml` shape.

1. **`src/serving/manager.py` (new).** Owns the single sglang subprocess.
   - State: `EMPTY | LOADING(id) | READY(id) | UNLOADING(id)`.
   - One `asyncio.Lock` guards all transitions.
   - `async ensure_loaded(model_id) -> (host, port)`: returns the running
     sglang address for `model_id`. Spawns if EMPTY, swaps if a different
     model is loaded, fast-path returns if already READY for `model_id`.
     Concurrent same-model callers coalesce on the same spawn.
   - `async unload_if(model_id) -> bool`: SIGKILL only if the loaded model
     is `model_id`. No-op (return False) if EMPTY or a different model is
     loaded.
   - `current() -> SlotSnapshot`: read-only `(state, id, port, pid,
     last_error)`. Used by `/v1/models`, `/readyz`, metrics.
   - **Spawn:** `asyncio.create_subprocess_exec(spawn_python, "-m",
     "sglang.launch_server", *flags)` where flags come from the same
     YAML→argv translator that `scripts/launch_worker.py` uses. **Extract
     the translator into a shared function** and call it from both places
     (it currently lives in `scripts/launch_worker.py::build_command`).
   - **Health wait:** poll `http://127.0.0.1:<port>/health` every
     `health_poll_interval_s` (default 0.5s) until 200 or `load_timeout_s`
     (default 120s) elapses.
   - **Kill:** `process.kill()` then `await process.wait()`. No drain in v1
     (unless the user opts into drain-on-implicit-swap).
   - **Cleanup on gateway shutdown:** lifespan must call `await
     manager.shutdown()` which kills any live child and waits.
2. **`src/serving/config.py` (rewrite).**
   - Replace `WorkerEntry` with `ModelEntry(id: str, config_ref: Path)`.
   - `GatewayConfig.workers` → `GatewayConfig.models`.
   - Add `ManagerConfig(load_timeout_s=120.0, health_poll_interval_s=0.5,
     spawn_python: str | None = None)` on `GatewayConfig`.
   - Drop `RegistryConfig`.
   - Keep `WorkerFileConfig` schema (per-model YAML is unchanged).
3. **`src/serving/registry.py`** — delete.
4. **`src/serving/app.py` (mostly rewrite).**
   - Lifespan owns a `ModelManager`, not a `WorkerRegistry`. Pass it the
     catalog + httpx client.
   - `/v1/models` returns the **whole catalog** with each entry's
     capability block and a `state` field from `manager.current()`.
   - `/v1/models/{id}` returns one entry (404 if absent from catalog).
   - `POST /v1/models/{id}/load` → `await manager.ensure_loaded(id)`; 200.
   - `POST /v1/models/{id}/unload` → `await manager.unload_if(id)`; 200.
   - `/v1/chat/completions` and `/v1/embeddings`: validate `model` is in
     the catalog (404 otherwise), call `manager.ensure_loaded(model)` to
     get `(host, port)`, then proxy.
   - `/readyz`: 200 if `manager.current().state == "ready"`, 503 otherwise
     (include the state in the body).
   - Auth dependency, middleware, exception handlers, SIGHUP reload all
     stay.
5. **`src/serving/metrics.py` (extend).** Add `verkos_model_state`,
   `verkos_model_loads_total{result}`, `verkos_model_load_duration_seconds`.
   The manager calls these on transitions.
6. **`src/serving/proxy.py` (minor).** No structural change — caller now
   passes `upstream_url` from `manager.ensure_loaded` instead of from a
   static `WorkerState`. The function signature already takes `upstream_url`
   as a parameter, so the change is mostly in the caller.
7. **`gateway.yaml.example` (rewrite).** Use the new `models:` schema and
   include the `manager:` block. See doc §B.4 for the canonical example.
8. **Verify per doc §Verification Phase 4** (requires a real sglang
   worker on the box, which is not currently available — ask the user
   what they want to test against).

### Phase 5 — multi-model swap verification

After the rewrite, add a second catalog entry (`workers/bge-embed.yaml`) and
verify the implicit swap path end-to-end. See doc §Verification Phase 5.

### Phase 6 — production hardening

- `deploy/systemd/verkos-gateway.service` (single unit; sglang is the
  gateway's child).
- Prometheus scrape config + dashboard.
- README "Deployment" section.

---

## Files to read first

1. `docs/INFERENCE_API_ARCHITECTURE.md` — design, source of truth.
2. `CLAUDE.md` — repo behavioral guidelines. Notably:
   "Don't assume. Don't hide confusion. Surface tradeoffs." and "Minimum
   code that solves the problem. Nothing speculative."
3. `src/serving/auth.py`, `src/serving/errors.py`, `src/serving/proxy.py` —
   keep-as-is, gives you the patterns and helpers to reuse.
4. `src/serving/registry.py` — what you're replacing. The polling pattern
   doesn't apply, but the state-mutation-under-lock pattern does.
5. `scripts/launch_worker.py` — extract the YAML→argv translator from this
   file into a shared helper that `manager.py` also uses.

## How to set up

```bash
uv sync
uv run ruff check src/ scripts/
uv run ruff format --check src/ scripts/
uv run mypy src/serving src/gpu_poller.py scripts/

# Mint a test API key:
uv run python scripts/generate_api_key.py --name test --keys-file ./keys.json

# After implementing Phase 4 rewrite, the gateway is:
uv run python -m src.serving --config gateway.yaml
```

## Things the user has been clear about

- **Don't recreate the bench harness or any Streamlit / bbox-rendering
  code.** API only.
- **Phase-wise implementation, not all at once.** Stop and check in between
  phases.
- **Don't run real-model end-to-end tests speculatively.** A mock-worker
  harness was started and the user pushed back ("Why are you setting up a
  worker in a background?"). Reach for it only when the user explicitly
  asks to verify against a running model.
- **No unauthorized destructive deletions.** When you needed to delete
  `Dockerfile` and `config.yaml*`, you asked first.
- **Architecture choice: gateway is correct for multi-model. Single-model
  deployments don't need it.** If the user reopens this question, the
  honest answer is "for one model, just run sglang directly with
  `--api-key`". The doc justifies the gateway by v1 having ≥2 models.

## Things that might trip you up

- `pyproject.toml` declares console scripts that point at modules. Make
  sure `scripts/__init__.py` exists (it does) so the script entrypoints
  resolve.
- `httpx.AsyncClient` is shared across the app lifespan — don't create a
  new one per request.
- SIGHUP is Unix-only — the existing `_install_sighup_reload` no-ops on
  Windows; keep that pattern.
- The user noticed and fixed a doc inconsistency: §C.1 says key prefix
  `sk-flytbase-airr-` and §Verification said `sk-verkos-`. The committed
  truth is `sk-flytbase-airr-` (already implemented in the script).
