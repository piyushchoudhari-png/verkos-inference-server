# Local Model Inference Server — PRD

*Status: Draft v0.1*
*Owner: [you]*
*Doc type: Technical build spec (infrastructure component, no end-user surface)*

---

## 1. Why this exists

AI-R V2 needs to run AI inference on customer-owned hardware. Verkos today calls OpenRouter, a cloud API. To unblock on-prem deployment, we need a local inference service that Verkos (and later other consumers) can call instead.

This document specifies the **first deliverable in that direction**: a Docker container that hosts a vision-language model behind a controllable HTTP API. It is the foundation for the later model router, BYOM support, and the air-gapped deployment story — but those are not in scope here. This thing has one job: serve a model.

## 2. What this is not

To keep scope tight, this deliverable explicitly does **not** include:

- Verkos integration (that's the next iteration; this server is the building block)
- Multi-model loading or hot-swap (architecture must not preclude it, but no implementation)
- LRU model caching (same — architecture allows, build doesn't implement)
- Authentication or rate limiting
- Production hardening (HA, graceful shutdown for in-flight requests, etc.)
- A Forensic Search inference path (separate workload, separate PRD later)
- The full PR FAQ's "BYOM" feature (the hooks must exist; the customer-facing flow does not)

The goal here is **a thing we can install on the RTX 5090 tower this week and start measuring against**.

## 3. Goals

In priority order:

1. **Run any configured vision-language (or compatible) model** in a Docker container on the tower — model identity is declared entirely in external config, not baked into the image.
2. **Expose an OpenAI-compatible inference API** so Verkos can point at us with only a base URL change.
3. **Surface all backend runtime options as configuration**, so we can do the VRAM/throughput experiments without rebuilding the image for every test.
4. **Support model download and weight management via config** — a companion download script or sidecar reads the same config file and fetches weights before the server starts; the server itself never pulls from the network.
5. **Be observable** — VRAM, latency, request count, model state — readable from outside the container.
6. **Be reproducible** — same image + same model config + same weights = byte-identical behavior on any tower.
7. **Be air-gappable** — once the image and weights are local, no outbound network calls during inference.

## 4. Non-goals

- Performance optimization beyond what the backend gives us by default. We're measuring, not tuning.
- Supporting non-VLM models in v0.1 (text-only LLMs, embedding models, etc.). Plumbing should allow it later.
- Cross-platform builds. Linux x86_64 with NVIDIA GPU only.

## 5. Users of this service

| Consumer | What they need from us | When |
|---|---|---|
| Me, this week | An endpoint to throw frames at, with VRAM/latency readout | Now |
| Verkos pipeline | OpenAI-compatible `/v1/chat/completions` with vision input | After v0.1 lands |
| Future model router | Lifecycle endpoints (load, unload, list) | Later phase |
| Future BYOM flow | Ability to load a customer-supplied model artifact | Later phase |

The v0.1 build needs to serve consumer 1 cleanly and not preclude consumers 2–4.

## 6. Architecture decision: backend choice

The container wraps one of two inference backends behind a FastAPI layer:

**Option A — vLLM.** Production-grade, widely deployed, broad model support (VLMs, text-only LLMs, AWQ/GPTQ quantization), OpenAI-compatible server built in, active community.

**Option B — SGLang.** Newer, claims better throughput on certain workloads, also OpenAI-compatible, growing VLM support.

**Recommendation: vLLM for v0.1**, on the grounds that it is more mature, has wider production deployment, and the Verkos team's parallel "self-hosted on cloud GPU" work is more likely to land on vLLM by default. Reuse what they build.

The FastAPI wrapper should treat the backend as a swappable component so v0.2 can A/B vLLM vs SGLang without breaking the consumer contract. Neither the wrapper nor the Dockerfile should reference any specific model — all model identity flows through config.

## 7. Architecture sketch

```
  Host filesystem
  ┌─────────────────────────────────────────────────┐
  │  /etc/inference-server/config.yaml  (any model) │
  │  /opt/models/<name>/  (weights, manifest)        │
  │         ▲                                        │
  │         │ written by                             │
  │  scripts/download_model.py  (runs on host)       │
  └──────────────────┬──────────────────────────────┘
                     │ volume-mounted (read-only)
                     ▼
┌─────────────────────── Docker container ───────────────────────┐
│                                                                │
│   Config loader  ◄── /etc/inference-server/config.yaml         │
│         │                                                      │
│   FastAPI (uvicorn)                                            │
│      │                                                         │
│      ├── /v1/chat/completions ──► vLLM engine ──► GPU          │
│      ├── /v1/models                                            │
│      ├── /health                                               │
│      ├── /metrics                                              │
│      └── /admin/config (read-only in v0.1)                     │
│                                                                │
│   No model identity baked in — all from config at runtime      │
└────────────────────────────────────────────────────────────────┘
              │
              ▼
       Host network port (default 8000)
```

**Why FastAPI in front of vLLM at all** (vLLM ships its own server):

- Lets us add lifecycle/metrics endpoints vLLM doesn't have.
- Lets us swap the backend later without changing the consumer contract.
- Lets us add the model router on top of this same surface in a future iteration.
- Lets us version our API independently of vLLM's release cadence.

The cost is one extra hop per request. For Verkos's 5–30s sampling rate, that latency cost is negligible.

## 8. API surface (v0.1)

### Inference

- `POST /v1/chat/completions` — OpenAI-compatible. Must accept image inputs (base64 inline; URL not required for v0.1 since we're air-gapped). Pass-through to vLLM with minimal massaging.
- `POST /v1/completions` — OpenAI-compatible text completion. Lower priority but free if vLLM provides it.

### Introspection

- `GET /v1/models` — list currently loaded models (v0.1: always one). OpenAI-compatible shape.
- `GET /health` — liveness (always 200 if process is up) and readiness (200 only if model loaded and responding).
- `GET /metrics` — Prometheus-formatted: VRAM used/total, request count, request latency histogram, in-flight requests, model name, uptime.
- `GET /admin/config` — read-only dump of effective runtime config (sanitized — no secrets if any).

### Reserved for future (define shape now, return 501)

- `POST /v1/models/load`
- `DELETE /v1/models/{id}`
- `POST /v1/models/{id}/warmup`

Defining these as 501 in v0.1 reserves the URLs so consumers don't have to migrate later.

## 9. Configuration

All model selection, download parameters, and backend runtime options must be settable without rebuilding the image. The single source of truth is a YAML config file mounted from the host; environment variables can override individual fields for quick iteration.

### 9.1 Mounted config file (`/etc/inference-server/config.yaml`)

This file drives everything: which model to serve, where its weights live or how to fetch them, and how vLLM should be tuned. The server refuses to start if this file is absent or malformed.

```yaml
model:
  # Human-readable name surfaced in /v1/models and logs
  name: "my-vlm"

  # Where weights are expected at runtime (inside container, via volume mount).
  # The server reads from this path; it does not download weights itself.
  path: "/models/my-vlm"

  # Optional: populated by the download helper (see §9.2).
  # Recorded here so the server can log provenance on startup.
  source:
    type: "huggingface"          # huggingface | s3 | local-copy
    repo_id: "org/model-name"    # HF repo, S3 URI, or absolute host path
    revision: "main"             # git ref, S3 version, or omit for local

  dtype: "auto"                  # auto | float16 | bfloat16
  quantization: "none"           # none | awq | gptq
  max_model_len: null            # null = vLLM default
  max_num_seqs: 256

engine:
  gpu_memory_utilization: 0.90
  tensor_parallel_size: 1        # reserved; always 1 in v0.1

server:
  port: 8000
  log_level: "info"              # debug | info | warn | error
```

Any `EngineArgs` field not listed above can be added under `engine:` as a free-form key; the server passes unknown keys through to vLLM verbatim. The server fails fast with a clear error on unknown keys that vLLM also rejects.

**Environment variable overrides** — for quick iteration without editing the file:

| Var | Overrides |
|---|---|
| `MODEL_PATH` | `model.path` |
| `MODEL_DTYPE` | `model.dtype` |
| `QUANTIZATION` | `model.quantization` |
| `GPU_MEMORY_UTILIZATION` | `engine.gpu_memory_utilization` |
| `MAX_MODEL_LEN` | `model.max_model_len` |
| `MAX_NUM_SEQS` | `model.max_num_seqs` |
| `TENSOR_PARALLEL_SIZE` | `engine.tensor_parallel_size` |
| `PORT` | `server.port` |
| `LOG_LEVEL` | `server.log_level` |

Env vars take precedence over the YAML file. Any conflict is logged at startup with both the file value and the override value so it's auditable.

### 9.2 Model download helper (`scripts/download_model.py`)

A standalone script (not part of the server process) reads the same `config.yaml` and fetches weights before the container starts. It is the only component that makes outbound network calls.

```bash
# On the host, before docker run:
python scripts/download_model.py --config /etc/inference-server/config.yaml
```

Behavior:
- Reads `model.source` from the config file.
- Downloads weights to `model.path` on the host (the directory that will be volume-mounted).
- Resumes interrupted downloads (HF Hub's built-in resume; S3 uses multipart).
- Writes a `.download_manifest.json` into the weights directory with sha256 checksums and the source ref, so the server can verify integrity on startup.
- Is a no-op if the manifest already exists and checksums pass (idempotent).
- Once weights are on disk, the container runs fully air-gapped.

## 10. Observability

The whole point of this build is to measure. Bake in:

- **`/metrics` endpoint** Prometheus-format. Include at minimum:
  - `gpu_memory_used_bytes`, `gpu_memory_total_bytes` (per GPU)
  - `inference_requests_total{status=...}`
  - `inference_latency_seconds` (histogram, with prompt-token-count and image-count labels)
  - `inference_in_flight_requests`
  - `model_loaded{name=...}` gauge
- **Structured JSON logs** to stdout. One line per request with: timestamp, latency, prompt tokens, completion tokens, image count, model, status.
- **Startup log line** with effective config dump (sanitized).

Out of scope for v0.1 but worth mentioning: hooking these into a real Prometheus + Grafana setup. The endpoint exists; the dashboard is a later piece of work.

## 11. Container requirements

- Base image: official NVIDIA CUDA runtime (e.g., `nvidia/cuda:12.4.1-runtime-ubuntu24.04`) to match the AI-R OS choice (Ubuntu 24.04 LTS).
- Python 3.11+.
- vLLM pinned to a specific version. No floating tags.
- Model weights **not baked into the image** — mount as volume. Image stays small and reusable across model changes.
- Image must build offline once dependencies are cached (matters for the air-gapped deployment story; not blocking for v0.1 but design for it).
- Runs as non-root inside the container. GPU access via NVIDIA Container Toolkit.

## 12. Deployment shape (v0.1, single tower)

**Step 1 — write a config file on the host:**

```yaml
# /etc/inference-server/config.yaml
model:
  name: "my-vlm"
  path: "/models/my-vlm"
  source:
    type: "huggingface"
    repo_id: "org/model-name"
    revision: "main"
  dtype: "auto"
  quantization: "none"
engine:
  gpu_memory_utilization: 0.90
  tensor_parallel_size: 1
server:
  port: 8000
  log_level: "info"
```

**Step 2 — download weights (one-time, on the host):**

```bash
python scripts/download_model.py --config /etc/inference-server/config.yaml
# weights land in /opt/models/my-vlm on the host
```

**Step 3 — run the server:**

```bash
docker run -d \
  --gpus all \
  --shm-size=16g \
  -v /opt/models:/models:ro \
  -v /etc/inference-server:/etc/inference-server:ro \
  -p 8000:8000 \
  inference-server:0.1.0
```

No model-specific flags on the `docker run` line. Swapping to a different model means editing `config.yaml` and re-running the download helper; the image and the `docker run` invocation are unchanged. This shape is what gets reused (with different config files) when this lands on AI-R production hardware later.

## 13. Open questions

These are the blockers I need answers on before or during the build:

1. **First model to test with.** The server is model-agnostic, but we need at least one model to validate against. Which checkpoint does the Verkos team want to test first (VLM vs text-only, quantization scheme)? This is a test fixture decision, not an architectural one. **Owner: Verkos team.**
2. **Image input format.** How does Verkos send images today via OpenRouter — base64-embedded in the message, or as URLs? v0.1 needs to accept whatever they're already sending. **Owner: Verkos team.**
3. **Weight storage location on the tower.** Where should the download helper deposit weights on the host? Impacts the volume-mount path in the config template. **Owner: infra.**
4. **Gemini fallback.** Out of scope for this container, but determines whether one model config is sufficient or whether a second server instance (different config file, different port) needs to be reachable simultaneously. **Owner: Verkos team + me, via accuracy comparison.**

## 14. Acceptance criteria

v0.1 ships when:

- [ ] Docker image builds reproducibly from a tagged commit with no model-specific content.
- [ ] `scripts/download_model.py` downloads weights to the host path declared in `config.yaml` and writes a checksum manifest.
- [ ] Container starts on the tower with only a `config.yaml` and a pre-downloaded weights directory; no model flags on `docker run`.
- [ ] Swapping to a different model requires only editing `config.yaml` (and re-running the download helper) — no image rebuild, no `docker run` change.
- [ ] `GET /health` returns ready within 60s of container start.
- [ ] `POST /v1/chat/completions` with a real test image returns a coherent response under 10s for the configured model.
- [ ] `GET /metrics` reports VRAM and request latency.
- [ ] `curl` round-trip from another machine on the LAN works.
- [ ] Effective config (model name, path, all engine args) is dumped to structured logs on startup.
- [ ] OpenAI Python client pointed at `http://tower:8000/v1` works end-to-end with vision input.

The core invariant: the image is a runtime, not a model bundle. Changing what model runs is a config and data operation, not a build operation.

## 15. Out of scope, for explicit acknowledgment

The following come up in adjacent docs (Tracks, PR FAQ, GPU Sizing Plan) and are **not** addressed here:

- The K3s deployment layer (this v0.1 runs as a bare Docker container; K3s comes later).
- Mender OTA (image-based updates ship in the AI-R OS layer, not in this container).
- Storage quota enforcement (this container reads weights and writes logs; quota management is the host's job).
- The MQTT-based unified comms layer (this server is a callee, not a publisher).
- The model router (the LRU/swap layer that will eventually sit in front of multiple inference servers).