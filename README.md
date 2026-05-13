# verkos-inference-server

FastAPI wrapper around vLLM that serves a single vision-language model behind an OpenAI-compatible API. Config-driven: no model identity is baked into the image. Changing what model runs is a config and data operation, not a build operation.

---

## Table of contents

- [How it works](#how-it-works)
- [Quick start (local, no GPU)](#quick-start-local-no-gpu)
- [Deploying on the tower (with GPU)](#deploying-on-the-tower-with-gpu)
- [Configuration reference](#configuration-reference)
- [Environment variable overrides](#environment-variable-overrides)
- [API reference](#api-reference)
- [Observability](#observability)
- [Project layout](#project-layout)

---

## How it works

```
  Host filesystem
  ┌──────────────────────────────────────────────┐
  │  config.yaml          (model identity)        │
  │  /opt/models/<name>/  (weights)               │
  │         ▲                                     │
  │         │ written by                          │
  │  scripts/download_model.py                    │
  └───────────────────┬──────────────────────────┘
                      │ volume-mounted read-only
                      ▼
  ┌──────────────── container ──────────────────┐
  │  FastAPI (uvicorn)                          │
  │   ├── POST /v1/chat/completions ──► vLLM   │
  │   ├── GET  /v1/models                       │
  │   ├── GET  /health                          │
  │   ├── GET  /metrics                         │
  │   └── GET  /admin/config                    │
  └─────────────────────────────────────────────┘
```

1. Write a `config.yaml` on the host describing the model.
2. Run `scripts/download_model.py` to fetch weights to the path named in the config.
3. Start the server (Docker or bare uvicorn). It reads the config, loads the engine, and begins serving.

The container never pulls from the network. Once weights are on disk it runs fully air-gapped.

---

## Quick start (local, no GPU)

Useful for verifying the API surface and config loading without a GPU. The engine will not load (vLLM requires CUDA), but all non-inference routes respond normally.

**1. Install dependencies**

```bash
uv sync
```

**2. Point at a config file**

```bash
cp config.yaml.example my-config.yaml
# edit model.name and model.path as needed — the path doesn't have to exist locally
```

**3. Start the server**

```bash
CONFIG_PATH=my-config.yaml .venv/bin/uvicorn app.main:app --reload
```

**4. Verify**

```bash
curl http://localhost:8000/health
# → {"status":"ok","model":null,"ready":false}
# ready:false is expected — no GPU, no model loaded

curl http://localhost:8000/v1/models
curl http://localhost:8000/metrics
curl http://localhost:8000/admin/config
```

---

## Deploying on the tower (with GPU)

### Step 1 — write a config file on the host

```yaml
# /etc/inference-server/config.yaml
model:
  name: "qwen2-vl-7b"
  path: "/opt/models/qwen2-vl-7b"
  source:
    type: "huggingface"
    repo_id: "Qwen/Qwen2-VL-7B-Instruct"
    revision: "main"
  dtype: "auto"
  quantization: "none"
  max_model_len: 4096
  max_num_seqs: 256

engine:
  gpu_memory_utilization: 0.90
  tensor_parallel_size: 1

server:
  port: 8000
  log_level: "info"
```

### Step 2 — download weights (one-time)

```bash
python scripts/download_model.py --config /etc/inference-server/config.yaml
```

The script is idempotent: re-running it skips the download if the manifest exists and all SHA-256 checksums pass. For gated HuggingFace models (e.g. Llama):

```bash
huggingface-cli login   # or set HF_TOKEN=hf_xxx
python scripts/download_model.py --config /etc/inference-server/config.yaml
```

### Step 3 — build the image

```bash
uv export --no-dev > requirements.txt
docker build -t inference-server:0.1.0 .
```

### Step 4 — run

```bash
docker run -d \
  --gpus all \
  --shm-size=16g \
  -v /opt/models:/models:ro \
  -v /etc/inference-server:/etc/inference-server:ro \
  -p 8000:8000 \
  inference-server:0.1.0
```

Model weights and config are always volume-mounted, never baked into the image. To swap models: edit `config.yaml`, re-run the download script for the new path, restart the container. No image rebuild.

### Step 5 — wait for ready

Engine load takes 30–120 s depending on model size:

```bash
watch -n 5 'curl -s http://localhost:8000/health'
# wait until ready:true
```

---

## Configuration reference

The server reads a single YAML file (default path `/etc/inference-server/config.yaml`, overridable via `CONFIG_PATH` env var). It refuses to start if the file is absent or fails validation.

```yaml
model:
  name: "my-vlm"           # label surfaced in /v1/models and logs
  path: "/models/my-vlm"   # absolute path to weights directory inside the container

  # Populated by scripts/download_model.py. Used for logging provenance on startup.
  source:
    type: "huggingface"    # huggingface | s3 | local-copy
    repo_id: "org/name"    # HF repo ID, s3://bucket/prefix, or absolute host path
    revision: "main"       # git ref or S3 version; omit for local-copy

  dtype: "auto"            # auto | float16 | bfloat16
  quantization: "none"     # none | awq | gptq
  max_model_len: null      # null = vLLM default; set to cap context length
  max_num_seqs: 256        # max concurrent sequences in the engine

engine:
  gpu_memory_utilization: 0.90   # fraction of GPU VRAM vLLM may use
  tensor_parallel_size: 1        # number of GPUs; always 1 in v0.1

  # Any additional vLLM EngineArgs field can be added here and is passed through verbatim.
  # Unknown keys that vLLM also rejects will cause a startup error.
  # enforce_eager: true
  # max_num_batched_tokens: 8192

server:
  port: 8000
  log_level: "info"        # debug | info | warn | error
```

**Only `model.name` and `model.path` are required.** Everything else has a sensible default.

---

## Environment variable overrides

Env vars take precedence over the YAML file. When a conflict is detected, both values are logged at startup so the override is auditable.

| Variable | Overrides |
|---|---|
| `CONFIG_PATH` | Path to the config file itself (default: `/etc/inference-server/config.yaml`) |
| `MODEL_PATH` | `model.path` |
| `MODEL_DTYPE` | `model.dtype` |
| `QUANTIZATION` | `model.quantization` |
| `GPU_MEMORY_UTILIZATION` | `engine.gpu_memory_utilization` |
| `MAX_MODEL_LEN` | `model.max_model_len` |
| `MAX_NUM_SEQS` | `model.max_num_seqs` |
| `TENSOR_PARALLEL_SIZE` | `engine.tensor_parallel_size` |
| `PORT` | `server.port` |
| `LOG_LEVEL` | `server.log_level` |

Useful for quick experiments without editing the file:

```bash
GPU_MEMORY_UTILIZATION=0.75 MAX_MODEL_LEN=2048 \
  CONFIG_PATH=/etc/inference-server/config.yaml \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## API reference

### `POST /v1/chat/completions`

OpenAI-compatible chat completion. Accepts vision input as base64 data URIs (no URL fetching — air-gapped).

**Request**

```json
{
  "model": "qwen2-vl-7b",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "What is in this image?" },
        { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,/9j/4AAQ..." } }
      ]
    }
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": false
}
```

- `model` — optional; defaults to the name in config.
- `content` — either a plain string or an array of `text` / `image_url` parts.
- `image_url.url` — **must** be a `data:image/...;base64,...` URI. URL references are rejected (air-gapped constraint).
- `stream: true` — returns SSE chunks in OpenAI format, terminated with `data: [DONE]`.

**Response (non-streaming)**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1718000000,
  "model": "qwen2-vl-7b",
  "choices": [{ "index": 0, "message": { "role": "assistant", "content": "..." }, "finish_reason": "stop" }],
  "usage": { "prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59 }
}
```

Returns `503` if the engine has not finished loading.

---

### `GET /v1/models`

Lists the currently loaded model. OpenAI-compatible shape.

```json
{ "object": "list", "data": [{ "id": "qwen2-vl-7b", "object": "model", "created": 1718000000, "owned_by": "local" }] }
```

---

### `GET /health`

Liveness and readiness in one response.

```json
{ "status": "ok", "model": "qwen2-vl-7b", "ready": true }
```

- `ready: false` — process is up but the engine is still loading (or the model path was missing).
- `ready: true` — engine is loaded and accepting requests.

---

### `GET /metrics`

JSON snapshot of runtime state. Useful for manual inspection and scripted monitoring.

```json
{
  "uptime_seconds": 142.3,
  "model": "qwen2-vl-7b",
  "requests": { "success": 12, "error": 1 },
  "in_flight": 0,
  "avg_latency_seconds": 4.2,
  "gpus": [
    { "index": 0, "memory_used_bytes": 12884901888, "memory_total_bytes": 25769803776 }
  ]
}
```

`gpus` is empty on machines without an NVIDIA GPU (or without `pynvml` available).

---

### `GET /admin/config`

Returns the full effective config as JSON — the merged result of the YAML file and any env var overrides applied at startup. Use this to confirm what the server is actually running with.

---

### Reserved endpoints (return 501)

These URLs are defined now so consumers don't need to migrate later when lifecycle management is implemented:

| Method | Path |
|---|---|
| `POST` | `/v1/models/load` |
| `DELETE` | `/v1/models/{id}` |
| `POST` | `/v1/models/{id}/warmup` |

---

## Observability

### Structured request logs

Every completed inference request emits a JSON line to stdout:

```json
{
  "asctime": "2026-05-13 12:00:00,123",
  "levelname": "INFO",
  "name": "inference_server.chat",
  "message": "chat_completion",
  "latency_ms": 3842.1,
  "prompt_tokens": 42,
  "completion_tokens": 17,
  "image_count": 1,
  "model": "qwen2-vl-7b",
  "status": "success"
}
```

Startup emits an `"Effective config"` line with the full merged config so the running state is always in the log.

### `/metrics` endpoint

Poll this to track VRAM consumption and throughput without a Prometheus setup:

```bash
watch -n 10 'curl -s http://localhost:8000/metrics | python3 -m json.tool'
```

---

## Project layout

```
app/
  main.py              # FastAPI app, lifespan, router registration
  config.py            # YAML + env var merge, Pydantic settings
  engine.py            # vLLM AsyncLLMEngine wrapper
  routes/
    chat.py            # POST /v1/chat/completions (streaming + non-streaming)
    models.py          # GET /v1/models
    health.py          # GET /health
    metrics.py         # GET /metrics + in-memory counters
    admin.py           # GET /admin/config
    reserved.py        # 501 stubs
  schemas/
    chat.py            # ChatCompletionRequest/Response, Message, Usage, chunks
    models.py          # ModelCard, ModelList
    health.py          # HealthResponse
    metrics.py         # GpuInfo, MetricsResponse
    admin.py           # AdminConfigResponse
    reserved.py        # NotImplementedResponse
scripts/
  download_model.py    # fetches weights from HuggingFace / S3 / local path
Dockerfile
config.yaml.example
```
