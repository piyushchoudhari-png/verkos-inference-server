# Verkos Inference Gateway — Full Setup & Operations Walkthrough

This is the soup-to-nuts version: every step, every flag, every "what should I see when it works." Run it in order.

---

## Step 0 · Prerequisites

The gateway itself is small Python, but the sglang subprocess needs a GPU. Verify these before you start:

| Requirement | How to check | What you should see |
|---|---|---|
| NVIDIA GPU + driver | `nvidia-smi` | Tables of GPUs, driver version, no errors |
| CUDA runtime sglang can use | `nvidia-smi` Driver Version column ≥ 535 | sglang needs CUDA 12.x toolchain at runtime |
| Python 3.12+ | `python3 --version` | `Python 3.12.x` or newer |
| `uv` package manager | `uv --version` | A version string; if not installed: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `curl` and `jq` | `curl --version`, `jq --version` | Both print versions. Needed for the test suite. |
| Model weights on disk | `ls /path/to/your-model` | Files like `config.json`, `tokenizer.json`, weight shards |

The gateway is **not bundled with sglang** — it shells out to `python -m sglang.launch_server`. You'll install sglang into the same venv in Step 2.

---

## Step 1 · Get the code & install gateway dependencies

```bash
# 1.1 — clone (you've already done this; included for completeness)
git clone <your-fork-url> verkos-inference-server
cd verkos-inference-server

# 1.2 — install the gateway's own Python deps (fastapi, httpx, pyyaml, prometheus-client, nvidia-ml-py, etc.)
uv sync
```

**What `uv sync` does:** creates `.venv/` in the repo root, installs the dependencies declared in `pyproject.toml` and pinned in `uv.lock`. Re-running it is idempotent.

**What you should see:** a line like `Resolved N packages` followed by `Prepared N packages` and then `Installed N packages`. No errors.

---

## Step 2 · Install sglang into the same venv

The gateway spawns sglang by running `python -m sglang.launch_server` *inside its own venv* (by default — `manager.spawn_python: null` means "use my interpreter"). So sglang must be importable from `.venv/bin/python`.

```bash
# 2.1 — install sglang and its runtime deps into the gateway's venv
uv pip install "sglang[all]"
```

**Why `[all]`:** pulls in the torch + flashinfer + cuda toolchain that sglang needs at runtime. This download is large (several GB).

**Verify it's importable:**

```bash
uv run python -c "import sglang; print(sglang.__version__)"
# → some version string, e.g. 0.4.x
```

**Alternative if you want sglang in a different venv:** install it wherever you like, then in `gateway.yaml` set:

```yaml
manager:
  spawn_python: "/absolute/path/to/that/venv/bin/python"
```

The gateway will use that interpreter to spawn sglang while continuing to run itself out of `.venv`.

---

## Step 3 · Configure the gateway (`gateway.yaml`)

The gateway needs **one** YAML file that tells it (a) where the bearer-key store lives, (b) which models clients are allowed to request, and (c) how to bind the HTTP server.

### 3.1 Copy the template

```bash
# Two starting points:
#   gateway.yaml.template  — pure placeholders, blank slate
#   gateway.yaml.example   — working two-model example (chat-vlm + bge-embed)

# If you want a clean slate to fill in:
cp gateway.yaml.template gateway.yaml

# If you want the worked example to tweak:
cp gateway.yaml.example gateway.yaml
```

### 3.2 Open and fill in every `<PLACEHOLDER>`

`gateway.yaml` after editing should look like this (yours will differ in paths/ids):

```yaml
auth:
  keys_file: "/home/youruser/verkos/keys.json"   # absolute path is safest

models:
  - id: "chat-vlm"                               # the value clients send in the OpenAI `model` field
    config_ref: "workers/chat-vlm.yaml"          # path relative to gateway.yaml's directory
  - id: "bge-embed"
    config_ref: "workers/bge-embed.yaml"

server:
  host: "0.0.0.0"        # 0.0.0.0 = LAN-reachable; 127.0.0.1 = localhost-only
  port: 8000
  log_level: "info"

gpu:
  poll_interval_s: 5.0

manager:
  load_timeout_s: 120.0           # widen this if your model takes longer than 2 min to cold-start
  health_poll_interval_s: 0.5
  spawn_python: null              # null → use this venv; set to a path to spawn from a different one
```

**What each block does:**

- **`auth.keys_file`** — the gateway reads this at startup. If the file doesn't exist or isn't valid JSON, **the gateway refuses to start**. We'll create it in Step 5.
- **`models[]`** — the catalog. Every entry must have an `id` (the client-facing name) and a `config_ref` (the path to a worker YAML). An inference request for an `id` not in this list returns 404.
- **`server.host`** — `0.0.0.0` makes the gateway reachable from other machines on the LAN; `127.0.0.1` restricts it to this box.
- **`server.log_level`** — `info` is the production default. Use `debug` if you want to see every internal log line.
- **`manager.load_timeout_s`** — if sglang's `/health` doesn't return 200 within this many seconds after spawn, the gateway returns 504 and the slot goes back to EMPTY. Large models with slow disks may need 300+.

---

## Step 4 · Configure each model (`workers/*.yaml`)

For every entry in `gateway.yaml::models[]`, there's a `workers/<name>.yaml` file telling the manager *how to launch sglang* for that model.

### 4.1 Copy the template

```bash
cp workers/template.yaml workers/chat-vlm.yaml
```

(Repeat once per model. If you started from `gateway.yaml.example`, `workers/chat-vlm.yaml` and `workers/bge-embed.yaml` already exist as concrete starting points.)

### 4.2 Fill in the placeholders

A complete worker file for a chat/VLM model looks like:

```yaml
worker:
  name: "chat-vlm"          # logging label
  host: "127.0.0.1"         # MUST stay 127.0.0.1
  port: 30001               # unique per catalog entry; gateway connects here after sglang is up

sglang:
  model_path: "/data/models/qwen2-vl-7b"   # absolute path to the weights directory
  served_model_name: "chat-vlm"            # what sglang advertises; usually matches worker.name
  dtype: "bfloat16"                        # bfloat16 | float16 | awq_marlin | ...
  mem_fraction_static: 0.85                # 0.0–1.0 of VRAM sglang reserves on startup
  chat_template: "chatml"                  # named template OR path to a Jinja file

capabilities:
  chat: true
  vision: true
  embeddings: false
  max_context_tokens: 32768
  max_image_pixels: 2000000
```

For an embedding model, the differences are:

```yaml
sglang:
  is_embedding: true                       # bare --is-embedding flag
  # NO chat_template

capabilities:
  chat: false
  embeddings: true
  max_context_tokens: 512
```

### 4.3 Important rules

- **Ports must be unique across the catalog.** chat-vlm on 30001, bge-embed on 30002, etc.
- **`worker.host` MUST be `127.0.0.1`.** The launcher refuses anything else — sglang has no auth, so it stays on loopback.
- **`served_model_name` is what sglang's `/v1/models` advertises.** The gateway accepts whatever id you put in `gateway.yaml::models[].id`; they're conventionally the same but don't have to be.
- **Anything under `sglang:` that isn't a known field is passed through verbatim to `sglang.launch_server`.** snake_case becomes `--kebab-case`. So `tensor_parallel_size: 2` becomes `--tensor-parallel-size 2`. `quantization: "awq"` becomes `--quantization awq`.

### 4.4 Dry-run check (optional but recommended)

Before running the gateway, verify the YAML translates to the sglang argv you expect:

```bash
uv run python -m scripts.launch_worker --config workers/chat-vlm.yaml --dry-run
```

**What you should see:** a single line like

```
/home/.../.venv/bin/python3 -m sglang.launch_server --host 127.0.0.1 --port 30001 --model-path /data/models/qwen2-vl-7b --served-model-name chat-vlm --dtype bfloat16 --mem-fraction-static 0.85 --chat-template chatml
```

If you see `ValueError: ... worker.host must be 127.0.0.1`, you've got a config bug — fix the worker file.

### 4.5 (Optional) Boot sglang standalone to verify weights load

```bash
uv run python -m scripts.launch_worker --config workers/chat-vlm.yaml
```

This runs the same argv from §4.4 but actually execs it. sglang will load the weights and start serving on `127.0.0.1:30001`. You should see sglang's own startup logs. Ctrl-C kills it.

Test it from another shell:

```bash
curl http://127.0.0.1:30001/health
# → ok
curl http://127.0.0.1:30001/v1/models
# → {"object":"list","data":[{"id":"chat-vlm",...}]}
```

If both work, your weights and flags are correct. Kill sglang (Ctrl-C in the first shell) before proceeding — the gateway will spawn its own.

---

## Step 5 · Mint API keys (`keys.json`)

The gateway authenticates every `/v1/*` request via `Authorization: Bearer <key>`. Keys live in `keys.json`. You never edit this file by hand.

### 5.1 Mint your first key

```bash
uv run python scripts/generate_api_key.py --name dev --keys-file /home/youruser/verkos/keys.json
```

**What you should see — copy the printed key now, it's only shown once:**

```
sk-flytbase-airr-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789-aAbBcCdD
```

The script also creates `/home/youruser/verkos/keys.json` with mode `0600` (only your user can read it). After this, the file looks like `keys.json.example` — a JSON array of records with `name`, `key`, `created_at`, `revoked_at`.

### 5.2 Other key-script operations

```bash
# List active keys (full key is NEVER shown again — only the first 12 chars)
uv run python scripts/generate_api_key.py --list --keys-file /home/youruser/verkos/keys.json

# Mint a second key for a different consumer
uv run python scripts/generate_api_key.py --name alice --keys-file /home/youruser/verkos/keys.json

# Revoke a key by name (marks revoked_at; gateway 401s any request using it after SIGHUP)
uv run python scripts/generate_api_key.py --revoke dev --keys-file /home/youruser/verkos/keys.json
```

### 5.3 Make sure `gateway.yaml` points at this file

In `gateway.yaml`:

```yaml
auth:
  keys_file: "/home/youruser/verkos/keys.json"
```

The path you pass to `--keys-file` and the path in `gateway.yaml` **must be the same file**, or the gateway will reject every request as 401.

---

## Step 6 · Start the gateway

```bash
uv run python -m src.serving --config gateway.yaml
```

**What happens in order:**

1. The CLI parses `--config` and reads `gateway.yaml`.
2. `KeyStore.reload()` reads `keys.json`. If missing or malformed, **the gateway exits immediately with an error.**
3. `ModelManager.from_config` reads each `workers/*.yaml` referenced by the catalog. If any worker YAML is malformed, **the gateway exits immediately.**
4. uvicorn binds to `server.host:server.port` (e.g. `0.0.0.0:8000`).
5. The slot starts in state **EMPTY**. No sglang process exists yet.
6. Background tasks start: SIGHUP handler (key reload) + GPU NVML poller.

**What you should see** (JSON-formatted log lines on stdout):

```json
{"time":"...","name":"uvicorn","level":"INFO","message":"Started server process [12345]"}
{"time":"...","name":"uvicorn","level":"INFO","message":"Uvicorn running on http://0.0.0.0:8000"}
```

The terminal is now blocked serving requests. Open another shell for everything below. Ctrl-C in this shell triggers graceful shutdown: any live sglang child is SIGKILL'd, then the gateway exits.

---

## Step 7 · Smoke-test the gateway (from another shell)

```bash
# 7.1 Liveness — no auth needed.
curl http://localhost:8000/healthz
# → {"status":"ok"}

# 7.2 Readiness — should be 503 because no model is loaded yet.
curl -i http://localhost:8000/readyz
# → HTTP/1.1 503 Service Unavailable
#   {"status":"not_ready","slot_state":"empty","model":null}

# 7.3 Metrics — Prometheus exposition format.
curl http://localhost:8000/metrics | grep verkos_model_state
# → verkos_model_state{model="chat-vlm",state="empty"} 1.0
#   verkos_model_state{model="bge-embed",state="empty"} 1.0
#   (others at 0.0)

# 7.4 List the catalog (auth required). Replace KEY with the one you minted in Step 5.
export KEY="sk-flytbase-airr-AbCdEfGh..."
curl -H "Authorization: Bearer $KEY" http://localhost:8000/v1/models | jq
# → {"object":"list","data":[
#       {"id":"chat-vlm","object":"model","state":"empty","capabilities":{...},"last_error":null},
#       {"id":"bge-embed","object":"model","state":"empty","capabilities":{...},"last_error":null}
#    ]}
```

### 7.5 Trigger the first cold start

```bash
# Explicit load — easier to time, since it returns when ready.
time curl -XPOST -H "Authorization: Bearer $KEY" http://localhost:8000/v1/models/chat-vlm/load
# → {"status":"ready","model":"chat-vlm","host":"127.0.0.1","port":30001}
#   real    0m23.412s   ← typical 7B-class cold start
```

**While this is happening**, in the gateway log shell you'll see:

```json
{"...","message":"load_started","model":"chat-vlm","port":30001}
... sglang's own logs interleaved ...
{"...","message":"load_complete","model":"chat-vlm","pid":67890,"duration_ms":23412.18}
```

Confirm the slot is now READY:

```bash
curl http://localhost:8000/readyz
# → {"status":"ready","model":"chat-vlm"}
curl -H "Authorization: Bearer $KEY" http://localhost:8000/v1/models/chat-vlm | jq .state
# → "ready"
```

### 7.6 First inference

```bash
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"chat-vlm","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":8,"temperature":0}' \
  http://localhost:8000/v1/chat/completions | jq
```

You should get an OpenAI-shaped response with `choices[0].message.content` populated.

### 7.7 Streaming

```bash
curl -N -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"chat-vlm","stream":true,"messages":[{"role":"user","content":"Count to 3."}],"max_tokens":24}' \
  http://localhost:8000/v1/chat/completions
```

Note `-N` (no buffering). You should see a stream of `data: {...}` lines ending in `data: [DONE]`.

### 7.8 Implicit swap

```bash
# Currently loaded: chat-vlm. Request bge-embed → swap.
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"bge-embed","input":"hello world"}' \
  http://localhost:8000/v1/embeddings | jq '.data[0].embedding | length'
# → 768 (or whatever dimension bge-embed has)
```

In the gateway logs you'll see `unload_started → unload_complete → load_started → load_complete`, all for the model swap.

### 7.9 Explicit unload

```bash
curl -XPOST -H "Authorization: Bearer $KEY" http://localhost:8000/v1/models/bge-embed/unload
# → {"status":"unloaded","model":"bge-embed"}
nvidia-smi  # VRAM should drop; sglang process is gone
```

---

## Step 8 · Run the automated test suite

In a shell separate from the gateway, with the gateway still running:

```bash
export GATEWAY_URL=http://localhost:8000
export API_KEY="sk-flytbase-airr-AbCdEfGh..."   # the key you minted in Step 5

./tests/api/run.sh
```

**What the suite does, in order:**

1. **Preflight** — confirms `curl` + `jq` are present, `API_KEY` is set, and `GATEWAY_URL/healthz` returns 200. Bails out with a clear error if any of those fail.
2. **`health, readyz, metrics`** — verifies the three no-auth endpoints work and `/metrics` exposes the `verkos_*` collectors.
3. **`auth`** — sends `/v1/models` with no bearer, a bogus bearer, and a real bearer. Expects 401 / 401 / 200 with OpenAI-shaped error envelopes for the failures.
4. **`catalog`** — `/v1/models` returns `object=list`, every entry has `id`/`state`/`capabilities`, every state is one of `{empty,loading,ready}`, per-model lookup works, unknown id returns 404 with `code=model_not_found`.
5. **`inference errors`** — chat completion with no `model` field → 400; with an unknown model → 404; with malformed JSON → 400.
6. **`explicit load / unload`** — `POST /v1/models/chat-vlm/load` cold-starts (uses `COLD_START_TIMEOUT`, default 180s); `/readyz` flips to 200; `/v1/models` shows `state=ready`; second `/load` is idempotent; `/unload` returns `status=unloaded`; `/readyz` flips back to 503; second `/unload` returns `status=noop`.
7. **`chat completion (non-streaming)`** — buffered JSON response with `object=chat.completion`, non-empty content and usage.
8. **`chat completion (SSE)`** — verifies ≥2 `data:` chunks arrive and the stream terminator is `data: [DONE]`.
9. **`embeddings`** — `object=list`, non-empty embedding vector.
10. **`implicit swap`** — load chat-vlm → embed inference swaps to bge-embed → catalog reflects new state → chat inference swaps back → catalog reflects again.

**Output:** colorized pass/fail lines, then a summary. Non-zero exit if anything failed; the failed test names are listed.

**Useful variants:**

```bash
# No-load smoke — just auth, catalog, error envelopes. Runs in seconds.
SKIP_INFERENCE=1 ./tests/api/run.sh

# Skip the swap dance (still does cold-start + chat + embed).
SKIP_SWAP=1 ./tests/api/run.sh

# Bigger timeout per cold start (default 180s). Use for very large models.
COLD_START_TIMEOUT=600 ./tests/api/run.sh

# Test against a non-default catalog (your gateway.yaml uses different ids).
CHAT_MODEL=qwen-vlm EMBED_MODEL=nomic-embed ./tests/api/run.sh
```

---

## Step 9 · Open the Swagger / ReDoc documentation UI

FastAPI auto-generates these. With the gateway running, in a browser:

| URL | What it is |
|---|---|
| `http://localhost:8000/docs`         | **Swagger UI** — interactive, lets you click "Try it out" and execute requests |
| `http://localhost:8000/redoc`        | **ReDoc** — denser, reference-style rendering of the same schema |
| `http://localhost:8000/openapi.json` | Raw OpenAPI 3 schema; useful if you want to point another tool (Postman, etc.) at it |

**To exercise `/v1/*` endpoints from the browser:**

The Swagger UI shows every endpoint with its request/response schema, but **the current code does not declare a security scheme**, so the green "Authorize" button at the top doesn't pre-fill the bearer header. You have two options:

1. **Use Swagger to learn the shape, then `curl` from a terminal** with the `Authorization: Bearer ...` header. This is what I'd do.
2. **Wire up FastAPI's `HTTPBearer` security so Authorize works.** That's a small change in `src/serving/app.py`.

The `/healthz`, `/readyz`, and `/metrics` endpoints don't require auth, so those are clickable from Swagger straight away.

---

## Step 10 · Operational tasks

### Reload `keys.json` without restarting

```bash
# Find the gateway PID and send SIGHUP.
kill -HUP $(pgrep -f 'src.serving --config')
```

**What happens:** the gateway re-reads `keys.json` and swaps in the new snapshot atomically. You'll see `SIGHUP key reload complete` in the logs with the new active-keys count. Any in-flight requests complete unaffected; any *new* request with a revoked key gets 401.

### Run the gateway as a systemd unit

The repo ships `deploy/systemd/verkos-gateway.service`. Install it:

```bash
sudo install -d -o verkos -g verkos -m 0750 /etc/verkos /var/log/verkos
sudo cp gateway.yaml /etc/verkos/gateway.yaml
sudo cp keys.json /etc/verkos/keys.json
sudo chown -R verkos:verkos /etc/verkos
sudo cp deploy/systemd/verkos-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now verkos-gateway

# Operational commands
sudo systemctl status verkos-gateway
sudo systemctl restart verkos-gateway
sudo systemctl reload verkos-gateway     # SIGHUP — reloads keys.json
sudo journalctl -u verkos-gateway -f     # follow logs (JSON lines)
```

### Hook into Prometheus

Merge `deploy/prometheus/verkos.yaml` into your prometheus.yml's `scrape_configs:`. Update the `targets:` line to point at your gateway's host:port. Reload Prometheus; you'll see `verkos_*` metrics within one scrape interval (default 15s).

### Shutdown

`Ctrl-C` in the foreground or `systemctl stop verkos-gateway` for the service. Either way:

1. uvicorn stops accepting new connections.
2. The lifespan teardown calls `manager.shutdown()` → SIGKILL the sglang child → `await process.wait()` (no orphan).
3. The GPU poller task is cancelled.
4. Process exits.

`TimeoutStopSec=30` in the systemd unit gives this 30s before systemd escalates to SIGKILL on the gateway itself.

---

## Step 11 · Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Gateway exits at startup with `FileNotFoundError: keys.json` | `auth.keys_file` path doesn't exist or isn't readable | Mint a key (Step 5) at the path `gateway.yaml` expects. |
| `/v1/models/{id}/load` returns 502 `worker_error` | sglang exited during cold start | Check the gateway log for sglang stderr. Common causes: wrong `model_path`, OOM (lower `mem_fraction_static`), `chat_template` doesn't match the model. |
| `/v1/models/{id}/load` returns 504 `timeout` | sglang took longer than `manager.load_timeout_s` to come up | Increase `load_timeout_s` in `gateway.yaml`; large models on slow disks can need 300+ s. |
| Every `/v1/*` request returns 401 | The key you're sending doesn't match a non-revoked record in `keys.json`, OR you minted a key in a different `keys.json` than `gateway.yaml` points at | `--list` the keys file the gateway is using; re-mint if needed. |
| `/v1/models` returns 401 even though `/healthz` works | Probes are unauthenticated; `/v1/*` requires `Authorization: Bearer …`. Add the header. |
| sglang spawn fails with `ModuleNotFoundError: No module named 'sglang'` | sglang isn't installed in the venv the gateway runs from | `uv pip install sglang[all]` (Step 2), or point `manager.spawn_python` at a venv that has it. |
| Implicit swap drops a streaming client's connection | Working as designed in v1 (`hard-kill on swap`). Clients should treat unexpected stream termination as retryable. | If you need drain-on-swap, that's a v2 item. |
| Two models try to use the same port | Both `workers/*.yaml` set `worker.port` to the same value | Pick unique ports (30001, 30002, ...). |

To get more detail in the logs, set `server.log_level: "debug"` in `gateway.yaml` and restart.

---

That's the entire lifecycle: prereqs → install → configure → mint keys → start → exercise → test → docs → operate → debug.
