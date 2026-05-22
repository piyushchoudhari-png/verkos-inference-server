#!/usr/bin/env bash
# End-to-end API tests for the Verkos inference gateway, driven by curl.
#
# Exercises every endpoint in docs/INFERENCE_API_ARCHITECTURE.md §A.2 and
# verifies the v1 lifecycle contract (implicit load, swap, hard-kill on unload,
# streaming pass-through, OpenAI-shaped errors).
#
# Usage:
#   GATEWAY_URL=http://localhost:8000 \
#   API_KEY=sk-flytbase-airr-XXXX \
#   ./tests/api/run.sh
#
# Optional env vars:
#   CHAT_MODEL=qwen3-8b-vl-instruct      # catalog id of the chat/VLM model
#   EMBED_MODEL=qwen3-vl-embedding-2b    # catalog id of the embedding model
#   SKIP_INFERENCE=1                     # only run no-load tests (health, auth, catalog, errors)
#   SKIP_SWAP=1                          # skip the chat→embed→chat swap test
#   COLD_START_TIMEOUT=180               # seconds curl waits on a /load (default 180)
#
# Requires: curl, jq.

set -u
set -o pipefail

# ---------------------------------------------------------------------------
# Configuration

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-}"
CHAT_MODEL="${CHAT_MODEL:-qwen3-8b-vl-instruct}"
EMBED_MODEL="${EMBED_MODEL:-qwen3-vl-embedding-2b}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"
SKIP_SWAP="${SKIP_SWAP:-0}"
COLD_START_TIMEOUT="${COLD_START_TIMEOUT:-180}"

TMP_DIR="$(mktemp -d -t verkos-api-tests.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

PASSED=0
FAILED=0
FAILED_NAMES=()

# ---------------------------------------------------------------------------
# Output helpers

if [ -t 1 ]; then
  C_GREEN=$'\033[32m'
  C_RED=$'\033[31m'
  C_YELLOW=$'\033[33m'
  C_DIM=$'\033[2m'
  C_RESET=$'\033[0m'
else
  C_GREEN=""; C_RED=""; C_YELLOW=""; C_DIM=""; C_RESET=""
fi

section()   { printf '\n%s── %s ──%s\n' "$C_DIM" "$1" "$C_RESET"; }
pass()      { PASSED=$((PASSED+1)); printf '  %s✓%s %s\n' "$C_GREEN" "$C_RESET" "$1"; }
fail()      { FAILED=$((FAILED+1)); FAILED_NAMES+=("$1"); printf '  %s✗%s %s\n     %s\n' "$C_RED" "$C_RESET" "$1" "$2"; }
skip_note() { printf '  %s↷%s %s %s\n' "$C_YELLOW" "$C_RESET" "$1" "$C_DIM($2)$C_RESET"; }

# ---------------------------------------------------------------------------
# HTTP helpers

# Run curl, capture body to $1 and headers to $2, echo HTTP status to stdout.
http_call() {
  local body_file=$1 hdr_file=$2
  shift 2
  curl --silent --show-error \
    --max-time 60 \
    --output "$body_file" \
    --dump-header "$hdr_file" \
    --write-out "%{http_code}" \
    "$@"
}

# Same as http_call but with a longer timeout for cold-start endpoints.
http_call_slow() {
  local body_file=$1 hdr_file=$2
  shift 2
  curl --silent --show-error \
    --max-time "$COLD_START_TIMEOUT" \
    --output "$body_file" \
    --dump-header "$hdr_file" \
    --write-out "%{http_code}" \
    "$@"
}

expect_status() {
  local label=$1 expected=$2 actual=$3 body_file=$4
  if [ "$actual" = "$expected" ]; then
    pass "$label (HTTP $actual)"
    return 0
  fi
  local body
  body=$(head -c 400 "$body_file" 2>/dev/null || true)
  fail "$label" "expected HTTP $expected, got $actual; body: ${body}"
  return 1
}

expect_jq() {
  local label=$1 jq_filter=$2 expected=$3 body_file=$4
  local actual
  actual=$(jq -r "$jq_filter" <"$body_file" 2>/dev/null || true)
  if [ "$actual" = "$expected" ]; then
    pass "$label"
    return 0
  fi
  fail "$label" "filter '$jq_filter' expected '$expected', got '$actual'"
  return 1
}

# Check that jq filter returns a non-empty / non-null value.
expect_jq_nonempty() {
  local label=$1 jq_filter=$2 body_file=$3
  local actual
  actual=$(jq -r "$jq_filter" <"$body_file" 2>/dev/null || true)
  if [ -n "$actual" ] && [ "$actual" != "null" ]; then
    pass "$label"
    return 0
  fi
  fail "$label" "filter '$jq_filter' returned null/empty"
  return 1
}

auth_header() { printf 'Authorization: Bearer %s' "$API_KEY"; }

# ---------------------------------------------------------------------------
# Preflight

preflight() {
  command -v curl >/dev/null 2>&1 || { echo "error: curl is required" >&2; exit 2; }
  command -v jq   >/dev/null 2>&1 || { echo "error: jq is required"   >&2; exit 2; }
  if [ -z "$API_KEY" ]; then
    echo "error: API_KEY is required" >&2
    exit 2
  fi
  local body hdr status
  body="$TMP_DIR/preflight.body"; hdr="$TMP_DIR/preflight.hdr"
  status=$(http_call "$body" "$hdr" "$GATEWAY_URL/healthz" 2>/dev/null || true)
  if [ "$status" != "200" ]; then
    echo "error: gateway is not reachable at $GATEWAY_URL/healthz (got HTTP ${status:-???})" >&2
    exit 2
  fi
  printf '%sGateway:%s %s   %sChat:%s %s   %sEmbed:%s %s\n' \
    "$C_DIM" "$C_RESET" "$GATEWAY_URL" \
    "$C_DIM" "$C_RESET" "$CHAT_MODEL" \
    "$C_DIM" "$C_RESET" "$EMBED_MODEL"
}

# ---------------------------------------------------------------------------
# Tests — no model load required

test_health_and_metrics() {
  section "health, readyz, metrics"
  local body hdr status

  body="$TMP_DIR/healthz.json"; hdr="$TMP_DIR/healthz.hdr"
  status=$(http_call "$body" "$hdr" "$GATEWAY_URL/healthz")
  expect_status "GET /healthz" 200 "$status" "$body"
  expect_jq    "  status=ok"  ".status" "ok" "$body"

  body="$TMP_DIR/metrics.txt"; hdr="$TMP_DIR/metrics.hdr"
  status=$(http_call "$body" "$hdr" "$GATEWAY_URL/metrics")
  expect_status "GET /metrics" 200 "$status" "$body"
  if grep -q '^verkos_requests_total' "$body"; then
    pass "  /metrics exposes verkos_* collectors"
  else
    fail "/metrics collectors" "verkos_requests_total not found in body"
  fi

  body="$TMP_DIR/readyz.json"; hdr="$TMP_DIR/readyz.hdr"
  status=$(http_call "$body" "$hdr" "$GATEWAY_URL/readyz")
  # 200 if a model happens to be loaded already; 503 otherwise. Both are valid.
  if [ "$status" = "200" ] || [ "$status" = "503" ]; then
    pass "GET /readyz (HTTP $status — valid)"
  else
    fail "GET /readyz" "expected 200 or 503, got $status"
  fi
}

test_auth() {
  section "auth"
  local body hdr status

  body="$TMP_DIR/auth_missing.json"; hdr="$TMP_DIR/auth_missing.hdr"
  status=$(http_call "$body" "$hdr" "$GATEWAY_URL/v1/models")
  expect_status "GET /v1/models without bearer" 401 "$status" "$body"
  expect_jq    "  envelope.error.type=authentication_error" ".error.type" "authentication_error" "$body"

  body="$TMP_DIR/auth_bad.json"; hdr="$TMP_DIR/auth_bad.hdr"
  status=$(http_call "$body" "$hdr" -H "Authorization: Bearer sk-not-a-real-key" "$GATEWAY_URL/v1/models")
  expect_status "GET /v1/models with bogus bearer" 401 "$status" "$body"

  body="$TMP_DIR/auth_ok.json"; hdr="$TMP_DIR/auth_ok.hdr"
  status=$(http_call "$body" "$hdr" -H "$(auth_header)" "$GATEWAY_URL/v1/models")
  expect_status "GET /v1/models with valid bearer" 200 "$status" "$body"
}

test_catalog() {
  section "catalog"
  local body hdr status

  body="$TMP_DIR/models.json"; hdr="$TMP_DIR/models.hdr"
  status=$(http_call "$body" "$hdr" -H "$(auth_header)" "$GATEWAY_URL/v1/models")
  expect_status "GET /v1/models" 200 "$status" "$body"
  expect_jq    "  object=list" ".object" "list" "$body"
  expect_jq_nonempty "  data is non-empty" ".data | length > 0 // empty" "$body"
  # Each entry has id+object+state+capabilities
  local missing
  missing=$(jq -r '[.data[] | select(.id == null or .state == null or .capabilities == null)] | length' <"$body")
  if [ "$missing" = "0" ]; then
    pass "  every entry has id, state, capabilities"
  else
    fail "/v1/models entry shape" "$missing entry/entries missing required fields"
  fi
  # state is one of empty|loading|ready
  local bad_states
  bad_states=$(jq -r '[.data[].state | select(. != "empty" and . != "loading" and . != "ready")] | length' <"$body")
  if [ "$bad_states" = "0" ]; then
    pass "  every state in {empty, loading, ready}"
  else
    fail "/v1/models state values" "$bad_states entry/entries have an out-of-range state"
  fi

  body="$TMP_DIR/model_one.json"; hdr="$TMP_DIR/model_one.hdr"
  status=$(http_call "$body" "$hdr" -H "$(auth_header)" "$GATEWAY_URL/v1/models/$CHAT_MODEL")
  expect_status "GET /v1/models/$CHAT_MODEL" 200 "$status" "$body"
  expect_jq    "  id=$CHAT_MODEL" ".id" "$CHAT_MODEL" "$body"

  body="$TMP_DIR/model_missing.json"; hdr="$TMP_DIR/model_missing.hdr"
  status=$(http_call "$body" "$hdr" -H "$(auth_header)" "$GATEWAY_URL/v1/models/does-not-exist")
  expect_status "GET /v1/models/does-not-exist" 404 "$status" "$body"
  expect_jq    "  envelope.error.code=model_not_found" ".error.code" "model_not_found" "$body"
}

test_inference_errors() {
  section "inference errors (no load required)"
  local body hdr status

  body="$TMP_DIR/missing_model.json"; hdr="$TMP_DIR/missing_model.hdr"
  status=$(http_call "$body" "$hdr" -X POST -H "$(auth_header)" -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"hi"}]}' "$GATEWAY_URL/v1/chat/completions")
  expect_status "POST /v1/chat/completions without model field" 400 "$status" "$body"
  expect_jq    "  envelope.error.type=invalid_request_error" ".error.type" "invalid_request_error" "$body"

  body="$TMP_DIR/bad_model.json"; hdr="$TMP_DIR/bad_model.hdr"
  status=$(http_call "$body" "$hdr" -X POST -H "$(auth_header)" -H "Content-Type: application/json" \
    -d '{"model":"not-in-catalog","messages":[{"role":"user","content":"hi"}]}' "$GATEWAY_URL/v1/chat/completions")
  expect_status "POST /v1/chat/completions with unknown model" 404 "$status" "$body"
  expect_jq    "  envelope.error.code=model_not_found" ".error.code" "model_not_found" "$body"

  body="$TMP_DIR/bad_json.json"; hdr="$TMP_DIR/bad_json.hdr"
  status=$(http_call "$body" "$hdr" -X POST -H "$(auth_header)" -H "Content-Type: application/json" \
    -d 'not-json' "$GATEWAY_URL/v1/chat/completions")
  expect_status "POST /v1/chat/completions with invalid JSON" 400 "$status" "$body"
}

# ---------------------------------------------------------------------------
# Tests — require model load

test_explicit_load_and_unload() {
  section "explicit load / unload ($CHAT_MODEL)"
  local body hdr status

  body="$TMP_DIR/load.json"; hdr="$TMP_DIR/load.hdr"
  status=$(http_call_slow "$body" "$hdr" -X POST -H "$(auth_header)" "$GATEWAY_URL/v1/models/$CHAT_MODEL/load")
  expect_status "POST /v1/models/$CHAT_MODEL/load (cold start)" 200 "$status" "$body"
  expect_jq    "  status=ready" ".status" "ready" "$body"
  expect_jq    "  model=$CHAT_MODEL" ".model" "$CHAT_MODEL" "$body"

  # /readyz should now be 200.
  body="$TMP_DIR/readyz_after_load.json"; hdr="$TMP_DIR/readyz_after_load.hdr"
  status=$(http_call "$body" "$hdr" "$GATEWAY_URL/readyz")
  expect_status "GET /readyz after load" 200 "$status" "$body"
  expect_jq    "  model=$CHAT_MODEL" ".model" "$CHAT_MODEL" "$body"

  # /v1/models entry for chat model is now state=ready.
  body="$TMP_DIR/models_after_load.json"; hdr="$TMP_DIR/models_after_load.hdr"
  status=$(http_call "$body" "$hdr" -H "$(auth_header)" "$GATEWAY_URL/v1/models")
  expect_status "GET /v1/models after load" 200 "$status" "$body"
  expect_jq    "  $CHAT_MODEL state=ready" \
    ".data[] | select(.id == \"$CHAT_MODEL\") | .state" "ready" "$body"

  # Second /load is idempotent.
  body="$TMP_DIR/load2.json"; hdr="$TMP_DIR/load2.hdr"
  status=$(http_call_slow "$body" "$hdr" -X POST -H "$(auth_header)" "$GATEWAY_URL/v1/models/$CHAT_MODEL/load")
  expect_status "POST /load is idempotent on already-loaded model" 200 "$status" "$body"

  # Unload.
  body="$TMP_DIR/unload.json"; hdr="$TMP_DIR/unload.hdr"
  status=$(http_call "$body" "$hdr" -X POST -H "$(auth_header)" "$GATEWAY_URL/v1/models/$CHAT_MODEL/unload")
  expect_status "POST /v1/models/$CHAT_MODEL/unload" 200 "$status" "$body"
  expect_jq    "  status=unloaded" ".status" "unloaded" "$body"

  # /readyz should now be 503.
  body="$TMP_DIR/readyz_after_unload.json"; hdr="$TMP_DIR/readyz_after_unload.hdr"
  status=$(http_call "$body" "$hdr" "$GATEWAY_URL/readyz")
  expect_status "GET /readyz after unload" 503 "$status" "$body"

  # Second /unload is a no-op (returns 200 with status=noop per the contract).
  body="$TMP_DIR/unload2.json"; hdr="$TMP_DIR/unload2.hdr"
  status=$(http_call "$body" "$hdr" -X POST -H "$(auth_header)" "$GATEWAY_URL/v1/models/$CHAT_MODEL/unload")
  expect_status "POST /unload when nothing is loaded" 200 "$status" "$body"
  expect_jq    "  status=noop" ".status" "noop" "$body"
}

test_chat_completion() {
  section "chat completion ($CHAT_MODEL) — non-streaming"
  local body hdr status
  body="$TMP_DIR/chat.json"; hdr="$TMP_DIR/chat.hdr"
  # Triggers implicit load if the slot is empty.
  status=$(http_call_slow "$body" "$hdr" -X POST -H "$(auth_header)" -H "Content-Type: application/json" \
    -d "{\"model\":\"$CHAT_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say only the word: pong\"}],\"max_tokens\":8,\"temperature\":0}" \
    "$GATEWAY_URL/v1/chat/completions")
  expect_status "POST /v1/chat/completions (non-streaming)" 200 "$status" "$body"
  expect_jq    "  object=chat.completion" ".object" "chat.completion" "$body"
  expect_jq_nonempty "  choices[0].message.content present" ".choices[0].message.content" "$body"
  expect_jq_nonempty "  usage.completion_tokens present" ".usage.completion_tokens" "$body"
}

test_chat_streaming() {
  section "chat completion ($CHAT_MODEL) — streaming SSE"
  local raw chunks last
  raw="$TMP_DIR/chat_stream.txt"
  # Buffered to a file so we can grep the SSE chunks. curl returns 0 once the
  # stream closes; if it errors we'll see an empty file.
  if ! curl --silent --show-error --max-time "$COLD_START_TIMEOUT" \
      -X POST -H "$(auth_header)" -H "Content-Type: application/json" \
      -d "{\"model\":\"$CHAT_MODEL\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Count: 1, 2, 3.\"}],\"max_tokens\":16,\"temperature\":0}" \
      "$GATEWAY_URL/v1/chat/completions" >"$raw" 2>/dev/null; then
    fail "streaming request" "curl exited non-zero"
    return
  fi
  chunks=$(grep -c '^data: ' "$raw" || true)
  if [ "$chunks" -ge 2 ]; then
    pass "  received $chunks SSE chunks"
  else
    fail "streaming chunk count" "expected ≥2 'data: ' lines, got $chunks"
  fi
  last=$(grep '^data: ' "$raw" | tail -n 1)
  if [ "$last" = "data: [DONE]" ]; then
    pass "  stream ends with data: [DONE]"
  else
    fail "streaming terminator" "last data line was: $last"
  fi
}

test_embeddings() {
  section "embeddings ($EMBED_MODEL)"
  local body hdr status
  body="$TMP_DIR/embed.json"; hdr="$TMP_DIR/embed.hdr"
  status=$(http_call_slow "$body" "$hdr" -X POST -H "$(auth_header)" -H "Content-Type: application/json" \
    -d "{\"model\":\"$EMBED_MODEL\",\"input\":\"hello world\"}" \
    "$GATEWAY_URL/v1/embeddings")
  expect_status "POST /v1/embeddings" 200 "$status" "$body"
  expect_jq    "  object=list" ".object" "list" "$body"
  expect_jq_nonempty "  data[0].embedding is a non-empty vector" ".data[0].embedding | length > 0 // empty" "$body"
}

test_implicit_swap() {
  section "implicit swap ($CHAT_MODEL → $EMBED_MODEL → $CHAT_MODEL)"
  local body hdr status

  # Step 1: ensure chat is loaded.
  body="$TMP_DIR/swap_load_chat.json"; hdr="$TMP_DIR/swap_load_chat.hdr"
  status=$(http_call_slow "$body" "$hdr" -X POST -H "$(auth_header)" "$GATEWAY_URL/v1/models/$CHAT_MODEL/load")
  expect_status "load $CHAT_MODEL" 200 "$status" "$body"

  # Step 2: an embed inference triggers the swap.
  body="$TMP_DIR/swap_embed.json"; hdr="$TMP_DIR/swap_embed.hdr"
  status=$(http_call_slow "$body" "$hdr" -X POST -H "$(auth_header)" -H "Content-Type: application/json" \
    -d "{\"model\":\"$EMBED_MODEL\",\"input\":\"swap me\"}" \
    "$GATEWAY_URL/v1/embeddings")
  expect_status "POST /v1/embeddings triggers swap" 200 "$status" "$body"

  body="$TMP_DIR/models_mid.json"; hdr="$TMP_DIR/models_mid.hdr"
  http_call "$body" "$hdr" -H "$(auth_header)" "$GATEWAY_URL/v1/models" >/dev/null
  expect_jq "  $EMBED_MODEL state=ready" \
    ".data[] | select(.id == \"$EMBED_MODEL\") | .state" "ready" "$body"
  expect_jq "  $CHAT_MODEL state=empty"  \
    ".data[] | select(.id == \"$CHAT_MODEL\")  | .state" "empty" "$body"

  # Step 3: chat inference swaps back.
  body="$TMP_DIR/swap_chat.json"; hdr="$TMP_DIR/swap_chat.hdr"
  status=$(http_call_slow "$body" "$hdr" -X POST -H "$(auth_header)" -H "Content-Type: application/json" \
    -d "{\"model\":\"$CHAT_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4,\"temperature\":0}" \
    "$GATEWAY_URL/v1/chat/completions")
  expect_status "POST /v1/chat/completions triggers swap back" 200 "$status" "$body"

  body="$TMP_DIR/models_end.json"; hdr="$TMP_DIR/models_end.hdr"
  http_call "$body" "$hdr" -H "$(auth_header)" "$GATEWAY_URL/v1/models" >/dev/null
  expect_jq "  $CHAT_MODEL state=ready (back)" \
    ".data[] | select(.id == \"$CHAT_MODEL\")  | .state" "ready" "$body"
  expect_jq "  $EMBED_MODEL state=empty (back)" \
    ".data[] | select(.id == \"$EMBED_MODEL\") | .state" "empty" "$body"

  # Cleanup so a subsequent run starts from a known state.
  http_call "$TMP_DIR/swap_cleanup.json" "$TMP_DIR/swap_cleanup.hdr" \
    -X POST -H "$(auth_header)" "$GATEWAY_URL/v1/models/$CHAT_MODEL/unload" >/dev/null
}

# ---------------------------------------------------------------------------
# Main

main() {
  preflight
  test_health_and_metrics
  test_auth
  test_catalog
  test_inference_errors

  if [ "$SKIP_INFERENCE" = "1" ]; then
    skip_note "inference tests" "SKIP_INFERENCE=1"
  else
    test_explicit_load_and_unload
    test_chat_completion
    test_chat_streaming
    test_embeddings
    if [ "$SKIP_SWAP" = "1" ]; then
      skip_note "swap test" "SKIP_SWAP=1"
    else
      test_implicit_swap
    fi
  fi

  section "summary"
  printf '  passed: %s%d%s\n' "$C_GREEN" "$PASSED" "$C_RESET"
  if [ "$FAILED" -gt 0 ]; then
    printf '  failed: %s%d%s\n' "$C_RED" "$FAILED" "$C_RESET"
    for name in "${FAILED_NAMES[@]}"; do
      printf '    - %s\n' "$name"
    done
    exit 1
  fi
  printf '  failed: 0\n'
}

main "$@"
