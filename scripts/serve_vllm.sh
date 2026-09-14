#!/usr/bin/env bash
# Launch the full serving stack: vLLM behind the FastAPI app with guardrails.
#
#   vLLM (:8001)          -- paged-attention inference over the MERGED weights
#     ^
#   FastAPI (:8000)       -- the canonical prompt template + NeMo guardrail policy
#     ^
#   client
#
# Why two processes rather than serving straight from vLLM:
#
#   * vLLM's OpenAI chat endpoint would apply the tokenizer's own chat template. The model was
#     fine-tuned and measured under exactly one template, owned by csbot.serve.template. The
#     FastAPI layer renders that template and calls vLLM's raw /v1/completions, so the bytes the
#     model sees at serve time are the bytes it saw in training.
#   * The guardrail layer has to run before generation, and must be able to answer without
#     generating at all.
#
# vLLM lives in its own virtualenv (.venv-vllm) because it pins a different torch build than the
# training environment. Installing it into the main venv would have downgraded torch underneath a
# finished, measured training run.

set -euo pipefail
cd "$(dirname "$0")/.."

MERGED_DIR="${MERGED_DIR:-artifacts/merged/final-polished}"
# Set ADAPTER_DIR to serve the BASE weights and the adapter side by side from one process, as
# "base" and "csbot". That is how the evaluation and the UI's comparison tab get a like-for-like
# base arm: two 4.8 GB servers do not fit on an 8 GB card, and loading them in sequence doubles
# the wall clock. Leave it unset to serve the merged weights alone, which is what ships.
ADAPTER_DIR="${ADAPTER_DIR:-}"
MODEL_KEY="${MODEL_KEY:-granite-3.3-2b}"
VLLM_PORT="${VLLM_PORT:-8001}"
API_PORT="${API_PORT:-8000}"
# 0.97 and 1280 are measured, not guessed: at 0.88/2048 the KV cache was 0.57 GiB and
# concurrency capped at 3.6x. The p99 prompt is 640 tokens plus 512 generated, so 2048 reserved
# cache per request that could never be used. At these values the cache is 1.86 GiB and
# concurrency 19x -- throughput went 387 -> 957 tok/s.
GPU_FRAC="${GPU_FRAC:-0.97}"
MAX_LEN="${MAX_LEN:-1280}"
LOG_DIR="${LOG_DIR:-artifacts/serve}"

mkdir -p "$LOG_DIR"

if [ -n "$ADAPTER_DIR" ]; then
  [ -d "$ADAPTER_DIR" ] || { echo "adapter not found at $ADAPTER_DIR" >&2; exit 1; }
  VLLM_MODEL="ibm-granite/granite-3.3-2b-instruct"
  VLLM_EXTRA=(--served-model-name base --enable-lora
              --lora-modules "csbot=${ADAPTER_DIR}" --max-lora-rank 16)
  echo "serving BASE + adapter ${ADAPTER_DIR} as 'base' and 'csbot'"
else
  if [ ! -d "$MERGED_DIR" ]; then
    echo "merged weights not found at $MERGED_DIR" >&2
    echo "run: .venv/bin/python scripts/merge_and_push.py --run artifacts/runs/final-polished --out $MERGED_DIR" >&2
    exit 1
  fi
  VLLM_MODEL="$MERGED_DIR"
  VLLM_EXTRA=(--served-model-name csbot)
  echo "serving merged weights from ${MERGED_DIR} as 'csbot'"
fi

echo "[1/3] starting vLLM on :$VLLM_PORT from $MERGED_DIR"
# --max-model-len is capped deliberately: the measured p99 prompt+response is well under 1k
# tokens, and a smaller KV cache leaves room on an 8 GB card that the default would consume.
nohup setsid .venv-vllm/bin/vllm serve "$VLLM_MODEL" \
  "${VLLM_EXTRA[@]}" \
  --port "$VLLM_PORT" \
  --gpu-memory-utilization "$GPU_FRAC" \
  --max-model-len "$MAX_LEN" \
  --dtype bfloat16 \
  --max-num-seqs "${MAX_SEQS:-32}" \
  > "$LOG_DIR/vllm.log" 2>&1 &
echo $! > "$LOG_DIR/vllm.pid"
echo "    pid $! -> $LOG_DIR/vllm.log"

echo "[2/3] waiting for vLLM to report a served model"
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
    echo "    ready after ${i}s: $(curl -fsS "http://127.0.0.1:$VLLM_PORT/v1/models" | head -c 200)"
    break
  fi
  if [ "$i" -eq 180 ]; then echo "    vLLM did not come up; see $LOG_DIR/vllm.log" >&2; exit 1; fi
  sleep 1
done

echo "[3/3] starting the API on :$API_PORT (template + guardrails, proxying to vLLM)"
CSBOT_ENGINE=proxy \
CSBOT_PROXY_URL="http://127.0.0.1:$VLLM_PORT" \
CSBOT_MODEL_KEY="$MODEL_KEY" \
CSBOT_MERGED_DIR="${MERGED_DIR}" \
CSBOT_PROXY_MODEL=csbot \
CSBOT_GUARDRAILS="${CSBOT_GUARDRAILS:-1}" \
PYTHONPATH=src \
nohup setsid .venv/bin/uvicorn csbot.serve.app:app --host 0.0.0.0 --port "$API_PORT" \
  > "$LOG_DIR/api.log" 2>&1 &
echo "    pid $! -> $LOG_DIR/api.log"

for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
    echo "    ready after ${i}s"
    exit 0
  fi
  sleep 1
done
echo "    API did not come up; see $LOG_DIR/api.log" >&2
exit 1
