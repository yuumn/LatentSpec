#!/usr/bin/env bash
set -euo pipefail

TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-4B}
DRAFT_MODEL=${DRAFT_MODEL:-}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3-4b-latentspec}

python -m sglang.launch_server \
  --model-path ${TARGET_MODEL} \
  --served-model-name ${SERVED_MODEL_NAME} \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --trust-remote-code \
  --speculative-algorithm LATENTSPEC \
  --speculative-draft-model-path ${DRAFT_MODEL} \
  --speculative-num-draft-tokens 8


