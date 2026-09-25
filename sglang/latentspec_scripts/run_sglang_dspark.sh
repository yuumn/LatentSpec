
#!/usr/bin/env bash
set -euo pipefail

TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-4B}
DRAFT_MODEL=${DRAFT_MODEL:-}

python -m sglang.launch_server \
  --model-path ${TARGET_MODEL} \
  --served-model-name qwen3-4b-dspark \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --trust-remote-code \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path ${DRAFT_MODEL}

