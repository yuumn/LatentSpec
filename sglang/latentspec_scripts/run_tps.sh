#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source ${SGLANG_DIR}/.sglangenv_07/bin/activate

PORT=${PORT:-30000}
TEMP=${TEMP:-1.0}
SUFFIX=${SUFFIX:-baseline}
MAX_TOKEN=${MAX_TOKEN:-512}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOWER_MODEL_NAME=${LOWER_MODEL_NAME:-qwen3-4b}

OUTPUT_DIR=${SCRIPT_DIR}/tps_results/${LOWER_MODEL_NAME}_${SUFFIX}
mkdir -p $OUTPUT_DIR

cd $SCRIPT_DIR

bash wait.sh $PORT 

python test_tps.py \
    --host 127.0.0.1 \
    --port $PORT \
    --dataset-dir ${SGLANG_DIR}/datasets \
    --datasets gsm8k math500 aime25 humaneval mbpp livecodebench mt-bench alpaca arena-hard-v2 \
    --concurrencies 1 2 4 8 16 32 \
    --requests-per-dataset 0 \
    --warmup-requests 32 \
    --max-tokens $MAX_TOKEN \
    --temperature $TEMP \
    --top-p 1.0 \
    --top-k -1 \
    --min-p 0.0 \
    --sampling-seed 980406 \
    --output-dir $OUTPUT_DIR \
    --output-filename ${TIMESTAMP}_temp-${TEMP}_max-${MAX_TOKEN}.jsonl \
    2>&1 | tee -a $OUTPUT_DIR/${TIMESTAMP}_temp-${TEMP}_max-${MAX_TOKEN}_print.log
