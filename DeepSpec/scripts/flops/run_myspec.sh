#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

python "${SCRIPT_DIR}/calculate_myspec_flops.py" \
    --context-lengths 512 1024 2048 4096 \
    --new-context-tokens 8
