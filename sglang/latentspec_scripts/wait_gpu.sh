#!/usr/bin/env bash

# Wait until no compute process is using any NVIDIA GPU.
# Usage: source wait_gpu.sh && wait_for_gpu_idle
wait_for_gpu_idle() {
    local check_interval=60
    local gpu_pids

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "Error: nvidia-smi was not found." >&2
        return 1
    fi

    while true; do
        if ! gpu_pids=$(nvidia-smi --query-compute-apps=pid \
            --format=csv,noheader,nounits 2>/dev/null); then
            echo "[$(date '+%F %T')] Failed to query GPU processes; retrying in ${check_interval} seconds..." >&2
            sleep "${check_interval}"
            continue
        fi

        # Ignore blank output and remove duplicate PIDs reported by multiple GPUs.
        gpu_pids=$(printf '%s\n' "${gpu_pids}" | awk 'NF' | sort -u)
        if [[ -z "${gpu_pids}" ]]; then
            echo "[$(date '+%F %T')] No GPU process is running."
            return 0
        fi

        echo "[$(date '+%F %T')] GPU process detected (PID: $(printf '%s' "${gpu_pids}" | paste -sd, -)); checking again in ${check_interval} seconds..."
        sleep "${check_interval}"
    done
}

wait_for_gpu_idle