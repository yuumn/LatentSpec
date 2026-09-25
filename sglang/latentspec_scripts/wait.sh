
PORT=${1:-8080}
TIMEOUT=${2:-600}
wait_for_health() {
    local port=$1
    local timeout=${2:-600}   # 默认最多等 600 秒
    local elapsed=0
    echo "等待 server http://127.0.0.1:$port 就绪..."
    while true; do
        if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            echo "  ✓ http://127.0.0.1:$port 已就绪 (${elapsed}s)"
            return 0
        fi
        if [ $elapsed -ge $timeout ]; then
            echo "  ✗ http://127.0.0.1:$port 超时 (${timeout}s)，退出"
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo "  等待中... (${elapsed} / ${timeout}s)"
    done
} 
wait_for_health $PORT $TIMEOUT
