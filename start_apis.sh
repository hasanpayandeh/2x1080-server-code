#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p logs run

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python executable '$PYTHON_BIN' not found."
    echo "Set PYTHON_BIN explicitly, for example: PYTHON_BIN=/usr/bin/python3 ./start_apis.sh"
    exit 1
fi

start_service() {
    local name="$1"
    local port="$2"
    local script="$3"
    local pid_file="run/${name}.pid"
    local log_file="logs/${name}.log"

    if [[ ! -f "$script" ]]; then
        echo "Error: script '$script' not found for service '$name'."
        return 1
    fi

    if [[ -f "$pid_file" ]]; then
        local old_pid
        old_pid="$(cat "$pid_file")"
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "$name already running with PID $old_pid"
            return
        else
            rm -f "$pid_file"
        fi
    fi

    nohup env HOST=0.0.0.0 PORT="$port" "$PYTHON_BIN" "$script" > "$log_file" 2>&1 &
    local new_pid=$!

    # Give the process a moment to fail fast (missing deps, model load errors, etc).
    sleep 1

    if ! kill -0 "$new_pid" 2>/dev/null; then
        echo "Failed to start $name on port $port."
        echo "Last log lines from $log_file:"
        tail -n 40 "$log_file" 2>/dev/null || true
        rm -f "$pid_file"
        return 1
    fi

    echo "$new_pid" > "$pid_file"
    echo "Started $name on port $port (PID $new_pid)"
}

start_service "rf-detr" "8000" "rf-detr.py"
start_service "gemma4" "8001" "gemma4.py"

echo ""
echo "Health checks:"
echo "  RF-DETR: http://<server-ip>:8000/health"
echo "  Gemma4:  http://<server-ip>:8001/health"
