#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p logs run

start_service() {
    local name="$1"
    local port="$2"
    local script="$3"
    local pid_file="run/${name}.pid"
    local log_file="logs/${name}.log"

    if [[ -f "$pid_file" ]]; then
        local old_pid
        old_pid="$(cat "$pid_file")"
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "$name already running with PID $old_pid"
            return
        fi
    fi

    nohup env HOST=0.0.0.0 PORT="$port" python "$script" > "$log_file" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$pid_file"
    echo "Started $name on port $port (PID $new_pid)"
}

start_service "rf-detr" "8000" "rf-detr.py"
start_service "gemma4" "8001" "gemma4.py"

echo ""
echo "Health checks:"
echo "  RF-DETR: http://<server-ip>:8000/health"
echo "  Gemma4:  http://<server-ip>:8001/health"
