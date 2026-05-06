#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

stop_service() {
    local name="$1"
    local pid_file="run/${name}.pid"

    if [[ ! -f "$pid_file" ]]; then
        echo "$name is not running (no pid file)"
        return
    fi

    local pid
    pid="$(cat "$pid_file")"

    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        echo "Stopped $name (PID $pid)"
    else
        echo "$name PID $pid not running"
    fi

    rm -f "$pid_file"
}

stop_service "rf-detr"
stop_service "gemma4"
