#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

status_service() {
    local name="$1"
    local pid_file="run/${name}.pid"

    if [[ ! -f "$pid_file" ]]; then
        echo "$name: stopped"
        return
    fi

    local pid
    pid="$(cat "$pid_file")"

    if kill -0 "$pid" 2>/dev/null; then
        echo "$name: running (PID $pid)"
    else
        echo "$name: stale pid file (PID $pid not running)"
    fi
}

status_service "rf-detr"
status_service "gemma4"
