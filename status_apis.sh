#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SHOW_LOGS="${SHOW_LOGS:-0}"

show_logs_if_requested() {
    local name="$1"
    local log_file="logs/${name}.log"

    if [[ "$SHOW_LOGS" != "1" ]]; then
        return
    fi

    if [[ -f "$log_file" ]]; then
        echo "--- recent logs for $name ---"
        tail -n 30 "$log_file" || true
    else
        echo "--- no log file for $name ---"
    fi
}

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
        echo "$name: stopped (cleaned stale pid $pid)"
        rm -f "$pid_file"
        show_logs_if_requested "$name"
    fi
}

status_service "rf-detr"
status_service "gemma4"
