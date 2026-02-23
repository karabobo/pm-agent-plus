#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT_DIR/pm-agent-plus.pid"
LOG_FILE="$ROOT_DIR/pm-agent-plus.log"

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    echo "pm-agent-plus is running (pid=$PID)"
    echo "log: $LOG_FILE"
    exit 0
  fi
fi

echo "pm-agent-plus is not running"
echo "log: $LOG_FILE"
