#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$ROOT_DIR/src"
LOG_FILE="$ROOT_DIR/pm-agent-plus.log"
PID_FILE="$ROOT_DIR/pm-agent-plus.pid"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "pm-agent-plus is already running (pid=$OLD_PID)"
    exit 0
  fi
fi

cd "$SRC_DIR"
nohup bash -lc "exec -a pm-agent-plus env PYTHONPATH=. python3.11 main.py" >>"$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" >"$PID_FILE"
echo "started pm-agent-plus (pid=$NEW_PID), log=$LOG_FILE"
