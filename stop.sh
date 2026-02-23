#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT_DIR/pm-agent-plus.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "pm-agent-plus is not running (no pid file)"
  exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "$PID" ]]; then
  rm -f "$PID_FILE"
  echo "pm-agent-plus is not running (empty pid file)"
  exit 0
fi

if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  sleep 1
  if kill -0 "$PID" 2>/dev/null; then
    kill -9 "$PID"
  fi
  echo "stopped pm-agent-plus (pid=$PID)"
else
  echo "pm-agent-plus process not found (pid=$PID)"
fi

rm -f "$PID_FILE"
