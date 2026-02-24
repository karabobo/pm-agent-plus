#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

INSTANCE_ID="${PM_AGENT_INSTANCE_ID:-default}"
INSTANCE_SLUG="$(printf '%s' "$INSTANCE_ID" | tr -cd '[:alnum:]_.-')"
if [[ -z "$INSTANCE_SLUG" ]]; then
  INSTANCE_SLUG="default"
fi
if [[ "$INSTANCE_SLUG" == "default" ]]; then
  DEFAULT_PID_FILE="$ROOT_DIR/pm-agent-plus.pid"
  DEFAULT_PROCESS_NAME="pm-agent-plus"
else
  DEFAULT_PID_FILE="$ROOT_DIR/pm-agent-plus-${INSTANCE_SLUG}.pid"
  DEFAULT_PROCESS_NAME="pm-agent-plus-${INSTANCE_SLUG}"
fi
PID_FILE="${PM_AGENT_PID_FILE:-$DEFAULT_PID_FILE}"
PROCESS_NAME="${PM_AGENT_PROCESS_NAME:-$DEFAULT_PROCESS_NAME}"

if [[ ! -f "$PID_FILE" ]]; then
  echo "$PROCESS_NAME is not running (no pid file)"
  exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "$PID" ]]; then
  rm -f "$PID_FILE"
  echo "$PROCESS_NAME is not running (empty pid file)"
  exit 0
fi

if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  sleep 1
  if kill -0 "$PID" 2>/dev/null; then
    kill -9 "$PID"
  fi
  echo "stopped $PROCESS_NAME (pid=$PID)"
else
  echo "$PROCESS_NAME process not found (pid=$PID)"
fi

rm -f "$PID_FILE"
