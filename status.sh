#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./status.sh [--verbose]

Options:
  -v, --verbose   Show recent log tail.
  -h, --help      Show this help message.
EOF
}

SHOW_LOG=0
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "-v" || "${1:-}" == "--verbose" ]]; then
  SHOW_LOG=1
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "Unknown option: $1"
  usage
  exit 1
fi

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
  DEFAULT_LOG_FILE="$ROOT_DIR/pm-agent-plus.log"
  DEFAULT_PID_FILE="$ROOT_DIR/pm-agent-plus.pid"
  DEFAULT_PROCESS_NAME="pm-agent-plus"
else
  DEFAULT_LOG_FILE="$ROOT_DIR/pm-agent-plus-${INSTANCE_SLUG}.log"
  DEFAULT_PID_FILE="$ROOT_DIR/pm-agent-plus-${INSTANCE_SLUG}.pid"
  DEFAULT_PROCESS_NAME="pm-agent-plus-${INSTANCE_SLUG}"
fi
LOG_FILE="${PM_AGENT_LOG_FILE:-$DEFAULT_LOG_FILE}"
PID_FILE="${PM_AGENT_PID_FILE:-$DEFAULT_PID_FILE}"
PROCESS_NAME="${PM_AGENT_PROCESS_NAME:-$DEFAULT_PROCESS_NAME}"

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    echo "$PROCESS_NAME is running (pid=$PID)"
    ps -p "$PID" -o pid=,etime=,pcpu=,pmem=,cmd= || true
    echo "log: $LOG_FILE"
    if [[ "$SHOW_LOG" -eq 1 ]]; then
      echo "--- last 40 log lines ---"
      if [[ -f "$LOG_FILE" ]]; then
        tail -n 40 "$LOG_FILE"
      else
        echo "(log file not found yet)"
      fi
    fi
    exit 0
  fi
fi

echo "$PROCESS_NAME is not running"
echo "log: $LOG_FILE"
if [[ "$SHOW_LOG" -eq 1 ]]; then
  echo "--- last 40 log lines ---"
  if [[ -f "$LOG_FILE" ]]; then
    tail -n 40 "$LOG_FILE"
  else
    echo "(log file not found yet)"
  fi
fi
