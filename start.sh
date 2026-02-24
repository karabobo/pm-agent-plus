#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./start.sh [--foreground]

Options:
  -f, --foreground   Run in foreground and stream logs to terminal.
  -h, --help         Show this help message.
EOF
}

MODE="background"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "-f" || "${1:-}" == "--foreground" ]]; then
  MODE="foreground"
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "Unknown option: $1"
  usage
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$ROOT_DIR/src"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ENV_FILE="$ROOT_DIR/.env"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

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
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "pm-agent-plus is already running (pid=$OLD_PID)"
    exit 0
  fi
fi

if [[ "$MODE" == "foreground" ]]; then
  # Foreground mode should not leave stale pid files from previous background runs.
  rm -f "$PID_FILE"
  cd "$SRC_DIR"
  echo "starting pm-agent-plus in foreground, log=$LOG_FILE"
  set +e
  PYTHONPATH=. "$PYTHON_BIN" main.py 2>&1 | tee -a "$LOG_FILE"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e
  exit "$EXIT_CODE"
fi

cd "$SRC_DIR"
nohup bash -lc "exec -a \"$PROCESS_NAME\" env PYTHONPATH=. \"$PYTHON_BIN\" main.py" >>"$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" >"$PID_FILE"
echo "started $PROCESS_NAME (pid=$NEW_PID), log=$LOG_FILE"
