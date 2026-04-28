#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${REPO_ROOT}/gui_console/runtime"
SUPERVISOR_ROOT="${RUNTIME_ROOT}/supervisor"

BACKEND_LOG="${SUPERVISOR_ROOT}/backend.log"
FRONTEND_LOG="${SUPERVISOR_ROOT}/frontend.log"
STATUS_FILE="${SUPERVISOR_ROOT}/status.json"
SUPERVISOR_PID_FILE="${SUPERVISOR_ROOT}/supervisor.pid"
BACKEND_PID_FILE="${SUPERVISOR_ROOT}/backend.pid"
FRONTEND_PID_FILE="${SUPERVISOR_ROOT}/frontend.pid"

BACKEND_HOST="${GUI_BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${GUI_BACKEND_PORT:-8001}"
FRONTEND_HOST="${GUI_FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${GUI_FRONTEND_PORT:-5173}"
FRONTEND_MODE="${GUI_CONSOLE_FRONTEND_MODE:-preview}"
MONITOR_INTERVAL="${GUI_MONITOR_INTERVAL:-3}"

BACKEND_PID=""
FRONTEND_PID=""
SHUTTING_DOWN=0

mkdir -p "${SUPERVISOR_ROOT}"

json_escape() {
  python3 - "$1" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1]))
PY
}

process_alive() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

find_listener_pid() {
  local port="${1:-}"
  lsof -t -nP -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null | head -n 1 || true
}

port_listener_info() {
  local port="${1:-}"
  lsof -nP -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null | tail -n +2 | head -n 1 || true
}

ensure_port_available() {
  local port="${1:-}"
  local name="${2:-service}"
  local info
  info="$(port_listener_info "${port}")"
  if [[ -z "${info}" ]]; then
    return 0
  fi

  echo "[supervisor] ${name} port ${port} is already in use"
  echo "[supervisor] listener: ${info}"
  echo "[supervisor] stop the existing process first, or run ./gui_console/bin/stop_console.sh"
  exit 1
}

resource_snapshot() {
  local pid="${1:-}"
  if ! process_alive "${pid}"; then
    printf '{"pid": null, "cpu_percent": null, "rss_mb": null, "elapsed": null, "command": null}'
    return
  fi

  local raw
  raw="$(ps -p "${pid}" -o pid=,%cpu=,rss=,etime=,command= | head -n 1 || true)"
  if [[ -z "${raw}" ]]; then
    printf '{"pid": null, "cpu_percent": null, "rss_mb": null, "elapsed": null, "command": null}'
    return
  fi

  python3 - "${raw}" <<'PY'
import json
import sys

line = sys.argv[1].rstrip()
parts = line.split(None, 4)
if len(parts) < 5:
    print(json.dumps({
        "pid": None,
        "cpu_percent": None,
        "rss_mb": None,
        "elapsed": None,
        "command": None,
    }))
    raise SystemExit(0)

pid, cpu, rss_kb, elapsed, command = parts
print(json.dumps({
    "pid": int(pid),
    "cpu_percent": float(cpu),
    "rss_mb": round(int(rss_kb) / 1024.0, 1),
    "elapsed": elapsed,
    "command": command,
}))
PY
}

write_status() {
  local backend_state="stopped"
  local frontend_state="stopped"
  local backend_log_json
  local frontend_log_json
  if process_alive "${BACKEND_PID}"; then
    backend_state="running"
  fi
  if process_alive "${FRONTEND_PID}"; then
    frontend_state="running"
  fi
  backend_log_json="$(json_escape "${BACKEND_LOG}")"
  frontend_log_json="$(json_escape "${FRONTEND_LOG}")"

  cat > "${STATUS_FILE}" <<EOF
{
  "supervisor_pid": $$,
  "backend": {
    "state": "${backend_state}",
    "log_path": ${backend_log_json},
    "metrics": $(resource_snapshot "${BACKEND_PID}")
  },
  "frontend": {
    "state": "${frontend_state}",
    "log_path": ${frontend_log_json},
    "metrics": $(resource_snapshot "${FRONTEND_PID}")
  },
  "urls": {
    "backend": "http://${BACKEND_HOST}:${BACKEND_PORT}",
    "frontend": "http://${FRONTEND_HOST}:${FRONTEND_PORT}"
  }
}
EOF
}

terminate_process() {
  local pid="${1:-}"
  local name="${2:-process}"
  if ! process_alive "${pid}"; then
    return
  fi

  echo "[supervisor] stopping ${name} (pid=${pid})"
  kill -TERM "${pid}" 2>/dev/null || true

  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if ! process_alive "${pid}"; then
      return
    fi
    sleep 0.5
  done

  if process_alive "${pid}"; then
    echo "[supervisor] force killing ${name} (pid=${pid})"
    kill -KILL "${pid}" 2>/dev/null || true
  fi
}

terminate_listener_on_port() {
  local port="${1:-}"
  local name="${2:-service}"
  local pid
  pid="$(find_listener_pid "${port}")"
  if [[ -z "${pid}" ]]; then
    return
  fi

  echo "[supervisor] stopping ${name} listener on port ${port} (pid=${pid})"
  kill -TERM "${pid}" 2>/dev/null || true
  for _ in 1 2 3 4 5 6; do
    if ! process_alive "${pid}"; then
      return
    fi
    sleep 0.5
  done
  if process_alive "${pid}"; then
    echo "[supervisor] force killing ${name} listener on port ${port} (pid=${pid})"
    kill -KILL "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  local reason="${1:-shutdown}"
  if [[ "${SHUTTING_DOWN}" -eq 1 ]]; then
    return
  fi
  SHUTTING_DOWN=1

  echo "[supervisor] ${reason}"
  terminate_process "${FRONTEND_PID}" "frontend"
  terminate_process "${BACKEND_PID}" "backend"
  terminate_listener_on_port "${FRONTEND_PORT}" "frontend"
  terminate_listener_on_port "${BACKEND_PORT}" "backend"

  rm -f "${BACKEND_PID_FILE}" "${FRONTEND_PID_FILE}" "${SUPERVISOR_PID_FILE}"
  write_status
}

handle_signal() {
  cleanup "received Ctrl+C / termination signal"
  exit 130
}

trap handle_signal INT TERM
trap 'cleanup "supervisor exiting"' EXIT

if [[ -f "${SUPERVISOR_PID_FILE}" ]]; then
  OLD_PID="$(cat "${SUPERVISOR_PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "SafeBenchHK GUI supervisor is already running (pid=${OLD_PID})." >&2
    echo "Use ./gui_console/bin/stop_console.sh first if you want to restart it." >&2
    exit 1
  fi
fi

printf '%s\n' "$$" > "${SUPERVISOR_PID_FILE}"

ensure_port_available "${BACKEND_PORT}" "backend"
ensure_port_available "${FRONTEND_PORT}" "frontend"

echo "[supervisor] starting backend on http://${BACKEND_HOST}:${BACKEND_PORT}"
GUI_BACKEND_HOST="${BACKEND_HOST}" \
GUI_BACKEND_PORT="${BACKEND_PORT}" \
"${REPO_ROOT}/gui_console/bin/start_backend.sh" >"${BACKEND_LOG}" 2>&1 &
BACKEND_PID=$!
printf '%s\n' "${BACKEND_PID}" > "${BACKEND_PID_FILE}"

echo "[supervisor] starting frontend on http://${FRONTEND_HOST}:${FRONTEND_PORT} (mode=${FRONTEND_MODE})"
GUI_FRONTEND_HOST="${FRONTEND_HOST}" \
GUI_FRONTEND_PORT="${FRONTEND_PORT}" \
GUI_FRONTEND_MODE="${FRONTEND_MODE}" \
"${REPO_ROOT}/gui_console/bin/start_frontend.sh" >"${FRONTEND_LOG}" 2>&1 &
FRONTEND_PID=$!
printf '%s\n' "${FRONTEND_PID}" > "${FRONTEND_PID_FILE}"

echo "[supervisor] backend log: ${BACKEND_LOG}"
echo "[supervisor] frontend log: ${FRONTEND_LOG}"
echo "[supervisor] press Ctrl+C at any time to stop backend and frontend together"

while true; do
  if ! process_alive "${BACKEND_PID}"; then
    echo "[supervisor] backend exited unexpectedly"
    echo "[supervisor] last backend log lines:"
    tail -n 20 "${BACKEND_LOG}" || true
    exit 1
  fi

  if ! process_alive "${FRONTEND_PID}"; then
    echo "[supervisor] frontend exited unexpectedly"
    echo "[supervisor] last frontend log lines:"
    tail -n 20 "${FRONTEND_LOG}" || true
    exit 1
  fi

  write_status

  echo "[supervisor] backend $(resource_snapshot "${BACKEND_PID}")"
  echo "[supervisor] frontend $(resource_snapshot "${FRONTEND_PID}")"

  sleep "${MONITOR_INTERVAL}"
done
