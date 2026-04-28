#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SUPERVISOR_ROOT="${REPO_ROOT}/gui_console/runtime/supervisor"

SUPERVISOR_PID_FILE="${SUPERVISOR_ROOT}/supervisor.pid"
BACKEND_PID_FILE="${SUPERVISOR_ROOT}/backend.pid"
FRONTEND_PID_FILE="${SUPERVISOR_ROOT}/frontend.pid"
BACKEND_PORT="${GUI_BACKEND_PORT:-8001}"
FRONTEND_PORT="${GUI_FRONTEND_PORT:-5173}"

kill_if_alive() {
  local pid="${1:-}"
  local signal="${2:-TERM}"
  [[ -n "${pid}" ]] || return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill "-${signal}" "${pid}" 2>/dev/null || true
  fi
}

read_pid() {
  local path="${1:-}"
  [[ -f "${path}" ]] || return 0
  cat "${path}" 2>/dev/null || true
}

find_listener_pid() {
  local port="${1:-}"
  lsof -t -nP -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null | head -n 1 || true
}

SUPERVISOR_PID="$(read_pid "${SUPERVISOR_PID_FILE}")"
BACKEND_PID="$(read_pid "${BACKEND_PID_FILE}")"
FRONTEND_PID="$(read_pid "${FRONTEND_PID_FILE}")"

if [[ -z "${SUPERVISOR_PID}" && -z "${BACKEND_PID}" && -z "${FRONTEND_PID}" ]]; then
  echo "SafeBenchHK GUI console is not running."
  exit 0
fi

echo "[stop] stopping SafeBenchHK GUI console"

if [[ -n "${SUPERVISOR_PID}" ]]; then
  kill_if_alive "${SUPERVISOR_PID}" INT
  sleep 1
fi

kill_if_alive "${FRONTEND_PID}" TERM
kill_if_alive "${BACKEND_PID}" TERM
sleep 1
kill_if_alive "${FRONTEND_PID}" KILL
kill_if_alive "${BACKEND_PID}" KILL

EXTRA_FRONTEND_PID="$(find_listener_pid "${FRONTEND_PORT}")"
EXTRA_BACKEND_PID="$(find_listener_pid "${BACKEND_PORT}")"

if [[ -n "${EXTRA_FRONTEND_PID}" ]]; then
  echo "[stop] stopping frontend listener on port ${FRONTEND_PORT} (pid=${EXTRA_FRONTEND_PID})"
  kill_if_alive "${EXTRA_FRONTEND_PID}" TERM
  sleep 1
  kill_if_alive "${EXTRA_FRONTEND_PID}" KILL
fi

if [[ -n "${EXTRA_BACKEND_PID}" ]]; then
  echo "[stop] stopping backend listener on port ${BACKEND_PORT} (pid=${EXTRA_BACKEND_PID})"
  kill_if_alive "${EXTRA_BACKEND_PID}" TERM
  sleep 1
  kill_if_alive "${EXTRA_BACKEND_PID}" KILL
fi

rm -f "${SUPERVISOR_PID_FILE}" "${BACKEND_PID_FILE}" "${FRONTEND_PID_FILE}"
echo "[stop] done"
