#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRONTEND_ROOT="${REPO_ROOT}/gui_console/frontend"
VITE_BIN="${FRONTEND_ROOT}/node_modules/.bin/vite"

HOST="${GUI_FRONTEND_HOST:-127.0.0.1}"
PORT="${GUI_FRONTEND_PORT:-5173}"
MODE="${GUI_FRONTEND_MODE:-preview}"

if [[ ! -x "${VITE_BIN}" ]]; then
  echo "Missing frontend dependencies. Run: cd ${FRONTEND_ROOT} && npm install" >&2
  exit 1
fi

cd "${FRONTEND_ROOT}"

if [[ "${MODE}" == "dev" ]]; then
  echo "[frontend] starting in dev mode with file watcher"
  exec -a "SafeBenchHK-gui-frontend" "${VITE_BIN}" --host "${HOST}" --port "${PORT}" --strictPort "$@"
fi

echo "[frontend] starting in stable preview mode (no file watcher)"
"${VITE_BIN}" build
exec -a "SafeBenchHK-gui-frontend" "${VITE_BIN}" preview --host "${HOST}" --port "${PORT}" --strictPort "$@"
