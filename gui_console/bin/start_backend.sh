#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

HOST="${GUI_BACKEND_HOST:-127.0.0.1}"
PORT="${GUI_BACKEND_PORT:-8001}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXEC="${PYTHON_BIN}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_EXEC="${CONDA_PREFIX}/bin/python"
else
  PYTHON_EXEC="python"
fi

cd "${REPO_ROOT}"
exec -a "SafeBenchHK-gui-backend" "${PYTHON_EXEC}" -m uvicorn gui_console.backend.main:app --host "${HOST}" --port "${PORT}" "$@"
