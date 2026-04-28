"""Static paths and constants for the GUI console."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = REPO_ROOT / "gui_console"
BACKEND_ROOT = GUI_ROOT / "backend"
FRONTEND_ROOT = GUI_ROOT / "frontend"
RUNTIME_ROOT = GUI_ROOT / "runtime"
RUN_ROOT = RUNTIME_ROOT / "run_root"
EXPERIMENTS_ROOT = RUNTIME_ROOT / "experiments"
JOBS_ROOT = RUNTIME_ROOT / "jobs"
STATE_FILE = RUNTIME_ROOT / "app_state.json"

BUILDER_ROOT = REPO_ROOT / "tools" / "CarlaScenariosBuilder"
BUILDER_MAP_WAYPOINTS_DIR = BUILDER_ROOT / "map_waypoints"
BUILDER_SCENARIO_ORIGIN_DIR = BUILDER_ROOT / "scenario_origin"
BUILDER_SCENARIO_DATA_DIR = BUILDER_ROOT / "scenario_data"

AGENT_CONFIG_DIR = REPO_ROOT / "safebench" / "agent" / "config"
SCENARIO_CONFIG_DIR = REPO_ROOT / "safebench" / "scenario" / "config"

RUN_ROOT_AGENT_DIR = RUN_ROOT / "safebench" / "agent"
RUN_ROOT_AGENT_CONFIG_DIR = RUN_ROOT_AGENT_DIR / "config"
RUN_ROOT_AGENT_MODEL_LINK = RUN_ROOT_AGENT_DIR / "model_ckpt"

RUN_ROOT_SCENARIO_DIR = RUN_ROOT / "safebench" / "scenario"
RUN_ROOT_SCENARIO_CONFIG_DIR = RUN_ROOT_SCENARIO_DIR / "config"
RUN_ROOT_SCENARIO_DATA_DIR = RUN_ROOT_SCENARIO_DIR / "scenario_data"
RUN_ROOT_SCENARIO_MODEL_LINK = RUN_ROOT_SCENARIO_DATA_DIR / "model_ckpt"

SCENARIO_TEMPLATE_FILES = {
    "standard": "standard.yaml",
    "lc": "LC.yaml",
}

STANDARD_SCENARIOS = [
    {"id": 1, "name": "DynamicObjectCrossing"},
    {"id": 2, "name": "VehicleTurningRoute"},
    {"id": 3, "name": "OtherLeadingVehicle"},
    {"id": 4, "name": "ManeuverOppositeDirection"},
    {"id": 5, "name": "OppositeVehicleRunningRedLight"},
    {"id": 6, "name": "SignalizedJunctionLeftTurn"},
    {"id": 7, "name": "SignalizedJunctionRightTurn"},
    {"id": 8, "name": "NoSignalJunctionCrossingRoute"},
]

DEFAULT_ALLOWED_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]


def _conda_executable_candidates() -> List[str]:
    candidates = []
    for candidate in [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        str(Path.home() / "miniconda3" / "condabin" / "conda"),
    ]:
        if candidate and Path(candidate).exists() and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _infer_env_name_from_python(python_exec: Path) -> str:
    parts = python_exec.resolve().parts
    if "envs" in parts:
        index = parts.index("envs")
        if index + 1 < len(parts):
            return parts[index + 1]
    conda_name = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_name:
        return conda_name
    base_dir = python_exec.resolve().parent.parent.name
    return base_dir or python_exec.name


def is_valid_python_exec(value: str) -> bool:
    if not value:
        return False
    path = Path(value).expanduser()
    return path.exists() and path.is_file() and os.access(path, os.X_OK)


def python_env_options() -> List[Dict[str, Any]]:
    options: List[Dict[str, Any]] = []
    seen = set()

    def add_option(
        python_exec: Path,
        *,
        name: str,
        source: str,
        active: bool = False,
        recommended: bool = False,
    ) -> None:
        resolved = str(python_exec.resolve())
        if resolved in seen or not is_valid_python_exec(resolved):
            return
        label = name
        if recommended:
            label += " (Recommended)"
        elif active:
            label += " (Active)"
        options.append(
            {
                "id": resolved,
                "name": name,
                "label": label,
                "python_exec": resolved,
                "source": source,
                "active": active,
                "recommended": recommended,
            }
        )
        seen.add(resolved)

    current_python = Path(sys.executable).resolve()
    add_option(
        current_python,
        name=_infer_env_name_from_python(current_python),
        source="backend",
        active=True,
        recommended=True,
    )

    for conda_exe in _conda_executable_candidates():
        try:
            completed = subprocess.run(
                [conda_exe, "env", "list", "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            continue
        if completed.returncode != 0:
            continue
        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError:
            continue
        details = payload.get("envs_details", {})
        for env_path in payload.get("envs", []):
            env_root = Path(env_path)
            python_exec = env_root / ("python.exe" if os.name == "nt" else "bin/python")
            detail = details.get(env_path, {})
            name = detail.get("name") or ("base" if env_root.name == "miniconda3" else env_root.name)
            active = bool(detail.get("active")) or python_exec.resolve() == current_python
            add_option(python_exec, name=name, source="conda", active=active)
        break

    add_option(Path("/usr/bin/python3"), name="system-python3", source="system")
    return options


def default_python_suggestions() -> List[str]:
    """Return local Python executables worth suggesting in the UI."""
    return [item["python_exec"] for item in python_env_options()]


def default_python_exec() -> str:
    options = python_env_options()
    return options[0]["python_exec"] if options else "python3"
