"""Filesystem, configuration, and experiment helpers for the GUI."""

from __future__ import annotations

from collections import Counter, deque
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading
import time
from typing import Any, Dict, Iterable, List, Optional
import uuid

import yaml

from . import settings

# Prevents multiple concurrent probe_current_world_map subprocesses from
# piling up and overwhelming the CARLA server with simultaneous connections.
_catalog_lock = threading.Lock()
_catalog_cache: Dict[str, Any] = {}
_catalog_cache_ts: float = 0.0
_CATALOG_TTL_SECONDS = 20.0


STOP_REASON_PATTERN = re.compile(r"Scenario stops due to (.+)")
SHELL_EXPORT_PATTERN = re.compile(r"^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.+)$")
CARLA_ROOT_KEYS = ["CARLA_UE4_ROOT", "CARLA_ROOT", "CARLA_0916_ROOT", "OP_CARLA_ROOT"]
CARLA_MAP_IGNORE_PARTS = {"Sublevels", "TestMaps", "BaseLargeMap", "Weathers"}
CARLA_MAP_IGNORE_STEMS = {"OpenDriveMap", "LargeMap", "DigitalTwinsTemplate", "BaseMap", "EmptyMap"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(str(link_path)):
        if link_path.is_symlink() and os.readlink(str(link_path)) == str(target_path):
            return
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(str(link_path))
        else:
            link_path.unlink()
    os.symlink(str(target_path), str(link_path))


def ensure_runtime_layout() -> Dict[str, str]:
    settings.RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    settings.EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)
    settings.JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    settings.RUN_ROOT_AGENT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    settings.RUN_ROOT_SCENARIO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    settings.RUN_ROOT_SCENARIO_DATA_DIR.mkdir(parents=True, exist_ok=True)

    ensure_symlink(settings.RUN_ROOT / "scripts", settings.REPO_ROOT / "scripts")
    ensure_symlink(
        settings.RUN_ROOT_AGENT_MODEL_LINK,
        settings.REPO_ROOT / "safebench" / "agent" / "model_ckpt",
    )
    ensure_symlink(
        settings.RUN_ROOT_SCENARIO_MODEL_LINK,
        settings.REPO_ROOT / "safebench" / "scenario" / "scenario_data" / "model_ckpt",
    )

    return {
        "repo_root": str(settings.REPO_ROOT),
        "gui_root": str(settings.GUI_ROOT),
        "runtime_root": str(settings.RUNTIME_ROOT),
        "run_root": str(settings.RUN_ROOT),
        "experiments_root": str(settings.EXPERIMENTS_ROOT),
        "jobs_root": str(settings.JOBS_ROOT),
    }


def ensure_map_link(map_name: str) -> Dict[str, Any]:
    builder_map_dir = settings.BUILDER_SCENARIO_DATA_DIR / map_name
    builder_map_dir.mkdir(parents=True, exist_ok=True)
    run_link = settings.RUN_ROOT_SCENARIO_DATA_DIR / map_name
    ensure_symlink(run_link, builder_map_dir)
    return {
        "builder_target": str(builder_map_dir),
        "run_link": str(run_link),
        "run_link_target": str(builder_map_dir),
        "run_link_exists": os.path.lexists(str(run_link)),
    }


def _prepend_env_path(env: Dict[str, str], key: str, value: str) -> None:
    current = env.get(key, "")
    parts = [item for item in current.split(os.pathsep) if item] if current else []
    if value in parts:
        parts = [item for item in parts if item != value]
    env[key] = os.pathsep.join([value, *parts]) if parts else value


def build_python_env(repo_root: Path, python_exec: Optional[str] = None) -> Dict[str, str]:
    env = os.environ.copy()
    repo_root_str = str(repo_root)
    current = env.get("PYTHONPATH")
    if current:
        paths = current.split(os.pathsep)
        if repo_root_str not in paths:
            env["PYTHONPATH"] = os.pathsep.join([repo_root_str, current])
    else:
        env["PYTHONPATH"] = repo_root_str
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.pop("PYTHONHOME", None)

    for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV"):
        env.pop(key, None)

    if python_exec and settings.is_valid_python_exec(python_exec):
        python_path = Path(python_exec).expanduser().resolve()
        bin_dir = python_path.parent
        _prepend_env_path(env, "PATH", str(bin_dir))

        prefix = bin_dir.parent if bin_dir.name in {"bin", "Scripts"} else None
        if prefix:
            env["VIRTUAL_ENV"] = str(prefix)
            if (prefix / "conda-meta").exists() or "envs" in python_path.parts:
                env["CONDA_PREFIX"] = str(prefix)
                env["CONDA_DEFAULT_ENV"] = prefix.name
            lib_dir = prefix / "lib"
            if lib_dir.exists():
                _prepend_env_path(env, "LD_LIBRARY_PATH", str(lib_dir))
    return env


def resolve_runtime_python_exec(
    preferred_python_exec: Optional[str],
    fallback_python_exec: Optional[str] = None,
) -> Optional[str]:
    for candidate in [preferred_python_exec, fallback_python_exec]:
        if candidate and settings.is_valid_python_exec(candidate):
            return str(Path(candidate).expanduser().resolve())
    return None


def rewrite_command_python(command: Iterable[Any], python_exec: str) -> List[str]:
    rewritten = [str(item) for item in list(command or [])]
    if rewritten:
        rewritten[0] = python_exec
    return rewritten


def persist_manifest_python_exec(manifest: Dict[str, Any], python_exec: str) -> Dict[str, Any]:
    updated = dict(manifest)
    updated["python_exec"] = python_exec
    updated["command"] = rewrite_command_python(updated.get("command", []), python_exec)
    write_json(Path(updated["snapshot_dir"]) / "manifest.json", updated)
    return updated


def build_rerun_request(
    manifest: Dict[str, Any],
    *,
    render: Optional[bool] = None,
    save_video: Optional[bool] = None,
    exp_name: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "agent_name": manifest["agent_name"],
        "scenario_template": manifest["scenario_template"],
        "map_name": manifest["map"],
        "scenario_id": manifest["scenario_id"],
        "exp_name": exp_name or manifest["exp_name"],
        "seed": manifest["seed"],
        "render": manifest["render"] if render is None else render,
        "save_video": manifest["save_video"] if save_video is None else save_video,
        "route_id": manifest.get("route_id"),
        "port": manifest["port"],
        "tm_port": manifest["tm_port"],
        "mode": manifest.get("mode", "eval"),
    }


def check_tcp_port(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def run_json_probe(
    command: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": "{name}: {message}".format(name=type(exc).__name__, message=str(exc)),
        }
    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        return {
            "ok": False,
            "error": completed.stderr.strip() or stdout or "probe failed",
        }
    try:
        payload = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": "probe did not return valid JSON",
            "stdout": stdout,
        }
    return payload


def probe_safebench_import(repo_root: Path, python_exec: str) -> Dict[str, Any]:
    code = """
import json
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1]).resolve()
result = {"ok": False}
try:
    import safebench
    module_path = pathlib.Path(safebench.__file__).resolve()
    result = {
        "ok": str(module_path).startswith(str(repo_root)),
        "module_path": str(module_path),
    }
except Exception as exc:
    result = {
        "ok": False,
        "error": "{name}: {msg}".format(name=type(exc).__name__, msg=str(exc)),
    }
print(json.dumps(result))
""".strip()
    return run_json_probe(
        [python_exec, "-c", code, str(repo_root)],
        cwd=repo_root,
        env=build_python_env(repo_root, python_exec),
    )


def discover_maps() -> List[str]:
    names = set()
    for root in [
        settings.BUILDER_MAP_WAYPOINTS_DIR,
        settings.BUILDER_SCENARIO_ORIGIN_DIR,
        settings.BUILDER_SCENARIO_DATA_DIR,
    ]:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir():
                names.add(child.name)
    return sorted(names)


def _load_shell_exports(shell_rc: Path) -> Dict[str, str]:
    exports: Dict[str, str] = {}
    if not shell_rc.exists():
        return exports
    for line in shell_rc.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = SHELL_EXPORT_PATTERN.match(stripped)
        if not match:
            continue
        key, raw_value = match.groups()
        value = raw_value.strip().strip('"').strip("'")
        exports[key] = os.path.expandvars(os.path.expanduser(value))
    return exports


def _candidate_carla_content_roots(carla_root: Path) -> List[Path]:
    candidates = [
        carla_root / "Unreal" / "CarlaUE4" / "Content",
        carla_root / "CarlaUE4" / "Content",
        carla_root / "Content",
    ]
    return [path for path in candidates if path.exists()]


def _collect_carla_root_candidates() -> List[str]:
    """Gather CARLA root candidates from environment, shell configs, and common paths."""
    candidates: List[str] = []

    # 1. environment variables (os.environ)
    for key in CARLA_ROOT_KEYS:
        value = os.environ.get(key)
        if value:
            candidates.append(value)

    # 2. shell configuration files (bashrc, profile, bash_profile)
    for rc_name in (".bashrc", ".profile", ".bash_profile"):
        shell_exports = _load_shell_exports(Path.home() / rc_name)
        for key in CARLA_ROOT_KEYS:
            value = shell_exports.get(key)
            if value:
                candidates.append(value)

    # 3. try to infer CARLA root from the carla Python package location
    try:
        import carla  # type: ignore
        carla_init = getattr(carla, "__file__", None)
        if carla_init:
            carla_pkg = Path(carla_init).resolve().parent
            # carla package is typically at <carla_root>/PythonAPI/carla/
            for ancestor in carla_pkg.parents:
                if _candidate_carla_content_roots(ancestor):
                    candidates.append(str(ancestor))
                    break
    except Exception:
        pass

    # 4. common CARLA installation paths (fallback)
    home = Path.home()
    common_paths = [
        home / "carla",
        home / "Carla" / "carla",
        home / "Carla",
        Path("/opt/carla"),
        Path("/opt/Carla"),
        Path("/usr/local/carla"),
    ]
    for p in common_paths:
        candidates.append(str(p))

    return candidates


def discover_carla_root() -> Optional[Path]:
    candidates = _collect_carla_root_candidates()

    seen = set()
    for value in candidates:
        candidate = Path(value).expanduser().resolve()
        if str(candidate) in seen or not candidate.exists():
            continue
        seen.add(str(candidate))
        if _candidate_carla_content_roots(candidate):
            return candidate
    return None


def probe_current_world_map(host: str, port: int, python_exec: str, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    code = """
import json
import sys

host = sys.argv[1]
port = int(sys.argv[2])
result = {"ok": False}
try:
    import carla
    client = carla.Client(host, port)
    client.set_timeout(5.0)
    world = client.get_world()
    raw_name = world.get_map().name
    result = {
        "ok": True,
        "raw_name": raw_name,
        "map_name": raw_name.split("/")[-1] if "/" in raw_name else raw_name,
    }
except Exception as exc:
    result = {
        "ok": False,
        "error": "{name}: {msg}".format(name=type(exc).__name__, msg=str(exc)),
    }
print(json.dumps(result))
""".strip()
    return run_json_probe(
        [python_exec, "-c", code, host, str(port)],
        cwd=repo_root or settings.REPO_ROOT,
        env=build_python_env(repo_root or settings.REPO_ROOT, python_exec),
    )


def restore_carla_async_mode(host: str, port: int, python_exec: str, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Connect to CARLA and forcibly reset synchronous_mode to False.

    This is used when a run.py process was killed externally (SIGKILL) and
    CARLA is left stuck in synchronous mode — waiting for a world.tick() that
    will never come.  Calling this endpoint unfreezes CARLA so it becomes
    usable again without needing a full CARLA restart.
    """
    code = """
import json
import sys

host = sys.argv[1]
port = int(sys.argv[2])
result = {"ok": False}
try:
    import carla
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    settings_obj = world.get_settings()
    was_sync = settings_obj.synchronous_mode
    settings_obj.synchronous_mode = False
    settings_obj.fixed_delta_seconds = None
    world.apply_settings(settings_obj)
    # Tick once so CARLA can process the settings change and unblock.
    world.tick()
    result = {"ok": True, "was_sync": was_sync}
except Exception as exc:
    result = {
        "ok": False,
        "error": "{name}: {msg}".format(name=type(exc).__name__, msg=str(exc)),
    }
print(json.dumps(result))
""".strip()
    return run_json_probe(
        [python_exec, "-c", code, host, str(port)],
        cwd=repo_root or settings.REPO_ROOT,
        env=build_python_env(repo_root or settings.REPO_ROOT, python_exec),
    )


def discover_carla_maps(carla_root: Optional[Path]) -> List[Dict[str, Any]]:
    if not carla_root:
        return []

    content_roots = _candidate_carla_content_roots(carla_root)
    seen = set()
    maps: List[Dict[str, Any]] = []

    for content_root in content_roots:
        for path in sorted(content_root.rglob("*.umap")):
            stem = path.stem
            relative_path = path.relative_to(carla_root)
            relative_parts = set(relative_path.parts)
            if any(part in CARLA_MAP_IGNORE_PARTS for part in relative_parts):
                continue
            if "_Tile_" in stem or stem in CARLA_MAP_IGNORE_STEMS:
                continue
            map_id = stem
            if map_id in seen:
                continue
            seen.add(map_id)
            maps.append(
                {
                    "id": map_id,
                    "label": map_id,
                    "source": "carla",
                    "relative_path": str(relative_path),
                    "umap_path": str(path),
                }
            )
    return maps


def discover_map_catalog(host: str, port: int, python_exec: str) -> Dict[str, Any]:
    global _catalog_cache, _catalog_cache_ts

    # Use cached result if still fresh to avoid spawning multiple concurrent
    # carla.Client subprocesses which can overwhelm and crash the CARLA server.
    cache_key = "{host}:{port}:{exec}".format(host=host, port=port, exec=python_exec)
    with _catalog_lock:
        age = time.monotonic() - _catalog_cache_ts
        if _catalog_cache.get("_cache_key") == cache_key and age < _CATALOG_TTL_SECONDS:
            return {k: v for k, v in _catalog_cache.items() if not k.startswith("_")}

    carla_root = discover_carla_root()
    current_world = None
    current_world_raw_name = None
    current_world_error = None

    if python_exec and check_tcp_port(host, port) and settings.is_valid_python_exec(python_exec):
        probe = probe_current_world_map(host, port, python_exec, settings.REPO_ROOT)
        if probe.get("ok"):
            current_world = probe.get("map_name")
            current_world_raw_name = probe.get("raw_name")
        else:
            current_world_error = probe.get("error")

    carla_maps = discover_carla_maps(carla_root)
    workspace_maps = discover_maps()
    merged: List[Dict[str, Any]] = []
    seen = set()

    def add_map(entry: Dict[str, Any]) -> None:
        map_id = entry["id"]
        if not map_id or map_id in seen:
            return
        seen.add(map_id)
        merged.append(entry)

    if current_world:
        add_map(
            {
                "id": current_world,
                "label": "{name} (当前 world)".format(name=current_world),
                "source": "current-world",
                "relative_path": None,
                "umap_path": None,
            }
        )

    for item in carla_maps:
        add_map(item)

    for map_name in workspace_maps:
        add_map(
            {
                "id": map_name,
                "label": "{name} (workspace)".format(name=map_name),
                "source": "workspace",
                "relative_path": None,
                "umap_path": None,
            }
        )

    result = {
        "carla_root": str(carla_root) if carla_root else None,
        "current_world_map": current_world,
        "current_world_raw_name": current_world_raw_name,
        "current_world_error": current_world_error,
        "maps": merged,
    }

    with _catalog_lock:
        _catalog_cache.clear()
        _catalog_cache.update(result)
        _catalog_cache["_cache_key"] = cache_key
        _catalog_cache_ts = time.monotonic()

    return result


def list_agents() -> List[Dict[str, Any]]:
    agents = []
    for path in sorted(settings.AGENT_CONFIG_DIR.glob("*.yaml")):
        config = read_yaml(path)
        agents.append(
            {
                "id": path.stem,
                "file": path.name,
                "policy_type": config.get("policy_type", path.stem),
            }
        )
    return agents


def list_scenario_templates() -> List[Dict[str, Any]]:
    templates = []
    for template_id, filename in settings.SCENARIO_TEMPLATE_FILES.items():
        path = settings.SCENARIO_CONFIG_DIR / filename
        if not path.exists():
            continue
        config = read_yaml(path)
        templates.append(
            {
                "id": template_id,
                "file": filename,
                "policy_type": config.get("policy_type", template_id),
            }
        )
    return templates


def map_status(map_name: str) -> Dict[str, Dict[str, Any]]:
    paths = {
        "waypoints": settings.BUILDER_MAP_WAYPOINTS_DIR / map_name,
        "scenario_origin": settings.BUILDER_SCENARIO_ORIGIN_DIR / map_name,
        "builder_scenario_data": settings.BUILDER_SCENARIO_DATA_DIR / map_name,
        "run_link": settings.RUN_ROOT_SCENARIO_DATA_DIR / map_name,
    }
    payload = {}
    for key, path in paths.items():
        info = {
            "path": str(path),
            "exists": path.exists(),
        }
        if path.is_symlink():
            info["is_symlink"] = True
            info["link_target"] = os.readlink(str(path))
        payload[key] = info
    return payload


def _count_files(directory: Path, predicate) -> int:
    if not directory.exists():
        return 0
    count = 0
    for child in directory.iterdir():
        if child.is_file() and predicate(child.name):
            count += 1
    return count


def _latest_mtime(paths: Iterable[Path]) -> Optional[str]:
    timestamps = []
    for path in paths:
        if path.exists():
            timestamps.append(path.stat().st_mtime)
    if not timestamps:
        return None
    return datetime.fromtimestamp(max(timestamps)).astimezone().isoformat(timespec="seconds")


def standard_cards(map_name: str, scenario_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    scenario_ids = scenario_ids or [item["id"] for item in settings.STANDARD_SCENARIOS]
    standard_lookup = {item["id"]: item["name"] for item in settings.STANDARD_SCENARIOS}
    cards = []
    for scenario_id in scenario_ids:
        route_dir = settings.BUILDER_SCENARIO_ORIGIN_DIR / map_name / "scenario_{sid:02d}_routes".format(sid=scenario_id)
        scenario_dir = settings.BUILDER_SCENARIO_ORIGIN_DIR / map_name / "scenario_{sid:02d}_scenarios".format(sid=scenario_id)
        export_route_dir = settings.BUILDER_SCENARIO_DATA_DIR / map_name / "scenario_{sid:02d}_routes".format(sid=scenario_id)
        export_scenario_json = settings.BUILDER_SCENARIO_DATA_DIR / map_name / "scenarios" / "scenario_{sid:02d}.json".format(sid=scenario_id)
        export_index_json = settings.BUILDER_SCENARIO_DATA_DIR / map_name / "standard_scenario_{sid:02d}.json".format(sid=scenario_id)
        run_link = settings.RUN_ROOT_SCENARIO_DATA_DIR / map_name

        route_count = _count_files(route_dir, lambda name: name.startswith("route_") and name.endswith(".npy"))
        scenario_count = _count_files(
            scenario_dir,
            lambda name: name.startswith("scenario_") and name.endswith(".npy") and not name.endswith("_sides.npy"),
        )
        sides_count = _count_files(scenario_dir, lambda name: name.startswith("scenario_") and name.endswith("_sides.npy"))

        if route_count > 0:
            route_status = "Route Ready"
        else:
            route_status = "No Route"

        if scenario_count == 0:
            scenario_status = "Scenario Missing"
        elif route_count and (scenario_count < route_count or sides_count < scenario_count):
            scenario_status = "Scenario Partial"
        else:
            scenario_status = "Scenario Ready"

        export_ready = export_route_dir.exists() and export_scenario_json.exists() and export_index_json.exists()
        link_ready = run_link.exists() or run_link.is_symlink()
        export_status = "Export Ready" if export_ready else "Export Missing"
        if export_ready and link_ready:
            overall_status = "Run Ready"
        elif export_ready:
            overall_status = "Export Ready"
        elif scenario_status == "Scenario Ready":
            overall_status = "Scenario Ready"
        elif route_status == "Route Ready":
            overall_status = "Route Ready"
        else:
            overall_status = "Not Started"

        cards.append(
            {
                "scenario_id": scenario_id,
                "name": standard_lookup.get(scenario_id, "Scenario {sid}".format(sid=scenario_id)),
                "map_name": map_name,
                "route_count": route_count,
                "scenario_count": scenario_count,
                "sides_count": sides_count,
                "route_status": route_status,
                "scenario_status": scenario_status,
                "export_status": export_status,
                "overall_status": overall_status,
                "latest_updated_at": _latest_mtime(
                    [route_dir, scenario_dir, export_route_dir, export_scenario_json, export_index_json]
                ),
                "paths": {
                    "route_dir": {"path": str(route_dir), "exists": route_dir.exists()},
                    "scenario_dir": {"path": str(scenario_dir), "exists": scenario_dir.exists()},
                    "export_route_dir": {"path": str(export_route_dir), "exists": export_route_dir.exists()},
                    "export_scenario_json": {"path": str(export_scenario_json), "exists": export_scenario_json.exists()},
                    "export_index_json": {"path": str(export_index_json), "exists": export_index_json.exists()},
                    "run_link": {
                        "path": str(run_link),
                        "exists": run_link.exists() or run_link.is_symlink(),
                        "is_symlink": run_link.is_symlink(),
                        "link_target": os.readlink(str(run_link)) if run_link.is_symlink() else None,
                    },
                },
            }
        )
    return cards


def _normalize_agent_name(agent_name: str) -> str:
    return agent_name[:-5] if agent_name.endswith(".yaml") else agent_name


def _build_experiment_id(exp_name: str, agent_name: str, scenario_id: int) -> str:
    safe_exp_name = re.sub(r"[^A-Za-z0-9._-]+", "-", exp_name).strip("-_.") or "exp"
    return "guiexp-{stamp}-{exp}-{agent}-s{sid:02d}-{suffix}".format(
        stamp=datetime.now().strftime("%Y%m%d-%H%M%S"),
        exp=safe_exp_name,
        agent=_normalize_agent_name(agent_name),
        sid=scenario_id,
        suffix=uuid.uuid4().hex[:6],
    )


def _scenario_json_path(map_name: str, scenario_id: int) -> Path:
    return settings.BUILDER_SCENARIO_DATA_DIR / map_name / "standard_scenario_{sid:02d}.json".format(sid=scenario_id)


def _count_total_data(map_name: str, scenario_id: int, route_id: Optional[int]) -> int:
    path = _scenario_json_path(map_name, scenario_id)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        items = json.load(handle)
    if route_id is None:
        return len(items)
    return len([item for item in items if item.get("route_id") == route_id])


def _predict_output_paths(
    exp_name: str,
    agent_policy: str,
    scenario_policy: str,
    seed: int,
    output_base_dir: Path,
) -> Dict[str, str]:
    output_dir = output_base_dir / exp_name / "{exp}_{agent}_{scenario}_seed_{seed}".format(
        exp=exp_name,
        agent=agent_policy,
        scenario=scenario_policy,
        seed=seed,
    )
    eval_results = output_dir / "eval_results"
    return {
        "output_dir": str(output_dir),
        "runtime_log_path": str(output_dir / "runtime.log"),
        "progress_path": str(output_dir / "progress.txt"),
        "records_path": str(eval_results / "records.pkl"),
        "results_path": str(eval_results / "results.pkl"),
        "batch_results_path": str(eval_results / "batch_results.jsonl"),
    }


def create_run_snapshot(
    python_exec: str,
    request: Dict[str, Any],
) -> Dict[str, Any]:
    ensure_runtime_layout()
    ensure_map_link(request["map_name"])

    agent_name = _normalize_agent_name(request["agent_name"])
    agent_source = settings.AGENT_CONFIG_DIR / "{name}.yaml".format(name=agent_name)
    if not agent_source.exists():
        raise FileNotFoundError("Agent config not found: {path}".format(path=agent_source))

    template_key = request["scenario_template"]
    if template_key not in settings.SCENARIO_TEMPLATE_FILES:
        raise ValueError("Unsupported scenario template: {value}".format(value=template_key))
    scenario_source = settings.SCENARIO_CONFIG_DIR / settings.SCENARIO_TEMPLATE_FILES[template_key]
    if not scenario_source.exists():
        raise FileNotFoundError("Scenario template not found: {path}".format(path=scenario_source))

    export_route_dir = settings.BUILDER_SCENARIO_DATA_DIR / request["map_name"] / "scenario_{sid:02d}_routes".format(
        sid=request["scenario_id"]
    )
    export_scenario_json = (
        settings.BUILDER_SCENARIO_DATA_DIR
        / request["map_name"]
        / "scenarios"
        / "scenario_{sid:02d}.json".format(sid=request["scenario_id"])
    )
    export_index_json = _scenario_json_path(request["map_name"], request["scenario_id"])
    if not export_route_dir.exists():
        raise FileNotFoundError(
            "Exported route directory does not exist: {path}".format(path=export_route_dir)
        )
    if not export_scenario_json.exists():
        raise FileNotFoundError(
            "Exported scenario definition does not exist: {path}".format(path=export_scenario_json)
        )
    if not export_index_json.exists():
        raise FileNotFoundError(
            "Exported standard scenario index does not exist: {path}".format(path=export_index_json)
        )

    agent_config = read_yaml(agent_source)
    scenario_config = read_yaml(scenario_source)
    scenario_config.update(
        {
            "scenario_type_dir": "safebench/scenario/scenario_data/{map_name}".format(
                map_name=request["map_name"]
            ),
            "scenario_type": "standard_scenario_{sid:02d}.json".format(sid=request["scenario_id"]),
            "route_dir": "safebench/scenario/scenario_data/{map_name}".format(
                map_name=request["map_name"]
            ),
            "scenario_id": request["scenario_id"],
            "route_id": request.get("route_id"),
        }
    )

    agent_policy = agent_config.get("policy_type", agent_name)
    scenario_policy = scenario_config.get("policy_type", template_key)
    experiment_id = _build_experiment_id(request["exp_name"], agent_policy, request["scenario_id"])
    snapshot_dir = settings.EXPERIMENTS_ROOT / experiment_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    output_base_dir = snapshot_dir / "outputs"

    agent_snapshot_path = snapshot_dir / "agent_resolved.yaml"
    scenario_snapshot_path = snapshot_dir / "scenario_resolved.yaml"
    write_yaml(agent_snapshot_path, agent_config)
    write_yaml(scenario_snapshot_path, scenario_config)

    run_agent_config_path = settings.RUN_ROOT_AGENT_CONFIG_DIR / "gui_{experiment_id}.yaml".format(
        experiment_id=experiment_id
    )
    run_scenario_config_path = settings.RUN_ROOT_SCENARIO_CONFIG_DIR / "gui_{experiment_id}.yaml".format(
        experiment_id=experiment_id
    )
    ensure_symlink(run_agent_config_path, agent_snapshot_path)
    ensure_symlink(run_scenario_config_path, scenario_snapshot_path)

    port = request.get("port") or 2000
    tm_port = request.get("tm_port") or 8000
    predicted_paths = _predict_output_paths(
        request["exp_name"],
        agent_policy,
        scenario_policy,
        request["seed"],
        output_base_dir,
    )
    total_data = _count_total_data(request["map_name"], request["scenario_id"], request.get("route_id"))

    command = [
        python_exec,
        str(settings.REPO_ROOT / "scripts" / "run.py"),
        "--ROOT_DIR",
        str(settings.RUN_ROOT),
        "--output_dir",
        str(output_base_dir),
        "--mode",
        request.get("mode", "eval"),
        "--agent_cfg",
        run_agent_config_path.name,
        "--scenario_cfg",
        run_scenario_config_path.name,
        "--exp_name",
        request["exp_name"],
        "--seed",
        str(request["seed"]),
        "--port",
        str(port),
        "--tm_port",
        str(tm_port),
        "--render",
        str(bool(request.get("render", False))).lower(),
        "--save_video",
        str(bool(request.get("save_video", False))).lower(),
    ]

    manifest = {
        "experiment_id": experiment_id,
        "created_at": now_iso(),
        "snapshot_dir": str(snapshot_dir),
        "python_exec": python_exec,
        "map": request["map_name"],
        "scenario_id": request["scenario_id"],
        "scenario_name": next(
            (item["name"] for item in settings.STANDARD_SCENARIOS if item["id"] == request["scenario_id"]),
            "Scenario {sid}".format(sid=request["scenario_id"]),
        ),
        "agent_cfg_source": str(agent_source),
        "agent_name": agent_name,
        "agent_policy": agent_policy,
        "scenario_template": template_key,
        "scenario_template_source": str(scenario_source),
        "scenario_policy": scenario_policy,
        "exp_name": request["exp_name"],
        "seed": request["seed"],
        "render": bool(request.get("render", False)),
        "save_video": bool(request.get("save_video", False)),
        "route_id": request.get("route_id"),
        "mode": request.get("mode", "eval"),
        "port": port,
        "tm_port": tm_port,
        "output_base_dir": str(output_base_dir),
        "total_data": total_data,
        "run_root": str(settings.RUN_ROOT),
        "run_root_agent_cfg": str(run_agent_config_path),
        "run_root_scenario_cfg": str(run_scenario_config_path),
        "agent_snapshot_path": str(agent_snapshot_path),
        "scenario_snapshot_path": str(scenario_snapshot_path),
        "records_path": predicted_paths["records_path"],
        "results_path": predicted_paths["results_path"],
        "runtime_log_path": predicted_paths["runtime_log_path"],
        "progress_path": predicted_paths["progress_path"],
        "batch_results_path": predicted_paths["batch_results_path"],
        "output_dir": predicted_paths["output_dir"],
        "command": command,
    }
    write_json(snapshot_dir / "manifest.json", manifest)
    return manifest


def load_manifest(experiment_id: str) -> Dict[str, Any]:
    manifest_path = settings.EXPERIMENTS_ROOT / experiment_id / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Experiment manifest not found: {path}".format(path=manifest_path))
    return read_json(manifest_path)


def _manifest_created_timestamp(manifest: Dict[str, Any]) -> float:
    created_at = manifest.get("created_at")
    if not created_at:
        return 0.0
    try:
        return datetime.fromisoformat(str(created_at)).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _manifest_activity_timestamp(manifest: Dict[str, Any]) -> float:
    timestamps: List[float] = []
    for key in [
        "runtime_log_path",
        "batch_results_path",
        "records_path",
        "results_path",
        "progress_path",
    ]:
        path_value = manifest.get(key)
        if not path_value:
            continue
        path = Path(path_value)
        if path.exists():
            timestamps.append(path.stat().st_mtime)
    if timestamps:
        return max(timestamps)
    return _manifest_created_timestamp(manifest)


def list_manifests() -> List[Dict[str, Any]]:
    manifests = []
    if not settings.EXPERIMENTS_ROOT.exists():
        return manifests
    for path in settings.EXPERIMENTS_ROOT.glob("*/manifest.json"):
        manifests.append(read_json(path))
    return sorted(
        manifests,
        key=lambda item: (
            _manifest_activity_timestamp(item),
            _manifest_created_timestamp(item),
            item.get("experiment_id") or "",
        ),
        reverse=True,
    )


def probe_experiment_pickles(manifest: Dict[str, Any], python_exec: str) -> Dict[str, Any]:
    code = """
import json
import os
import sys

repo_root, records_path, results_path = sys.argv[1:4]
sys.path.insert(0, repo_root)
payload = {"ok": True, "records_count": 0, "results": {}}
try:
    import joblib
    if os.path.exists(records_path):
        payload["records_count"] = len(joblib.load(records_path))
    if os.path.exists(results_path):
        value = joblib.load(results_path)
        if isinstance(value, dict):
            payload["results"] = value
except Exception as exc:
    payload = {
        "ok": False,
        "error": "{name}: {message}".format(name=type(exc).__name__, message=str(exc)),
        "records_count": 0,
        "results": {},
    }
print(json.dumps(payload, ensure_ascii=False, default=str))
""".strip()
    return run_json_probe(
        [
            python_exec,
            "-c",
            code,
            str(settings.REPO_ROOT),
            manifest["records_path"],
            manifest["results_path"],
        ],
        cwd=settings.REPO_ROOT,
        env=build_python_env(settings.REPO_ROOT, python_exec),
    )


def parse_stop_reasons(runtime_log_path: Path) -> Dict[str, int]:
    counter = Counter()
    if not runtime_log_path.exists():
        return {}
    with runtime_log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = STOP_REASON_PATTERN.search(line)
            if match:
                counter[match.group(1).strip()] += 1
    return dict(counter)


def tail_text_file(path: Path, lines: int = 80) -> List[str]:
    if not path.exists():
        return []
    tail = deque(maxlen=max(1, lines))
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tail.append(line.rstrip("\n"))
    return list(tail)


def parse_batch_summaries(path: Path, limit: int = 20) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    tail = deque(maxlen=max(1, limit))
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                tail.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(tail)
