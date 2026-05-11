"""FastAPI application for the SafeBenchHK GUI console."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import settings
from .jobs import JobStore
from .repository import (
    build_rerun_request,
    build_python_env,
    check_tcp_port,
    create_run_snapshot,
    discover_map_catalog,
    discover_maps,
    ensure_map_link,
    ensure_runtime_layout,
    list_agents,
    list_manifests,
    list_scenario_templates,
    load_manifest,
    map_status,
    parse_batch_summaries,
    parse_stop_reasons,
    persist_manifest_python_exec,
    probe_experiment_pickles,
    probe_safebench_import,
    resolve_runtime_python_exec,
    restore_carla_async_mode,
    standard_cards,
    tail_text_file,
)
from .schemas import (
    EnvironmentCheckRequest,
    EnvironmentCheckResponse,
    ExperimentDetailResponse,
    ExportRequest,
    JobInfo,
    MapCatalogResponse,
    MapPrepareRequest,
    MapStatusResponse,
    OpenDirRequest,
    OptionsResponse,
    RerunExperimentRequest,
    RunExperimentRequest,
    StandardCardResponse,
    StandardJobRequest,
)
from .state import AppStateStore, now_iso


app = FastAPI(title="SafeBenchHK GUI Console")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.DEFAULT_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state_store = AppStateStore()
job_store = JobStore()


@app.on_event("startup")
def _startup():
    ensure_runtime_layout()


def _current_state():
    return state_store.load()


def _failed_environment_checks(state) -> List[str]:
    failed_checks = []
    if not state.python_exists:
        failed_checks.append("Python 解释器路径无效")
    if not state.carla_reachable:
        failed_checks.append(
            "无法连接 CARLA ({host}:{port})".format(
                host=state.carla_host,
                port=state.carla_port,
            )
        )
    if not state.safebench_import_ok:
        failed_checks.append("当前 Python 未能导入当前仓库的 safebench")
    return failed_checks


def _refresh_environment_state(state, *, refresh_import: bool = False, connect_session: bool = False, preserve_session: bool = False):
    repo_root = Path(state.repo_root).resolve()
    state.python_exists = settings.is_valid_python_exec(state.python_exec)
    state.carla_reachable = check_tcp_port(state.carla_host, state.carla_port)
    state.last_checked_at = now_iso()

    if state.python_exists and (refresh_import or not state.safebench_import_ok or not state.safebench_module_path):
        import_probe = probe_safebench_import(repo_root, state.python_exec)
        state.safebench_import_ok = bool(import_probe.get("ok"))
        state.safebench_module_path = import_probe.get("module_path")
        state.error = import_probe.get("error")
    elif not state.python_exists:
        state.safebench_import_ok = False
        state.safebench_module_path = None
        state.error = "Python 解释器路径无效。"

    ready = state.python_exists and state.carla_reachable and state.safebench_import_ok
    if connect_session:
        state.carla_session_connected = ready
    elif not preserve_session and state.carla_session_connected and not ready:
        state.carla_session_connected = False

    if ready:
        state.error = None

    state_store.save(state)
    return state, _failed_environment_checks(state), ready


def _effective_python_exec(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return _current_state().python_exec


def _effective_port(explicit: Optional[int]) -> int:
    if explicit is not None:
        return explicit
    return _current_state().carla_port


def _job_env(python_exec: str):
    return build_python_env(settings.REPO_ROOT, python_exec)


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "item"


def _job_process_name(job_type: str, map_name: Optional[str] = None, scenario_id: Optional[int] = None) -> str:
    parts = ["SafeBenchHK", _slug(job_type)]
    if map_name:
        parts.append(_slug(map_name))
    if scenario_id is not None:
        parts.append("s{sid:02d}".format(sid=scenario_id))
    return "-".join(parts)


def _run_process_name(exp_name: str, scenario_id: int, resume: bool = False) -> str:
    role = "resume-run" if resume else "run"
    return "-".join(
        [
            "SafeBenchHK",
            _slug(role),
            _slug(exp_name),
            "s{sid:02d}".format(sid=scenario_id),
        ]
    )


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "repo_root": str(settings.REPO_ROOT),
        "gui_root": str(settings.GUI_ROOT),
        "runtime_root": str(settings.RUNTIME_ROOT),
        "timestamp": now_iso(),
    }


@app.get("/api/options", response_model=OptionsResponse)
def options():
    return {
        "repo_root": str(settings.REPO_ROOT),
        "gui_root": str(settings.GUI_ROOT),
        "runtime_root": str(settings.RUNTIME_ROOT),
        "state": _current_state().to_dict(),
        "python_envs": settings.python_env_options(),
        "maps": discover_maps(),
        "agents": list_agents(),
        "scenario_templates": list_scenario_templates(),
        "standards": settings.STANDARD_SCENARIOS,
    }


@app.post("/api/environment/check", response_model=EnvironmentCheckResponse)
def check_environment(request: EnvironmentCheckRequest):
    repo_root = Path(request.repo_root).resolve()
    if repo_root != settings.REPO_ROOT.resolve():
        raise HTTPException(
            status_code=400,
            detail="GUI 当前只支持当前仓库：{repo}".format(repo=settings.REPO_ROOT),
        )

    runtime_paths = ensure_runtime_layout()

    state = _current_state()
    state.repo_root = str(repo_root)
    state.python_exec = request.python_exec
    state.carla_host = request.carla_host
    state.carla_port = request.carla_port
    state.tm_port = request.tm_port
    state, failed_checks, ok = _refresh_environment_state(
        state,
        refresh_import=True,
        connect_session=True,
    )
    if ok:
        message = "CARLA 与 Python 环境已就绪。"
    else:
        message = "环境检查未全部通过：{reasons}".format(reasons="；".join(failed_checks))
    return {
        "ok": ok,
        "message": message,
        "state": state.to_dict(),
        "runtime_paths": runtime_paths,
        "failed_checks": failed_checks,
    }


@app.get("/api/environment/status", response_model=EnvironmentCheckResponse)
def environment_status():
    runtime_paths = ensure_runtime_layout()
    state, failed_checks, ok = _refresh_environment_state(_current_state(), refresh_import=False, preserve_session=True)
    if state.carla_session_connected and ok:
        message = "GUI 与 CARLA 的会话已连接。"
    elif state.carla_reachable:
        message = "CARLA 服务在线，但 GUI 当前未建立会话。"
    else:
        message = "CARLA 服务当前不可达。"
    return {
        "ok": ok,
        "message": message,
        "state": state.to_dict(),
        "runtime_paths": runtime_paths,
        "failed_checks": failed_checks,
    }


@app.post("/api/environment/disconnect", response_model=EnvironmentCheckResponse)
def disconnect_environment():
    runtime_paths = ensure_runtime_layout()
    state = _current_state()
    state.carla_session_connected = False
    state.last_checked_at = now_iso()
    state_store.save(state)
    return {
        "ok": False,
        "message": "已断开 GUI 与 CARLA 的会话；CARLA 服务本身未关闭。",
        "state": state.to_dict(),
        "runtime_paths": runtime_paths,
        "failed_checks": [],
    }


@app.post("/api/environment/restore-async")
def restore_async():
    """Reset CARLA synchronous mode back to False (async).

    When a run.py process is killed externally (SIGKILL / OOM), CARLA is left
    stuck in synchronous mode waiting for world.tick() forever.  This endpoint
    connects to CARLA, disables synchronous mode, and fires a single tick so
    CARLA becomes responsive again — no CARLA restart required.
    """
    state = _current_state()
    if not state.python_exec or not state.carla_host or not state.carla_port:
        raise HTTPException(status_code=400, detail="环境尚未配置，请先完成环境检查。")
    result = restore_carla_async_mode(
        state.carla_host, state.carla_port, state.python_exec
    )
    if result.get("ok"):
        was_sync = result.get("was_sync", False)
        message = (
            "CARLA 已恢复异步模式（原先处于同步模式）。" if was_sync
            else "CARLA 已确认处于异步模式，无需修改。"
        )
        return {"ok": True, "message": message, "was_sync": was_sync}
    else:
        raise HTTPException(
            status_code=502,
            detail="无法恢复 CARLA 异步模式：{err}".format(err=result.get("error", "未知错误")),
        )


@app.get("/api/maps/{map_name}/status", response_model=MapStatusResponse)
def get_map_status(map_name: str):
    return {"map_name": map_name, "paths": map_status(map_name)}


@app.get("/api/maps/catalog", response_model=MapCatalogResponse)
def get_map_catalog():
    state = _current_state()
    return discover_map_catalog(state.carla_host, state.carla_port, state.python_exec)


@app.get("/api/maps/{map_name}/standards", response_model=List[StandardCardResponse])
def get_standard_cards(
    map_name: str,
    scenario_ids: Optional[str] = Query(default=None, description="comma-separated scenario ids"),
):
    ids = None
    if scenario_ids:
        ids = [int(item) for item in scenario_ids.split(",") if item.strip()]
    return standard_cards(map_name, ids)


@app.post("/api/jobs/map-prepare", response_model=JobInfo)
def start_map_prepare(request: MapPrepareRequest):
    state = _current_state()
    python_exec = _effective_python_exec(request.python_exec)
    port = request.port if request.port is not None else state.carla_port
    host = request.host or state.carla_host
    link_info = ensure_map_link(request.map_name)
    job = job_store.start_job(
        job_type="map-prepare",
        command=[
            python_exec,
            "get_map_data.py",
            "--host",
            host,
            "--port",
            str(port),
            "--map",
            request.map_name,
        ],
        cwd=settings.BUILDER_ROOT,
        metadata={"map_name": request.map_name},
        output_hints=link_info,
        env=_job_env(python_exec),
        process_name=_job_process_name("map-prepare", request.map_name),
    )
    return job


@app.post("/api/jobs/route-editor", response_model=JobInfo)
def start_route_editor(request: StandardJobRequest):
    ensure_map_link(request.map_name)
    python_exec = _effective_python_exec(request.python_exec)
    job = job_store.start_job(
        job_type="route-editor",
        command=[
            python_exec,
            "create_routes.py",
            "--map",
            request.map_name,
            "--scenario",
            str(request.scenario_id),
            "--route",
            "-1",
        ],
        cwd=settings.BUILDER_ROOT,
        metadata={"map_name": request.map_name, "scenario_id": request.scenario_id},
        env=_job_env(python_exec),
        process_name=_job_process_name("route-editor", request.map_name, request.scenario_id),
    )
    return job


@app.post("/api/jobs/scenario-editor", response_model=JobInfo)
def start_scenario_editor(request: StandardJobRequest):
    ensure_map_link(request.map_name)
    python_exec = _effective_python_exec(request.python_exec)
    job = job_store.start_job(
        job_type="scenario-editor",
        command=[
            python_exec,
            "create_scenarios.py",
            "--map",
            request.map_name,
            "--scenario",
            str(request.scenario_id),
            "--route_idx",
            "-1",
        ],
        cwd=settings.BUILDER_ROOT,
        metadata={"map_name": request.map_name, "scenario_id": request.scenario_id},
        env=_job_env(python_exec),
        process_name=_job_process_name("scenario-editor", request.map_name, request.scenario_id),
    )
    return job


@app.delete("/api/maps/{map_name}/standards/{scenario_id}/route")
def clear_route(map_name: str, scenario_id: int):
    """Delete all route .npy files for a scenario, leaving the directory intact."""
    import shutil as _shutil
    route_dir = settings.BUILDER_SCENARIO_ORIGIN_DIR / map_name / "scenario_{sid:02d}_routes".format(sid=scenario_id)
    deleted = 0
    if route_dir.exists():
        for path in list(route_dir.iterdir()):
            if path.name.startswith("route_") and path.name.endswith(".npy"):
                path.unlink()
                deleted += 1
    return {"deleted": deleted, "dir": str(route_dir)}


@app.delete("/api/maps/{map_name}/standards/{scenario_id}/scenario")
def clear_scenario(map_name: str, scenario_id: int):
    """Delete all scenario/sides .npy files for a scenario, leaving the directory intact."""
    scenario_dir = settings.BUILDER_SCENARIO_ORIGIN_DIR / map_name / "scenario_{sid:02d}_scenarios".format(sid=scenario_id)
    deleted = 0
    if scenario_dir.exists():
        for path in list(scenario_dir.iterdir()):
            if path.name.startswith("scenario_") and path.name.endswith(".npy"):
                path.unlink()
                deleted += 1
    return {"deleted": deleted, "dir": str(scenario_dir)}


@app.post("/api/jobs/export", response_model=JobInfo)
def start_export(request: ExportRequest):
    ensure_map_link(request.map_name)
    python_exec = _effective_python_exec(request.python_exec)
    job = job_store.start_job(
        job_type="export",
        command=[
            python_exec,
            "export.py",
            "--map",
            request.map_name,
            "--scenario",
            str(request.scenario_id),
            "--format",
            request.export_format,
        ],
        cwd=settings.BUILDER_ROOT,
        metadata={
            "map_name": request.map_name,
            "scenario_id": request.scenario_id,
            "export_format": request.export_format,
        },
        env=_job_env(python_exec),
        process_name=_job_process_name("export", request.map_name, request.scenario_id),
    )
    return job


@app.get("/api/jobs", response_model=List[JobInfo])
def list_jobs():
    return job_store.list_jobs()


@app.get("/api/jobs/{job_id}", response_model=JobInfo)
def get_job(job_id: str):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/log")
def get_job_log(job_id: str, lines: int = Query(default=200, ge=10, le=2000)):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "lines": job_store.tail_log(job_id, lines)}


@app.post("/api/runs", response_model=JobInfo)
def start_run(request: RunExperimentRequest):
    state = _current_state()
    python_exec = resolve_runtime_python_exec(state.python_exec)
    if not python_exec:
        raise HTTPException(status_code=400, detail="Step 0 选择的 Python 解释器无效，请重新检查环境。")
    request_payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    manifest = create_run_snapshot(python_exec, request_payload)
    job = job_store.start_job(
        job_type="run",
        command=manifest["command"],
        cwd=settings.REPO_ROOT,
        metadata={
            "experiment_id": manifest["experiment_id"],
            "map_name": manifest["map"],
            "scenario_id": manifest["scenario_id"],
            "agent_name": manifest["agent_name"],
            "scenario_template": manifest["scenario_template"],
        },
        output_hints={
            "experiment_snapshot": manifest["snapshot_dir"],
            "output_dir": manifest["output_dir"],
        },
        env=_job_env(python_exec),
        process_name=_run_process_name(manifest["exp_name"], manifest["scenario_id"]),
        supports_control=True,
    )
    return job


@app.get("/api/experiments")
def experiments():
    state = _current_state()
    manifests = list_manifests()
    jobs = job_store.list_jobs()
    items = []
    for manifest in manifests:
        probe_python = resolve_runtime_python_exec(state.python_exec, manifest.get("python_exec"))
        probe = probe_experiment_pickles(manifest, probe_python or state.python_exec)
        records_count = int(probe.get("records_count", 0) or 0)
        total_data = int(manifest.get("total_data", 0) or 0)
        related_jobs = [
            job for job in jobs if job.get("metadata", {}).get("experiment_id") == manifest["experiment_id"]
        ]
        running = next((job for job in related_jobs if job.get("status") == "running"), None)
        items.append(
            {
                "manifest": manifest,
                "records_count": records_count,
                "remaining_count": max(total_data - records_count, 0),
                "total_data": total_data,
                "results": probe.get("results", {}),
                "active_job": running,
            }
        )
    return items


@app.get("/api/experiments/{experiment_id}", response_model=ExperimentDetailResponse)
def experiment_detail(experiment_id: str):
    state = _current_state()
    manifest = load_manifest(experiment_id)
    probe_python = resolve_runtime_python_exec(state.python_exec, manifest.get("python_exec"))
    probe = probe_experiment_pickles(manifest, probe_python or state.python_exec)
    records_count = int(probe.get("records_count", 0) or 0)
    total_data = int(manifest.get("total_data", 0) or 0)
    related_jobs = [
        job for job in job_store.list_jobs() if job.get("metadata", {}).get("experiment_id") == experiment_id
    ]
    return {
        "manifest": manifest,
        "results": probe.get("results", {}),
        "progress": {
            "records_count": records_count,
            "total_data": total_data,
            "remaining_count": max(total_data - records_count, 0),
            "resume_ready": total_data > records_count,
            "probe_ok": probe.get("ok", False),
            "probe_error": probe.get("error"),
        },
        "batch_summaries": parse_batch_summaries(Path(manifest["batch_results_path"])),
        "stop_reason_counts": parse_stop_reasons(Path(manifest["runtime_log_path"])),
        "runtime_log_tail": tail_text_file(Path(manifest["runtime_log_path"]), 80),
        "related_jobs": related_jobs,
        "video_dir": str(Path(manifest["output_dir"]) / "video") if manifest.get("save_video") else None,
    }


@app.post("/api/experiments/{experiment_id}/open-video-dir")
def open_video_dir(experiment_id: str):
    manifest = load_manifest(experiment_id)
    if not manifest.get("save_video"):
        raise HTTPException(status_code=400, detail="该实验未开启视频录制。")
    video_dir = Path(manifest["output_dir"]) / "video"
    if not video_dir.exists():
        raise HTTPException(status_code=404, detail=f"视频目录不存在：{video_dir}")
    subprocess.Popen(["xdg-open", str(video_dir)])
    return {"ok": True, "video_dir": str(video_dir)}


@app.post("/api/open-dir")
def open_dir(request: OpenDirRequest):
    target = Path(request.path).resolve()
    repo_root = settings.REPO_ROOT.resolve()
    if not str(target).startswith(str(repo_root)):
        raise HTTPException(status_code=400, detail="只允许打开仓库内的目录。")
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(["xdg-open", str(target)])
    return {"ok": True, "dir": str(target)}


@app.post("/api/experiments/{experiment_id}/resume", response_model=JobInfo)
def resume_experiment(experiment_id: str):
    manifest = load_manifest(experiment_id)
    ensure_map_link(manifest["map"])
    state = _current_state()
    python_exec = resolve_runtime_python_exec(state.python_exec, manifest.get("python_exec"))
    if not python_exec:
        raise HTTPException(status_code=400, detail="无法确定可用的 Python 解释器，请先在 Step 0 完成环境检查。")
    manifest = persist_manifest_python_exec(manifest, python_exec)
    job = job_store.start_job(
        job_type="resume-run",
        command=manifest["command"],
        cwd=settings.REPO_ROOT,
        metadata={
            "experiment_id": manifest["experiment_id"],
            "map_name": manifest["map"],
            "scenario_id": manifest["scenario_id"],
            "agent_name": manifest["agent_name"],
            "scenario_template": manifest["scenario_template"],
            "resume": True,
        },
        output_hints={
            "experiment_snapshot": manifest["snapshot_dir"],
            "output_dir": manifest["output_dir"],
        },
        env=_job_env(python_exec),
        process_name=_run_process_name(manifest["exp_name"], manifest["scenario_id"], resume=True),
        supports_control=True,
    )
    return job


@app.post("/api/experiments/{experiment_id}/rerun", response_model=JobInfo)
def rerun_experiment(experiment_id: str, request: RerunExperimentRequest):
    manifest = load_manifest(experiment_id)
    ensure_map_link(manifest["map"])
    state = _current_state()
    python_exec = resolve_runtime_python_exec(state.python_exec, manifest.get("python_exec"))
    if not python_exec:
        raise HTTPException(status_code=400, detail="无法确定可用的 Python 解释器，请先在 Step 0 完成环境检查。")

    request_payload = build_rerun_request(
        manifest,
        render=request.render,
        save_video=request.save_video,
        exp_name=request.exp_name,
    )
    new_manifest = create_run_snapshot(python_exec, request_payload)
    job = job_store.start_job(
        job_type="rerun",
        command=new_manifest["command"],
        cwd=settings.REPO_ROOT,
        metadata={
            "experiment_id": new_manifest["experiment_id"],
            "source_experiment_id": experiment_id,
            "map_name": new_manifest["map"],
            "scenario_id": new_manifest["scenario_id"],
            "agent_name": new_manifest["agent_name"],
            "scenario_template": new_manifest["scenario_template"],
            "rerun": True,
        },
        output_hints={
            "experiment_snapshot": new_manifest["snapshot_dir"],
            "output_dir": new_manifest["output_dir"],
        },
        env=_job_env(python_exec),
        process_name=_run_process_name(new_manifest["exp_name"], new_manifest["scenario_id"]),
        supports_control=True,
    )
    return job


@app.post("/api/jobs/{job_id}/pause", response_model=JobInfo)
def pause_job(job_id: str):
    try:
        return job_store.request_control(job_id, "pause")
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/jobs/{job_id}/stop", response_model=JobInfo)
def stop_job(job_id: str):
    try:
        return job_store.request_control(job_id, "stop")
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
