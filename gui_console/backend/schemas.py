"""Pydantic models used by the GUI backend."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EnvironmentCheckRequest(BaseModel):
    repo_root: str
    python_exec: str
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    tm_port: int = 8000


class MapPrepareRequest(BaseModel):
    map_name: str = Field(..., description="Target CARLA map name")
    python_exec: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None


class StandardJobRequest(BaseModel):
    map_name: str
    scenario_id: int
    python_exec: Optional[str] = None


class ExportRequest(BaseModel):
    map_name: str
    scenario_id: int
    export_format: str = "standard"
    python_exec: Optional[str] = None


class RunExperimentRequest(BaseModel):
    agent_name: str
    scenario_template: str = "standard"
    map_name: str
    scenario_id: int
    exp_name: str
    seed: int = 0
    render: bool = True
    save_video: bool = True
    route_id: Optional[int] = None
    port: Optional[int] = None
    tm_port: Optional[int] = None
    mode: str = "eval"


class RerunExperimentRequest(BaseModel):
    render: bool = True
    save_video: bool = True
    exp_name: Optional[str] = None


class OpenDirRequest(BaseModel):
    path: str


class EnvironmentCheckResponse(BaseModel):
    ok: bool
    message: str
    state: Dict[str, Any]
    runtime_paths: Dict[str, str]
    failed_checks: List[str] = Field(default_factory=list)


class JobInfo(BaseModel):
    id: str
    type: str
    status: str
    command: str
    process_name: Optional[str] = None
    cwd: str
    pid: Optional[int] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    log_path: str
    return_code: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    output_hints: Dict[str, Any] = Field(default_factory=dict)
    supports_control: bool = False
    control_requested: Optional[str] = None
    control_requested_at: Optional[str] = None
    error: Optional[str] = None


class MapStatusResponse(BaseModel):
    map_name: str
    paths: Dict[str, Dict[str, Any]]


class MapOptionResponse(BaseModel):
    id: str
    label: str
    source: str
    relative_path: Optional[str] = None
    umap_path: Optional[str] = None


class MapCatalogResponse(BaseModel):
    carla_root: Optional[str] = None
    current_world_map: Optional[str] = None
    current_world_raw_name: Optional[str] = None
    current_world_error: Optional[str] = None
    maps: List[MapOptionResponse] = Field(default_factory=list)


class StandardCardResponse(BaseModel):
    scenario_id: int
    name: str
    map_name: str
    route_count: int
    scenario_count: int
    sides_count: int
    route_status: str
    scenario_status: str
    export_status: str
    export_stale: bool = False
    export_route_count: int = 0
    overall_status: str
    latest_updated_at: Optional[str] = None
    paths: Dict[str, Dict[str, Any]]


class PythonEnvOption(BaseModel):
    id: str
    name: str
    label: str
    python_exec: str
    source: str
    active: bool = False
    recommended: bool = False


class OptionsResponse(BaseModel):
    repo_root: str
    gui_root: str
    runtime_root: str
    state: Dict[str, Any]
    python_envs: List[PythonEnvOption]
    maps: List[str]
    agents: List[Dict[str, Any]]
    scenario_templates: List[Dict[str, Any]]
    standards: List[Dict[str, Any]]


class ExperimentDetailResponse(BaseModel):
    manifest: Dict[str, Any]
    results: Dict[str, Any]
    progress: Dict[str, Any]
    batch_summaries: List[Dict[str, Any]]
    stop_reason_counts: Dict[str, int]
    runtime_log_tail: List[str]
    related_jobs: List[Dict[str, Any]]
    video_dir: Optional[str] = None
