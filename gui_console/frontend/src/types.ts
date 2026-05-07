export interface EnvironmentState {
  repo_root: string;
  python_exec: string;
  carla_host: string;
  carla_port: number;
  tm_port: number;
  carla_session_connected: boolean;
  last_checked_at?: string | null;
  carla_reachable: boolean;
  safebench_import_ok: boolean;
  safebench_module_path?: string | null;
  python_exists: boolean;
  error?: string | null;
}

export interface AgentOption {
  id: string;
  file: string;
  policy_type: string;
}

export interface ScenarioTemplateOption {
  id: string;
  file: string;
  policy_type: string;
}

export interface StandardOption {
  id: number;
  name: string;
}

export interface PythonEnvOption {
  id: string;
  name: string;
  label: string;
  python_exec: string;
  source: string;
  active: boolean;
  recommended: boolean;
}

export interface OptionsResponse {
  repo_root: string;
  gui_root: string;
  runtime_root: string;
  state: EnvironmentState;
  python_envs: PythonEnvOption[];
  maps: string[];
  agents: AgentOption[];
  scenario_templates: ScenarioTemplateOption[];
  standards: StandardOption[];
}

export interface JobInfo {
  id: string;
  type: string;
  status: string;
  command: string;
  process_name?: string | null;
  cwd: string;
  pid?: number | null;
  started_at?: string | null;
  ended_at?: string | null;
  log_path: string;
  return_code?: number | null;
  metadata: Record<string, unknown>;
  output_hints: Record<string, unknown>;
  supports_control?: boolean;
  control_requested?: string | null;
  control_requested_at?: string | null;
  error?: string | null;
}

export interface StandardCard {
  scenario_id: number;
  name: string;
  map_name: string;
  route_count: number;
  scenario_count: number;
  sides_count: number;
  route_status: string;
  scenario_status: string;
  export_status: string;
  overall_status: string;
  latest_updated_at?: string | null;
  paths: Record<string, { path: string; exists: boolean; is_symlink?: boolean; link_target?: string | null }>;
}

export interface MapStatusResponse {
  map_name: string;
  paths: Record<string, { path: string; exists: boolean; is_symlink?: boolean; link_target?: string | null }>;
}

export interface MapOption {
  id: string;
  label: string;
  source: string;
  relative_path?: string | null;
  umap_path?: string | null;
}

export interface MapCatalogResponse {
  carla_root?: string | null;
  current_world_map?: string | null;
  current_world_raw_name?: string | null;
  current_world_error?: string | null;
  maps: MapOption[];
}

export interface ExperimentSummary {
  manifest: ExperimentManifest;
  records_count: number;
  remaining_count: number;
  total_data: number;
  results: Record<string, number>;
  active_job?: JobInfo | null;
}

export interface ExperimentManifest {
  experiment_id: string;
  created_at: string;
  snapshot_dir: string;
  python_exec: string;
  map: string;
  scenario_id: number;
  scenario_name: string;
  agent_cfg_source: string;
  agent_name: string;
  agent_policy: string;
  scenario_template: string;
  scenario_template_source: string;
  scenario_policy: string;
  exp_name: string;
  seed: number;
  render: boolean;
  save_video: boolean;
  route_id?: number | null;
  mode: string;
  port: number;
  tm_port: number;
  output_base_dir: string;
  total_data: number;
  run_root: string;
  run_root_agent_cfg: string;
  run_root_scenario_cfg: string;
  agent_snapshot_path: string;
  scenario_snapshot_path: string;
  records_path: string;
  results_path: string;
  runtime_log_path: string;
  progress_path: string;
  batch_results_path: string;
  output_dir: string;
  command: string[];
}

export interface ExperimentDetail {
  manifest: ExperimentManifest;
  results: Record<string, number>;
  progress: {
    records_count: number;
    total_data: number;
    remaining_count: number;
    resume_ready: boolean;
    probe_ok: boolean;
    probe_error?: string | null;
  };
  batch_summaries: Array<Record<string, unknown>>;
  stop_reason_counts: Record<string, number>;
  runtime_log_tail: string[];
  related_jobs: JobInfo[];
  video_dir?: string | null;
}
