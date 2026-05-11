import { useEffect, useRef, useState } from "react";

import { apiGet, apiPost, apiDelete } from "./api";
import { ExperimentList, Step7Tab } from "./components/ExperimentList";
import { StandardCard, CardSubStep } from "./components/StandardCard";
import { TaskConsole } from "./components/TaskConsole";
import type {
  ExperimentDetail,
  ExperimentSummary,
  JobInfo,
  MapCatalogResponse,
  MapStatusResponse,
  OptionsResponse,
  StandardCard as StandardCardType
} from "./types";

const STEP_TITLES = [
  "Step 0 环境确认",
  "Step 1 地图与 Waypoint",
  "Step 2 标准选择",
  "Step 3-5 标准工作区",
  "Step 6 运行中心",
  "Step 7 结果与续跑"
];

function pickExperimentFocusJob(detail: ExperimentDetail | null): JobInfo | null {
  if (!detail || detail.related_jobs.length === 0) {
    return null;
  }
  return (
    detail.related_jobs.find((job) => job.status === "running" || job.status === "starting") ??
    detail.related_jobs.find((job) => job.status === "failed" || job.status === "stale") ??
    detail.related_jobs[0] ??
    null
  );
}

const STEP_GUIDES: React.ReactNode[] = [
  /* Step 0 */ (
    <div className="step-guide-content">
      <h4>Step 0 · 环境确认</h4>
      <ol>
        <li>确认<strong>仓库根目录</strong>和 <strong>Python 解释器</strong>已正确显示。</li>
        <li>按需填写 <strong>CARLA Host / Port / TM Port</strong>（默认本机 2000/8000）。</li>
        <li>点击绿色的<strong>「检查 CARLA 与 Python 环境」</strong>按钮。</li>
        <li>三项全部通过后，右侧导航的后续步骤会解锁。</li>
      </ol>
      <p className="muted">若 CARLA 未启动，「CARLA 连通性」会失败，先启动 CARLA 再重试即可。</p>
    </div>
  ),
  /* Step 1 */ (
    <div className="step-guide-content">
      <h4>Step 1 · 地图与 Waypoint</h4>
      <ol>
        <li>在下拉框中选择目标<strong>地图</strong>，确认仓库中对应的数据目录存在。</li>
        <li>点击<strong>「开始构建」</strong>，等待 waypoint 生成任务完成。</li>
        <li>若已有构建结果，点击<strong>「跳过，继续」</strong>可直接进入 Step 2。</li>
      </ol>
      <p className="muted">构建任务启动后，右侧会自动切换到「任务监控」显示进度。</p>
    </div>
  ),
  /* Step 2 */ (
    <div className="step-guide-content">
      <h4>Step 2 · 选择测试标准</h4>
      <ol>
        <li>勾选需要评测的<strong>测试标准</strong>（可多选）。</li>
        <li>确认选择后点击<strong>「下一步」</strong>进入标准工作区。</li>
      </ol>
      <p className="muted">每个标准在 Step 3-5 中会生成独立的工作卡片，可分别配置。</p>
    </div>
  ),
  /* Step 3 */ (
    <div className="step-guide-content">
      <h4>Step 3-5 · 编辑器操作说明</h4>
      <p className="muted">路线编辑器（create_routes）与场景编辑器（create_scenarios）共用以下快捷键：</p>
      <ul>
        <li><strong>左键点击</strong>：在地图上选取路径点 / Trigger / Actor 位置；再次点击同一点可取消选中。</li>
        <li><strong>右键点击</strong>：保存当前已选点为一条 route 或 scenario，保存后自动清空选点，可继续绘制下一条。</li>
        <li><strong>R 键</strong>：撤销 / 删除。如果有未保存的选点，清空当前选点；如果没有未保存的选点，删除上一条已保存的记录。</li>
        <li><strong>Home 键</strong>：将视角重置回地图中心。</li>
        <li><strong>ESC</strong>：自动保存当前选点（如果有效）并退出编辑器。</li>
      </ul>
      <p className="muted">鼠标滚轮可缩放地图；按住鼠标中键或拖动可平移视角。</p>
    </div>
  ),
  /* Step 4 */ (
    <div className="step-guide-content">
      <h4>Step 6 · 运行中心</h4>
      <ol>
        <li>选择<strong>智能体</strong>和<strong>场景模板</strong>，填写实验名称与随机种子。</li>
        <li>点击<strong>「提交运行」</strong>，任务进入队列后自动切换到「任务监控」。</li>
        <li>运行期间可在「任务监控—当前」中查看实时日志与资源占用。</li>
      </ol>
      <p className="muted">勾选「保存视频」会产生额外磁盘占用，大规模评测时可关闭。</p>
    </div>
  ),
  /* Step 5 */ (
    <div className="step-guide-content">
      <h4>Step 7 · 结果与续跑</h4>
      <ol>
        <li>在左侧列表中选择一个<strong>实验</strong>查看详情。</li>
        <li>若实验未完成，点击<strong>「续跑」</strong>从断点继续；已完成可点击<strong>「重跑」</strong>。</li>
        <li>展开各场景卡片可查看每条 route 的得分与失败原因。</li>
      </ol>
      <p className="muted">「续跑」会复用原实验 ID，跳过已完成的场景。</p>
    </div>
  ),
];

function App() {
  const alertedRuntimeEventsRef = useRef<Set<string>>(new Set());
  const waypointJobIdRef = useRef<string | null>(null);
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeJobLog, setActiveJobLog] = useState<string[]>([]);
  const [experimentJobLog, setExperimentJobLog] = useState<string[]>([]);
  const [taskConsoleView, setTaskConsoleView] = useState<"current" | "history">("current");
  const [rightPanel, setRightPanel] = useState<"guide" | "console">("guide");
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selectedExperimentId, setSelectedExperimentId] = useState<string | null>(null);
  const [experimentDetail, setExperimentDetail] = useState<ExperimentDetail | null>(null);
  const [selectedStep, setSelectedStep] = useState<number>(0);
  const [selectedMap, setSelectedMap] = useState<string>("");
  const [selectedStandards, setSelectedStandards] = useState<number[]>([]);
  const [cardSubStep, setCardSubStep] = useState<CardSubStep>("route");
  const [step7Tab, setStep7Tab] = useState<Step7Tab>("list");
  const [runFormCollapsed, setRunFormCollapsed] = useState(false);
  const [cards, setCards] = useState<StandardCardType[]>([]);
  const [mapCatalog, setMapCatalog] = useState<MapCatalogResponse | null>(null);
  const [mapStatus, setMapStatus] = useState<MapStatusResponse | null>(null);
  const [runScenarioId, setRunScenarioId] = useState<number | "">("");
  const [notice, setNotice] = useState<string>("欢迎使用 SafeBenchHK GUI Console。");
  const [error, setError] = useState<string | null>(null);
  const [environmentChecked, setEnvironmentChecked] = useState(false);
  const [runtimeAlert, setRuntimeAlert] = useState<{
    key: string;
    tone: "warning" | "error";
    title: string;
    message: string;
    jobId?: string | null;
    experimentId?: string | null;
  } | null>(null);

  const [envForm, setEnvForm] = useState({
    repo_root: "",
    python_exec: "",
    carla_host: "127.0.0.1",
    carla_port: 2000,
    tm_port: 8000
  });

  const [runForm, setRunForm] = useState({
    agent_name: "behavior",
    scenario_template: "standard",
    exp_name: "gui_eval",
    seed: 0,
    render: true,
    save_video: true,
    route_id: "",
    port: 2000,
    tm_port: 8000
  });

  const selectedCards = selectedStandards.length
    ? cards.filter((card) => selectedStandards.includes(card.scenario_id))
    : [];
  const selectedReadyCards = selectedCards.filter(
    (card) => card.overall_status === "Run Ready" && !card.export_stale
  );
  const selectedStaleCards = selectedCards.filter((card) => card.export_stale);
  const selectedExperiment =
    experiments.find((item) => item.manifest.experiment_id === selectedExperimentId) ?? null;
  const experimentFocusJob = pickExperimentFocusJob(experimentDetail);
  const activeJob = jobs.find((job) => job.id === activeJobId) ?? jobs[0] ?? null;
  const activeJobLogPollMs =
    activeJob?.status === "running" || activeJob?.status === "starting" ? 1200 : 4000;
  const experimentDetailPollMs =
    experimentFocusJob?.status === "running" || experimentFocusJob?.status === "starting" ? 2500 : 5000;
  const experimentJobLogPollMs =
    experimentFocusJob?.status === "running" || experimentFocusJob?.status === "starting" ? 1200 : 4000;
  const latestFailedJob = jobs.find((job) => job.status === "failed" || job.status === "stale") ?? null;
  const latestRunningJob =
    jobs.find((job) => job.status === "running" || job.status === "starting") ?? null;
  const alertJob = runtimeAlert?.jobId ? jobs.find((job) => job.id === runtimeAlert.jobId) ?? null : null;
  const alertExperiment = runtimeAlert?.experimentId
    ? experiments.find((item) => item.manifest.experiment_id === runtimeAlert.experimentId) ?? null
    : null;
  const canControlAlertJob = Boolean(
    alertJob?.supports_control &&
      (alertJob.status === "running" || alertJob.status === "starting") &&
      !alertJob.control_requested
  );
  const canResumeAlertExperiment = Boolean(alertExperiment && alertExperiment.remaining_count > 0);
  const carlaSessionConnected = Boolean(options?.state.carla_session_connected);
  const carlaServiceOnline = Boolean(options?.state.carla_reachable);
  const environmentReady = Boolean(
    options?.state.python_exists &&
      options?.state.carla_reachable &&
      options?.state.safebench_import_ok &&
      options?.state.carla_session_connected
  );
  const environmentConfirmed = environmentChecked && environmentReady;
  const runningJobsCount = jobs.filter((job) => job.status === "running" || job.status === "starting").length;
  const lastProbeAt = options?.state.last_checked_at ?? "尚未探测";
  const globalConnectionTone = carlaSessionConnected
    ? "connected"
    : carlaServiceOnline
      ? "idle"
      : "offline";
  const environmentChecks = options
    ? [
        {
          key: "python",
          order: 1,
          label: "Python 解释器",
          ok: options.state.python_exists,
          detail: options.state.python_exec
        },
        {
          key: "safebench",
          order: 2,
          label: "safebench 导入",
          ok: options.state.safebench_import_ok,
          detail: options.state.safebench_module_path ?? options.state.error ?? "未通过"
        },
        {
          key: "carla",
          order: 3,
          label: "CARLA 连接",
          ok: options.state.carla_reachable,
          detail: `${options.state.carla_host}:${options.state.carla_port}`
        }
      ]
        .sort((left, right) => {
          if (left.ok !== right.ok) {
            return left.ok ? 1 : -1;
          }
          return left.order - right.order;
        })
    : [];
  const failedEnvironmentChecks = environmentChecks.filter((item) => !item.ok);
  const environmentStatus = !environmentChecked
    ? {
        tone: "pending",
        label: "等待检查",
        message: "尚未执行环境检查，请先检查 CARLA 与 Python 环境。"
      }
    : environmentReady
      ? {
          tone: "ready",
          label: "就绪",
          message: "GUI 会话已连接到 CARLA，可以继续场景构建和评测。"
        }
      : options?.state.python_exists && options?.state.safebench_import_ok && carlaServiceOnline
        ? {
            tone: "fail",
            label: "未通过",
            message: "GUI 当前尚未连接到 CARLA，会话未建立；请重新执行环境检查。"
          }
        : {
            tone: "fail",
            label: "未通过",
            message:
              failedEnvironmentChecks.length > 0
                ? `当前未通过：${failedEnvironmentChecks.map((item) => item.label).join("、")}`
                : options?.state.error ?? "环境检查未通过，请重新检查。"
          };
  const environmentNoticePattern =
    /CARLA 与 Python 环境已就绪|环境检查未全部通过|已断开 GUI 与 CARLA 的会话|GUI 与 CARLA 的会话已连接|CARLA 服务在线，但 GUI 当前未建立会话|CARLA 服务当前不可达/;
  const pageNotice = notice && !environmentNoticePattern.test(notice) ? notice : null;
  const existingMapCards = cards
    .filter((card) => card.overall_status !== "Not Started")
    .sort((left, right) => left.scenario_id - right.scenario_id);
  const mapExperiments = experiments.filter((item) => item.manifest.map === selectedMap);
  const unfinishedMapExperiments = mapExperiments.filter((item) => item.remaining_count > 0);
  const completedMapExperiments = mapExperiments.filter(
    (item) => item.total_data > 0 && item.remaining_count === 0
  );
  const mapPathEntries = mapStatus ? Object.entries(mapStatus.paths) : [];
  const readyMapPathCount = mapPathEntries.filter(([, value]) => value.exists).length;
  const hasExistingMapState =
    readyMapPathCount > 0 || existingMapCards.length > 0 || mapExperiments.length > 0;
  const mapOptions =
    mapCatalog?.maps.length
      ? mapCatalog.maps
      : (options?.maps ?? []).map((item) => ({
          id: item,
          label: `${item} (workspace fallback)`,
          source: "workspace-fallback",
          relative_path: null,
          umap_path: null
        }));
  const previousStepRef = useRef(selectedStep);

  async function loadOptions() {
    try {
      const data = await apiGet<OptionsResponse>("/api/options");
      setOptions(data);
      setEnvForm((current) => ({
        repo_root: current.repo_root || data.state.repo_root,
        python_exec: current.python_exec || data.state.python_exec,
        carla_host: current.carla_host || data.state.carla_host,
        carla_port: current.carla_port || data.state.carla_port,
        tm_port: current.tm_port || data.state.tm_port
      }));
      setRunForm((current) => ({
        ...current,
        agent_name: current.agent_name || data.agents[0]?.id || "behavior",
        scenario_template: current.scenario_template || "standard",
        port: current.port || data.state.carla_port,
        tm_port: current.tm_port || data.state.tm_port
      }));
      if (!selectedMap && data.maps.length > 0) {
        setSelectedMap(data.maps[0]);
      }
    } catch (fetchError) {
      setError((fetchError as Error).message);
    }
  }

  async function loadMapCatalog(preferCurrentWorld = false) {
    try {
      const data = await apiGet<MapCatalogResponse>("/api/maps/catalog");
      setMapCatalog(data);
      const availableIds = data.maps.map((item) => item.id);
      if (preferCurrentWorld && data.current_world_map) {
        setSelectedMap(data.current_world_map);
        return;
      }
      if (!selectedMap || !availableIds.includes(selectedMap)) {
        if (data.current_world_map) {
          setSelectedMap(data.current_world_map);
        } else if (data.maps.length > 0) {
          setSelectedMap(data.maps[0].id);
        }
      }
    } catch (fetchError) {
      const message = (fetchError as Error).message;
      const fallbackMaps = (options?.maps ?? []).map((item) => ({
        id: item,
        label: `${item} (workspace fallback)`,
        source: "workspace-fallback",
        relative_path: null,
        umap_path: null
      }));
      if (fallbackMaps.length > 0) {
        setMapCatalog({
          carla_root: null,
          current_world_map: null,
          current_world_raw_name: null,
          current_world_error: message,
          maps: fallbackMaps
        });
        if (!selectedMap) {
          setSelectedMap(fallbackMaps[0].id);
        }
      }
      setError(
        message === "Not Found"
          ? "地图目录接口未找到。后端大概率还没有重启到最新版本，当前先回退到 workspace 地图列表。"
          : message
      );
    }
  }

  async function refreshEnvironmentStatus(showNotice = false) {
    try {
      const response = await apiGet<{
        ok: boolean;
        message: string;
        state: OptionsResponse["state"];
        failed_checks: string[];
      }>("/api/environment/status");
      setOptions((current) => (current ? { ...current, state: response.state } : current));
      if (showNotice) {
        setEnvironmentChecked(true);
        setNotice(response.message);
      }
    } catch (fetchError) {
      setError((fetchError as Error).message);
    }
  }

  async function loadJobs() {
    try {
      const data = await apiGet<JobInfo[]>("/api/jobs");
      setJobs(data);
      if (!activeJobId && data.length > 0) {
        setActiveJobId(data[0].id);
      }
    } catch (fetchError) {
      setError((fetchError as Error).message);
    }
  }

  async function loadExperiments() {
    try {
      const data = await apiGet<ExperimentSummary[]>("/api/experiments");
      setExperiments(data);
      if (!selectedExperimentId && data.length > 0) {
        setSelectedExperimentId(data[0].manifest.experiment_id);
      }
    } catch (fetchError) {
      setError((fetchError as Error).message);
    }
  }

  async function fetchJobLog(jobId: string, lines = 200) {
    const payload = await apiGet<{ lines: string[] }>(`/api/jobs/${jobId}/log?lines=${lines}`);
    return payload.lines;
  }

  async function loadActiveJobLog(jobId: string) {
    try {
      setActiveJobLog(await fetchJobLog(jobId, 200));
    } catch (fetchError) {
      setError((fetchError as Error).message);
    }
  }

  async function loadExperimentJobLog(jobId: string) {
    try {
      setExperimentJobLog(await fetchJobLog(jobId, 220));
    } catch (fetchError) {
      setError((fetchError as Error).message);
    }
  }

  async function loadMapStatus(mapName: string) {
    try {
      const data = await apiGet<MapStatusResponse>(`/api/maps/${encodeURIComponent(mapName)}/status`);
      setMapStatus(data);
    } catch (fetchError) {
      setError((fetchError as Error).message);
    }
  }

  async function loadCards(mapName: string) {
    if (!mapName) {
      setCards([]);
      return;
    }
    try {
      const data = await apiGet<StandardCardType[]>(`/api/maps/${encodeURIComponent(mapName)}/standards`);
      setCards(data);
    } catch (fetchError) {
      setError((fetchError as Error).message);
    }
  }

  async function loadExperimentDetail(experimentId: string) {
    try {
      const data = await apiGet<ExperimentDetail>(`/api/experiments/${experimentId}`);
      setExperimentDetail(data);
    } catch (fetchError) {
      setError((fetchError as Error).message);
    }
  }

  function pushRuntimeAlert(payload: {
    key: string;
    tone: "warning" | "error";
    title: string;
    message: string;
    jobId?: string | null;
    experimentId?: string | null;
  }) {
    if (alertedRuntimeEventsRef.current.has(payload.key)) {
      return;
    }
    alertedRuntimeEventsRef.current.add(payload.key);
    setRuntimeAlert(payload);
  }

  useEffect(() => {
    void loadOptions();
    void loadJobs();
    void loadExperiments();
  }, []);

  useEffect(() => {
    if (!options) {
      return;
    }
    void refreshEnvironmentStatus(false);
    const timer = window.setInterval(() => {
      void refreshEnvironmentStatus(false);
    }, 5000);
    return () => window.clearInterval(timer);
  }, [options?.repo_root]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void loadJobs();
      void loadExperiments();
    }, 3000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!selectedMap) {
      return;
    }
    const refresh = () => {
      void loadMapStatus(selectedMap);
      void loadCards(selectedMap);
    };
    refresh();
    const timer = window.setInterval(refresh, 4000);
    return () => window.clearInterval(timer);
  }, [selectedMap]);

  useEffect(() => {
    if (!environmentChecked || !options) {
      return;
    }
    const staleCheck =
      envForm.repo_root !== options.state.repo_root ||
      envForm.python_exec !== options.state.python_exec ||
      envForm.carla_host !== options.state.carla_host ||
      envForm.carla_port !== options.state.carla_port ||
      envForm.tm_port !== options.state.tm_port;
    if (staleCheck) {
      setEnvironmentChecked(false);
    }
  }, [environmentChecked, envForm, options]);

  useEffect(() => {
    if (!environmentConfirmed) {
      return;
    }
    void loadMapCatalog(selectedMap === "");
  }, [environmentConfirmed]);

  useEffect(() => {
    if (selectedReadyCards.length === 0) {
      setRunScenarioId("");
      return;
    }
    const found = selectedReadyCards.some((card) => card.scenario_id === runScenarioId);
    if (!found) {
      setRunScenarioId(selectedReadyCards[0].scenario_id);
    }
  }, [selectedReadyCards, runScenarioId]);

  useEffect(() => {
    if (!activeJobId) {
      setActiveJobLog([]);
      return;
    }
    void loadActiveJobLog(activeJobId);
    const timer = window.setInterval(() => {
      void loadActiveJobLog(activeJobId);
    }, activeJobLogPollMs);
    return () => window.clearInterval(timer);
  }, [activeJobId, activeJobLogPollMs]);

  useEffect(() => {
    if (!selectedExperimentId) {
      setExperimentDetail(null);
      setExperimentJobLog([]);
      return;
    }
    void loadExperimentDetail(selectedExperimentId);
    const timer = window.setInterval(() => {
      void loadExperimentDetail(selectedExperimentId);
    }, experimentDetailPollMs);
    return () => window.clearInterval(timer);
  }, [selectedExperimentId, experimentDetailPollMs]);

  useEffect(() => {
    if (!experimentFocusJob?.id) {
      setExperimentJobLog([]);
      return;
    }
    void loadExperimentJobLog(experimentFocusJob.id);
    const timer = window.setInterval(() => {
      void loadExperimentJobLog(experimentFocusJob.id);
    }, experimentJobLogPollMs);
    return () => window.clearInterval(timer);
  }, [experimentFocusJob?.id, experimentJobLogPollMs]);

  useEffect(() => {
    const failedJob = jobs.find((job) => job.status === "failed");
    if (!failedJob) {
      return;
    }
    const experimentId =
      typeof failedJob.metadata?.experiment_id === "string"
        ? (failedJob.metadata.experiment_id as string)
        : null;
    pushRuntimeAlert({
      key: `job-failed:${failedJob.id}`,
      tone: "error",
      title: "运行任务失败",
      message:
        failedJob.error ??
        "任务已失败。你可以先查看日志，再决定是否继续运行、重新运行或手动处理当前实验。",
      jobId: failedJob.id,
      experimentId
    });
  }, [jobs]);

  useEffect(() => {
    if (!activeJob || (activeJob.status !== "running" && activeJob.status !== "starting")) {
      return;
    }
    if (!activeJobLog.some((line) => line.includes("Scenario stops due to timeout"))) {
      return;
    }
    const experimentId =
      typeof activeJob.metadata?.experiment_id === "string"
        ? (activeJob.metadata.experiment_id as string)
        : null;
    pushRuntimeAlert({
      key: `job-timeout:${activeJob.id}`,
      tone: "warning",
      title: "检测到场景 timeout",
      message:
        "当前单个场景因 timeout 正常结束，系统会保存这一批的记录和结果，然后继续下一条。你可以查看日志，或请求暂停 / 停止后续运行。",
      jobId: activeJob.id,
      experimentId
    });
  }, [activeJob, activeJobLog]);

  function handleApiError(fetchError: unknown) {
    setError((fetchError as Error).message);
  }

  async function handleEnvironmentCheck() {
    setError(null);
    try {
      const response = await apiPost<{ ok: boolean; message: string; failed_checks: string[] }>(
        "/api/environment/check",
        envForm
      );
      setEnvironmentChecked(true);
      setNotice(response.message);
      await loadOptions();
      await loadMapCatalog(true);
      if (selectedMap) {
        await loadMapStatus(selectedMap);
      }
      if (response.ok) {
        setSelectedStep(1);
      } else {
        setSelectedStep(0);
      }
    } catch (fetchError) {
      handleApiError(fetchError);
    }
  }

  async function handleDisconnectCarla() {
    setError(null);
    try {
      const response = await apiPost<{
        ok: boolean;
        message: string;
        state: OptionsResponse["state"];
        failed_checks: string[];
      }>("/api/environment/disconnect", {});
      setOptions((current) => (current ? { ...current, state: response.state } : current));
      setEnvironmentChecked(true);
      setNotice(response.message);
      setSelectedStep(0);
    } catch (fetchError) {
      handleApiError(fetchError);
    }
  }

  async function handleRestoreAsync() {
    setError(null);
    try {
      const response = await apiPost<{ ok: boolean; message: string; was_sync: boolean }>(
        "/api/environment/restore-async",
        {}
      );
      setNotice(response.message);
      // Refresh environment status so UI shows updated connection state.
      await refreshEnvironmentStatus(true);
    } catch (fetchError) {
      handleApiError(fetchError);
    }
  }

  async function handleGenerateWaypoint() {
    if (!selectedMap) {
      setError("请先选择地图。");
      return;
    }
    try {
      const job = await apiPost<JobInfo>("/api/jobs/map-prepare", {
        map_name: selectedMap,
        host: envForm.carla_host,
        port: envForm.carla_port
      });
      setNotice(`已启动 waypoint 生成任务：${job.id}`);
      setTaskConsoleView("current");
      setActiveJobId(job.id);
      waypointJobIdRef.current = job.id;
      await loadJobs();
      await loadMapStatus(selectedMap);
    } catch (fetchError) {
      handleApiError(fetchError);
    }
  }

  function toggleStandard(standardId: number) {
    setSelectedStandards((current) =>
      current.includes(standardId)
        ? current.filter((value) => value !== standardId)
        : [...current, standardId].sort((left, right) => left - right)
    );
  }

  async function handleRouteEditor(scenarioId: number) {
    try {
      const job = await apiPost<JobInfo>("/api/jobs/route-editor", {
        map_name: selectedMap,
        scenario_id: scenarioId
      });
      setNotice(`已启动路线编辑器：${job.id}`);
      setTaskConsoleView("current");
      setActiveJobId(job.id);
      await loadJobs();
    } catch (fetchError) {
      handleApiError(fetchError);
    }
  }

  async function handleScenarioEditor(scenarioId: number) {
    try {
      const job = await apiPost<JobInfo>("/api/jobs/scenario-editor", {
        map_name: selectedMap,
        scenario_id: scenarioId
      });
      setNotice(`已启动场景编辑器：${job.id}`);
      setTaskConsoleView("current");
      setActiveJobId(job.id);
      await loadJobs();
    } catch (fetchError) {
      handleApiError(fetchError);
    }
  }

  async function handleExport(scenarioId: number, exportFormat: "standard" | "adv" | "both") {
    try {
      const job = await apiPost<JobInfo>("/api/jobs/export", {
        map_name: selectedMap,
        scenario_id: scenarioId,
        export_format: exportFormat
      });
      setNotice(`已启动导出任务：${job.id}`);
      setTaskConsoleView("current");
      setActiveJobId(job.id);
      await loadJobs();
    } catch (fetchError) {
      handleApiError(fetchError);
    }
  }

  async function handleOpenDir(path: string) {
    try {
      await apiPost("/api/open-dir", { path });
    } catch (fetchError) {
      handleApiError(fetchError);
    }
  }

  async function handleClearRoute(scenarioId: number) {
    if (!window.confirm(`确认清零 Scenario ${scenarioId.toString().padStart(2, "0")} 的路线数据？\n此操作同时会清除该场景的 Trigger/Actor 数据。`)) return;
    try {
      // Also clear scenario first (cascade)
      await apiDelete(`/api/maps/${encodeURIComponent(selectedMap)}/standards/${scenarioId}/scenario`);
      await apiDelete(`/api/maps/${encodeURIComponent(selectedMap)}/standards/${scenarioId}/route`);
      setNotice(`已清零 Scenario ${scenarioId.toString().padStart(2, "0")} 的路线及场景数据`);
      await loadCards(selectedMap);
    } catch (fetchError) {
      handleApiError(fetchError);
    }
  }

  async function handleClearScenario(scenarioId: number) {
    if (!window.confirm(`确认清零 Scenario ${scenarioId.toString().padStart(2, "0")} 的 Trigger/Actor 数据？`)) return;
    try {
      await apiDelete(`/api/maps/${encodeURIComponent(selectedMap)}/standards/${scenarioId}/scenario`);
      setNotice(`已清零 Scenario ${scenarioId.toString().padStart(2, "0")} 的场景数据`);
      await loadCards(selectedMap);
    } catch (fetchError) {
      handleApiError(fetchError);
    }
  }

  async function handleRun() {
    if (!selectedMap || runScenarioId === "") {
      setError("请先选择地图，并确保至少有一个标准卡片达到 Run Ready。");
      return;
    }
    try {
      const job = await apiPost<JobInfo>("/api/runs", {
        agent_name: runForm.agent_name,
        scenario_template: runForm.scenario_template,
        map_name: selectedMap,
        scenario_id: runScenarioId,
        exp_name: runForm.exp_name,
        seed: runForm.seed,
        render: runForm.render,
        save_video: runForm.save_video,
        route_id: runForm.route_id === "" ? null : Number(runForm.route_id),
        port: runForm.port,
        tm_port: runForm.tm_port,
        mode: "eval"
      });
      setNotice(`已启动实验：${job.id}`);
      setTaskConsoleView("current");
      setActiveJobId(job.id);
      setRunFormCollapsed(true);
      const newExpId =
        typeof job.metadata?.experiment_id === "string" ? (job.metadata.experiment_id as string) : null;
      setExperimentDetail(null);
      setExperimentJobLog([]);
      if (newExpId) {
        setSelectedExperimentId(newExpId);
      }
      setSelectedStep(4);
      await loadJobs();
      await loadExperiments();
    } catch (fetchError) {
      handleApiError(fetchError);
    }
  }

  async function handleResumeExperiment(experimentId: string) {
    try {
      setSelectedExperimentId(experimentId);
      setExperimentDetail(null);
      setRunFormCollapsed(true);
      setSelectedStep(4);
      const job = await apiPost<JobInfo>(`/api/experiments/${experimentId}/resume`, {});
      setNotice(`已发起续跑任务：${job.id}`);
      setTaskConsoleView("current");
      setActiveJobId(job.id);
      await loadExperimentDetail(experimentId);
      await loadJobs();
      await loadExperiments();
    } catch (fetchError) {
      handleApiError(fetchError);
    }
  }

  async function handleRerunExperiment(experimentId: string) {
    try {
      setSelectedExperimentId(experimentId);
      setExperimentDetail(null);
      setRunFormCollapsed(true);
      setSelectedStep(4);
      const job = await apiPost<JobInfo>(`/api/experiments/${experimentId}/rerun`, {
        render: true,
        save_video: true
      });
      const newExperimentId =
        typeof job.metadata?.experiment_id === "string" ? (job.metadata.experiment_id as string) : null;
      if (newExperimentId) {
        setSelectedExperimentId(newExperimentId);
      }
      setNotice(`已重新创建实验：${job.id}。将按当前默认设置开启窗口并录制视频。`);
      setTaskConsoleView("current");
      setActiveJobId(job.id);
      await loadJobs();
      await loadExperiments();
      if (newExperimentId) {
        await loadExperimentDetail(newExperimentId);
      }
    } catch (fetchError) {
      handleApiError(fetchError);
    }
  }

  async function handlePauseJob(jobId: string) {
    try {
      const job = await apiPost<JobInfo>(`/api/jobs/${jobId}/pause`, {});
      setNotice(`已请求暂停任务：${job.id}。会在当前场景完成后暂停。`);
      setTaskConsoleView("current");
      await loadJobs();
    } catch (fetchError) {
      handleApiError(fetchError);
    }
  }

  async function handlePauseFromAlert() {
    if (!runtimeAlert?.jobId) {
      return;
    }
    await handlePauseJob(runtimeAlert.jobId);
    setRuntimeAlert(null);
  }

  async function handleStopJob(jobId: string) {
    try {
      const job = await apiPost<JobInfo>(`/api/jobs/${jobId}/stop`, {});
      setNotice(`已请求停止任务：${job.id}。会在当前场景完成后停止。`);
      setTaskConsoleView("current");
      await loadJobs();
    } catch (fetchError) {
      handleApiError(fetchError);
    }
  }

  async function handleStopFromAlert() {
    if (!runtimeAlert?.jobId) {
      return;
    }
    await handleStopJob(runtimeAlert.jobId);
    setRuntimeAlert(null);
  }

  function handleContinueStandard(scenarioId: number) {
    setSelectedStandards([scenarioId]);
    setSelectedStep(3);
  }

  function handleOpenExperiment(experimentId: string) {
    setSelectedExperimentId(experimentId);
    setExperimentDetail(null);
    setStep7Tab("detail");
    setSelectedStep(5);
  }

  function handleOpenAlertJob() {
    if (!runtimeAlert?.jobId) {
      return;
    }
    setTaskConsoleView("current");
    setActiveJobId(runtimeAlert.jobId);
    setRuntimeAlert(null);
  }

  function handleOpenAlertExperiment() {
    if (!runtimeAlert?.experimentId) {
      return;
    }
    setSelectedExperimentId(runtimeAlert.experimentId);
    setStep7Tab("detail");
    setSelectedStep(5);
    setRuntimeAlert(null);
  }

  async function handleResumeFromAlert() {
    if (!runtimeAlert?.experimentId) {
      return;
    }
    const experimentId = runtimeAlert.experimentId;
    setRuntimeAlert(null);
    await handleResumeExperiment(experimentId);
  }

  function handleContinueWithExistingMapData() {
    setSelectedStep(2);
  }

  function handleOpenFirstResumeCandidate() {
    const target = unfinishedMapExperiments[0] ?? mapExperiments[0] ?? null;
    if (mapExperiments.length <= 1 && target) {
      setSelectedExperimentId(target.manifest.experiment_id);
    } else {
      setSelectedExperimentId(null);
      setExperimentDetail(null);
    }
    setStep7Tab("list");
    setSelectedStep(5);
  }

  const runningJobs = jobs.filter((j) => j.status === "running" || j.status === "starting");

  useEffect(() => {
    if (runningJobs.length > 0) {
      setRightPanel("console");
    }
  }, [runningJobs.length]);

  // Watch for waypoint job completion and refresh cards + show success notice
  useEffect(() => {
    const waypointJobId = waypointJobIdRef.current;
    if (!waypointJobId) return;
    const waypointJob = jobs.find((j) => j.id === waypointJobId);
    if (!waypointJob) return;
    if (waypointJob.status === "finished") {
      waypointJobIdRef.current = null;
      setNotice(`Waypoint 生成任务 ${waypointJobId} 已完成，数据已就绪，可以继续下一步。`);
      if (selectedMap) {
        void loadMapStatus(selectedMap);
        void loadCards(selectedMap);
      }
    } else if (waypointJob.status === "failed" || waypointJob.status === "stale") {
      waypointJobIdRef.current = null;
      setNotice(`Waypoint 生成任务 ${waypointJobId} 失败：${waypointJob.error ?? "未知错误"}。请查看任务日志后重试。`);
    }
  }, [jobs, selectedMap]);

  useEffect(() => {
    setRightPanel("guide");
  }, [selectedStep]);

  // Clear stale experiment detail when entering Step 6 so previous run data doesn't linger
  useEffect(() => {
    if (selectedStep === 4) {
      setExperimentDetail(null);
      setExperimentJobLog([]);
    }
  }, [selectedStep]);

  useEffect(() => {
    const previousStep = previousStepRef.current;
    if (selectedStep === 1 && previousStep !== 1) {
      if (latestFailedJob) {
        setTaskConsoleView("history");
        setActiveJobId(latestFailedJob.id);
      } else if (latestRunningJob) {
        setTaskConsoleView("current");
        setActiveJobId(latestRunningJob.id);
      }
    }
    previousStepRef.current = selectedStep;
  }, [latestFailedJob, latestRunningJob, selectedStep]);

  return (
    <div className="app-shell">
      <aside className="step-nav">
        <div className="brand-block">
          <p className="eyebrow">SafeBenchHK</p>
          <h1>GUI Console</h1>
          <p className="muted">零侵入任务编排与结果恢复界面</p>
        </div>

          <nav>
            {STEP_TITLES.map((title, index) => (
              <button
                key={title}
                className={`step-link ${selectedStep === index ? "active" : ""}`}
                disabled={
                  (index > 0 && !environmentConfirmed) ||
                  (index === 3 && selectedStandards.length === 0)
                }
                onClick={() => setSelectedStep(index)}
              >
                {title}
              </button>
            ))}
        </nav>
      </aside>

      <main className="main-panel">
        <section className={`panel global-toolbar tone-${globalConnectionTone}`}>
          <div className="global-toolbar-row">
            <div className="global-toolbar-main">
              <div className="global-toolbar-topline">
                <span className="global-step-chip">{STEP_TITLES[selectedStep]}</span>
                <div className="global-status-group">
                  <span
                    className={`status-pill monitor-pill ${
                      carlaSessionConnected ? "status-connected" : "status-disconnected"
                    }`}
                  >
                    GUI 会话：{carlaSessionConnected ? "已连接" : "已断开"}
                  </span>
                  <span
                    className={`status-pill monitor-pill ${
                      carlaServiceOnline ? "status-online" : "status-offline"
                    }`}
                  >
                    CARLA 服务：{carlaServiceOnline ? "在线" : "离线"}
                  </span>
                </div>
              </div>

              <div className="global-toolbar-summary">
                <article className="global-stat-card global-stat-card-map">
                  <span>地图</span>
                  <strong>{selectedMap || "未选择"}</strong>
                </article>
                <article className="global-stat-card">
                  <span>活跃任务</span>
                  <strong>{runningJobsCount}</strong>
                </article>
                <article className="global-stat-card">
                  <span>待续跑实验</span>
                  <strong>{unfinishedMapExperiments.length}</strong>
                </article>
              </div>
            </div>

            <div className="global-toolbar-side">
              <div className={`global-environment-card state-${environmentStatus.tone}`}>
                <span className={`global-environment-label state-${environmentStatus.tone}`}>
                  {environmentStatus.label}
                </span>
                <div className="global-environment-copy">
                  <strong>{environmentStatus.message}</strong>
                </div>
              </div>
              <div className="global-toolbar-actions">
                <button
                  className="button-secondary toolbar-action-button toolbar-refresh"
                  onClick={() => void refreshEnvironmentStatus(true)}
                >
                  刷新状态
                </button>
                <button
                  className="button-danger toolbar-action-button toolbar-disconnect"
                  onClick={handleDisconnectCarla}
                  disabled={!carlaSessionConnected}
                >
                  断开 CARLA 链接
                </button>
              </div>
            </div>
          </div>

          <details className="global-details">
            <summary>
              <span className="global-details-title">查看最近情况</span>
              <span className="global-details-meta">最近探测：{lastProbeAt}</span>
            </summary>
            <div className="global-detail-grid">
              <article className="global-detail-item">
                <span>Repo</span>
                <strong>{options?.repo_root ?? "加载中..."}</strong>
              </article>
              <article className="global-detail-item">
                <span>Python</span>
                <strong>{options?.state.python_exec ?? "未选择"}</strong>
              </article>
              <article className="global-detail-item">
                <span>CARLA Endpoint</span>
                <strong>
                  {options?.state.carla_host ?? envForm.carla_host}:{options?.state.carla_port ?? envForm.carla_port}
                </strong>
              </article>
              <article className="global-detail-item">
                <span>会话说明</span>
                <strong>
                  {carlaSessionConnected
                    ? "当前 GUI 会话正常，可继续使用后续面板。"
                    : "断开只会清除 GUI 会话，不会关闭 CARLA 服务进程。"}
                </strong>
              </article>
            </div>
          </details>
        </section>

        {pageNotice ? <div className="notice notice-info">{pageNotice}</div> : null}
        {error ? <div className="notice notice-error">{error}</div> : null}

        {selectedStep === 0 ? (
          <section className="panel">
            <h3>Step 0. 环境确认</h3>
            <div className="environment-setup-shell">
              <div className="environment-form-stack">
                <div className="form-grid environment-form-grid">
                  <label>
                    仓库根目录
                    <input
                      value={envForm.repo_root}
                      readOnly
                      disabled
                    />
                  </label>
                  <label>
                    Python 解释器
                    <select
                      value={envForm.python_exec}
                      onChange={(event) => setEnvForm({ ...envForm, python_exec: event.target.value })}
                    >
                      {options?.python_envs.map((item) => (
                        <option key={item.id} value={item.python_exec}>
                          {item.label} · {item.python_exec}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>

                <div className="environment-endpoint-group">
                  <span className="environment-group-label">连接参数</span>
                  <div className="environment-endpoint-row">
                    <label>
                      CARLA Host
                      <input
                        value={envForm.carla_host}
                        onChange={(event) => setEnvForm({ ...envForm, carla_host: event.target.value })}
                      />
                    </label>
                    <label>
                      CARLA Port
                      <input
                        type="number"
                        value={envForm.carla_port}
                        onChange={(event) =>
                          setEnvForm({ ...envForm, carla_port: Number(event.target.value) })
                        }
                      />
                    </label>
                    <label>
                      TM Port
                      <input
                        type="number"
                        value={envForm.tm_port}
                        onChange={(event) => setEnvForm({ ...envForm, tm_port: Number(event.target.value) })}
                      />
                    </label>
                  </div>
                </div>
              </div>

            </div>
            <div className="environment-check-action">
              <button className="environment-check-button" onClick={handleEnvironmentCheck}>
                检查 CARLA 与 Python 环境
              </button>
              {options?.state.carla_reachable ? (
                <button
                  className="btn-restore-async"
                  title="当运行任务被强制终止后，CARLA 可能卡在同步模式。点击此按钮可恢复 CARLA 为正常的异步模式。"
                  onClick={handleRestoreAsync}
                >
                  恢复 CARLA 异步模式
                </button>
              ) : null}
            </div>

            {options ? (
              <div className="result-section">
                <h4>检查结果</h4>
                <p className="muted">
                  {failedEnvironmentChecks.length === 0
                    ? "当前三项检查均已通过。"
                    : `当前未通过：${failedEnvironmentChecks.map((item) => item.label).join("、")}`}
                </p>
                <div className="env-check-grid">
                  {environmentChecks.map((item) => (
                    <article key={item.key} className={`env-check-card ${item.ok ? "ok" : "fail"}`}>
                      <div className="env-check-header">
                        <h5>{item.label}</h5>
                        <span className={`status-pill mini ${item.ok ? "env-pass" : "env-fail"}`}>
                          {item.ok ? "通过" : "未通过"}
                        </span>
                      </div>
                      <p className="path env-check-detail">{item.detail}</p>
                    </article>
                  ))}
                </div>
                {!environmentReady && options.state.error ? (
                  <div className="result-section subtle-block">
                    <h4>失败原因</h4>
                    <p className="path">{options.state.error}</p>
                  </div>
                ) : null}
              </div>
            ) : null}
          </section>
        ) : null}

        {selectedStep === 1 ? (
          <section className="panel">
            <h3>Step 1. 选择地图并生成 waypoint</h3>
            <div className="step1-map-layout">
              <section className="step1-map-panel step1-map-picker">
                <div className="section-heading">
                  <h4>地图选择</h4>
                  <span className="muted">地图列表每 20 秒自动刷新，或点击刷新按钮手动更新</span>
                </div>
                <label>
                  地图
                  <select
                    value={selectedMap}
                    onChange={(event) => setSelectedMap(event.target.value)}
                  >
                    {mapOptions.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="step1-map-selection-summary">
                  <article className="step1-map-highlight">
                    <span>当前选择</span>
                    <strong>{selectedMap || "未选择地图"}</strong>
                    <p className="muted">将作为 read waypoints 的目标地图</p>
                  </article>
  
                </div>
              </section>

              <aside className="step1-map-panel step1-map-help">
                <div className="section-heading">
                  <h4>地图来源与状态</h4>
                  <span className="muted">用于确认当前读取目标</span>
                </div>
                <p className="muted">
                  默认会优先选中当前已经加载到 CARLA 的地图；你也可以从 CARLA 安装目录扫描到的主地图中切换 waypoint 生成目标。
                </p>
                {mapCatalog ? (
                  <div className="step1-map-source-grid">
                    <article className="step1-map-source-item">
                      <span>当前 world</span>
                      <strong>{mapCatalog.current_world_map ?? "未读取"}</strong>
                    </article>
                    <article className="step1-map-source-item">
                      <span>CARLA 根目录</span>
                      <strong>{mapCatalog.carla_root ?? "未检测到"}</strong>
                    </article>
                  </div>
                ) : null}
                {mapCatalog?.current_world_error ? (
                  <details className="data-viewer compact-data-viewer">
                    <summary>查看读取异常</summary>
                    {mapCatalog.current_world_error ? (
                      <p className="path">{mapCatalog.current_world_error}</p>
                    ) : null}
                  </details>
                ) : null}
              </aside>
            </div>

            {mapStatus ? (
              <>
                {hasExistingMapState ? (
                  <section className="step1-existing-shell">
                    <div className="step1-existing-header">
                      <div>
                        <p className="eyebrow">已有数据 / 断点</p>
                        <h4>检测到这张地图已有数据</h4>
                        <p className="muted">
                          当前已就绪 {readyMapPathCount} 个基础目录，已有标准 {existingMapCards.length} 个，相关实验{" "}
                          {mapExperiments.length} 个。
                        </p>
                      </div>
                      <div className="button-row compact-row">
                        <button onClick={handleContinueWithExistingMapData}>继续使用已有数据</button>
                        {mapExperiments.length > 0 ? (
                          <button className="button-secondary" onClick={handleOpenFirstResumeCandidate}>
                            查看断点 / 续跑
                          </button>
                        ) : null}
                        <button className="button-secondary" onClick={handleGenerateWaypoint}>
                          重新生成 waypoint
                        </button>
                      </div>
                    </div>

                    <section className="step1-existing-subsection">
                      <div className="section-heading">
                        <h5>基础目录状态</h5>
                        <span className="muted">仅显示 Ready / Missing，路径收进详情里</span>
                      </div>
                      <div className="paths-grid compact-status-grid">
                        {mapPathEntries.map(([key, value]) => (
                          <article key={key} className="metric-card path-status-card">
                            <span>{key}</span>
                            <span
                              className={`status-pill mini ${
                                value.exists ? "status-run-ready" : "status-failed"
                              }`}
                            >
                              {value.exists ? "Ready" : "Missing"}
                            </span>
                            <details className="path-card-details">
                              <summary>查看详情</summary>
                              <p className="path">{value.path}</p>
                              {value.link_target ? <p className="path">→ {value.link_target}</p> : null}
                            </details>
                          </article>
                        ))}
                      </div>
                    </section>

                    <section className="step1-existing-subsection">
                      <div className="section-heading">
                        <h5>当前可复用内容</h5>
                        <span className="muted">先看标准构建，再看实验断点</span>
                      </div>
                      <div className="metrics-grid step1-summary-grid">
                        <article className="metric-card">
                          <span>Waypoint</span>
                          <strong>{mapStatus.paths.waypoints?.exists ? "已存在" : "未生成"}</strong>
                          <p className="muted">用于后续 route / scenario 构建</p>
                        </article>
                        <article className="metric-card">
                          <span>已构建标准</span>
                          <strong>{existingMapCards.length}/8</strong>
                          <p className="muted">
                            Run Ready {cards.filter((card) => card.overall_status === "Run Ready").length} 个
                          </p>
                        </article>
                        <article className="metric-card">
                          <span>未完成实验</span>
                          <strong>{unfinishedMapExperiments.length}</strong>
                          <p className="muted">可直接断点续跑</p>
                        </article>
                        <article className="metric-card">
                          <span>已完成实验</span>
                          <strong>{completedMapExperiments.length}</strong>
                          <p className="muted">可直接查看结果</p>
                        </article>
                      </div>
                    </section>

                    <div className="map-existing-layout step1-existing-groups">
                      <section className="map-existing-block">
                        <div className="section-heading">
                          <h5>已有标准数据</h5>
                          <span className="muted">如果已经有 route / scenario / export，可直接继续</span>
                        </div>
                        {existingMapCards.length === 0 ? (
                          <p className="muted">这张地图当前还没有已保存的标准构建数据。</p>
                        ) : (
                          <div className="resume-list">
                            {existingMapCards.map((card) => (
                              <article key={card.scenario_id} className="resume-item">
                                <div>
                                  <p className="eyebrow">Scenario {card.scenario_id.toString().padStart(2, "0")}</p>
                                  <h5>{card.name}</h5>
                                  <p className="muted">
                                    {card.overall_status} · route {card.route_count} · scenario {card.scenario_count}
                                  </p>
                                </div>
                                <div className="button-row compact-row">
                                  <button onClick={() => handleContinueStandard(card.scenario_id)}>继续构建</button>
                                </div>
                              </article>
                            ))}
                          </div>
                        )}
                      </section>

                      <section className="map-existing-block">
                        <div className="section-heading">
                          <h5>已有实验与断点续跑</h5>
                          <span className="muted">优先显示这张地图下尚未完成的实验</span>
                        </div>
                        {mapExperiments.length === 0 ? (
                          <p className="muted">这张地图目前还没有通过 GUI 创建的实验记录。</p>
                        ) : (
                          <div className="resume-list">
                            {mapExperiments.map((item) => (
                              <article key={item.manifest.experiment_id} className="resume-item">
                                <div>
                                  <p className="eyebrow">{item.manifest.exp_name}</p>
                                  <h5>
                                    S{item.manifest.scenario_id.toString().padStart(2, "0")} ·{" "}
                                    {item.manifest.agent_policy}
                                  </h5>
                                  <p className="muted">
                                    {item.records_count}/{item.total_data} 已完成
                                    {item.remaining_count > 0 ? ` · 剩余 ${item.remaining_count}` : " · 已完成"}
                                  </p>
                                </div>
                                <div className="button-row compact-row">
                                  <button onClick={() => handleOpenExperiment(item.manifest.experiment_id)}>
                                    查看结果
                                  </button>
                                  {item.remaining_count > 0 ? (
                                    <button onClick={() => handleResumeExperiment(item.manifest.experiment_id)}>
                                      断点续跑
                                    </button>
                                  ) : null}
                                </div>
                              </article>
                            ))}
                          </div>
                        )}
                      </section>
                    </div>
                  </section>
                ) : (
                  <section className="step1-empty-shell">
                    <div>
                      <h4>这张地图还没有现成数据</h4>
                      <p className="muted">
                        当前未检测到可复用的 waypoint、标准构建结果或实验断点；可以直接开始生成 waypoint 并建立运行侧软链接。
                      </p>
                    </div>
                    <div className="button-row compact-row">
                      <button onClick={handleGenerateWaypoint}>生成 waypoint 并建立 run_root 软链接</button>
                    </div>
                  </section>
                )}
              </>
            ) : null}
          </section>
        ) : null}

        {selectedStep === 2 ? (
          <section className="panel">
            <h3>Step 2. 选择测试标准</h3>
            <div className="standards-picker">
              {options?.standards.map((item) => (
                <label key={item.id} className="checkbox-card">
                  <input
                    type="checkbox"
                    checked={selectedStandards.includes(item.id)}
                    onChange={() => toggleStandard(item.id)}
                  />
                  <span className="checkbox-card-label">
                    {item.id}. {item.name}
                  </span>
                </label>
              ))}
            </div>
            <div className="standards-footer">
              <p className="muted">
                选中的标准会在 Step 3-5 里各自生成一张工作卡片，之后可分别重复 route / scenario / export。
              </p>
              <button
                disabled={selectedStandards.length === 0}
                onClick={() => setSelectedStep(3)}
              >
                下一步 →
              </button>
            </div>
          </section>
        ) : null}

        {selectedStep === 3 ? (
          <section className="panel">
            <h3>Step 3-5. 标准工作区</h3>
            {selectedStandards.length === 0 ? (
              <p className="muted">先回到 Step 2 勾选一个或多个标准功能场景。</p>
            ) : null}
            {/* Sub-step tabs */}
            {selectedCards.length > 0 && (
              <div className="substep-tabs">
                {(["route", "scenario", "export"] as CardSubStep[]).map((s, i) => {
                  const labels = ["Step 3 · 路线绘制", "Step 4 · Trigger/Actor", "Step 5 · 导出路线"];
                  const doneCount = s === "route"
                    ? selectedCards.filter(c => c.route_count > 0).length
                    : s === "scenario"
                    ? selectedCards.filter(c => c.scenario_count > 0 && c.sides_count > 0).length
                    : selectedCards.filter(c => c.export_status === "Export Ready").length;
                  return (
                    <button
                      key={s}
                      className={`substep-tab${cardSubStep === s ? " substep-tab--active" : ""}`}
                      onClick={() => setCardSubStep(s)}
                    >
                      {labels[i]}
                      <span className="substep-tab-badge">{doneCount}/{selectedCards.length}</span>
                    </button>
                  );
                })}
              </div>
            )}
            <div className="cards-grid">
              {selectedCards.map((card) => (
                <StandardCard
                  key={card.scenario_id}
                  card={card}
                  subStep={cardSubStep}
                  onRoute={handleRouteEditor}
                  onScenario={handleScenarioEditor}
                  onExport={handleExport}
                  onClearRoute={handleClearRoute}
                  onClearScenario={handleClearScenario}
                  onOpenDir={handleOpenDir}
                />
              ))}
            </div>
            {selectedCards.length > 0 && selectedReadyCards.length === selectedCards.length && (
              <div className="standards-footer">
                <p className="muted">
                  全部 {selectedCards.length} 个标准均已达到 Run Ready，可以进入 Step 6 开始评测。
                </p>
                <button onClick={() => setSelectedStep(4)}>前往 Step 6 · 运行中心 →</button>
              </div>
            )}
          </section>
        ) : null}

        {selectedStep === 4 ? (
          <section className="panel">
            <h3>Step 6. 运行中心</h3>
            {selectedStaleCards.length > 0 && (
              <div className="notice notice-warning">
                以下标准导出已过期（源路线/场景数据已更新但未重新导出）：
                {selectedStaleCards.map((card) => `S${card.scenario_id.toString().padStart(2, "0")} ${card.name}`).join("、")}
                。请返回 Step 5 重新导出后再运行。
              </div>
            )}
            {runFormCollapsed ? (
              <div className="run-form-summary">
                <div className="run-form-summary-chips">
                  <span className="task-chip">{runForm.agent_name}</span>
                  <span className="task-chip">{runForm.scenario_template}</span>
                  {runScenarioId !== "" && <span className="task-chip">S{String(runScenarioId).padStart(2, "0")}</span>}
                  <span className="task-chip">{runForm.exp_name}</span>
                  <span className="task-chip">seed {runForm.seed}</span>
                  {runForm.route_id && <span className="task-chip">route {runForm.route_id}</span>}
                  {runForm.render && <span className="task-chip">render</span>}
                  {runForm.save_video && <span className="task-chip">video</span>}
                </div>
                <button
                  type="button"
                  className="secondary-button run-form-expand-btn"
                  onClick={() => {
                    setExperimentDetail(null);
                    setExperimentJobLog([]);
                    setRunFormCollapsed(false);
                  }}
                >
                  修改配置 ▾
                </button>
              </div>
            ) : (
              <>
                <div className="form-grid">
                  <label>
                    Agent
                    <select
                      value={runForm.agent_name}
                      onChange={(event) => setRunForm({ ...runForm, agent_name: event.target.value })}
                    >
                      {options?.agents.map((item) => (
                        <option key={item.id} value={item.id}>
                          {item.id} ({item.policy_type})
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    运行模板
                    <select
                      value={runForm.scenario_template}
                      onChange={(event) =>
                        setRunForm({ ...runForm, scenario_template: event.target.value })
                      }
                    >
                      {options?.scenario_templates.map((item) => (
                        <option key={item.id} value={item.id}>
                          {item.id} ({item.policy_type})
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    目标标准
                    <select
                      id="run-standard-select"
                      value={runScenarioId}
                      onChange={(event) =>
                        setRunScenarioId(event.target.value === "" ? "" : Number(event.target.value))
                      }
                    >
                      <option value="">请选择</option>
                      {selectedReadyCards.map((card) => (
                        <option key={card.scenario_id} value={card.scenario_id}>
                          {card.scenario_id}. {card.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    实验名
                    <input
                      value={runForm.exp_name}
                      onChange={(event) => setRunForm({ ...runForm, exp_name: event.target.value })}
                    />
                  </label>
                  <label>
                    Seed
                    <input
                      type="number"
                      value={runForm.seed}
                      onChange={(event) => setRunForm({ ...runForm, seed: Number(event.target.value) })}
                    />
                  </label>
                  <label>
                    Route ID
                    <input
                      placeholder="默认 null"
                      value={runForm.route_id}
                      onChange={(event) => setRunForm({ ...runForm, route_id: event.target.value })}
                    />
                  </label>
                  <label>
                    CARLA Port
                    <input
                      type="number"
                      value={runForm.port}
                      onChange={(event) => setRunForm({ ...runForm, port: Number(event.target.value) })}
                    />
                  </label>
                  <label>
                    TM Port
                    <input
                      type="number"
                      value={runForm.tm_port}
                      onChange={(event) => setRunForm({ ...runForm, tm_port: Number(event.target.value) })}
                    />
                  </label>
                  <label className="switch">
                    <input
                      type="checkbox"
                      checked={runForm.render}
                      onChange={(event) => setRunForm({ ...runForm, render: event.target.checked })}
                    />
                    <span>Render</span>
                  </label>
                  <label className="switch">
                    <input
                      type="checkbox"
                      checked={runForm.save_video}
                      onChange={(event) => setRunForm({ ...runForm, save_video: event.target.checked })}
                    />
                    <span>Save Video</span>
                  </label>
                </div>
                <div className="button-row">
                  <button onClick={handleRun} disabled={selectedReadyCards.length === 0}>
                    启动评测
                  </button>
                </div>
              </>
            )}
            {experimentDetail ? (
              <section className="run-center-monitor">
                <div className="run-center-monitor-header">
                  <div>
                    <p className="eyebrow">实时监控</p>
                    <h4>{experimentDetail.manifest.exp_name}</h4>
                  </div>
                  <div className="run-center-monitor-meta">
                    <span>{experimentDetail.manifest.map} · S{experimentDetail.manifest.scenario_id.toString().padStart(2, "0")} · seed {experimentDetail.manifest.seed}</span>
                    {experimentFocusJob && (
                      <span className={`status-pill mini status-${experimentFocusJob.status}`}>
                        {experimentFocusJob.type} · {experimentFocusJob.status}
                        {experimentFocusJob.pid ? ` · PID ${experimentFocusJob.pid}` : ""}
                      </span>
                    )}
                  </div>
                </div>
                <div className="run-center-monitor-progress">
                  <strong>{experimentDetail.progress.records_count} / {experimentDetail.progress.total_data}</strong>
                  <span>已完成 · 剩余 {experimentDetail.progress.remaining_count} 条</span>
                </div>
                {Object.keys(experimentDetail.results).length > 0 && (
                  <div className="metrics-grid run-center-metrics">
                    {Object.entries(experimentDetail.results).map(([key, value]) => (
                      <article key={key} className="metric-card">
                        <span>{key}</span>
                        <strong>{value}</strong>
                      </article>
                    ))}
                  </div>
                )}
                {experimentFocusJob?.error && (
                  <div className="experiment-log-alert error">
                    <strong>任务返回错误</strong>
                    <p>{experimentFocusJob.error}</p>
                  </div>
                )}
                {experimentDetail.progress.probe_error && (
                  <div className="experiment-log-alert error">
                    <strong>结果探测异常</strong>
                    <p>{experimentDetail.progress.probe_error}</p>
                  </div>
                )}
                {experimentJobLog.length > 0 && (
                  <div className="log-panel run-center-log">
                    <pre>{experimentJobLog.slice(-40).join("\n")}</pre>
                  </div>
                )}
                <div className="button-row">
                  <button
                    className="secondary-button"
                    onClick={() => handleOpenExperiment(experimentDetail.manifest.experiment_id)}
                  >
                    实验记录 →
                  </button>
                  {experimentFocusJob && (
                    <button
                      className="secondary-button danger-button"
                      onClick={() => handleStopJob(experimentFocusJob.id)}
                      disabled={experimentFocusJob.status !== "running" && experimentFocusJob.status !== "starting"}
                    >
                      停止任务
                    </button>
                  )}
                  <button
                    className="secondary-button"
                    onClick={() => handleRerunExperiment(experimentDetail.manifest.experiment_id)}
                  >
                    重新运行
                  </button>
                  <button
                    onClick={() => handleResumeExperiment(experimentDetail.manifest.experiment_id)}
                    disabled={!experimentDetail.progress.resume_ready}
                  >
                    继续运行
                  </button>
                </div>
              </section>
            ) : (
              <p className="muted">
                启动评测后将在此处显示实时进度与日志。续跑请到 Step 7 实验记录。
              </p>
            )}
          </section>
        ) : null}

        {selectedStep === 5 ? (
          <section className="panel">
            <h3>Step 7. 实验记录</h3>
            <ExperimentList
              experiments={experiments}
              selectedExperimentId={selectedExperimentId}
              detail={experimentDetail}
              activeTab={step7Tab}
              onSelect={setSelectedExperimentId}
              onResume={handleResumeExperiment}
              onRerun={handleRerunExperiment}
              onTabChange={setStep7Tab}
            />
          </section>
        ) : null}
      </main>

      <div className="right-rail">
        <div className="right-rail-toggle">
          <div className="task-console-switch">
            <button
              className={`task-tab-button ${rightPanel === "guide" ? "active" : ""}`}
              onClick={() => setRightPanel("guide")}
            >
              操作指引
            </button>
            <button
              className={`task-tab-button ${rightPanel === "console" ? "active" : ""}${runningJobs.length > 0 ? " has-badge" : ""}`}
              onClick={() => setRightPanel("console")}
            >
              任务监控{runningJobs.length > 0 ? <span className="rail-badge">{runningJobs.length}</span> : null}
            </button>
          </div>
        </div>
        {rightPanel === "guide" ? (
          <div className="step-guide-panel">
            {STEP_GUIDES[selectedStep]}
          </div>
        ) : (
          <TaskConsole
            jobs={jobs}
            activeJobId={activeJobId}
            activeJobLog={activeJobLog}
            viewMode={taskConsoleView}
            onSelectJob={setActiveJobId}
            onChangeView={setTaskConsoleView}
            onPauseJob={handlePauseJob}
            onStopJob={handleStopJob}
          />
        )}
      </div>

      {runtimeAlert ? (
        <div className="app-modal-overlay" onClick={() => setRuntimeAlert(null)}>
          <div className={`app-modal tone-${runtimeAlert.tone}`} onClick={(event) => event.stopPropagation()}>
            <div className="app-modal-header">
              <div>
                <span className={`status-pill mini ${runtimeAlert.tone === "error" ? "env-fail" : "env-pass"}`}>
                  {runtimeAlert.tone === "error" ? "错误" : "警告"}
                </span>
                <h3>{runtimeAlert.title}</h3>
              </div>
              <button type="button" className="secondary-button" onClick={() => setRuntimeAlert(null)}>
                关闭
              </button>
            </div>
            <p className="app-modal-message">{runtimeAlert.message}</p>
            <div className="app-modal-meta">
              {runtimeAlert.jobId ? <span>任务：{runtimeAlert.jobId}</span> : null}
              {runtimeAlert.experimentId ? <span>实验：{runtimeAlert.experimentId}</span> : null}
            </div>
            <div className="app-modal-actions">
              {runtimeAlert.jobId ? (
                <button type="button" className="secondary-button" onClick={handleOpenAlertJob}>
                  查看任务日志
                </button>
              ) : null}
              {runtimeAlert.experimentId ? (
                <button type="button" className="secondary-button" onClick={handleOpenAlertExperiment}>
                  打开实验详情
                </button>
              ) : null}
              {canControlAlertJob ? (
                <button type="button" className="secondary-button" onClick={handlePauseFromAlert}>
                  暂停当前任务
                </button>
              ) : null}
              {canControlAlertJob ? (
                <button type="button" className="danger-button" onClick={handleStopFromAlert}>
                  停止当前任务
                </button>
              ) : null}
              {canResumeAlertExperiment ? (
                <button type="button" onClick={handleResumeFromAlert}>
                  继续运行
                </button>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default App;
