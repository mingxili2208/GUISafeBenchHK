import { useEffect, useMemo, useRef } from "react";
import { JobInfo } from "../types";

interface TaskConsoleProps {
  jobs: JobInfo[];
  activeJobId: string | null;
  activeJobLog: string[];
  viewMode: "current" | "history";
  onSelectJob: (jobId: string) => void;
  onChangeView: (view: "current" | "history") => void;
  onPauseJob: (jobId: string) => void;
  onStopJob: (jobId: string) => void;
}

export function TaskConsole({
  jobs,
  activeJobId,
  activeJobLog,
  viewMode,
  onSelectJob,
  onChangeView,
  onPauseJob,
  onStopJob
}: TaskConsoleProps) {
  const logPanelRef = useRef<HTMLDivElement | null>(null);
  const currentJobs = useMemo(
    () => jobs.filter((job) => job.status === "running" || job.status === "starting"),
    [jobs]
  );
  const historyJobs = useMemo(
    () => jobs.filter((job) => job.status !== "running" && job.status !== "starting"),
    [jobs]
  );
  const effectiveView =
    viewMode === "current"
      ? currentJobs.length > 0
        ? "current"
        : "history"
      : historyJobs.length > 0
        ? "history"
        : "current";
  const visibleJobs = effectiveView === "current" ? currentJobs : historyJobs;
  const shouldRenderHistoryList = effectiveView === "history";

  useEffect(() => {
    if (visibleJobs.length === 0) {
      return;
    }
    if (!activeJobId || !visibleJobs.some((job) => job.id === activeJobId)) {
      onSelectJob(visibleJobs[0].id);
    }
  }, [activeJobId, onSelectJob, viewMode, visibleJobs]);

  const activeJob = visibleJobs.find((job) => job.id === activeJobId) ?? visibleJobs[0] ?? null;
  const canControlActiveJob = Boolean(
    activeJob?.supports_control && (activeJob.status === "running" || activeJob.status === "starting")
  );
  const pendingControlLabel =
    activeJob?.control_requested === "pause"
      ? "已请求暂停，会在当前场景完成后暂停。"
      : activeJob?.control_requested === "stop"
        ? "已请求停止，会在当前场景完成后停止。"
        : null;

  useEffect(() => {
    if (!logPanelRef.current) {
      return;
    }
    logPanelRef.current.scrollTop = logPanelRef.current.scrollHeight;
  }, [activeJob?.id, activeJobLog]);

  return (
    <aside className="task-console">
      <div className="task-console-header">
        <p className="eyebrow">Task Console</p>
        <h2>任务监控</h2>
      </div>

      {currentJobs.length > 0 ? (
        <section className="task-live-section">
          <div className="task-section-head">
            <strong>当前运行</strong>
            <span className="task-section-count">{currentJobs.length}</span>
          </div>
          <div className="task-list task-list-current">
            {currentJobs.map((job) => (
              <button
                key={job.id}
                className={`task-item ${activeJob?.id === job.id ? "active" : ""}`}
                onClick={() => {
                  onChangeView("current");
                  onSelectJob(job.id);
                }}
              >
                <div className="task-item-head">
                  <span className="task-type">{job.type}</span>
                  <span className={`status-pill mini status-${job.status}`}>{job.status}</span>
                </div>
                <span className="task-id">{job.id}</span>
              </button>
            ))}
          </div>
        </section>
      ) : null}

      <div className="task-console-toolbar">
        <div className="task-console-switch" role="tablist" aria-label="任务列表视图">
          <button
            type="button"
            className={`task-tab-button ${effectiveView === "current" ? "active" : ""}`}
            onClick={() => onChangeView("current")}
            disabled={currentJobs.length === 0}
          >
            当前
          </button>
          <button
            type="button"
            className={`task-tab-button ${effectiveView === "history" ? "active" : ""}`}
            onClick={() => onChangeView("history")}
            disabled={historyJobs.length === 0}
          >
            历史
          </button>
        </div>
        <span className="task-toolbar-note">
          {effectiveView === "current"
            ? currentJobs.length > 0
              ? "当前视图直接显示正在运行任务的详情。"
              : "当前没有正在运行的任务。"
            : `历史记录 ${historyJobs.length} 条`}
        </span>
      </div>

      {jobs.length === 0 ? <p className="muted">当前还没有任务。</p> : null}

      {effectiveView === "current" && currentJobs.length > 0 ? (
        <div className="task-current-summary">
          <p className="muted">上方卡片用于切换当前任务，这里不再重复显示同一批运行中任务。</p>
        </div>
      ) : null}

      {shouldRenderHistoryList ? (
        <div className="task-history-section">
          <div className="task-section-head">
            <strong>历史记录</strong>
            <span className="task-section-count">{historyJobs.length}</span>
          </div>
          <div className="task-list">
            {historyJobs.length === 0 ? <p className="muted">历史分组里暂时没有任务。</p> : null}
            {historyJobs.map((job) => (
              <button
                key={job.id}
                className={`task-item ${activeJob?.id === job.id ? "active" : ""}`}
                onClick={() => {
                  onChangeView("history");
                  onSelectJob(job.id);
                }}
              >
                <div className="task-item-head">
                  <span className="task-type">{job.type}</span>
                  <span className={`status-pill mini status-${job.status}`}>{job.status}</span>
                </div>
                <span className="task-id">{job.id}</span>
              </button>
            ))}
          </div>
        </div>
      ) : null}

      {activeJob ? (
        <div className="task-detail">
          <div className="task-detail-header">
            <div>
              <p className="eyebrow">{effectiveView === "current" ? "当前任务" : "历史任务"}</p>
              <h3>{activeJob.type}</h3>
            </div>
            <span className={`status-pill status-${activeJob.status}`}>{activeJob.status}</span>
          </div>

          <div className="task-chip-row">
            <span className="task-chip">PID {activeJob.pid ?? "N/A"}</span>
            <span className="task-chip">{activeJob.process_name ?? "未命名进程"}</span>
          </div>

          {activeJob.error ? <p className="task-error-summary">{activeJob.error}</p> : null}

          {canControlActiveJob ? (
            <div className="task-control-card">
              <div className="task-control-header">
                <strong>运行控制</strong>
                <span className="task-control-note">会等待当前 route / 场景完成后再停下</span>
              </div>
              <div className="task-control-row">
                <button
                  type="button"
                  className="secondary-button task-control-button"
                  disabled={Boolean(activeJob.control_requested)}
                  onClick={() => onPauseJob(activeJob.id)}
                >
                  暂停
                </button>
                <button
                  type="button"
                  className="danger-button task-control-button"
                  disabled={Boolean(activeJob.control_requested)}
                  onClick={() => onStopJob(activeJob.id)}
                >
                  停止
                </button>
              </div>
              {pendingControlLabel ? <p className="task-control-state">{pendingControlLabel}</p> : null}
            </div>
          ) : null}

          <dl className="detail-grid compact-detail-grid">
            <dt>ID</dt>
            <dd>{activeJob.id}</dd>
            <dt>CWD</dt>
            <dd className="path">{activeJob.cwd}</dd>
          </dl>

          <details className="data-viewer">
            <summary>查看命令与日志路径</summary>
            <div className="data-viewer-stack">
              <div className="task-inline-block">
                <h4>Command</h4>
                <p className="path">{activeJob.command}</p>
              </div>
              <div className="task-inline-block">
                <h4>Log</h4>
                <p className="path">{activeJob.log_path}</p>
              </div>
            </div>
          </details>

          <details className="data-viewer" open>
            <summary>stdout / stderr</summary>
            <div ref={logPanelRef} className="log-panel compact-log-panel">
              <pre>{activeJobLog.length > 0 ? activeJobLog.join("\n") : "暂无日志输出。"}</pre>
            </div>
          </details>
        </div>
      ) : null}
    </aside>
  );
}
