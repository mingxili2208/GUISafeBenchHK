import { useEffect, useMemo, useRef } from "react";

import { ExperimentDetail, ExperimentSummary, JobInfo } from "../types";

interface ExperimentListProps {
  experiments: ExperimentSummary[];
  selectedExperimentId: string | null;
  detail: ExperimentDetail | null;
  focusJob: JobInfo | null;
  focusJobLog: string[];
  onSelect: (experimentId: string) => void;
  onResume: (experimentId: string) => void;
  onRerun: (experimentId: string) => void;
}

function getExperimentState(item: ExperimentSummary) {
  if (item.active_job?.status === "running" || item.active_job?.status === "starting") {
    return { label: "运行中", className: "status-running" };
  }
  if (item.remaining_count > 0) {
    return { label: "待续跑", className: "status-paused" };
  }
  return { label: "已完成", className: "status-succeeded" };
}

function formatScenarioLabel(scenarioId: number) {
  return `S${scenarioId.toString().padStart(2, "0")}`;
}

function extractErrorExcerpt(lines: string[]) {
  const patterns = [
    /Traceback/i,
    /Exception/i,
    /Error:/i,
    /RuntimeError/i,
    /AssertionError/i,
    /failed!/i,
    /CRITICAL/i,
    /SIGSEGV/i,
    /Segmentation fault/i,
    /TimeoutException/i,
    /Killed/i,
    /Aborted/i
  ];
  const index = [...lines].reverse().findIndex((line) => patterns.some((pattern) => pattern.test(line)));
  if (index === -1) {
    return [];
  }
  const matchedLineIndex = lines.length - 1 - index;
  const start = Math.max(0, matchedLineIndex - 4);
  const end = Math.min(lines.length, matchedLineIndex + 16);
  return lines.slice(start, end);
}

export function ExperimentList({
  experiments,
  selectedExperimentId,
  detail,
  focusJob,
  focusJobLog,
  onSelect,
  onResume,
  onRerun
}: ExperimentListProps) {
  const liveLogRef = useRef<HTMLDivElement | null>(null);
  const selectedSummary = useMemo(
    () => experiments.find((item) => item.manifest.experiment_id === selectedExperimentId) ?? null,
    [experiments, selectedExperimentId]
  );
  const hasSelection = Boolean(selectedSummary);
  const queueExperiments = useMemo(
    () =>
      hasSelection && selectedSummary
        ? experiments.filter((item) => item.manifest.experiment_id !== selectedSummary.manifest.experiment_id)
        : experiments,
    [experiments, hasSelection, selectedSummary]
  );
  const focusJobRunning = focusJob?.status === "running" || focusJob?.status === "starting";
  const displayLogLines = focusJob ? focusJobLog : detail?.runtime_log_tail ?? [];
  const errorExcerpt = useMemo(() => extractErrorExcerpt(displayLogLines), [displayLogLines]);
  const showErrorState =
    Boolean(focusJob?.error) ||
    Boolean(detail?.progress.probe_error) ||
    focusJob?.status === "failed" ||
    focusJob?.status === "stale" ||
    errorExcerpt.length > 0;
  const spotlightMetrics = useMemo(() => Object.entries(detail?.results ?? {}).slice(0, 2), [detail?.results]);
  const spotlightState = selectedSummary ? getExperimentState(selectedSummary) : null;

  useEffect(() => {
    if (!liveLogRef.current) {
      return;
    }
    liveLogRef.current.scrollTop = liveLogRef.current.scrollHeight;
  }, [focusJob?.id, focusJobLog]);

  return (
    <div className={`experiment-layout ${hasSelection ? "has-selection" : ""}`}>
      <aside className={`experiment-sidebar ${hasSelection ? "has-selection" : ""}`}>
        {detail && selectedSummary ? (
          <section className="experiment-current-card">
            <div className="experiment-current-top">
              <div>
                <p className="eyebrow">当前实验</p>
                <h4>{detail.manifest.exp_name}</h4>
              </div>
              {spotlightState ? (
                <span className={`status-pill mini ${spotlightState.className}`}>{spotlightState.label}</span>
              ) : null}
            </div>

            <p className="experiment-current-title">{detail.manifest.scenario_name}</p>

            <div className="experiment-current-progress">
              <strong>
                {detail.progress.records_count}/{detail.progress.total_data}
              </strong>
              <span>剩余 {detail.progress.remaining_count}</span>
            </div>

            <div className="experiment-current-meta">
              <span>{detail.manifest.map}</span>
              <span>{formatScenarioLabel(detail.manifest.scenario_id)}</span>
              <span>seed {detail.manifest.seed}</span>
            </div>

            <div className="experiment-current-meta">
              <span>{detail.manifest.agent_policy}</span>
              <span>{detail.manifest.scenario_policy}</span>
              <span>
                {focusJob ? `${focusJob.type} · ${focusJob.status}` : detail.progress.resume_ready ? "可续跑" : "已完成"}
              </span>
            </div>

            {spotlightMetrics.length > 0 ? (
              <div className="experiment-current-metrics">
                {spotlightMetrics.map(([key, value]) => (
                  <article key={key} className="experiment-current-metric">
                    <span>{key}</span>
                    <strong>{value}</strong>
                  </article>
                ))}
              </div>
            ) : (
              <p className="muted experiment-current-note">右侧主区域会优先展示这次实验的实时输出、结果和续跑操作。</p>
            )}
          </section>
        ) : null}

        {hasSelection ? (
          <div className="experiment-queue-header">
            <h4>其它实验</h4>
            <span className="muted">已压缩显示，点击即可切换</span>
          </div>
        ) : null}

        <div className={`experiment-list ${hasSelection ? "has-selection" : ""}`}>
          {experiments.length === 0 ? <p className="muted">还没有通过 GUI 创建的实验。</p> : null}
          {hasSelection && queueExperiments.length === 0 ? <p className="muted">当前没有其它实验可切换。</p> : null}
          {queueExperiments.map((item) => {
            const itemState = getExperimentState(item);
            return (
              <button
                key={item.manifest.experiment_id}
                className={`experiment-item ${hasSelection ? "collapsed" : ""}`}
                onClick={() => onSelect(item.manifest.experiment_id)}
              >
                <div className="experiment-item-main">
                  <div className="experiment-item-top">
                    <p className="eyebrow">{item.manifest.exp_name}</p>
                    <span className={`status-pill mini ${itemState.className}`}>{itemState.label}</span>
                  </div>
                  <h4>{item.manifest.scenario_name}</h4>
                  <p className="muted">
                    {item.manifest.map} · S{item.manifest.scenario_id.toString().padStart(2, "0")} · seed {item.manifest.seed}
                  </p>
                  <p className="muted experiment-item-secondary">
                    {item.manifest.agent_policy} / {item.manifest.scenario_policy}
                  </p>
                </div>
                <div className="experiment-progress">
                  <strong>
                    {item.records_count}/{item.total_data}
                  </strong>
                  <span>剩余 {item.remaining_count}</span>
                </div>
              </button>
            );
          })}
        </div>
      </aside>

      <div className="experiment-detail experiment-detail-focus">
        {detail ? (
          <>
            <section className="experiment-focus-hero">
              <header className="card-header experiment-hero-header">
                <div className="experiment-hero-copy">
                  <p className="eyebrow">{detail.manifest.experiment_id}</p>
                  <h3>
                    {detail.manifest.exp_name} · {detail.manifest.scenario_name}
                  </h3>
                  <p className="muted">
                    当前实验已前置展示，右侧主区域优先显示实时输出、关键结果和续跑操作。
                  </p>
                </div>
                <div className="experiment-detail-actions">
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => onRerun(detail.manifest.experiment_id)}
                  >
                    重新运行
                  </button>
                  <button
                    onClick={() => onResume(detail.manifest.experiment_id)}
                    disabled={!detail.progress.resume_ready}
                  >
                    继续运行
                  </button>
                </div>
              </header>

              <div className="experiment-focus-stats">
                <article className="experiment-focus-card">
                  <span>地图 / 标准</span>
                  <strong>{detail.manifest.map}</strong>
                  <p className="muted">
                    S{detail.manifest.scenario_id.toString().padStart(2, "0")} · seed {detail.manifest.seed}
                  </p>
                </article>
                <article className="experiment-focus-card">
                  <span>完成进度</span>
                  <strong>
                    {detail.progress.records_count}/{detail.progress.total_data}
                  </strong>
                  <p className="muted">剩余 {detail.progress.remaining_count}</p>
                </article>
                <article className="experiment-focus-card">
                  <span>关联任务</span>
                  <strong>{focusJob ? focusJob.type : "暂无"}</strong>
                  <p className="muted">
                    {focusJob ? `${focusJob.status}${focusJob.pid ? ` · PID ${focusJob.pid}` : ""}` : "当前实验没有关联 job"}
                  </p>
                </article>
              </div>

              <aside className="experiment-snapshot-rail">
                <div className="experiment-snapshot-rail-head">
                  <h4>配置快照</h4>
                  <span className="muted">进一步压缩，只保留运行时最常回看的关键信息</span>
                </div>

                <div className="experiment-snapshot-inline-grid">
                  <article className="experiment-snapshot-mini">
                    <span>Agent / 模板</span>
                    <strong>{detail.manifest.agent_policy}</strong>
                    <p className="muted">{detail.manifest.scenario_policy}</p>
                  </article>
                  <article className="experiment-snapshot-mini">
                    <span>Seed / Route</span>
                    <strong>seed {detail.manifest.seed}</strong>
                    <p className="muted">
                      {detail.manifest.route_id === null || detail.manifest.route_id === undefined
                        ? "使用全部 route"
                        : `route ${detail.manifest.route_id}`}
                    </p>
                  </article>
                  <article className="experiment-snapshot-mini">
                    <span>运行端口</span>
                    <strong>{detail.manifest.port}</strong>
                    <p className="muted">tm {detail.manifest.tm_port}</p>
                  </article>
                  <article className="experiment-snapshot-mini">
                    <span>视频 / 渲染</span>
                    <strong>{detail.manifest.save_video ? "录制开启" : "未录制"}</strong>
                    <p className="muted">{detail.manifest.render ? "渲染窗口开启" : "无可视化窗口"}</p>
                  </article>
                </div>

                <details className="data-viewer experiment-snapshot-details">
                  <summary>查看配置路径与原始 manifest</summary>
                  <div className="data-viewer-stack">
                    <div className="task-inline-block">
                      <h4>运行时配置副本</h4>
                      <p className="path">{detail.manifest.run_root_agent_cfg}</p>
                      <p className="path">{detail.manifest.run_root_scenario_cfg}</p>
                    </div>
                    <div className="task-inline-block">
                      <h4>实验输出路径</h4>
                      <p className="path">{detail.manifest.output_dir}</p>
                      <p className="path">{detail.manifest.runtime_log_path}</p>
                    </div>
                    <pre>{JSON.stringify(detail.manifest, null, 2)}</pre>
                  </div>
                </details>
              </aside>
            </section>

            <section className={`result-section experiment-live-section ${showErrorState ? "has-error" : ""}`}>
              <div className="section-heading">
                <h4>{focusJobRunning ? "当前运行 stdout / stderr" : "最近任务 stdout / stderr"}</h4>
                {focusJob ? (
                  <span className={`status-pill mini status-${focusJob.status}`}>
                    {focusJob.type} · {focusJob.status}
                  </span>
                ) : (
                  <span className="muted">暂无关联任务，先显示实验日志</span>
                )}
              </div>

              {focusJob ? (
                <>
                  <div className="experiment-job-meta">
                    <span className="task-chip">任务 {focusJob.id}</span>
                    <span className="task-chip">{focusJob.process_name ?? "未命名进程"}</span>
                    <span className="task-chip">日志 {focusJob.log_path}</span>
                  </div>

                  {focusJob.error ? (
                    <div className="experiment-log-alert error">
                      <strong>任务返回错误</strong>
                      <p>{focusJob.error}</p>
                    </div>
                  ) : null}

                  {detail.progress.probe_error ? (
                    <div className="experiment-log-alert error">
                      <strong>结果探测异常</strong>
                      <p>{detail.progress.probe_error}</p>
                    </div>
                  ) : null}

                  {errorExcerpt.length > 0 ? (
                    <details className="data-viewer experiment-error-viewer" open>
                      <summary>展开最近捕获的错误片段</summary>
                      <pre>{errorExcerpt.join("\n")}</pre>
                    </details>
                  ) : null}

                  <div ref={liveLogRef} className="log-panel experiment-live-log">
                    <pre>{focusJobLog.length > 0 ? focusJobLog.join("\n") : "暂无日志输出。"}</pre>
                  </div>
                </>
              ) : (
                <div className="log-panel experiment-live-log">
                  <pre>
                    {displayLogLines.length > 0
                      ? displayLogLines.join("\n")
                      : "当前实验还没有关联运行任务或实时日志。"}
                  </pre>
                </div>
              )}
            </section>

            <section className="metrics-grid experiment-results-grid">
              {Object.entries(detail.results).map(([key, value]) => (
                <article key={key} className="metric-card">
                  <span>{key}</span>
                  <strong>{value}</strong>
                </article>
              ))}
            </section>

            <section className="result-section">
              <h4>停止原因统计</h4>
              <div className="tag-row">
                {Object.entries(detail.stop_reason_counts).map(([key, value]) => (
                  <span key={key} className="result-tag">
                    {key}: {value}
                  </span>
                ))}
              </div>
            </section>

            <section className="result-section">
              <div className="section-heading">
                <h4>最近 Batch 摘要</h4>
                <span className="muted">最近 5 条</span>
              </div>
              <details className="data-viewer">
                <summary>展开原始内容</summary>
                <pre>{JSON.stringify(detail.batch_summaries.slice(-5), null, 2)}</pre>
              </details>
            </section>

            <section className="result-section">
              <div className="section-heading">
                <h4>实验 runtime.log 尾部</h4>
                <span className="muted">保留为实验级补充日志，避免和上面的实时输出重复</span>
              </div>
              <details className="data-viewer">
                <summary>展开 runtime.log</summary>
                <pre>{detail.runtime_log_tail.join("\n")}</pre>
              </details>
            </section>
          </>
        ) : (
          <p className="muted">选择一个实验后可查看详情与续跑状态。</p>
        )}
      </div>
    </div>
  );
}
