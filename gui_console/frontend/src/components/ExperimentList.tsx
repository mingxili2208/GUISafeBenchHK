import { useMemo } from "react";

import { apiPost } from "../api";
import { ExperimentDetail, ExperimentSummary } from "../types";

export type Step7Tab = "list" | "detail";

interface ExperimentListProps {
  experiments: ExperimentSummary[];
  selectedExperimentId: string | null;
  detail: ExperimentDetail | null;
  activeTab: Step7Tab;
  onSelect: (experimentId: string) => void;
  onResume: (experimentId: string) => void;
  onRerun: (experimentId: string) => void;
  onTabChange: (tab: Step7Tab) => void;
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
  activeTab,
  onSelect,
  onResume,
  onRerun,
  onTabChange
}: ExperimentListProps) {
  const selectedSummary = useMemo(
    () => experiments.find((item) => item.manifest.experiment_id === selectedExperimentId) ?? null,
    [experiments, selectedExperimentId]
  );

  return (
    <div className="experiment-layout">
      {/* ── Tab bar ── */}
      <div className="exp-tab-bar">
        <button
          className={`exp-tab ${activeTab === "list" ? "exp-tab--active" : ""}`}
          onClick={() => onTabChange("list")}
        >
          实验列表
          {experiments.length > 0 && (
            <span className="exp-tab-count">{experiments.length}</span>
          )}
        </button>
        <button
          className={`exp-tab ${activeTab === "detail" ? "exp-tab--active" : ""}`}
          disabled={!selectedSummary}
          onClick={() => onTabChange("detail")}
        >
          {selectedSummary ? `详情 · ${selectedSummary.manifest.exp_name}` : "详情"}
        </button>
      </div>

      {/* ── LIST ── */}
      {activeTab === "list" && (
        <div className="exp-panel">
          {experiments.length === 0 ? (
            <p className="muted">还没有通过 GUI 创建的实验。</p>
          ) : (
            <div className="experiment-list">
              {experiments.map((item) => {
                const itemState = getExperimentState(item);
                const isSelected = item.manifest.experiment_id === selectedExperimentId;
                return (
                  <button
                    key={item.manifest.experiment_id}
                    className={`experiment-item${isSelected ? " experiment-item--selected" : ""}`}
                    onClick={() => {
                      onSelect(item.manifest.experiment_id);
                      onTabChange("detail");
                    }}
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
                      <strong>{item.records_count}/{item.total_data}</strong>
                      <span>剩余 {item.remaining_count}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* ── DETAIL ── */}
      {activeTab === "detail" && (
        <div className="exp-panel">
          {!detail ? (
            <p className="muted">请先从「实验列表」选择一个实验。</p>
          ) : (
            <>
              <section className="experiment-focus-hero">
                <header className="card-header experiment-hero-header">
                  <div className="experiment-hero-copy">
                    <p className="eyebrow">{detail.manifest.experiment_id}</p>
                    <h3>{detail.manifest.exp_name} · {detail.manifest.scenario_name}</h3>
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
                    {detail.manifest.save_video && detail.video_dir && (
                      <button
                        type="button"
                        className="secondary-button"
                        onClick={() => {
                          apiPost(`/api/experiments/${detail.manifest.experiment_id}/open-video-dir`, {}).catch(
                            (err) => alert(`无法打开视频目录：${err.message}`)
                          );
                        }}
                      >
                        📁 查看视频目录
                      </button>
                    )}
                  </div>
                </header>

                <aside className="experiment-snapshot-rail">
                  <div className="experiment-snapshot-rail-head">
                    <h4>配置快照</h4>
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
                  <span className="muted">实验级补充日志</span>
                </div>
                <details className="data-viewer">
                  <summary>展开 runtime.log</summary>
                  <pre>{detail.runtime_log_tail.join("\n")}</pre>
                </details>
              </section>
            </>
          )}
        </div>
      )}
    </div>
  );
}
