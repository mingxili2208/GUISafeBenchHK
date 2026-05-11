import { useState } from "react";
import { apiPost } from "../api";
import { StandardCard as StandardCardType } from "../types";

export type CardSubStep = "route" | "scenario" | "export";

interface ScenarioGuide {
  route: string;
  trigger: string;
}

const SCENARIO_GUIDES: Record<number, ScenarioGuide> = {
  1: {
    route:
      "选取一段「直行路段」，起点和终点均在同一车道上，路旁须有人行道或路肩空间（行人需从此横穿）。",
    trigger:
      "编辑器操作：点击 2 个点后右键保存。点1 = Trigger 点（放在路线上；激活阈值 6m）；点2 = Actor 行人的 Spawn 点（放在路旁人行道或路肩）。激活后行人从 Spawn 点横穿马路。",
  },
  2: {
    route:
      "路线须包含一个「路口转弯」（左转或右转）：起点在转弯前直线段，终点在转弯后直线段。",
    trigger:
      "编辑器操作：点击 2 个点后右键保存。点1 = Trigger 点（放在路线上接近路口；激活阈值 6m）；点2 = Actor 骑行者的 Spawn 点（放在转弯出口的直行段上）。激活后骑行者向前同向行驶。",
  },
  3: {
    route:
      "选取一段「双车道直行路段」，路段须存在相邻平行车道（系统将自动在此生成旁车）。",
    trigger:
      "编辑器操作：点击 2 个点后右键保存。点1 = Trigger 点（放在路线上；激活阈值 6m）；点2 = Actor[0] 前车的 Spawn 点（放在主车前方同向车道）。Actor[1] 旁车由系统自动生成于前车相邻车道，无需额外点击。",
  },
  4: {
    route:
      "选取一段「双向行车直行路段」（须存在对向车道）。",
    trigger:
      "编辑器操作：只需点击 1 个点后右键保存。点1 = Trigger 点（同时作为 Actor[0] 慢速前车的 Spawn 位置；放在主车前方同向车道；激活阈值 10m）。Actor[1] 对向来车由系统自动生成于 Trigger 点前方 30m 的对向车道，无需手动点击。",
  },
  5: {
    route:
      "路线须「直行穿过一个有信号灯的路口」：起点在路口前直线段，终点在路口后。",
    trigger:
      "编辑器操作：点击 2 个点后右键保存。点1 = Trigger 点（放在路口前；主车速度 > 0.5m/s 即激活）；点2 = Actor 闯红灯来车的 Spawn 点（放在路口对向进入车道）。激活后来车在主车绿灯通行时闯红灯冲出。",
  },
  6: {
    route:
      "路线须在「有信号灯路口」左转：起点在路口前直线段，左转后终点在目标路段上。",
    trigger:
      "编辑器操作：点击 2 个点后右键保存。点1 = Trigger 点（放在路线上接近路口；激活阈值 6m）；点2 = Actor 对向直行来车的 Spawn 点（放在路口对向直行进入车道）。主车左转时对向来车直行进入路口，主车须让行。",
  },
  7: {
    route:
      "路线须在「有信号灯路口」右转：起点在路口前直线段，右转后终点在目标路段上。",
    trigger:
      "编辑器操作：点击 2 个点后右键保存。点1 = Trigger 点（放在路线上接近路口；主车速度 > 0.5m/s 即激活）；点2 = Actor 侧向来车的 Spawn 点（放在路口来车方向车道）。主车右转时来车进入路口，注意避让。",
  },
  8: {
    route:
      "路线须「直行穿过一个无信号灯的路口」，起点在路口前，终点在路口后。",
    trigger:
      "编辑器操作：点击 2 个点后右键保存。点1 = Trigger 点（须放在路口前约 35m 处；激活阈值为 35m，远大于其他场景的 6m）；点2 = Actor 侧向来车的 Spawn 点（放在与主车垂直的侧向进入车道）。激活后侧向来车以 10m/s 驶入路口。",
  },
};

interface StandardCardProps {
  card: StandardCardType;
  subStep: CardSubStep;
  onRoute: (scenarioId: number) => void;
  onScenario: (scenarioId: number) => void;
  onExport: (scenarioId: number, format: "standard" | "adv" | "both") => void;
  onClearRoute: (scenarioId: number) => void;
  onClearScenario: (scenarioId: number) => void;
  onOpenDir: (path: string) => void;
}

function StepDot({ done, active }: { done: boolean; active: boolean }) {
  const cls = done ? "step-dot step-dot--done" : active ? "step-dot step-dot--active" : "step-dot step-dot--pending";
  return <span className={cls} />;
}

export function StandardCard({ card, subStep, onRoute, onScenario, onExport, onClearRoute, onClearScenario, onOpenDir }: StandardCardProps) {
  const routeDone = card.route_count > 0;
  const scenarioDone = card.scenario_count > 0 && card.sides_count > 0;
  const exportDone = card.export_status === "Export Ready";

  const [guideOpen, setGuideOpen] = useState(true);
  const guide = SCENARIO_GUIDES[card.scenario_id];

  return (
    <article className="standard-card">
      <header className="card-header">
        <div>
          <p className="eyebrow">Scenario {card.scenario_id.toString().padStart(2, "0")}</p>
          <h3>{card.name}</h3>
          <p className="muted">Map: {card.map_name}</p>
        </div>
        <div className="card-header-right">
          <span className={`status-pill status-${card.overall_status.toLowerCase().replace(/\s+/g, "-")}`}>
            {card.overall_status}
          </span>
          <div className="card-step-dots">
            <StepDot done={routeDone} active={subStep === "route"} />
            <StepDot done={scenarioDone} active={subStep === "scenario"} />
            <StepDot done={exportDone} active={subStep === "export"} />
          </div>
        </div>
      </header>

      {/* Step 3: Route */}
      <section className={`card-block substep-block${subStep === "route" ? " substep-block--active" : ""}`}>
        <div className="substep-block-header">
          <span className="substep-label">Step 3 · 路线绘制</span>
          {routeDone
            ? <span className="substep-status substep-status--done">✓ {card.route_count} 条路线已就绪</span>
            : <span className="substep-status substep-status--missing">尚无路线</span>}
        </div>
        <p className="path">{card.paths.route_dir?.path}</p>
        {guide && (
          <div className="scenario-guide-tip">
            <button
              className="scenario-guide-toggle"
              onClick={() => setGuideOpen((v) => !v)}
            >
              {guideOpen ? "▾" : "▸"} 操作指南
            </button>
            {guideOpen && (
              <div className="scenario-guide-body">
                <div className="scenario-guide-row">
                  <span className="scenario-guide-label">🗺 路线绘制</span>
                  <span className="scenario-guide-text">{guide.route}</span>
                </div>
              </div>
            )}
          </div>
        )}
        <div className="button-row">
          <button onClick={() => onRoute(card.scenario_id)}>启动路线编辑器</button>
          {card.paths.route_dir?.exists && (
            <button className="button-secondary" onClick={() => onOpenDir(card.paths.route_dir!.path)}>查看文件</button>
          )}
          {routeDone && (
            <button
              className="btn-danger-ghost"
              onClick={() => onClearRoute(card.scenario_id)}
              title="清零此卡的路线（同时清除 Trigger/Actor 和导出数据）"
            >
              清零路线
            </button>
          )}
        </div>
      </section>

      {/* Step 4: Trigger / Actor */}
      <section className={`card-block substep-block${subStep === "scenario" ? " substep-block--active" : ""}${!routeDone ? " substep-block--locked" : ""}`}>
        <div className="substep-block-header">
          <span className="substep-label">Step 4 · Trigger / Actor</span>
          {!routeDone
            ? <span className="substep-status substep-status--locked">需先完成路线</span>
            : scenarioDone
            ? <span className="substep-status substep-status--done">✓ {card.scenario_count} 标注 / {card.sides_count} Sides</span>
            : <span className="substep-status substep-status--missing">尚无标注</span>}
        </div>
        <p className="path">{card.paths.scenario_dir?.path}</p>
        {guide && guideOpen && (
          <div className="scenario-guide-tip">
            <div className="scenario-guide-body">
              <div className="scenario-guide-row">
                <span className="scenario-guide-label">📍 Trigger / Actor</span>
                <span className="scenario-guide-text">{guide.trigger}</span>
              </div>
            </div>
          </div>
        )}
        <div className="button-row">
          <button onClick={() => onScenario(card.scenario_id)} disabled={!routeDone}>
            启动场景编辑器
          </button>
          {card.paths.scenario_dir?.exists && (
            <button className="button-secondary" onClick={() => onOpenDir(card.paths.scenario_dir!.path)}>查看文件</button>
          )}
          {scenarioDone && (
            <button
              className="btn-danger-ghost"
              onClick={() => onClearScenario(card.scenario_id)}
              title="清零此卡的 Trigger/Actor（同时清除导出数据）"
            >
              清零场景
            </button>
          )}
        </div>
      </section>

      {/* Step 5: Export */}
      <section className={`card-block substep-block${subStep === "export" ? " substep-block--active" : ""}${!scenarioDone ? " substep-block--locked" : ""}`}>
        <div className="substep-block-header">
          <span className="substep-label">Step 5 · 导出路线</span>
          {!scenarioDone
            ? <span className="substep-status substep-status--locked">需先完成 Trigger/Actor</span>
            : card.export_stale
            ? <span className="substep-status substep-status--stale">⚠ 导出已过期（源数据已更新）</span>
            : exportDone
            ? <span className="substep-status substep-status--done">✓ 已导出（{card.export_route_count} 条）</span>
            : <span className="substep-status substep-status--missing">待导出</span>}
        </div>
        {card.export_stale && (
          <p className="stale-warning">路线或场景数据已更新，请重新导出以使 Step 6 使用最新数据。</p>
        )}
        <p className="path">{card.paths.export_index_json?.path}</p>
        <div className="button-row">
          <button onClick={() => onExport(card.scenario_id, "standard")} disabled={!scenarioDone}>导出 Standard</button>
          <button onClick={() => onExport(card.scenario_id, "adv")} disabled={!scenarioDone}>导出 Adv</button>
          <button onClick={() => onExport(card.scenario_id, "both")} disabled={!scenarioDone}>导出 Both</button>
          {card.paths.export_route_dir?.exists && (
            <button className="button-secondary" onClick={() => onOpenDir(card.paths.export_route_dir!.path)}>查看文件</button>
          )}
        </div>
      </section>

      <footer className="card-footer">
        <span>Route：{card.route_status}</span>
        <span>Scenario：{card.scenario_status}</span>
        <span>更新：{card.latest_updated_at ?? "暂无"}</span>
      </footer>
    </article>
  );
}

