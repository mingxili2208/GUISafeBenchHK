import { StandardCard as StandardCardType } from "../types";

interface StandardCardProps {
  card: StandardCardType;
  onRoute: (scenarioId: number) => void;
  onScenario: (scenarioId: number) => void;
  onExport: (scenarioId: number, format: "standard" | "adv" | "both") => void;
}

export function StandardCard({ card, onRoute, onScenario, onExport }: StandardCardProps) {
  return (
    <article className="standard-card">
      <header className="card-header">
        <div>
          <p className="eyebrow">Scenario {card.scenario_id.toString().padStart(2, "0")}</p>
          <h3>{card.name}</h3>
          <p className="muted">Map: {card.map_name}</p>
        </div>
        <span className={`status-pill status-${card.overall_status.toLowerCase().replace(/\s+/g, "-")}`}>
          {card.overall_status}
        </span>
      </header>

      <section className="card-block">
        <h4>Route</h4>
        <p>数量：{card.route_count}</p>
        <p className="path">{card.paths.route_dir?.path}</p>
        <button onClick={() => onRoute(card.scenario_id)}>启动路线编辑器</button>
      </section>

      <section className="card-block">
        <h4>Trigger / Actor</h4>
        <p>
          标注文件：{card.scenario_count}，Sides：{card.sides_count}
        </p>
        <p className="path">{card.paths.scenario_dir?.path}</p>
        <button onClick={() => onScenario(card.scenario_id)}>启动场景编辑器</button>
      </section>

      <section className="card-block">
        <h4>Export</h4>
        <p>状态：{card.export_status}</p>
        <p className="path">{card.paths.export_index_json?.path}</p>
        <div className="button-row">
          <button onClick={() => onExport(card.scenario_id, "standard")}>导出 Standard</button>
          <button onClick={() => onExport(card.scenario_id, "adv")}>导出 Adv</button>
          <button onClick={() => onExport(card.scenario_id, "both")}>导出 Both</button>
        </div>
      </section>

      <footer className="card-footer">
        <span>Route 状态：{card.route_status}</span>
        <span>Scenario 状态：{card.scenario_status}</span>
        <span>最近更新：{card.latest_updated_at ?? "暂无"}</span>
      </footer>
    </article>
  );
}

