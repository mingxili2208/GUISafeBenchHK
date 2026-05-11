# Changelog

## 2026-05-11 fix: Step 6 导出过期检测

### 问题描述

用户在 Step 3-5 绘制新路线后，进入 Step 6 运行实验时仍然使用旧的路线数据（实验次数也是画新路线之前的），没有反映最新绘制的路线。

### 根因分析

`repository.py` 中的 `standard_cards()` 仅检查 `scenario_data/` 中的导出文件**是否存在**，不检查是否与 `scenario_origin/` 中的源数据**一致**。

数据流程：
1. Step 3 路线绘制 → 写入 `scenario_origin/<map>/scenario_<id>_routes/`
2. Step 4 Trigger/Actor → 写入 `scenario_origin/<map>/scenario_<id>_scenarios/`
3. Step 5 导出 → 从 `scenario_origin/` 读取源数据，导出到 `scenario_data/<map>/`
4. Step 6 运行 → 读取 `scenario_data/` 中的导出数据

当用户绘制了新路线（`scenario_origin/` 更新）但没有重新导出时，旧的导出文件仍然存在于 `scenario_data/`，导致卡片状态仍显示 "Export Ready" / "Run Ready"，Step 6 实际运行的是旧数据。

### 修复方案

通过比较源数据与导出数据的文件修改时间（mtime），检测导出是否过期：

| 文件 | 修改内容 |
|---|---|
| `gui_console/backend/repository.py` | 新增 `_latest_mtime_raw()` 辅助函数；`standard_cards()` 中比较 `scenario_origin/` 与 `scenario_data/` 的 mtime，若源数据更新则标记 `export_stale=True`，`overall_status` 设为 "Export Stale"；同时统计 `export_route_count`（导出的 XML 路线数） |
| `gui_console/backend/schemas.py` | `StandardCardResponse` 新增 `export_stale: bool` 和 `export_route_count: int` 字段 |
| `gui_console/frontend/src/types.ts` | `StandardCard` interface 新增 `export_stale` 和 `export_route_count` 字段 |
| `gui_console/frontend/src/components/StandardCard.tsx` | Step 5 区域：过期时显示 "⚠ 导出已过期" 状态标签和黄色警告横幅；正常时显示导出路线数 |
| `gui_console/frontend/src/App.tsx` | `selectedReadyCards` 过滤掉 `export_stale=true` 的场景；Step 6 顶部显示黄色警告提示用户重新导出 |
| `gui_console/frontend/src/styles.css` | 新增 `.notice-warning`、`.status-export-stale`、`.substep-status--stale`、`.stale-warning` 样式 |

### 修复后的行为

1. 用户绘制新路线 → `scenario_origin/` mtime 更新
2. 后端检测到 `source_mtime > export_mtime` → 卡片状态变为 **"Export Stale"**
3. Step 6 下拉列表**排除**过期场景，无法被选中运行
4. Step 6 顶部显示黄色警告：「请返回 Step 5 重新导出后再运行」
5. 用户返回 Step 5 重新导出 → `scenario_data/` mtime 更新 → 状态恢复 "Run Ready"
6. Step 6 允许使用最新数据运行

---

## 2026-05-11 feat: StandardCard 添加「查看文件」按钮

### 功能说明

在 Step 3-5 的标准工作区卡片中，每个子步骤新增「查看文件」按钮，点击后在系统文件管理器中打开对应的目录，方便用户直接查看和确认路线、场景标注、导出数据。

### 按钮位置

| 子步骤 | 打开的目录 | 显示条件 |
|---|---|---|
| Step 3 · 路线绘制 | `scenario_origin/<map>/scenario_<id>_routes/` | 路线目录存在时 |
| Step 4 · Trigger/Actor | `scenario_origin/<map>/scenario_<id>_scenarios/` | 场景目录存在时 |
| Step 5 · 导出路线 | `scenario_data/<map>/scenario_<id>_routes/` | 导出目录存在时 |

### 补充说明

- Step 3/4 的路线编辑器和场景编辑器**不会自动保存**，必须右键点击才会保存当前选点为一条 route 或 scenario
- Step 5 导出**不会生成独立的 session/config 文件**，只生成 XML 路线文件 + JSON 索引 + JSON 场景定义，被 `run.py` 直接消费
- 后端 `POST /api/open-dir` 接口限制只能打开仓库内的目录

### 修改文件

| 文件 | 修改内容 |
|---|---|
| `gui_console/backend/main.py` | 新增 `POST /api/open-dir` 接口，接收路径参数并用 `xdg-open` 打开，限制只能打开仓库内的目录 |
| `gui_console/backend/schemas.py` | 新增 `OpenDirRequest` schema |
| `gui_console/frontend/src/components/StandardCard.tsx` | 每个 substep 的 button-row 中新增「查看文件」按钮，条件显示 |
| `gui_console/frontend/src/App.tsx` | 新增 `handleOpenDir()` 并传入 StandardCard |

---

## 2026-05-11 feat: 路线/场景编辑器 R 键撤销 + ESC 自动保存

### 功能说明

**R 键撤销/删除**：在路线编辑器（create_routes）和场景编辑器（create_scenarios）中：
- 如果有未保存的选点 → 清空当前选点（重新绘制）
- 如果没有未保存的选点 → 删除上一条已保存的记录（撤销上次右键保存）
- 原有的 R 键视角重置功能移至 **Home 键**

**ESC 自动保存**：按 ESC 退出时，如果当前有有效选点（路线 >= 2 个点，场景 >= 1 个点），会自动保存后退出，不再丢失未保存的工作。

### 修改文件

| 文件 | 修改内容 |
|---|---|
| `tools/CarlaScenariosBuilder/create_routes.py` | R 键：清空选点或删除最后保存的路线文件；ESC 键：自动保存后退出；Home 键：视角重置；更新 overlay 提示文本 |
| `tools/CarlaScenariosBuilder/create_scenarios.py` | R 键：清空选点或删除当前 route 对应的场景文件；ESC 键：自动保存后退出（含 side_marks 计算）；Home 键：视角重置；更新 overlay 提示文本 |
| `gui_console/frontend/src/App.tsx` | 更新 Step 3-5 操作指南，反映新的快捷键说明 |
