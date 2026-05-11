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
