# GUI 控制台使用手册

SafeBenchHK 提供了一套浏览器 GUI，用于在本地可视化地管理 CARLA 自动驾驶评测实验。  
GUI 由 **FastAPI 后端**（端口 8001）和 **Vite/React 前端**（端口 5173）两部分组成，通过仓库内的 `gui_console/bin/` 脚本统一启停。

---

## 1. 依赖前提

| 依赖 | 说明 |
|---|---|
| Python 3.8+ | 后端运行环境，须与 `pip install -e .` 安装 safebench 的 Python 一致 |
| Node.js 18+ | 前端构建与预览 |
| npm | 前端依赖管理 |
| CARLA Simulator | 运行实验时需要已启动 |

安装前端依赖（首次克隆后执行一次即可）：

```bash
cd gui_console/frontend
npm install
```

安装后端 Python 依赖（若尚未安装仓库依赖）：

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 2. 启动与停止

### 2.1 一键启动（推荐）

在仓库根目录执行：

```bash
bash gui_console/bin/start_console.sh
```

脚本会同时启动后端和前端，并在后台以 supervisor 模式监控两个进程。  
启动完成后，在浏览器访问：

```
http://127.0.0.1:5173
```

### 2.2 停止

```bash
bash gui_console/bin/stop_console.sh
```

### 2.3 自定义端口（可选）

通过环境变量覆盖默认端口，例如：

```bash
GUI_BACKEND_PORT=9001 GUI_FRONTEND_PORT=5174 bash gui_console/bin/start_console.sh
```

前端会自动读取对应的 `VITE_API_BASE` 环境变量。

---

## 3. GUI 工作流（Step 0 → Step 7）

GUI 将实验流程拆分为 8 个步骤，从左侧导航栏依次进行。

### Step 0 — 环境确认

- 填写 Python 解释器路径、CARLA Host/Port/TM Port（默认 `127.0.0.1:2000/8000`）
- 点击「检查 CARLA 与 Python 环境」按钮
- 所有项目变绿后，顶部状态栏显示「GUI 会话：已连接」才可继续

> CARLA 需要在此之前启动。若 CARLA 未运行，「CARLA 连通性」项会失败。

### Step 1 — 地图 / Waypoint

- 选择要使用的地图（从 `tools/CarlaScenariosBuilder/map_waypoints/` 自动扫描）
- 点击「准备地图」，将地图数据软链接到运行时目录
- 状态变为「已就绪」后继续

### Step 2 — 标准选择

- 从地图关联的场景标准卡片中选择一组标准（对应 `standard_scenario_XX.json`）
- 点击「选定此标准」

### Step 3–5 — 配置工作区

- **Step 3**：选择 Agent 配置（`safebench/agent/config/*.yaml`）
- **Step 4**：选择场景配置（`safebench/scenario/config/*.yaml`）
- **Step 5**：设置实验名称、seed、端口，以及是否开启渲染/视频录制

### Step 6 — 运行中心

- 点击「开始实验」后，GUI 会创建一个 Job，在后台调用 `scripts/run.py` 运行评测
- 顶部卡片实时显示活跃任务数、待续跑实验数
- 「查看最近情况」可展开当前活跃 Job 的实时日志
- 「刷新状态」手动刷新；「断开 CARLA 链接」会清除会话（不会终止 CARLA 进程）

> GUI 会话（`carla_session_connected`）只由「环境检查」建立，由「断开链接」清除；  
> 状态轮询（每 5 秒）探活失败不会自动断开会话。

### Step 7 — 实验记录

- 列出所有通过 GUI 创建的历史实验
- 点击实验进入详情页，可查看：
  - 配置快照（Agent、场景、端口、seed、是否录制视频）
  - 评测指标（成功率等）
  - 停止原因统计
  - 最近 Batch 摘要
  - `runtime.log` 尾部日志
- 详情页操作按钮：
  - **继续运行**：在已有进度基础上续跑（剩余 > 0 时启用）
  - **重新运行**：从头清空重跑
  - **📁 查看视频目录**：仅在开启了视频录制时显示，点击后用系统文件管理器打开视频输出目录

---

## 4. 数据目录说明

| 目录 | 说明 |
|---|---|
| `gui_console/runtime/experiments/` | 每次实验的 manifest、输出结果 |
| `gui_console/runtime/jobs/` | Job 日志（`job-*.log`）|
| `gui_console/runtime/supervisor/` | 后端/前端进程 PID 与状态文件 |
| `gui_console/runtime/run_root/` | 实验运行时配置副本 |

> `gui_console/runtime/` 整个目录已被 `.gitignore` 排除，不纳入版本控制。

---

## 5. 常见问题

**Q: 启动后浏览器空白或 API 报错**  
A: 检查后端是否已启动，确认 `http://127.0.0.1:8001/docs` 可访问。若端口被占用，用 `stop_console.sh` 清理后重试。

**Q: Step 0 「CARLA 连通性」失败**  
A: 先启动 CARLA（`./CarlaUE4.sh -carla-rpc-port=2000`），再点「重新检查」。

**Q: Step 6 顶部显示「GUI 会话：已断开」**  
A: 这通常是会话未建立（未完成 Step 0 环境检查）。重新执行 Step 0 检查即可。

**Q: 实验卡在「运行中」但 CARLA 已崩溃**  
A: 在 Step 6 或 Step 7 中点「重新运行」，或手动 kill 对应的 `run.py` 进程后续跑。 