# SafeBenchHK GUI Console

这个目录是 SafeBenchHK 的零侵入 GUI 原型。

设计原则：

- 所有新代码只放在 `gui_console/`
- 不修改现有 `scripts/`、`safebench/`、`tools/CarlaScenariosBuilder/` 源文件
- Builder 继续调用原脚本
- 运行继续调用原 `scripts/run.py`
- 运行时配置、副本、软链接、任务状态都放在 `gui_console/runtime/`

## 目录结构

```text
gui_console/
├── backend/
├── frontend/
└── runtime/
```

## 后端启动

建议在可导入 `safebench` 的 Python 环境中启动：

```bash
cd /home/fsm/SafeBenchHK-zh-simulate-tag
pip install -r gui_console/backend/requirements.txt
./gui_console/bin/start_backend.sh
```

## 前端启动

```bash
cd /home/fsm/SafeBenchHK-zh-simulate-tag/gui_console/frontend
npm install
cd /home/fsm/SafeBenchHK-zh-simulate-tag
./gui_console/bin/start_frontend.sh
```

默认前端会以稳定的 `preview` 模式启动，避免 `vite` 文件监听导致的 `EMFILE` / `watch` 资源报错。
如果你需要热更新开发模式，可以显式指定：

```bash
GUI_FRONTEND_MODE=dev ./gui_console/bin/start_frontend.sh
```

默认前端开发地址为 `http://127.0.0.1:5173`，后端地址为 `http://127.0.0.1:8001`。

## 一键启动与生命周期监管

如果你希望前后端一键启动、统一监管生命周期和资源占用，并且在任意时刻通过 `Ctrl+C` 一起停掉，使用：

```bash
cd /home/fsm/SafeBenchHK-zh-simulate-tag
conda activate safebench
./gui_console/bin/start_console.sh
```

这个 supervisor 会：

- 同时拉起前端和后端
- 每隔几秒输出一次 backend / frontend 的 CPU、内存和运行时长
- 持续检查两个子进程是否异常退出
- 在 `Ctrl+C` 时一起停止前后端
- 将状态写到 `gui_console/runtime/supervisor/status.json`
- 将日志写到：
  - `gui_console/runtime/supervisor/backend.log`
  - `gui_console/runtime/supervisor/frontend.log`

为了避免 `vite` 文件监听导致的 `EMFILE` / `watch` 资源报错，`start_console.sh` 默认会强制让前端使用稳定的 `preview` 模式。
如果你确实想在一键启动里使用前端开发模式，可以显式指定：

```bash
GUI_CONSOLE_FRONTEND_MODE=dev ./gui_console/bin/start_console.sh
```

如果需要从另一个终端一键停止：

```bash
cd /home/fsm/SafeBenchHK-zh-simulate-tag
./gui_console/bin/stop_console.sh
```

## 进程命名

为便于在系统监视器、`ps`、`top` 中识别，GUI 相关进程统一使用 `SafeBenchHK-...` 前缀：

- `SafeBenchHK-gui-backend`
- `SafeBenchHK-gui-frontend`
- `SafeBenchHK-map-prepare-<map>`
- `SafeBenchHK-route-editor-<map>-sXX`
- `SafeBenchHK-scenario-editor-<map>-sXX`
- `SafeBenchHK-export-<map>-sXX`
- `SafeBenchHK-run-<exp>-sXX`
- `SafeBenchHK-resume-run-<exp>-sXX`

## 当前实现范围

- Step 0 环境确认
- Step 1 地图与 waypoint 生成任务发起
- Step 2 标准功能场景选择
- Step 3-5 标准卡片的 route/scenario/export 编排
- Step 6 单 agent 运行配置与实验启动
- Step 7 实验列表、结果查看、按实验恢复续跑

## 运行时目录

`gui_console/runtime/` 中会逐步产生：

- `run_root/`
  GUI 专用运行根目录
- `jobs/`
  任务日志与任务状态
- `experiments/`
  实验快照、配置副本和运行输出
