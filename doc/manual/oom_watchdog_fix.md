# OOM 卡死问题排查与修复

> 日期：2026-06-24 ~ 2026-06-25
> 实验：`guiexp-20260624-121943-gui_eval-behavior-s01-3e9697`（14C_new_2 / DynamicObjectCrossing / behavior agent / 184 场景）

## 问题现象

SafeBench 在连续评测时，跑到第 4 个 batch 左右就会卡死，最终被 OOM Killer 杀死：
- 进程 RSS 膨胀到 **34–44 GB**
- `dmesg` / `syslog` 显示 `Out of memory: Killed process`
- 终端 exit code = `-9`（SIGKILL）
- CARLA EBADF 修复已应用，非 EBADF 竞态导致

## 排查过程

### 1. 添加内存监控

在 `safebench/carla_runner.py` 的 `eval()` 循环中添加 RSS 日志点（读 `/proc/self/status`，零依赖）：

| 监控点 | 含义 |
|--------|------|
| `batch-start` | 每个 batch 开始 |
| `after-env-reset` | env.reset() 完成后 |
| `step-50/100/...` | 场景执行中每 50 步 |
| `before-cleanup` | env.clean_up() 之前 |
| `after-cleanup` | env.clean_up() 之后 |
| `after-video-save` | 视频保存后 |
| `batch-end` | batch 结束 |

### 2. 定位暴涨阶段

正常的 batch（蓝色）vs 出问题的 batch（红色）：

```
正常 batch 1-3:
  batch-start:          RSS= 965 MB
  step-100:             RSS= 967 MB  ← 几乎不变
  before-cleanup:       RSS= 968 MB
  after-cleanup:        RSS= 968 MB
  batch-end:            RSS= 968 MB

异常 batch 4:
  batch-start:          RSS= 968 MB
  after-env-reset:      RSS= 969 MB
  step-100:             RSS=2514 MB  ← 暴涨 +1545 MB
  before-cleanup:       RSS=2540 MB
  after-cleanup (R8a):  进程挂起 → 4 分钟后 RSS 达 35 GB → OOM Kill
```

**结论：暴涨发生在场景执行阶段（step 0→100），不是 cleanup。但 final blow 总是在 R8a。**

### 3. 定位 R8a 挂起

R8a 的 `actor.set_autopilot(False, tm_port)` 是向 CARLA Traffic Manager 发送 RPC 调用。
场景执行期间 CARLA/TM 连接被某些数据 ID 触发的内存 spike 破坏后，该 RPC 永久挂起。
挂起期间 CARLA 传感器回调和 pygame 渲染持续分配内存，4 分钟内从 2.5 GB 膨胀到 35 GB。

### 4. "每 3 batch 失败" 是巧合

初期日志显示每次第 4 个 batch 崩溃，看起来像规律：

```
Run 1: data ID 0-2 OK → data ID 3  spike
Run 2: data ID 3-5 OK → data ID 6  spike
Run 3: data ID 6-8 OK → data ID 9  spike
```

但后续大量运行后真相浮现——**spike 与特定数据 ID 相关，不是 batch 计数**：

```
33-batch run: data ID 44-76 OK (33 个!) → data ID 77 spike
9-batch run:  data ID 77-85 OK           → data ID 86 spike
98-batch run: data ID 86-183 OK (98 个!)  → 全部正常 ✅
```

data ID 0–86 之间的某些路由存在易于触发 CARLA 连接异常的特定条件（路线、spawn 位置、actor 数量等）。
data ID 86–183 完全不触发 spike，可以连续跑 98 个不出问题。

## 修复方案

### A. spawn 阶段保护（`carla_data_provider.py`）

**问题**：spawn 背景车的 batch 命令中链了 `SetAutopilot(FutureActor, True, tm_port)`，TM 异常时 `apply_batch_sync` 整体挂起，连车都 spawn 不出来。

**修复**：batch 命令只 spawn，不设 autopilot。spawn 后用 `signal.alarm(10)` 超时逐个设置：
```python
# 旧：一个 batch 同时 spawn + autopilot
batch.append(SpawnActor(blueprint, spawn_point).then(
    SetAutopilot(FutureActor, autopilot, tm_port)))

# 新：先 spawn，再逐个 autopilot + 超时
batch.append(SpawnActor(blueprint, spawn_point))
# spawn 完成后...
for actor in actors:
    _signal.alarm(10)
    actor.set_autopilot(True, tm_port)  # 超时抛异常，跳过该车
    _signal.alarm(0)
```

### B. R8a cleanup 保护（`route_scenario.py`）

**问题**：`set_autopilot(False)` 在 TM 异常时永久挂起，daemon 线程方式会泄漏内存。

**修复**：改回 `signal.alarm(8)` 超时 + 每 actor 日志：
```python
for actor_id in valid_background_actor_ids:
    _dbg(f'R8a: [{idx}/{total}] actor_id={actor_id}')
    _mem(f'R8a-actor-{idx}-before')
    _signal.alarm(8)
    actor.set_autopilot(False, tm_port)
    _signal.alarm(0)
    _mem(f'R8a-actor-{idx}-after')
```
超时时跳过该 actor，继续处理下一个，不阻塞 cleanup。

### C. 内存守卫 + 自动续跑（`carla_runner.py` + `jobs.py`）

上述 timeout 方案在 CARLA C 扩展完全阻塞信号时仍然失效（`signal.alarm` 无法投递）。作为最后防线：

**`carla_runner.py`**：daemon 线程每 3 秒检查 RSS，超过 4 GB 调用 `os._exit(99)`：
```python
WATCHDOG_THRESHOLD_MB = 4000
WATCHDOG_INTERVAL = 3

def _start_memory_watchdog(self):
    def _watch():
        while not stopped:
            if rss > 4000:
                os._exit(99)  # 特殊退出码
```

**`gui_console/backend/jobs.py`**：检测 `return_code == 99`，等 5 秒后自动启动续跑：
```python
if return_code == 99 and payload["type"] in ("run", "resume-run"):
    self._schedule_auto_resume(payload)  # 5s 后 start_job(resume-run)
```

### 自动续跑流程

```
batch1 → batch2 → batch3 → [spike → watchdog kill (exit 99)]
                                ↓
                           jobs.py 检测 ret=99
                                ↓
                           sleep 5s
                                ↓
                           自动 resume-run（加载 results.pkl 跳过已完成的 data ID）
                                ↓
                           batch4 → batch5 → ...
```

## 涉及文件

| 文件 | 修改内容 |
|------|----------|
| `safebench/carla_runner.py` | 内存监控日志、step 每 50 步记录、内存 watchdog daemon 线程 |
| `safebench/scenario/scenario_definition/route_scenario.py` | R8a per-actor 日志 + `signal.alarm` 超时 + 每步 RSS 记录 |
| `safebench/scenario/scenario_manager/carla_data_provider.py` | spawn batch 移除 `SetAutopilot` 链式调用，改为逐个超时设置 |
| `gui_console/backend/jobs.py` | 检测 `ret=99` 自动续跑 |

## 结果

- 184/184 测试全部完成
- watchdog 触发约 10 次，每次自动续跑成功
- 最终 run 连续完成 98 个 batch，RSS 从 973 MB 仅增至 1077 MB（+104 MB），无 spike
