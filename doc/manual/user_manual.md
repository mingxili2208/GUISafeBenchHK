# SafeBenchHK 使用手册

## 1. 这份手册解决什么问题

这份手册不是只做参数罗列，而是把仓库里真正可执行的两条工作流写清楚：

1. 使用仓库现成场景，跑一次标准测试或训练
2. 参考 `ScenarioManual.pdf`，自己创建测试路线、Trigger、Actor，并接入 SafeBenchHK 做测试

如果你想先了解代码关系，请先看 [module_relationship.md](module_relationship.md)。

---

## 2. 先记住两个工作目录

这份仓库有两个常用工作目录，很多“明明命令对了却跑不起来”的问题都出在这里。

### 2.1 仓库根目录

仓库根目录用于：

- 安装依赖
- 运行 `scripts/run.py`
- 修改 `safebench/agent/config/*.yaml`
- 修改 `safebench/scenario/config/*.yaml`
- 查看 `log/` 下的结果

下面这些命令都应在仓库根目录执行：

```bash
pip install -r requirements.txt
pip install -e .
python scripts/run.py ...
```

额外提醒：

- 尽量在当前仓库根目录执行 `pip install -e .`
- 如果你机器上还有别的 `SafeBenchHK` checkout，先确认不会从旧路径导入 `safebench`

### 2.2 `tools/CarlaScenariosBuilder`

`ScenarioManual.pdf` 里讲的地图采样、画路线、打 Trigger、导出场景，都对应这个目录。

这个目录下的脚本大量使用相对路径，例如：

- `map_waypoints/...`
- `scenario_origin/...`
- `scenario_data/...`

所以这些脚本建议先 `cd tools/CarlaScenariosBuilder` 再运行：

```bash
cd tools/CarlaScenariosBuilder
python get_map_data.py ...
python create_routes.py ...
python create_scenarios.py ...
python export.py --format standard ...
```

---

## 3. 项目里真正的运行主线

统一入口是 [scripts/run.py](../scripts/run.py)。

运行时主线如下：

1. `scripts/run.py` 解析命令行参数
2. 从 `safebench/agent/config/` 和 `safebench/scenario/config/` 读取 YAML
3. `CarlaRunner` 加载 Agent、Scenario Policy、CARLA 世界和场景数据
4. 从 `scenario_type` 指向的 JSON 中取出待测的 `scenario_id / route_id / weather_id`
5. 根据 `route_dir` 下的 XML 和 `scenarios/` 下的 JSON 创建具体场景
6. 运行评测或训练
7. 把日志、评测结果、视频写到 `log/` 下的实验目录

这里最重要的结论是：

- 真正决定“测哪些路线、哪些天气、哪类场景”的不是命令行，而是 `scenario/config/*.yaml` 里那几个路径字段
- `scripts/run.py` 的命令行参数主要是实验名、模式、端口、是否渲染、是否保存视频等运行期开关

---

## 4. 运行前准备

### 4.1 基础环境

建议准备：

- Linux
- Python 3.8+
- 与本仓库兼容的 CARLA Simulator
- 与 CARLA 版本匹配的 Python API
- GPU 环境（如果使用 TCP 或训练类任务）

Python 依赖安装：

```bash
pip install -r requirements.txt
pip install -e .
```

### 4.2 启动 CARLA

运行前需要先启动 CARLA 服务端，并保证以下端口可用：

- CARLA 端口：`2000`
- Traffic Manager 端口：`8000`

这两个端口也可以在运行时覆盖：

```bash
python scripts/run.py --port 2000 --tm_port 8000
```

注意：

- 本仓库不会替你启动 CARLA
- 运行时会根据待测路线 XML 中的 `town` 字段去加载地图，因此 CARLA 需要能够访问对应地图资源

---

## 5. 先跑通一次测试

这一节是最重要的“测试操作过程”。

### 5.1 先决定你要用哪套场景配置

当前最常用的场景配置文件有两个：

- [safebench/scenario/config/standard.yaml](../safebench/scenario/config/standard.yaml)：标准功能场景
- [safebench/scenario/config/LC.yaml](../safebench/scenario/config/LC.yaml)：`lc` 场景策略配置

当前仓库里这两个文件的默认指向分别是：

- `standard.yaml`：`center` 地图上的 `standard_scenario_04.json`
- `LC.yaml`：`14C_2` 目录下的 `standard_scenario_01.json`

需要重点理解这几个字段：

| 字段 | 作用 |
|---|---|
| `scenario_type_dir` | 存放“待测场景列表 JSON”的目录 |
| `scenario_type` | 待测场景列表 JSON 文件名，例如 `standard_scenario_04.json` |
| `route_dir` | 存放 `scenario_XX_routes/*.xml` 和 `scenarios/scenario_XX.json` 的目录 |
| `scenario_id` | 只跑某个场景编号；设为 `null` 表示按 JSON 全部跑 |
| `route_id` | 只跑某条路线；设为 `null` 表示按 JSON 全部跑 |
| `policy_type` | 场景策略类型，例如 `standard`、`lc` |

补充说明：

- `scenario_type` 对应的 JSON 只是“待测条目列表”
- 真正导入哪个场景定义模块，取决于这个 JSON 每一项里的 `scenario_folder`
- 例如 `standard_scenario_04.json` 里通常是 `scenario_folder: "standard"`

### 5.2 最小可执行示例：标准评测

如果你只想先确认系统能跑通，推荐直接使用 `standard.yaml` 当前默认配置。

命令：

```bash
python scripts/run.py \
  --mode eval \
  --agent_cfg behavior.yaml \
  --scenario_cfg standard.yaml \
  --exp_name behavior_standard_eval \
  --render false \
  --save_video false
```

这条命令的含义是：

- 用 `behavior.yaml` 作为被测自动驾驶策略
- 用 `standard.yaml` 指定待测场景
- 运行评测模式
- 不开渲染窗口
- 不保存视频

### 5.3 使用 TCP 跑评测

```bash
python scripts/run.py \
  --mode eval \
  --agent_cfg tcp.yaml \
  --scenario_cfg standard.yaml \
  --exp_name tcp_standard_eval \
  --render false \
  --save_video false
```

### 5.4 跑 `lc` 场景策略

```bash
python scripts/run.py \
  --mode eval \
  --agent_cfg tcp.yaml \
  --scenario_cfg LC.yaml \
  --exp_name tcp_lc_eval \
  --render false \
  --save_video false
```

### 5.5 如何只测一个场景或一条路线

`scenario_id` 和 `route_id` 不在命令行里单独提供，而是在 `scenario/config/*.yaml` 中设置。

例如要只测 `standard.yaml` 中的第 4 类场景、第 0 条路线，可以把：

```yaml
scenario_id: 4
route_id: 0
```

如果想跑该 JSON 中的全部路线，则把 `route_id` 设为：

```yaml
route_id: null
```

### 5.6 结果去哪看

当前实现会把结果写到 `log/` 下，而不是仓库根目录单独的 `video/`。

实验目录结构大致是：

```text
log/<exp_name>/<exp_name>_<agent>_<scenario>_seed_<seed>/
```

你最常看的文件通常是：

- `runtime.log`：运行过程日志
- `progress.txt`：表格化指标输出
- `eval_results/results.pkl`：评测汇总结果
- `eval_results/records.pkl`：逐条记录
- `eval_results/batch_results.jsonl`：按 batch 的结果摘要
- `video/<timestamp>/*.mp4`：视频，只有 `--save_video true` 时才会生成

### 5.7 训练命令示例

训练 Agent：

```bash
python scripts/run.py \
  --mode train_agent \
  --agent_cfg ppo.yaml \
  --scenario_cfg standard.yaml \
  --exp_name ppo_train
```

训练 Scenario Policy：

```bash
python scripts/run.py \
  --mode train_scenario \
  --agent_cfg behavior.yaml \
  --scenario_cfg LC.yaml \
  --exp_name lc_train
```

---

## 6. 场景编号与定义文件对照

`ScenarioManual.pdf` 提到每张地图通常设计 8 类场景。当前仓库实际对应关系如下。

| 场景编号 | 场景名称 | 定义文件 |
|---|---|---|
| 1 | `DynamicObjectCrossing` | [safebench/scenario/scenario_definition/standard/object_crash_vehicle.py](../safebench/scenario/scenario_definition/standard/object_crash_vehicle.py) |
| 2 | `VehicleTurningRoute` | [safebench/scenario/scenario_definition/standard/object_crash_intersection.py](../safebench/scenario/scenario_definition/standard/object_crash_intersection.py) |
| 3 | `OtherLeadingVehicle` | [safebench/scenario/scenario_definition/standard/other_leading_vehicle.py](../safebench/scenario/scenario_definition/standard/other_leading_vehicle.py) |
| 4 | `ManeuverOppositeDirection` | [safebench/scenario/scenario_definition/standard/maneuver_opposite_direction.py](../safebench/scenario/scenario_definition/standard/maneuver_opposite_direction.py) |
| 5 | `OppositeVehicleRunningRedLight` | [safebench/scenario/scenario_definition/standard/junction_crossing_route.py](../safebench/scenario/scenario_definition/standard/junction_crossing_route.py) |
| 6 | `SignalizedJunctionLeftTurn` | [safebench/scenario/scenario_definition/standard/junction_crossing_route.py](../safebench/scenario/scenario_definition/standard/junction_crossing_route.py) |
| 7 | `SignalizedJunctionRightTurn` | [safebench/scenario/scenario_definition/standard/junction_crossing_route.py](../safebench/scenario/scenario_definition/standard/junction_crossing_route.py) |
| 8 | `NoSignalJunctionCrossingRoute` | [safebench/scenario/scenario_definition/standard/junction_crossing_route.py](../safebench/scenario/scenario_definition/standard/junction_crossing_route.py) |

这里以仓库实际代码为准。`ScenarioManual.pdf` 中对部分文件的描述与当前仓库实现并不完全一致时，请优先以这里的对照表和代码路径为准。

---

## 7. 参考 `ScenarioManual.pdf` 创建测试场景

这一节把 PDF 里的内容改写成可执行操作。

### 7.1 先说明当前仓库与 PDF 的差异

`ScenarioManual.pdf` 里用的是 `yuanlang-stf` 地图示例，但当前仓库里场景构建器默认支持的地图中心定义在 [tools/CarlaScenariosBuilder/utilities.py](../tools/CarlaScenariosBuilder/utilities.py)，目前只内置：

- `center`
- `Town10HD_Opt`
- `WangJiao_2_w`

如果你要新加地图名，例如 PDF 里的 `yuanlang-stf` 或仓库里的 `14C_2`，至少要先补上 `get_map_centers(map_name)` 里的中心点定义，否则 `create_routes.py` 和 `create_scenarios.py` 无法正常工作。

### 7.2 第 1 步：生成地图 waypoint 数据

进入场景构建目录：

```bash
cd tools/CarlaScenariosBuilder
```

执行：

```bash
python get_map_data.py --map center --port 2000
```

生成结果：

- `map_waypoints/center/sparse.npy`
- `map_waypoints/center/dense.npy`

作用：

- `sparse.npy` 用于交互式选路
- `dense.npy` 用于后续场景导出和精细化处理

### 7.3 第 2 步：交互式创建测试路线

以场景 1 为例：

```bash
python create_routes.py --map center --scenario 1 --route -1
```

界面操作：

1. 鼠标左键：选择或取消某个 waypoint
2. 鼠标右键：保存当前路线
3. `ESC`：退出界面

常用操作习惯：

- 至少选 2 个点，分别作为起点和终点
- 可以连续保存多条路线
- 保存后会写到 `scenario_origin/<map>/scenario_XX_routes/`

以 `center` 地图、场景 1 为例，输出目录是：

```text
scenario_origin/center/scenario_01_routes/
```

### 7.4 第 3 步：为每条路线打 Trigger 和 Actor 点

继续以场景 1 为例：

```bash
python create_scenarios.py --map center --scenario 1 --route_idx -1
```

这里的 `--route_idx -1` 表示按顺序处理该场景下的所有路线。

界面操作：

1. 鼠标左键：依次选择 Trigger 点和 Actor 生成点
2. 鼠标右键：保存当前路线对应的场景配置
3. `ESC`：退出界面

选择规则一定要清楚：

1. 第一个选中的点会被当作 Trigger
2. 后续点会被当作其他交通参与者的生成点
3. 一条路线至少要有 1 个 Trigger 点才能保存

根据 PDF 中的经验，建议你这样理解：

- Trigger：主车接近到这里时，场景开始触发
- Actor 点：对抗车辆、行人或其他参与者的出生位置

对于场景 1 `DynamicObjectCrossing`，PDF 特别强调：

- Actor 通常应选在与主车行驶方向相关的旁侧车道或穿越位置

生成结果会写到：

```text
scenario_origin/<map>/scenario_XX_scenarios/
```

其中每条路线通常会得到：

- `scenario_00.npy`：Trigger 和 Actor 点
- `scenario_00_sides.npy`：Actor 左右侧标记

### 7.5 第 4 步：导出运行时所需的 XML 和场景 JSON

现在推荐统一使用 [tools/CarlaScenariosBuilder/export.py](../tools/CarlaScenariosBuilder/export.py) 作为正式入口。

标准场景导出：

```bash
python export.py --map center --save_dir scenario_data/center --scenario 1 --format standard
```

如果你不想每次都手写一长串参数，也可以把参数写进 YAML，再用一个脚本调用：

```bash
python export.py
```

默认会优先自动读取同目录下的 `export_config.yaml`；如果它不存在，则回退到仓库里的 `export_config.example.yaml`。只有你想切换到别的配置文件时，才需要显式传：

```bash
python export.py --config your_export_config.yaml
```

如果还想同时保留对抗链路所需的索引 JSON，可以改成：

```bash
python export.py --map center --save_dir scenario_data/center --scenario 1 --format both
```

导出后的关键产物通常是：

- `scenario_data/center/scenario_01_routes/*.xml`
- `scenario_data/center/scenarios/scenario_01.json`
- `scenario_data/center/standard_scenario_01.json`
- 可选：`scenario_data/center/adv_scenario_01.json`

`--format` 的含义：

- `standard`：生成 `standard_scenario_XX.json`
- `adv`：生成 `adv_scenario_XX.json`
- `both`：两者都生成

### 7.6 第 5 步：把导出的数据接入 `safebench/scenario/scenario_data`

完成导出后，需要把构建器目录里的结果同步到运行时目录。

如果你当前在仓库根目录，推荐执行：

```bash
mkdir -p safebench/scenario/scenario_data/center
cp -r tools/CarlaScenariosBuilder/scenario_data/center/* safebench/scenario/scenario_data/center/
```

完成后，运行时目录里至少应具备：

```text
safebench/scenario/scenario_data/<map>/
├── scenario_01_routes/
├── scenarios/
└── *.json
```

### 7.7 第 6 步：准备 `scenario_type` 对应的列表 JSON

这一步是很多人最容易漏的，也是原 `user_manual.md` 最没有讲清楚的地方。

`scripts/run.py` 不是直接扫目录跑的，它会先读取 `scenario_type` 指向的列表 JSON，例如：

- `standard_scenario_04.json`
- `standard_scenario_01.json`

这个文件的每一项至少描述：

- `scenario_folder`
- `scenario_id`
- `route_id`
- `weather_id`
- `parameters`

因此你在接入新场景时要主动检查：

1. `scenario/config/*.yaml` 的 `scenario_type` 到底写的是什么文件名
2. 该文件是否真实存在于 `scenario_type_dir` 中
3. 文件内的 `scenario_folder` 是否与你想跑的场景实现一致

如果你走标准场景评测，导出器应该直接产出与 `scenario_type` 同名的 `standard_scenario_XX.json`。更稳妥的检查方法是对照仓库里现成的 [safebench/scenario/scenario_data/center/standard_scenario_04.json](../safebench/scenario/scenario_data/center/standard_scenario_04.json) 看字段是否一致。

最低要求如下：

- 文件名与 `scenario_type` 一致
- `scenario_folder` 为 `standard`
- `scenario_id`、`route_id`、`weather_id` 能对应到你刚导出的 XML

### 7.8 第 7 步：修改 YAML 并执行测试

假设你刚刚在 `center` 地图上创建了场景 1，并准备好了：

- `safebench/scenario/scenario_data/center/scenario_01_routes/*.xml`
- `safebench/scenario/scenario_data/center/scenarios/scenario_01.json`
- `safebench/scenario/scenario_data/center/standard_scenario_01.json`

那么你需要把 [safebench/scenario/config/standard.yaml](../safebench/scenario/config/standard.yaml) 调整成类似这样：

```yaml
scenario_type_dir: 'safebench/scenario/scenario_data/center'
scenario_type: 'standard_scenario_01.json'
scenario_category: 'planning'
policy_type: 'standard'

route_dir: 'safebench/scenario/scenario_data/center'
scenario_id: 1
route_id: null
```

然后回到仓库根目录运行：

```bash
python scripts/run.py \
  --mode eval \
  --agent_cfg behavior.yaml \
  --scenario_cfg standard.yaml \
  --exp_name center_s01_eval \
  --render false \
  --save_video false
```

这就是“从场景制作到真正开跑测试”的完整闭环。

---

## 8. 常见文件和目录

### 8.1 运行配置

- [scripts/run.py](../scripts/run.py)
- [safebench/agent/config](../safebench/agent/config)
- [safebench/scenario/config](../safebench/scenario/config)

### 8.2 场景构建

- [tools/CarlaScenariosBuilder/get_map_data.py](../tools/CarlaScenariosBuilder/get_map_data.py)
- [tools/CarlaScenariosBuilder/create_routes.py](../tools/CarlaScenariosBuilder/create_routes.py)
- [tools/CarlaScenariosBuilder/create_scenarios.py](../tools/CarlaScenariosBuilder/create_scenarios.py)
- [tools/CarlaScenariosBuilder/export.py](../tools/CarlaScenariosBuilder/export.py)
- [tools/CarlaScenariosBuilder/export_scenarios.py](../tools/CarlaScenariosBuilder/export_scenarios.py)
- [tools/CarlaScenariosBuilder/readme.txt](../tools/CarlaScenariosBuilder/readme.txt)

### 8.3 运行时场景数据

- [safebench/scenario/scenario_data](../safebench/scenario/scenario_data)

### 8.4 输出结果

- [log](../log)

---

## 9. 常见问题

### 9.1 `create_routes.py` 或 `create_scenarios.py` 找不到文件

优先检查是不是在仓库根目录直接运行了脚本。

正确姿势通常是：

```bash
cd tools/CarlaScenariosBuilder
python create_routes.py ...
```

### 9.2 新地图名报 `Map xxx is not supported`

说明 [tools/CarlaScenariosBuilder/utilities.py](../tools/CarlaScenariosBuilder/utilities.py) 里的 `get_map_centers()` 还没有这个地图的中心点定义。

### 9.3 只导出了 XML，没有 `scenarios/scenario_XX.json`

优先检查是否在旧版本脚本或错误目录下执行了导出。

正确姿势通常是：

```bash
cd tools/CarlaScenariosBuilder
python export.py --map <map> --scenario <id> --format standard
```

如果这条命令执行后仍然没有 `scenarios/scenario_XX.json`，再去检查：

- `origin_dir` 里是否存在 `scenario_XX_scenarios/*.npy`
- `origin_dir` 里是否存在对应的 `scenario_XX_routes/*.npy`

### 9.4 YAML 已经改了，但运行时还是找不到场景

按这个顺序检查：

1. `scenario_type_dir` 是否正确
2. `scenario_type` 文件是否真实存在
3. `route_dir` 下是否存在对应的 `scenario_XX_routes/*.xml`
4. `route_dir/scenarios/` 下是否存在对应的 `scenario_XX.json`
5. `scenario_type` 列表中的 `scenario_id / route_id / weather_id` 是否与 XML 文件名匹配

### 9.5 为什么视频没有出现在仓库根目录 `video/`

当前实现把视频保存到了实验输出目录下，例如：

```text
log/<exp_name>/<exp_name>_<agent>_<scenario>_seed_<seed>/video/<timestamp>/
```

不是仓库根目录的 `video/`。

---

## 10. 推荐使用顺序

如果你是第一次接触这个仓库，建议按下面顺序操作：

1. 先直接用现成 `standard.yaml` 跑通一次评测
2. 确认会看 `log/.../runtime.log` 和 `eval_results/`
3. 再进入 `tools/CarlaScenariosBuilder` 生成 waypoint
4. 交互式创建路线
5. 为路线添加 Trigger 和 Actor
6. 导出 XML 和场景 JSON
7. 把导出结果复制到 `safebench/scenario/scenario_data/<map>/`
8. 准备或修改 `standard_scenario_XX.json`
9. 修改 `scenario/config/*.yaml`
10. 最后再运行 `scripts/run.py`

照这个顺序做，基本就能把“场景制作”和“测试执行”完整串起来。
