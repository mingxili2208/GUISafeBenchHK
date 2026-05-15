# Bench2Drive 离屏渲染与视频生成说明

本文说明 Bench2Drive 在不打开 CARLA 可视窗口的情况下，如何仍然获得相机图像，以及这些连续图像如何进一步生成视频。

核心结论：

- Bench2Drive 不需要打开 Unreal Editor，也不需要手动点击 Play。
- 它通过命令行启动 CARLA packaged server，并使用 `-RenderOffScreen` 让 CARLA 在后台渲染。
- `-RenderOffScreen` 不是“不渲染”，而是“不显示窗口地渲染”。
- RGB 相机图像来自 CARLA 的 `sensor.camera.rgb` actor。每次仿真 tick，相机 sensor 都会产生一帧图像。
- 视频脚本 `tools/generate_video.py` 不负责渲染，它只把已经保存好的连续图片写成视频。

## 1. 为什么没有 GUI 也能有图像

Bench2Drive 启动 CARLA 的地方在：

```text
leaderboard/leaderboard/leaderboard_evaluator.py
```

关键代码：

```python
cmd1 = f"{os.path.join(self.carla_path, 'CarlaUE4.sh')} -RenderOffScreen -nosound -carla-rpc-port={args.port} -graphicsadapter={args.gpu_rank}"
self.server = subprocess.Popen(cmd1, shell=True, preexec_fn=os.setsid)
```

这里启动的是 CARLA 的 packaged simulator，也就是 `CarlaUE4.sh`，不是 Unreal Editor。参数含义如下：

- `-RenderOffScreen`：使用离屏渲染，不创建可交互的可视窗口。
- `-nosound`：关闭声音。
- `-carla-rpc-port=...`：指定 Python API 连接的 RPC 端口。
- `-graphicsadapter=...`：指定使用哪个 GPU 图形适配器。

因此它规避的是“打开窗口和人工操作 Play”这一步，但没有规避渲染本身。CARLA 仍然会在 GPU 上渲染场景，相机 sensor 会把渲染结果通过 Python API 返回。

可以把它理解成：

```text
传统手动方式：
打开 CARLA/UE 窗口 -> 手动进入仿真 -> 看屏幕画面

Bench2Drive 方式：
命令行启动 CARLA server -> Python API 控制世界 -> camera sensor 离屏产生图像
```

## 2. Bench2Drive 的整体评测链路

Bench2Drive 评测时的大致流程是：

```text
run_evaluation.sh
  -> leaderboard_evaluator.py
    -> 启动 CARLA server, 使用 -RenderOffScreen
    -> carla.Client 连接 server
    -> load_world 加载 Town
    -> RouteScenario spawn ego vehicle
    -> AgentWrapper 根据 agent.sensors() spawn sensors
    -> ScenarioManager 同步 tick
    -> sensor.listen 回调接收图像
    -> agent.run_step(input_data, timestamp)
    -> apply_control 控制 ego vehicle
```

入口脚本在：

```text
leaderboard/scripts/run_evaluation.sh
```

它设置 `CARLA_ROOT`、`PYTHONPATH`、`ROUTES`、`TEAM_AGENT`、`TEAM_CONFIG`、`SAVE_PATH` 等环境变量，然后运行：

```text
leaderboard/leaderboard/leaderboard_evaluator.py
```

真正启动 CARLA server 的位置是：

```text
leaderboard/leaderboard/leaderboard_evaluator.py
```

加载地图和设置同步模式也在这里完成：

```python
client = carla.Client(args.host, args.port)
client.set_timeout(client_timeout)

settings = carla.WorldSettings(
    synchronous_mode=True,
    fixed_delta_seconds=1.0 / self.frame_rate,
    deterministic_ragdolls=True,
    spectator_as_ego=False
)
client.get_world().apply_settings(settings)
```

`synchronous_mode=True` 很重要。它表示仿真由 Python 明确调用 `world.tick()` 推进，每推进一帧，sensor 数据也对应产生一帧。

## 3. 相机 sensor 是如何创建的

在 leaderboard 评测链路中，用户 agent 会实现：

```python
def sensors(self):
    return [
        {
            "type": "sensor.camera.rgb",
            "x": 0.7,
            "y": 0.0,
            "z": 1.6,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "width": 800,
            "height": 600,
            "fov": 100,
            "id": "Center",
        }
    ]
```

Bench2Drive 的 sensor 创建逻辑在：

```text
leaderboard/leaderboard/autoagents/agent_wrapper.py
```

关键流程：

```python
bp = bp_library.find(type_)
for key, value in attributes.items():
    bp.set_attribute(str(key), str(value))
sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, vehicle)
sensor.listen(CallBack(id_, type_, sensor, self._agent.sensor_interface))
```

也就是说：

1. 根据 `sensor.camera.rgb` 找到 CARLA blueprint。
2. 设置分辨率、FOV 等参数。
3. 用 `spawn_actor` 把相机挂到 ego vehicle 上。
4. 用 `sensor.listen(...)` 注册异步回调。

这里的相机不是屏幕上的视角，也不是 spectator view。它是 CARLA world 中的一个传感器 actor，绑定在车体坐标系下。

## 4. 每一帧图像是如何产生的

仿真主循环在：

```text
leaderboard/leaderboard/scenarios/scenario_manager.py
```

关键代码：

```python
CarlaDataProvider.get_world().tick(self._timeout)
```

每次 `world.tick()`，CARLA 推进一帧。对于已经 spawn 的 camera sensor，CARLA 会在这一帧渲染出对应视角的图像，然后触发 `sensor.listen` 注册的回调。

回调处理在：

```text
leaderboard/leaderboard/envs/sensor_interface.py
```

关键代码：

```python
def _parse_image_cb(self, image, tag):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = copy.deepcopy(array)
    array = np.reshape(array, (image.height, image.width, 4))
    self._data_provider.update_sensor(tag, array, image.frame)
```

这里 `image.raw_data` 是 CARLA camera sensor 返回的原始 BGRA 字节流。代码把它转成 numpy 数组，形状是：

```text
(height, width, 4)
```

其中 4 个通道通常是 BGRA。之后图像被放进 `SensorInterface` 的队列里。

agent 每一帧调用时会取出这些 sensor 数据：

```text
leaderboard/leaderboard/autoagents/autonomous_agent.py
```

```python
input_data = self.sensor_interface.get_data(GameTime.get_frame())
control = self.run_step(input_data, timestamp)
```

因此用户 agent 的 `run_step` 收到的 `input_data` 里，就包含相机图像、LiDAR、radar、IMU、GNSS 等 sensor 输出。

## 5. 连续图片是如何保存到磁盘的

注意：leaderboard 评测框架负责把图像传给 agent，但不一定自动保存图片。图片是否落盘，取决于具体 agent 或采集脚本是否调用 `cv2.imwrite`。

Bench2Drive 仓库中保存连续图片的代表实现是：

```text
tools/data_collect.py
```

它定义了多路相机，例如：

```python
{
    "type": "sensor.camera.rgb",
    "x": 0.80,
    "y": 0.0,
    "z": 1.60,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "width": 1600,
    "height": 900,
    "fov": 70,
    "id": "CAM_FRONT",
}
```

在 `tick` 中，它从 `input_data` 里取出相机数据：

```python
cam_bgr_front = input_data["CAM_FRONT"][1][:, :, :3]
```

然后在 `save` 中写成连续图片：

```python
frame = self.count
cv2.imwrite(
    str(self.save_path / "camera" / "rgb_front" / (f"{frame:05}.jpg")),
    tick_data["cam_bgr_front"],
    [cv2.IMWRITE_JPEG_QUALITY, 20],
)
```

保存后的目录结构类似：

```text
scenario_name/
  TownXX_weatherXX_routeXX/
    camera/
      rgb_front/
        00000.jpg
        00001.jpg
        00002.jpg
      rgb_front_left/
      rgb_front_right/
      rgb_back/
      rgb_back_left/
      rgb_back_right/
      rgb_top_down/
```

所以“连续图片”的来源是：

```text
world.tick()
  -> CARLA camera sensor 离屏渲染
  -> sensor.listen 回调收到 raw_data
  -> raw_data 转 numpy
  -> data_collect.py 或 agent debug 代码调用 cv2.imwrite
  -> 00000.jpg, 00001.jpg, 00002.jpg ...
```

## 6. 视频是如何生成的

视频生成脚本是：

```text
tools/generate_video.py
```

它不连接 CARLA，也不做渲染。它只读取已有图片：

```python
images = [
    img
    for img in os.listdir(os.path.join(images_folder, "rgb_front"))
    if img.endswith(".jpg") or img.endswith(".png")
]
images.sort()
```

然后用第一张图片确定视频尺寸：

```python
frame = cv2.imread(os.path.join(images_folder, "rgb_front", images[0]))
height, width, layers = frame.shape
```

创建 OpenCV 视频写入器：

```python
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
```

逐帧读取图片，并叠加 debug 信息：

```python
meta = json.load(open(os.path.join(images_folder, f"meta/{i:04}.json"), "r"))
text = f"speed: {round(speed, 2)}, steer: {round(steer, 2)}, throttle: {round(throttle, 2)}, brake: {round(brake, 2)}"
cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
video.write(img)
```

整体链路是：

```text
rgb_front/00000.jpg
rgb_front/00001.jpg
rgb_front/00002.jpg
...
  -> cv2.VideoWriter
  -> output.mp4
```

## 7. `generate_video.py` 与当前数据目录的一个差异

当前 README 中写的是：

```bash
python tools/generate_video.py -f your_rgb_folder/
```

但仓库里的 `tools/generate_video.py` 当前没有实现 `argparse`，脚本底部仍是：

```python
images_folder = ""
output_video = ""
```

另外，`generate_video.py` 期待的目录是：

```text
images_folder/
  rgb_front/
  meta/
```

而 `tools/data_collect.py` 保存的数据目录是：

```text
save_path/
  camera/
    rgb_front/
  anno/
```

因此，当前视频脚本更像是服务于某种 agent debug 输出目录，而不是完全直接适配 `data_collect.py` 的原始数据目录。如果要直接用采集数据生成视频，需要做一层路径适配，例如读取：

```text
camera/rgb_front/
```

并从：

```text
anno/*.json.gz
```

或其他 meta 文件中读取速度、方向盘、油门、刹车信息。

## 8. 与手动点击 Play 的关系

CARLA 有两种常见使用方式：

### 8.1 手动可视方式

```text
打开 CARLA 窗口
看到渲染画面
人工或脚本连接 CARLA
```

这种方式适合调试，但不适合大规模评测。

### 8.2 Bench2Drive 后台评测方式

```text
Python 通过 subprocess 启动 CarlaUE4.sh
传入 -RenderOffScreen
Python API 连接端口
加载 Town
spawn ego vehicle 和 sensors
world.tick 推进仿真
sensor 回调得到图像
```

这个过程不需要人工点击 Play。CARLA packaged server 启动后，本身就是可被 Python API 控制的仿真服务。

## 9. `-RenderOffScreen` 和 no rendering 的区别

这两个概念很容易混淆：

```text
-RenderOffScreen:
  不显示窗口，但仍然渲染。
  camera sensor 可以输出 RGB 图像。

no rendering / null rendering:
  尽量关闭渲染。
  适合只需要状态、碰撞、控制、物理的场景。
  通常不适合依赖 RGB camera 的任务。
```

Bench2Drive 是自动驾驶感知和闭环评测任务，agent 通常需要相机输入，所以它使用的是 `-RenderOffScreen`，而不是完全关闭渲染。

## 10. 一句话总结

Bench2Drive 的方案是：

```text
不开可视窗口
但启动完整 CARLA 仿真服务
使用离屏渲染生成 camera sensor 图像
Python 同步 tick 获取每帧 sensor 数据
需要保存时用 cv2.imwrite 输出连续图片
最后用 cv2.VideoWriter 合成视频
```

这就是它能在无 GUI 或多 GPU 评测环境中运行，同时仍然获得 RGB 图像和视频调试材料的原因。
