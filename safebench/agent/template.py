"""
[你的模型名称] 自动驾驶算法的 Safebench 集成模板
基于 [论文名称/自研方案] 的端到端自动驾驶算法实现

模型需要的输入（根据你的模型修改）:
- RGB图像 (请填写尺寸, 如 900x256, FOV=100)
- 车辆状态 (速度、朝向等)
- 导航信息 (目标点、路线指令等)

模型的输出（根据你的模型修改）:
- throttle, steer, brake 控制信号
"""

import os
import sys
import math
import numpy as np
from collections import deque, OrderedDict

import torch
from torchvision import transforms as T

# 导入 SafeBench 基类
from safebench.agent.base_policy import BasePolicy


# -------------------------- 可选：添加你的模型路径 --------------------------
# 如果模型不在当前目录，添加模型根目录到 sys.path
# MODEL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'your_model_dir')
# if MODEL_ROOT not in sys.path:
#     sys.path.insert(0, MODEL_ROOT)

# 导入你的模型和配置（替换下面的注释）
# from your_model import YourModel
# from your_config import ModelConfig


# -------------------------- 通用路线规划器（可复用） --------------------------
class YourModelRoutePlanner:
    """
    通用路线规划器：将 CARLA 全局路点转换为车辆局部坐标系下的目标点
    适用于大多数端到端模型的导航需求
    """

    def __init__(self, min_distance=4.0, max_distance=50.0):
        self.route = deque()  # 存储 (路点坐标, 导航指令)
        self.min_distance = min_distance  # 已通过路点的距离阈值
        self.max_distance = max_distance  # 最远目标点距离

    def set_route(self, waypoints_list):
        """
        设置全局路线
        Args:
            waypoints_list: CARLA waypoint 对象列表
        """
        self.route.clear()
        for waypoint in waypoints_list:
            # 提取路点全局坐标 (x, y)
            pos = np.array([waypoint.transform.location.x, waypoint.transform.location.y])
            # 提取导航指令（默认 LANEFOLLOW=4）
            cmd = getattr(waypoint, 'road_option', 4)
            self.route.append((pos, cmd))

    def get_target_point(self, vehicle_location):
        """
        根据当前车辆位置，获取局部坐标系下的目标点和导航指令
        Args:
            vehicle_location: CARLA 车辆位置对象 (包含 x, y 属性)
        Returns:
            local_target: 局部坐标系下的目标点 [x, y]
            nav_cmd: 导航指令整数
        """
        if not self.route:
            return np.array([0.0, 0.0]), 4

        # 当前车辆全局坐标
        curr_pos = np.array([vehicle_location.x, vehicle_location.y])

        # 移除已通过的路点
        to_pop = 0
        for i in range(len(self.route)):
            wp_pos, _ = self.route[i]
            dist = np.linalg.norm(wp_pos - curr_pos)
            if dist <= self.min_distance:
                to_pop = i + 1
        for _ in range(min(to_pop, len(self.route) - 1)):
            self.route.popleft()

        # 获取下一个目标路点
        target_pos, nav_cmd = self.route[0]
        return target_pos - curr_pos, nav_cmd  # 转换为相对坐标


# -------------------------- 核心：模型集成类（修改重点） --------------------------
class YourModelAgent(BasePolicy):
    """
    端到端自动驾驶模型 SafeBench 集成模板
    替换 [你的模型名称] 为实际模型名，修改核心方法即可适配新模型
    """
    # -------------------------- 1. 必须修改：模型标识 --------------------------
    name = 'your_model_name'  # 模型名称，SafeBench 识别用
    type = 'unlearnable'  # 预训练模型填 'unlearnable'；可训练填 'rl'/'bc'

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.logger = logger
        self.num_scenario = config['num_scenario']  # 并行场景数
        self.ego_action_dim = config['ego_action_dim']  # 动作维度 (一般是 2/3)

        # -------------------------- 2. 必须修改：模型配置 --------------------------
        self.model_path = config['model_path']  # 模型权重路径
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model_config = ModelConfig()  # 替换为你的模型配置
        self.net = None  # 模型实例

        # -------------------------- 3. 可选：图像预处理（根据模型修改） --------------------------
        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Resize((256, 900)),  # 替换为你的模型输入尺寸 (H, W)
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 均值/std
        ])

        # -------------------------- 通用：多场景状态管理 --------------------------
        self.ego_vehicles = []  # 存储各场景的 ego 车辆
        self.planner_list = []  # 各场景独立的路线规划器
        self.step_count = []  # 各场景的步数计数

    # -------------------------- 4. 必须实现：SafeBench 接口 --------------------------
    def set_ego_and_route(self, ego_vehicles, info, static_obs=None):
        """
        设置 ego 车辆和路线信息（SafeBench 调用）
        """
        self.ego_vehicles = ego_vehicles
        self.planner_list = []
        self.step_count = [0] * len(ego_vehicles)

        # 为每个场景初始化独立的路线规划器
        for e_i in range(len(ego_vehicles)):
            planner = RoutePlanner()
            if 'route_waypoints' in info[e_i]:
                planner.set_route(info[e_i]['route_waypoints'])
            self.planner_list.append(planner)

    def train(self, replay_buffer):
        """
        训练接口：预训练模型直接 pass；可训练模型在此实现训练逻辑
        """
        pass

    def set_mode(self, mode):
        """
        设置模型模式：train/eval
        """
        if self.net is not None:
            self.net.train() if mode == 'train' else self.net.eval()

    def save_model(self, episode=None):
        """
        模型保存：预训练模型 pass；可训练模型在此实现保存逻辑
        """
        pass

    # -------------------------- 5. 必须修改：模型加载逻辑 --------------------------
    def load_model(self):
        """
        加载预训练模型权重
        重点：替换为你的模型初始化和权重加载代码
        """
        # 1. 初始化你的模型
        # self.net = YourModel(self.model_config).to(self.device)
        if self.net is None:
            self.logger.log(f">> 请先初始化 {self.name} 模型", 'yellow')
            return

        # 2. 加载权重文件
        if not os.path.exists(self.model_path):
            self.logger.log(f">> 模型权重不存在: {self.model_path}", 'red')
            return

        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)

        # 3. 权重兼容处理（可选：移除前缀、匹配层名）
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # 移除可能的前缀（如 'model.'）
            new_k = k.replace("model.", "")
            new_state_dict[new_k] = v

        # 4. 加载权重
        self.net.load_state_dict(new_state_dict, strict=False)
        self.net.eval()
        self.logger.log(f">> {self.name} 模型加载成功")

    # -------------------------- 6. 必须修改：输入预处理（核心） --------------------------
    def _preprocess_input(self, obs, ego_vehicle, planner):
        """
        预处理 SafeBench 观测数据，转换为模型输入格式
        Args:
            obs: SafeBench 单场景观测数据（字典/数组）
            ego_vehicle: 单场景 ego 车辆对象
            planner: 单场景路线规划器
        Returns:
            model_input: 模型可直接输入的张量/字典
        """
        # 1. 提取 RGB 图像（根据你的 obs 格式修改）
        if isinstance(obs, dict):
            rgb_img = obs.get('img', obs.get('camera'))
        else:
            rgb_img = obs
        rgb_img = np.array(rgb_img).astype(np.uint8)
        img_tensor = self.img_transform(rgb_img).unsqueeze(0).to(self.device)  # [1, 3, H, W]

        # 2. 提取车辆状态（速度示例，可添加朝向、加速度等）
        v = ego_vehicle.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        speed_tensor = torch.tensor([speed / 12.0], dtype=torch.float32).to(self.device)  # 归一化

        # 3. 提取导航信息（局部目标点 + 指令）
        target_point, nav_cmd = planner.get_target_point(ego_vehicle.transform.location)
        target_tensor = torch.tensor(target_point, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 4. 组合模型输入（根据你的模型输入格式修改）
        model_input = {
            'image': img_tensor,
            'speed': speed_tensor,
            'target_point': target_tensor,
            'nav_cmd': nav_cmd
        }
        return model_input

    # -------------------------- 7. 必须修改：模型推理 + 输出后处理（核心） --------------------------
    @torch.no_grad()
    def get_action(self, obs, infos, deterministic=False):
        """
        SafeBench 核心推理接口：输入观测，输出控制信号
        Args:
            obs: 所有场景的观测数据列表/数组
            infos: 所有场景的信息字典列表
            deterministic: 是否确定性推理
        Returns:
            actions: 控制信号数组，shape [num_scenario, action_dim]
        """
        # 初始化模型（如果未加载）
        if self.net is None:
            self.load_model()

        actions = []
        for e_i in range(len(self.ego_vehicles)):
            self.step_count[e_i] += 1
            ego_vehicle = self.ego_vehicles[e_i]
            planner = self.planner_list[e_i]

            # 1. 提取单场景观测
            curr_obs = obs[e_i] if isinstance(obs, (list, np.ndarray)) else obs

            # 2. 前 N 帧输出零动作（可选：等待模型预热）
            if self.step_count[e_i] <= 10:
                actions.append([0.0, 0.0])
                continue

            try:
                # 3. 输入预处理
                model_input = self._preprocess_input(curr_obs, ego_vehicle, planner)

                # 4. 模型推理（替换为你的模型前向逻辑）
                # pred = self.net(**model_input)
                # throttle, steer, brake = self._postprocess_output(pred, model_input)
                throttle, steer, brake = 0.0, 0.0, 0.0  # 临时占位

                # 5. 控制信号后处理（通用：裁剪范围、互斥逻辑）
                steer = np.clip(steer, -1.0, 1.0)
                throttle = np.clip(throttle, 0.0, 0.75)
                brake = np.clip(brake, 0.0, 1.0)

                # 刹车时关闭油门（安全逻辑）
                if brake > 0.1:
                    throttle = 0.0

                # 6. 收集动作（根据 action_dim 调整，一般是 [throttle, steer] 或 [throttle, steer, brake]）
                actions.append([throttle, steer])

            except Exception as e:
                self.logger.log(f">> 场景 {e_i} 推理错误: {str(e)}", 'red')
                actions.append([0.0, 0.0])

        return np.array(actions, dtype=np.float32)

    # -------------------------- 可选：输出后处理（根据模型修改） --------------------------
    def _postprocess_output(self, model_output, model_input):
        """
        模型输出转换为控制信号
        Args:
            model_output: 模型原始输出
            model_input: 预处理后的输入（可能需要用于辅助决策）
        Returns:
            throttle, steer, brake: 控制信号
        """
        # 替换为你的输出解析逻辑
        # 示例：如果模型直接输出 [throttle, steer, brake]
        # return model_output[:, 0].item(), model_output[:, 1].item(), model_output[:, 2].item()
        return 0.0, 0.0, 0.0
