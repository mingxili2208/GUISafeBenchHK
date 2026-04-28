"""
Pylot 自动驾驶算法在 Safebench 框架中的实现

该模块将 pylot 的规划和控制算法集成到 safebench 测试框架中。
主要使用 pylot 的 PID 控制器进行路径跟踪。
"""

import math
import numpy as np
from collections import deque

from safebench.agent.base_policy import BasePolicy

# 从 pylot 导入核心组件
import sys
import os
# 添加 pylot 路径
pylot_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'pylot')
if pylot_path not in sys.path:
    sys.path.insert(0, pylot_path)

from pylot.planning.waypoints import Waypoints
from pylot.control.pid import PIDLongitudinalController, PIDLateralController
import pylot.utils as pylot_utils


class PylotAgent(BasePolicy):
    """
    使用 Pylot 规划和控制算法的自动驾驶 Agent
    
    该 agent 使用 pylot 的 PID 控制器进行路径跟踪:
    - PIDLongitudinalController: 纵向控制，控制油门和刹车
    - PIDLateralController: 横向控制，控制转向
    """
    name = 'pylot'
    type = 'unlearnable'

    def __init__(self, config, logger):
        """
        初始化 Pylot Agent
        
        Args:
            config: 配置字典，包含 PID 参数等
            logger: 日志记录器
        """
        self.logger = logger
        self.num_scenario = config['num_scenario']
        self.ego_action_dim = config['ego_action_dim']
        self.model_path = config['model_path']
        self.mode = 'train'
        self.continue_episode = 0
        
        # PID 控制器参数
        self.pid_p = config.get('pid_p', 1.0)
        self.pid_i = config.get('pid_i', 0.0)
        self.pid_d = config.get('pid_d', 0.0)
        self.steer_gain = config.get('steer_gain', 0.7)
        
        # 目标速度 (m/s)
        self.target_speed = config.get('target_speed', 7.0)
        
        # 路点参数
        self.min_steer_waypoint_distance = config.get('min_steer_waypoint_distance', 5.0)
        self.min_speed_waypoint_distance = config.get('min_speed_waypoint_distance', 3.0)
        
        # 控制器列表 - 每个场景一个
        self.longitudinal_controllers = []
        self.lateral_controllers = []
        self.waypoints_list = []
        self.ego_vehicles = None
        
        # 时间步长
        self.dt = 0.05  # 20 FPS

    def set_ego_and_route(self, ego_vehicles, info, static_obs=None):
        """
        设置 ego 车辆和路线
        
        Args:
            ego_vehicles: ego 车辆列表
            info: 路线信息列表，每个元素包含 'route_waypoints'
            static_obs: 静态观测信息，包含目标速度等
        """
        self.ego_vehicles = ego_vehicles
        self.longitudinal_controllers = []
        self.lateral_controllers = []
        self.waypoints_list = []
        
        for e_i in range(len(ego_vehicles)):
            # 创建纵向 PID 控制器
            long_controller = PIDLongitudinalController(
                K_P=self.pid_p,
                K_D=self.pid_d,
                K_I=self.pid_i,
                dt=self.dt
            )
            self.longitudinal_controllers.append(long_controller)
            
            # 创建横向 PID 控制器
            lat_controller = PIDLateralController(
                K_P=1.0,
                K_D=0.0,
                K_I=0.0,
                dt=self.dt
            )
            self.lateral_controllers.append(lat_controller)
            
            # 转换路点为 pylot 格式
            route_waypoints = info[e_i]['route_waypoints']
            waypoints = self._convert_waypoints(route_waypoints, static_obs[e_i] if static_obs else None)
            self.waypoints_list.append(waypoints)

    def _convert_waypoints(self, route_waypoints, static_obs=None):
        """
        将 safebench 路点转换为 pylot Waypoints 格式
        
        Args:
            route_waypoints: safebench 路点列表 (carla.Waypoint)
            static_obs: 静态观测信息
            
        Returns:
            pylot Waypoints 对象
        """
        waypoint_deque = deque()
        target_speeds = deque()
        
        # 获取目标速度
        if static_obs is not None and 'target_speed' in static_obs:
            target_speed = static_obs['target_speed']
        else:
            target_speed = self.target_speed
        
        for wp in route_waypoints:
            # 从 carla Waypoint 提取位置和旋转
            location = wp.transform.location
            rotation = wp.transform.rotation
            
            # 创建 pylot Transform
            pylot_location = pylot_utils.Location(
                x=location.x,
                y=location.y,
                z=location.z
            )
            pylot_rotation = pylot_utils.Rotation(
                pitch=rotation.pitch,
                yaw=rotation.yaw,
                roll=rotation.roll
            )
            pylot_transform = pylot_utils.Transform(
                location=pylot_location,
                rotation=pylot_rotation
            )
            
            waypoint_deque.append(pylot_transform)
            target_speeds.append(target_speed)
        
        return Waypoints(waypoint_deque, target_speeds)

    def _get_vehicle_transform(self, vehicle):
        """
        获取车辆的 pylot Transform
        
        Args:
            vehicle: carla 车辆对象
            
        Returns:
            pylot Transform 对象
        """
        transform = vehicle.get_transform()
        location = pylot_utils.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z
        )
        rotation = pylot_utils.Rotation(
            pitch=transform.rotation.pitch,
            yaw=transform.rotation.yaw,
            roll=transform.rotation.roll
        )
        return pylot_utils.Transform(location=location, rotation=rotation)

    def _get_vehicle_speed(self, vehicle):
        """
        获取车辆当前速度 (m/s)
        
        Args:
            vehicle: carla 车辆对象
            
        Returns:
            速度 (m/s)
        """
        velocity = vehicle.get_velocity()
        return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    def _compute_steer(self, angle):
        """
        将角度转换为转向值
        
        Args:
            angle: 转向角度 (弧度)
            
        Returns:
            转向值 [-1, 1]
        """
        steer = self.steer_gain * angle
        return np.clip(steer, -1.0, 1.0)

    def train(self, replay_buffer):
        """训练方法 - Pylot 是非学习型 agent，不需要训练"""
        pass

    def set_mode(self, mode):
        """设置运行模式"""
        self.mode = mode

    def get_action(self, obs, infos, deterministic=False):
        """
        获取控制动作
        
        Args:
            obs: 观测值 (未使用，pylot 使用车辆状态)
            infos: 场景信息列表
            deterministic: 是否确定性策略 (Pylot 始终确定性)
            
        Returns:
            actions: numpy 数组，形状 (num_scenario, 2)，包含 [throttle, steer]
        """
        actions = []
        
        for e_i, info in enumerate(infos):
            vehicle = self.ego_vehicles[e_i]
            
            # 获取车辆当前状态
            vehicle_transform = self._get_vehicle_transform(vehicle)
            current_speed = self._get_vehicle_speed(vehicle)
            
            # 获取当前场景的路点和控制器
            waypoints = self.waypoints_list[e_i]
            long_controller = self.longitudinal_controllers[e_i]
            lat_controller = self.lateral_controllers[e_i]
            
            # 更新路点 - 移除已完成的路点
            try:
                waypoints.remove_completed(vehicle_transform.location, vehicle_transform)
            except ValueError:
                # 没有更多路点，停车
                actions.append([0.0, 0.0])
                continue
            
            # 如果路点为空，停车
            if waypoints.is_empty():
                actions.append([0.0, 0.0])
                continue
            
            try:
                # 计算目标速度
                target_speed = waypoints.get_target_speed(
                    vehicle_transform, 
                    self.min_speed_waypoint_distance
                )
                
                # 计算转向角度
                angle = waypoints.get_angle(
                    vehicle_transform,
                    self.min_steer_waypoint_distance
                )
                
                # 使用纵向 PID 控制器计算油门/刹车
                acceleration = long_controller.run_step(target_speed, current_speed)
                
                if acceleration >= 0:
                    throttle = min(acceleration, 1.0)
                    brake = 0.0
                else:
                    throttle = 0.0
                    brake = min(abs(acceleration), 1.0)
                
                # 当速度很低且目标速度为0时，保持刹车
                if current_speed < 1 and target_speed == 0:
                    brake = 1.0
                    throttle = 0.0
                
                # 计算转向
                steer = self._compute_steer(angle)
                
                # 注意：safebench 期望的动作格式是 [throttle, steer]
                # 这里我们将 brake 信息编码进 throttle（负值表示刹车）
                if brake > 0:
                    throttle = -brake
                
                actions.append([throttle, steer])
                
            except ValueError as e:
                # 路点计算出错，紧急停车
                self.logger.log(f'Pylot Agent: ValueError - {e}, emergency stop')
                actions.append([0.0, 0.0])
        
        actions = np.array(actions, dtype=np.float32)
        return actions

    def load_model(self):
        """加载模型 - Pylot 是非学习型 agent，不需要加载"""
        pass

    def save_model(self, episode=None):
        """保存模型 - Pylot 是非学习型 agent，不需要保存"""
        pass
