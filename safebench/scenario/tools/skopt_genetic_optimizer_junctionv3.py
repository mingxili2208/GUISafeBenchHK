"""
基于正确触发机制的场景优化器

触发逻辑：
1. Ego车辆行驶中
2. 当ego距离触发点 <= trigger_distance时，actor开始行动
3. Actor从静止开始加速移动
4. 目标：让actor在ego到达碰撞点时正好也到达

关键变量：
- trigger_distance: ego触发actor的距离阈值
- trigger_point: config中定义的触发点位置
- actor_speed: actor被触发后的移动速度
- collision_point: 预计的碰撞点位置
"""

import numpy as np
import math
import carla
from typing import Dict, List, Any, Tuple
from abc import ABC, abstractmethod
from sko.GA import GA


class BaseScenarioOptimizer(ABC):
    """场景优化器基类"""

    def __init__(self, scenario):
        self.scenario = scenario
        self.ego_vehicle = scenario.ego_vehicle

        # 获取ego的详细状态
        self.ego_transform = scenario.ego_vehicle.get_transform()
        self.ego_location = self.ego_transform.location

        self.ego_velocity = scenario.ego_vehicle.get_velocity()
        self.ego_speed = math.sqrt(self.ego_velocity.x ** 2 + self.ego_velocity.y ** 2)
        self.ego_heading = self.ego_transform.rotation.yaw

        if self.ego_speed < 1.0:
            self.ego_speed = 5.6

        # 计算ego的前进方向向量
        ego_heading_rad = math.radians(self.ego_heading)
        self.ego_forward = np.array([
            math.cos(ego_heading_rad),
            math.sin(ego_heading_rad)
        ])

        # 获取场景配置中的触发点（这是关键修正）
        if hasattr(scenario.config, 'trigger_points') and len(scenario.config.trigger_points) > 0:
            self.trigger_point = scenario.config.trigger_points[0].location
        else:
            # 如果没有触发点，使用ego前方30米作为默认触发点
            self.trigger_point = carla.Location(
                self.ego_location.x + self.ego_forward[0] * 30.0,
                self.ego_location.y + self.ego_forward[1] * 30.0,
                self.ego_location.z
            )

        # 获取场景配置中的actor基础位置
        if hasattr(scenario.config, 'other_actors') and len(scenario.config.other_actors) > 0:
            self.actor_base_transform = scenario.config.other_actors[0].transform
            self.actor_base_location = self.actor_base_transform.location
            self.actor_base_heading = self.actor_base_transform.rotation.yaw
        else:
            self.actor_base_location = carla.Location(
                self.ego_location.x + 30.0,
                self.ego_location.y,
                self.ego_location.z
            )
            self.actor_base_heading = self.ego_heading
            self.actor_base_transform = carla.Transform(
                self.actor_base_location,
                carla.Rotation(yaw=self.actor_base_heading)
            )

        # 遗传算法参数
        self.population_size = 30
        self.generations = 30
        self.mutation_rate = 0.2
        self.crossover_rate = 0.8

        self.best_params = None
        self.best_fitness = None
        self.ga = None

        print(f">> Ego状态:")
        print(f"   位置: ({self.ego_location.x:.2f}, {self.ego_location.y:.2f})")
        print(f"   速度: {self.ego_speed:.2f} m/s")
        print(f"   朝向: {self.ego_heading:.2f}°")
        print(f">> 触发点位置: ({self.trigger_point.x:.2f}, {self.trigger_point.y:.2f})")
        print(f">> Actor基础位置: ({self.actor_base_location.x:.2f}, {self.actor_base_location.y:.2f})")

    def _calculate_distance(self, loc1: carla.Location, loc2: carla.Location) -> float:
        """计算两点间的欧氏距离"""
        return math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)

    def _calculate_actor_position(self, genes: np.ndarray) -> carla.Location:
        """根据基因计算actor的实际起始位置"""
        params = self.genes_to_params(genes)
        offset = params['position_offset']

        # 基于actor的基础朝向计算偏移
        actor_heading_rad = math.radians(self.actor_base_heading)
        forward = np.array([math.cos(actor_heading_rad), math.sin(actor_heading_rad)])
        right = np.array([-math.sin(actor_heading_rad), math.cos(actor_heading_rad)])

        # 应用偏移
        offset_x = offset['x'] * right[0] + offset['y'] * forward[0]
        offset_y = offset['x'] * right[1] + offset['y'] * forward[1]

        return carla.Location(
            self.actor_base_location.x + offset_x,
            self.actor_base_location.y + offset_y,
            self.actor_base_location.z
        )

    @abstractmethod
    def get_bounds(self) -> List[Tuple[float, float]]:
        """获取优化参数的边界"""
        pass

    @abstractmethod
    def evaluate_fitness(self, genes: np.ndarray) -> float:
        """评估适应度（返回负值，越小越好 = 越危险）"""
        pass

    @abstractmethod
    def genes_to_params(self, genes: np.ndarray) -> Dict[str, Any]:
        """将基因转换为场景参数"""
        pass

    def optimize(self) -> Dict[str, Any]:
        """执行优化"""
        print(f">> 开始基于触发机制的碰撞优化: {self.scenario.name}")

        bounds = self.get_bounds()

        # 创建遗传算法实例
        self.ga = GA(
            func=self.evaluate_fitness,
            n_dim=len(bounds),
            size_pop=self.population_size,
            max_iter=self.generations,
            prob_mut=self.mutation_rate,
            lb=[b[0] for b in bounds],
            ub=[b[1] for b in bounds],
            precision=[1e-6] * len(bounds)
        )

        # 执行优化
        best_x, best_y = self.ga.run()

        self.best_params = best_x
        self.best_fitness = -best_y

        if isinstance(self.best_fitness, np.ndarray):
            self.best_fitness = self.best_fitness.item()

        # 验证参数有效性
        if np.any(np.isnan(self.best_params)) or np.any(np.isinf(self.best_params)):
            print(">> Warning: Invalid parameters, using defaults")
            self.best_params = np.array([(b[0] + b[1]) / 2 for b in bounds])

        result = self.genes_to_params(self.best_params)

        # 打印优化结果和预测
        actor_pos = self._calculate_actor_position(self.best_params)
        print(f">> 优化完成 - 碰撞风险评分: {self.best_fitness:.4f}")
        print(f"   Actor速度: {result['actor_speed']:.2f} m/s")
        print(f"   触发距离阈值: {result['trigger_distance']:.2f} m")
        print(f"   Actor起始位置: ({actor_pos.x:.2f}, {actor_pos.y:.2f})")

        return result


class OppositeVehicleOptimizer(BaseScenarioOptimizer):
    """对向车辆闯红灯场景优化器"""

    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        [0] actor_speed: 对向车速度 (5.5-22 m/s)
        [1] trigger_distance: ego触发actor的距离阈值 (9-35 m)
        [2] lateral_offset: actor横向偏移 (-2 to 2 m)
        [3] longitudinal_offset: actor纵向偏移 (-3 to 3 m)
        """
        return [
            (5.5, 22.0),
            (8.0, 35.0),
            (-2.0, 2.0),
            (-3.0, 3.0)
        ]

    def evaluate_fitness(self, genes: np.ndarray) -> float:
        """
        计算逻辑（修正版）：
        1. ego当前位置 → 到达触发点（trigger_point）
        2. 当 distance(ego, trigger_point) <= trigger_distance 时触发actor
        3. 触发后，ego继续前进到碰撞点
        4. actor从起始位置移动到碰撞点
        5. 目标：两者同时到达
        """
        try:
            params = self.genes_to_params(genes)
            actor_speed = params['actor_speed']
            trigger_distance_threshold = params['trigger_distance']

            # === 第一阶段：ego到达触发条件的时刻 ===
            # ego当前位置到触发点的距离
            distance_ego_to_trigger_point = self._calculate_distance(self.ego_location, self.trigger_point)

            # 触发条件：ego距离触发点 <= trigger_distance_threshold
            # 即：ego需要行驶到距离触发点trigger_distance_threshold的位置
            if distance_ego_to_trigger_point <= trigger_distance_threshold:
                # ego已经在触发范围内
                time_to_trigger = 0.0
                trigger_ego_location = self.ego_location
            else:
                # ego需要行驶 (distance_ego_to_trigger_point - trigger_distance_threshold) 才能进入触发范围
                distance_to_reach_trigger = distance_ego_to_trigger_point - trigger_distance_threshold
                time_to_trigger = distance_to_reach_trigger / self.ego_speed

                # 触发时ego的位置（沿着前进方向）
                trigger_ego_location = carla.Location(
                    self.ego_location.x + self.ego_forward[0] * distance_to_reach_trigger,
                    self.ego_location.y + self.ego_forward[1] * distance_to_reach_trigger,
                    self.ego_location.z
                )

            # === 第二阶段：触发后到碰撞点 ===
            # 碰撞点：在触发点附近（假设在触发点前方的路口中心）
            distance_trigger_to_collision = self.trigger_point.distance(self.actor_base_location)

            collision_point = carla.Location(
                self.trigger_point.x + self.ego_forward[0] * distance_trigger_to_collision * 0.5,
                self.trigger_point.y + self.ego_forward[1] * distance_trigger_to_collision * 0.5,
                self.trigger_point.z
            )

            # ego从触发位置到碰撞点的时间
            distance_trigger_ego_to_collision = self._calculate_distance(trigger_ego_location, collision_point)
            time_trigger_to_collision = distance_trigger_ego_to_collision / self.ego_speed

            # ego总时间
            ego_total_time = time_to_trigger + time_trigger_to_collision

            # === Actor时间计算 ===
            # actor起始位置（应用偏移）
            actor_start_pos = self._calculate_actor_position(genes)

            # actor到碰撞点的距离
            actor_distance_to_collision = self._calculate_distance(actor_start_pos, collision_point)

            # actor移动时间（从触发时刻开始）
            actor_travel_time = actor_distance_to_collision / actor_speed

            # actor总时间 = ego到达触发的时间 + actor移动时间
            actor_total_time = time_to_trigger + actor_travel_time

            # === 碰撞评分 ===
            time_diff = actor_total_time - ego_total_time

            # 理想：actor比ego早到0-0.5秒
            if 0.0 <= time_diff <= 0.5:
                collision_score = 20.0 - time_diff * 10.0
            elif -0.3 <= time_diff < 0.0:
                collision_score = 18.0 + time_diff * 20.0
            elif 0.5 < time_diff <= 1.5:
                collision_score = 15.0 - (time_diff - 0.5) * 8.0
            elif -1.0 <= time_diff < -0.3:
                collision_score = 12.0 - abs(time_diff + 0.3) * 10.0
            else:
                collision_score = max(0.1, 5.0 / (abs(time_diff) + 0.5))

            # 速度加成
            speed_bonus = (actor_speed - 12.0) / 10.0 * 3.0

            # 位置惩罚
            lateral_penalty = abs(params['position_offset']['x']) * 0.5

            total_score = collision_score + speed_bonus - lateral_penalty

            return -total_score

        except Exception as e:
            print(f">> Fitness error: {e}")
            return 1e6

    def genes_to_params(self, genes: np.ndarray) -> Dict[str, Any]:
        return {
            'actor_speed': float(genes[0]),
            'trigger_distance': float(genes[1]),
            'position_offset': {
                'x': float(genes[2]),
                'y': float(genes[3]),
                'yaw': 0.0
            }
        }


class SignalizedJunctionLeftTurnOptimizer(BaseScenarioOptimizer):
    """信号灯左转场景优化器"""

    def get_bounds(self) -> List[Tuple[float, float]]:
        return [
            (2.0, 10.0),  # actor速度
            (5.0, 45.0),  # 触发距离阈值
            (-3.0, 3.0),  # 横向偏移
            (-6.0, 2.0)  # 纵向偏移
        ]

    def evaluate_fitness(self, genes: np.ndarray) -> float:
        """
        正确的触发逻辑（修正版）：
        1. Actor在初始位置等待
        2. 当 distance(ego, trigger_point) <= trigger_distance 时，actor被触发
        3. 目标：计算何时触发，以及触发后ego和actor能否在碰撞点相遇
        """
        try:
            params = self.genes_to_params(genes)
            actor_speed = params['actor_speed']
            trigger_distance_threshold = params['trigger_distance']

            # Actor的起始位置
            actor_start_pos = self._calculate_actor_position(genes)

            # === 计算触发时刻 ===
            # Ego当前位置到触发点的距离
            distance_ego_to_trigger_point = self._calculate_distance(self.ego_location, self.trigger_point)

            # 判断是否需要移动才能触发
            if distance_ego_to_trigger_point <= trigger_distance_threshold:
                # 已经在触发范围内
                time_to_trigger = 0.0
                trigger_ego_location = self.ego_location
            else:
                # 需要移动到触发范围
                distance_to_reach_trigger = distance_ego_to_trigger_point - trigger_distance_threshold
                time_to_trigger = distance_to_reach_trigger / self.ego_speed

                # 触发时ego的位置
                trigger_ego_location = carla.Location(
                    self.ego_location.x + self.ego_forward[0] * distance_to_reach_trigger,
                    self.ego_location.y + self.ego_forward[1] * distance_to_reach_trigger,
                    self.ego_location.z
                )

            # === 预测碰撞点 ===
            # 左转场景：碰撞点在触发点附近的左转路径上
            distance_trigger_to_collision = 17.0
            ego_turn_speed = self.ego_speed * 0.65  # 左转减速

            # 碰撞点：触发点前方15米，左侧8米
            collision_point = carla.Location(
                self.trigger_point.x + self.ego_forward[0] * 15.0 - self.ego_forward[1] * 8.0,
                self.trigger_point.y + self.ego_forward[1] * 15.0 + self.ego_forward[0] * 8.0,
                self.trigger_point.z
            )

            # Ego从触发位置到碰撞点的时间
            distance_trigger_ego_to_collision = self._calculate_distance(trigger_ego_location, collision_point)
            time_trigger_to_collision = distance_trigger_ego_to_collision / ego_turn_speed

            # Ego总时间
            ego_total_time = time_to_trigger + time_trigger_to_collision

            # === Actor时间计算 ===
            actor_distance_to_collision = self._calculate_distance(actor_start_pos, collision_point)
            actor_travel_time = actor_distance_to_collision / actor_speed
            actor_total_time = time_to_trigger + actor_travel_time

            # === 碰撞评分 ===
            time_diff = actor_total_time - ego_total_time

            if -0.5 <= time_diff <= 0.5:
                collision_score = 25.0 - abs(time_diff) * 15.0
            elif -1.0 <= time_diff <= 1.5:
                collision_score = 18.0 - abs(time_diff) * 8.0
            elif -2.0 <= time_diff <= 2.5:
                collision_score = 10.0 - abs(time_diff) * 4.0
            else:
                collision_score = max(0.1, 5.0 / (abs(time_diff) + 0.5))

            speed_bonus = (actor_speed - 8.0) / 10.0 * 3.0

            if 25.0 <= trigger_distance_threshold <= 40.0:
                trigger_bonus = 2.0
            else:
                trigger_bonus = 0.5

            y_offset = params['position_offset']['y']
            if -6.0 <= y_offset <= 2.0:
                position_bonus = 1.5
            else:
                position_bonus = 0.3

            total_score = collision_score + speed_bonus + trigger_bonus + position_bonus

            if collision_score > 20:
                print(f"   [高危配置] time_diff={time_diff:.2f}s, score={total_score:.2f}, "
                      f"trigger_dist={trigger_distance_threshold:.1f}m, actor_speed={actor_speed:.1f}m/s")

            return -total_score

        except Exception as e:
            print(f">> Fitness error: {e}")
            return 1e6

    def genes_to_params(self, genes: np.ndarray) -> Dict[str, Any]:
        return {
            'actor_speed': float(genes[0]),
            'trigger_distance': float(genes[1]),
            'position_offset': {
                'x': float(genes[2]),
                'y': float(genes[3]),
                'yaw': 0.0
            }
        }


class SignalizedJunctionRightTurnOptimizer(BaseScenarioOptimizer):
    """信号灯右转场景优化器"""

    def get_bounds(self) -> List[Tuple[float, float]]:
        return [
            (2.0, 10.0),  # actor速度
            (5.0, 30.0),  # 触发距离阈值
            (-3.0, 3.0),  # 横向偏移
            (-5.0, 6.0)  # 纵向偏移
        ]

    def evaluate_fitness(self, genes: np.ndarray) -> float:
        try:
            params = self.genes_to_params(genes)
            actor_speed = params['actor_speed']
            trigger_distance_threshold = params['trigger_distance']

            # Actor的起始位置
            actor_start_pos = self._calculate_actor_position(genes)

            # === 计算触发时刻 ===
            distance_ego_to_trigger_point = self._calculate_distance(self.ego_location, self.trigger_point)

            if distance_ego_to_trigger_point <= trigger_distance_threshold:
                time_to_trigger = 0.0
                trigger_ego_location = self.ego_location
            else:
                distance_to_reach_trigger = distance_ego_to_trigger_point - trigger_distance_threshold
                time_to_trigger = distance_to_reach_trigger / self.ego_speed
                trigger_ego_location = carla.Location(
                    self.ego_location.x + self.ego_forward[0] * distance_to_reach_trigger,
                    self.ego_location.y + self.ego_forward[1] * distance_to_reach_trigger,
                    self.ego_location.z
                )

            # === 预测碰撞点 ===
            ego_turn_speed = self.ego_speed * 0.6
            distance_trigger_to_collision = 18.0

            # 右转碰撞点：触发点前方12米，右侧5米
            collision_point = carla.Location(
                self.trigger_point.x + self.ego_forward[0] * 12.0 + self.ego_forward[1] * 5.0,
                self.trigger_point.y + self.ego_forward[1] * 12.0 - self.ego_forward[0] * 5.0,
                self.trigger_point.z
            )

            distance_trigger_ego_to_collision = self._calculate_distance(trigger_ego_location, collision_point)
            time_trigger_to_collision = distance_trigger_ego_to_collision / ego_turn_speed

            ego_total_time = time_to_trigger + time_trigger_to_collision

            # === Actor时间计算 ===
            actor_distance_to_collision = self._calculate_distance(actor_start_pos, collision_point)
            actor_travel_time = actor_distance_to_collision / actor_speed
            actor_total_time = time_to_trigger + actor_travel_time

            # === 碰撞评分 ===
            time_diff = actor_total_time - ego_total_time

            if -0.3 <= time_diff <= 0.6:
                collision_score = 16.0 - abs(time_diff - 0.15) * 12.0
            elif -1.0 <= time_diff <= 1.2:
                collision_score = 10.0 - abs(time_diff - 0.15) * 6.0
            else:
                collision_score = max(0.1, 5.0 / (abs(time_diff) + 0.5))

            speed_bonus = (actor_speed - 10.0) / 8.0 * 1.5
            total_score = collision_score + speed_bonus

            return -total_score

        except Exception as e:
            return 1e6

    def genes_to_params(self, genes: np.ndarray) -> Dict[str, Any]:
        return {
            'actor_speed': float(genes[0]),
            'trigger_distance': float(genes[1]),
            'position_offset': {
                'x': float(genes[2]),
                'y': float(genes[3]),
                'yaw': 0.0
            }
        }


class NoSignalJunctionOptimizer(BaseScenarioOptimizer):
    """无信号路口场景优化器"""

    def get_bounds(self) -> List[Tuple[float, float]]:
        return [
            (8.0, 18.0),  # actor速度
            (10.0, 35.0),  # 触发距离阈值
            (-4.0, 4.0),  # 横向偏移
            (-8.0, 8.0)  # 纵向偏移
        ]

    def evaluate_fitness(self, genes: np.ndarray) -> float:
        try:
            params = self.genes_to_params(genes)
            actor_speed = params['actor_speed']
            trigger_distance_threshold = params['trigger_distance']

            # Actor的起始位置
            actor_start_pos = self._calculate_actor_position(genes)

            # === 计算触发时刻 ===
            distance_ego_to_trigger_point = self._calculate_distance(self.ego_location, self.trigger_point)

            if distance_ego_to_trigger_point <= trigger_distance_threshold:
                time_to_trigger = 0.0
                trigger_ego_location = self.ego_location
            else:
                distance_to_reach_trigger = distance_ego_to_trigger_point - trigger_distance_threshold
                time_to_trigger = distance_to_reach_trigger / self.ego_speed
                trigger_ego_location = carla.Location(
                    self.ego_location.x + self.ego_forward[0] * distance_to_reach_trigger,
                    self.ego_location.y + self.ego_forward[1] * distance_to_reach_trigger,
                    self.ego_location.z
                )

            # === 预测碰撞点 ===
            # 路口中心：触发点前方10米
            distance_to_collision = 10.0
            collision_point = carla.Location(
                self.trigger_point.x + self.ego_forward[0] * distance_to_collision,
                self.trigger_point.y + self.ego_forward[1] * distance_to_collision,
                self.trigger_point.z
            )

            distance_trigger_ego_to_collision = self._calculate_distance(trigger_ego_location, collision_point)
            time_to_collision = distance_trigger_ego_to_collision / self.ego_speed

            ego_total_time = time_to_trigger + time_to_collision

            # === Actor时间计算 ===
            actor_distance = self._calculate_distance(actor_start_pos, collision_point)
            actor_travel_time = actor_distance / actor_speed
            actor_total_time = time_to_trigger + actor_travel_time

            # === 碰撞评分 ===
            time_diff = abs(actor_total_time - ego_total_time)

            # 目标：完全同时到达
            if time_diff <= 0.15:
                collision_score = 20.0
            elif time_diff <= 0.4:
                collision_score = 18.0 - (time_diff - 0.15) * 20.0
            elif time_diff <= 1.0:
                collision_score = 12.0 - (time_diff - 0.4) * 8.0
            else:
                collision_score = max(0.1, 8.0 / time_diff)

            # 速度匹配
            speed_diff = abs(actor_speed - self.ego_speed)
            if speed_diff <= 2.0:
                speed_match_bonus = 3.0
            else:
                speed_match_bonus = max(0, 3.0 - speed_diff * 0.4)

            total_score = collision_score + speed_match_bonus

            return -total_score

        except Exception as e:
            return 1e6

    def genes_to_params(self, genes: np.ndarray) -> Dict[str, Any]:
        return {
            'actor_speed': float(genes[0]),
            'trigger_distance': float(genes[1]),
            'position_offset': {
                'x': float(genes[2]),
                'y': float(genes[3]),
                'yaw': 0.0
            }
        }


def create_optimizer(scenario) -> BaseScenarioOptimizer:
    """工厂函数：根据场景类型创建对应的优化器"""
    scenario_name = scenario.name.lower()

    if 'oppositevehicle' in scenario_name or 'runningredlight' in scenario_name:
        return OppositeVehicleOptimizer(scenario)
    elif 'leftturn' in scenario_name:
        return SignalizedJunctionLeftTurnOptimizer(scenario)
    elif 'rightturn' in scenario_name:
        return SignalizedJunctionRightTurnOptimizer(scenario)
    elif 'nosignal' in scenario_name or 'crossingroute' in scenario_name:
        return NoSignalJunctionOptimizer(scenario)
    else:
        print(f">> Warning: Unknown scenario type '{scenario_name}', using default optimizer")
        return SignalizedJunctionLeftTurnOptimizer(scenario)