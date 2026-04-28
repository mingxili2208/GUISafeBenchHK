"""
基于正确触发机制的场景优化器

触发逻辑：
1. Ego车辆行驶中
2. 当distance(ego, actor) <= trigger_distance时，actor开始行动
3. Actor从静止开始加速移动
4. 目标：让actor在ego行驶路径上制造碰撞

关键变量：
- trigger_distance: ego距离actor多近时触发
- actor_speed: actor被触发后的移动速度
- collision_point: 预计的碰撞点位置（在ego的行驶路径上）
"""

import numpy as np
import math
import carla
from typing import Dict, List, Any, Tuple
from abc import ABC, abstractmethod
from sko.GA import GA


class BaseScenarioOptimizer(ABC):
    """场景优化器基类 - 支持Yaw优化版"""

    def __init__(self, scenario):
        # ... (保持原有的初始化代码不变) ...
        self.scenario = scenario
        self.ego_vehicle = scenario.ego_vehicle

        self.ego_transform = scenario.ego_vehicle.get_transform()
        self.ego_location = self.ego_transform.location
        self.ego_velocity = scenario.ego_vehicle.get_velocity()
        self.ego_speed = math.sqrt(self.ego_velocity.x ** 2 + self.ego_velocity.y ** 2)
        self.ego_heading = self.ego_transform.rotation.yaw

        if self.ego_speed < 1.0:
            self.ego_speed = 5.6

        ego_heading_rad = math.radians(self.ego_heading)
        self.ego_forward = np.array([
            math.cos(ego_heading_rad),
            math.sin(ego_heading_rad)
        ])

        # 获取Actor基础信息
        if hasattr(scenario.config, 'other_actors') and len(scenario.config.other_actors) > 0:
            self.actor_base_transform = scenario.config.other_actors[0].transform
            self.actor_base_location = self.actor_base_transform.location
            self.actor_base_heading = self.actor_base_transform.rotation.yaw
        else:
            # Fallback
            self.actor_base_location = carla.Location(self.ego_location.x + 30, self.ego_location.y,
                                                      self.ego_location.z)
            self.actor_base_heading = self.ego_heading
            self.actor_base_transform = carla.Transform(self.actor_base_location,
                                                        carla.Rotation(yaw=self.actor_base_heading))

        # 基础方向向量 (用于计算位置偏移，保持基于车道的坐标系)
        base_heading_rad = math.radians(self.actor_base_heading)
        self.actor_forward = np.array([math.cos(base_heading_rad), math.sin(base_heading_rad)])
        self.actor_right = np.array([-math.sin(base_heading_rad), math.cos(base_heading_rad)])

        # 遗传算法参数
        self.population_size = 50  # 稍微增加种群以适应更多维度
        self.generations = 50
        self.mutation_rate = 0.2
        self.crossover_rate = 0.8
        self.best_params = None
        self.best_fitness = None
        self.ga = None

    def _calculate_distance(self, loc1, loc2):
        return math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)

    def _calculate_actor_position(self, genes: np.ndarray) -> carla.Location:
        """位置计算保持不变，基于基础坐标系偏移"""
        params = self.genes_to_params(genes)
        offset = params['position_offset']

        # 使用基础方向向量计算位置，这样X/Y偏移仍然是相对于车道的
        offset_x = offset['x'] * self.actor_right[0] + offset['y'] * self.actor_forward[0]
        offset_y = offset['x'] * self.actor_right[1] + offset['y'] * self.actor_forward[1]

        return carla.Location(
            self.actor_base_location.x + offset_x,
            self.actor_base_location.y + offset_y,
            self.actor_base_location.z
        )

    def _get_current_actor_direction(self, yaw_offset: float) -> np.ndarray:
        """[新增] 根据优化的yaw偏移量，计算actor实际的冲锋方向"""
        final_yaw = self.actor_base_heading + yaw_offset
        final_yaw_rad = math.radians(final_yaw)
        return np.array([
            math.cos(final_yaw_rad),
            math.sin(final_yaw_rad)
        ])

    def _find_intersection_point(self, ego_start, ego_direction, actor_start, actor_direction):
        # ... (保持原有的求交点逻辑不变) ...
        p1 = np.array([ego_start.x, ego_start.y])
        d1 = ego_direction
        p2 = np.array([actor_start.x, actor_start.y])
        d2 = actor_direction  # 注意：这里传入的必须是加入yaw偏移后的方向

        A = np.column_stack([d1, -d2])
        b = p2 - p1
        try:
            t = np.linalg.solve(A, b)
            t1, t2 = t[0], t[1]
            intersection = p1 + t1 * d1
            intersection_point = carla.Location(intersection[0], intersection[1], ego_start.z)
            return intersection_point, abs(t1), abs(t2)
        except:
            return carla.Location(0, 0, 0), 20.0, 9999.0

    @abstractmethod
    def get_bounds(self) -> List[Tuple[float, float]]:
        pass

    @abstractmethod
    def evaluate_fitness(self, genes: np.ndarray) -> float:
        pass

    @abstractmethod
    def genes_to_params(self, genes: np.ndarray) -> Dict[str, Any]:
        pass

    def optimize(self) -> Dict[str, Any]:
        print(f">> 开始包含Yaw角的碰撞优化: {self.scenario.name}")
        bounds = self.get_bounds()

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
        best_x, best_y = self.ga.run()
        self.best_params = best_x
        self.best_fitness = -best_y if not isinstance(best_y, np.ndarray) else -best_y.item()

        result = self.genes_to_params(self.best_params)

        # 打印结果时加入Yaw信息
        actor_pos = self._calculate_actor_position(self.best_params)
        yaw_offset = result['position_offset']['yaw']
        final_yaw = self.actor_base_heading + yaw_offset

        print(f">> 优化完成 - 评分: {self.best_fitness:.4f}")
        print(f"   Actor速度: {result['actor_speed']:.2f} m/s")
        print(f"   触发距离: {result['trigger_distance']:.2f} m")
        print(f"   位置偏移: x={result['position_offset']['x']:.2f}m, y={result['position_offset']['y']:.2f}m")
        print(f"   Yaw角偏移: {yaw_offset:.2f}° (最终朝向: {final_yaw:.2f}°)")

        return result


class OppositeVehicleOptimizer(BaseScenarioOptimizer):
    """对向车辆闯红灯场景优化器 - 含Yaw优化"""

    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        [0] actor_speed: 速度
        [1] trigger_distance: 触发距离
        [2] lateral_offset: 横向偏移
        [3] longitudinal_offset: 纵向偏移
        [4] yaw_offset: 角度偏移 (-30度 到 +30度) <--- 新增
        """
        return [
            (6.3, 22.0),
            (19.5, 45.0),
            (-3.0, 3.0),
            (-4.0, 4.0),
            (-10.0, 10.0)  # 新增 Yaw 范围，允许车头偏转攻击
        ]

    def genes_to_params(self, genes: np.ndarray) -> Dict[str, Any]:
        """将5维基因映射为参数"""
        return {
            'actor_speed': float(genes[0]),
            'trigger_distance': float(genes[1]),
            'position_offset': {
                'x': float(genes[2]),
                'y': float(genes[3]),
                'yaw': float(genes[4])  # <--- 获取 Yaw
            }
        }

    def evaluate_fitness(self, genes: np.ndarray) -> float:
        try:
            params = self.genes_to_params(genes)
            actor_speed = params['actor_speed']
            trigger_distance_threshold = params['trigger_distance']
            yaw_offset = params['position_offset']['yaw']  # 获取 yaw 偏移

            self.ego_speed = 6.9

            # 1. 计算起始位置 (基于基础坐标系偏移)
            actor_start_pos = self._calculate_actor_position(genes)

            # 2. 【关键】计算带有 yaw 偏移的实际前进方向
            current_actor_direction = self._get_current_actor_direction(yaw_offset)

            # === 第一阶段：计算触发时刻 (同原版) ===
            current_distance_to_actor = self._calculate_distance(self.ego_location, actor_start_pos)
            if current_distance_to_actor <= trigger_distance_threshold:
                time_to_trigger = 0.0
                trigger_ego_location = self.ego_location
            else:
                distance_to_reach_trigger = current_distance_to_actor - trigger_distance_threshold
                time_to_trigger = distance_to_reach_trigger / self.ego_speed
                trigger_ego_location = carla.Location(
                    self.ego_location.x + self.ego_forward[0] * distance_to_reach_trigger,
                    self.ego_location.y + self.ego_forward[1] * distance_to_reach_trigger,
                    self.ego_location.z
                )

            # === 第二阶段：找碰撞点 (使用新的 current_actor_direction) ===
            collision_point, ego_dist_to_collision, actor_dist_to_collision = self._find_intersection_point(
                trigger_ego_location, self.ego_forward,
                actor_start_pos, current_actor_direction  # <--- 传入这一帧实际的朝向向量
            )

            # 检查交点合理性
            if ego_dist_to_collision < 0 or ego_dist_to_collision > 100:
                ego_dist_to_collision = 25.0
                # 如果没有交点，假设一个虚拟碰撞点
                collision_point = carla.Location(
                    trigger_ego_location.x + self.ego_forward[0] * ego_dist_to_collision,
                    trigger_ego_location.y + self.ego_forward[1] * ego_dist_to_collision,
                    trigger_ego_location.z
                )
                actor_dist_to_collision = self._calculate_distance(actor_start_pos, collision_point)

            if actor_dist_to_collision < 0 or actor_dist_to_collision > 100:
                return 1e6

            # === 计算时间 ===
            time_ego_to_collision = ego_dist_to_collision / self.ego_speed
            ego_total_time = time_to_trigger + time_ego_to_collision

            time_actor_to_collision = actor_dist_to_collision / actor_speed +2.0  # 增加2秒反应时间,qidongxuyao
            actor_total_time = time_to_trigger + time_actor_to_collision

            # === 评分 ===
            time_diff = actor_total_time - ego_total_time

            # 时间评分 (保持原逻辑)
            if -0.2 <= time_diff <= 0.3:
                collision_score = 30.0 - abs(time_diff) * 20.0
            elif -0.5 <= time_diff <= 0.8:
                collision_score = 22.0 - abs(time_diff - 0.05) * 15.0
            else:
                collision_score = max(0.1, 8.0 / (abs(time_diff) + 0.5))

            # 速度评分
            speed_bonus = (actor_speed - 15.0) / 12.0 * 4.0

            # 距离评分
            distance_bonus = 3.0 if actor_dist_to_collision < 40.0 else 0.0

            # 触发评分
            trigger_bonus = 2.0 if 25.0 <= trigger_distance_threshold <= 45.0 else 0.5

            # 【新增】Yaw 角度评分：鼓励更有攻击性的角度
            # 计算 Ego 和 Actor 路径的夹角，越接近垂直(90度)或对冲(180度)通常越危险
            # dot_product: 1=平行同向, 0=垂直, -1=对冲
            dot_product = np.dot(self.ego_forward, current_actor_direction)
            if -0.8 < dot_product < 0.2:  # 稍微倾斜到垂直撞击
                angle_bonus = 3.0
            else:
                angle_bonus = 0.5

            total_score = collision_score + speed_bonus + distance_bonus + trigger_bonus + angle_bonus

            return -total_score

        except Exception as e:
            return 1e6


class SignalizedJunctionLeftTurnOptimizer(BaseScenarioOptimizer):
    """信号灯左转场景优化器 - 改进版"""

    def get_bounds(self) -> List[Tuple[float, float]]:
        return [
            (4.0, 14.0),  # actor速度 (提高下限，确保能追上ego)
            (9.8, 40.0),  # 触发距离阈值 (扩大范围)
            (-4.0, 4.0),  # 横向偏移 (扩大范围，允许更灵活的位置)
            (-8.0, 4.0),  # 纵向偏移 (扩大范围)
            (-15.0, 15.0)  # yaw偏移 (新增：允许actor调整冲击角度)
        ]

    def evaluate_fitness(self, genes: np.ndarray) -> float:
        """
        改进的左转场景评估：
        1. 使用多个候选碰撞点（左转路径上的不同位置）
        2. 考虑actor的实际行驶方向（加入yaw优化）
        3. 动态计算最佳碰撞时机
        """
        try:
            params = self.genes_to_params(genes)
            actor_speed = params['actor_speed']
            trigger_distance_threshold = params['trigger_distance']
            yaw_offset = params['position_offset']['yaw']

            # Actor的起始位置和实际行驶方向
            actor_start_pos = self._calculate_actor_position(genes)
            actor_direction = self._get_current_actor_direction(yaw_offset)

            # === 第一阶段：计算触发时刻 ===
            current_distance_to_actor = self._calculate_distance(self.ego_location, actor_start_pos)

            if current_distance_to_actor <= trigger_distance_threshold:
                time_to_trigger = 0.0
                trigger_ego_location = self.ego_location
            else:
                distance_to_reach_trigger = current_distance_to_actor - trigger_distance_threshold
                time_to_trigger = distance_to_reach_trigger / self.ego_speed
                trigger_ego_location = carla.Location(
                    self.ego_location.x + self.ego_forward[0] * distance_to_reach_trigger,
                    self.ego_location.y + self.ego_forward[1] * distance_to_reach_trigger,
                    self.ego_location.z
                )

            # === 第二阶段：在左转路径上寻找最佳碰撞点 ===
            ego_turn_speed = self.ego_speed * 0.7  # 左转减速

            # 定义左转路径上的多个候选点
            # 左转是逆时针90度转弯，分为3个阶段采样
            best_collision_score = -999
            best_time_diff = 999
            best_collision_point = None

            # 左转向量：ego_forward旋转-90度
            left_vector = np.array([-self.ego_forward[1], self.ego_forward[0]])

            # 在左转路径上采样5个候选碰撞点
            for i in range(5):
                # 前进距离：8-18米
                forward_dist = 8.0 + i * 2.5
                # 左偏距离：随前进距离增加而增加（模拟转弯）
                left_dist = (i / 4.0) * 8.0  # 0到8米的左偏

                # 计算候选碰撞点
                candidate_point = carla.Location(
                    trigger_ego_location.x + self.ego_forward[0] * forward_dist + left_vector[0] * left_dist,
                    trigger_ego_location.y + self.ego_forward[1] * forward_dist + left_vector[1] * left_dist,
                    trigger_ego_location.z
                )

                # 计算ego到这个点的路径长度（近似弧长）
                straight_part = forward_dist
                if left_dist > 0:
                    # 转弯部分用弧长公式：半径*角度(弧度)
                    turn_radius = forward_dist * 0.8  # 估算转弯半径
                    turn_angle_rad = left_dist / turn_radius if turn_radius > 0 else 0
                    arc_length = turn_radius * turn_angle_rad
                    ego_path_length = straight_part * 0.7 + arc_length
                else:
                    ego_path_length = straight_part

                time_ego_to_point = ego_path_length / ego_turn_speed
                ego_arrival_time = time_to_trigger + time_ego_to_point

                # 计算actor到这个点的距离和时间
                actor_dist_to_point = self._calculate_distance(actor_start_pos, candidate_point)

                # 检查actor是否能朝这个方向行驶（点积检查）
                actor_to_point = np.array([
                    candidate_point.x - actor_start_pos.x,
                    candidate_point.y - actor_start_pos.y
                ])
                actor_to_point_norm = np.linalg.norm(actor_to_point)
                if actor_to_point_norm > 0.1:
                    actor_to_point = actor_to_point / actor_to_point_norm
                    direction_match = np.dot(actor_direction, actor_to_point)

                    # 如果actor朝向与目标点夹角太大（>90度），跳过
                    if direction_match < -0.2:
                        continue

                    # 考虑方向偏差的实际行驶距离（夹角越大，实际距离越长）
                    angle_penalty = 1.0 / (0.5 + direction_match)  # direction_match越小，惩罚越大
                    effective_distance = actor_dist_to_point * angle_penalty
                else:
                    effective_distance = actor_dist_to_point

                actor_travel_time = effective_distance / actor_speed + 1.5  # 加1.5秒启动时间
                actor_arrival_time = time_to_trigger + actor_travel_time

                # 计算时间差
                time_diff = actor_arrival_time - ego_arrival_time

                # 评分：时间差越小越好
                if abs(time_diff) < abs(best_time_diff):
                    best_time_diff = time_diff
                    best_collision_point = candidate_point

                    # 距离合理性检查
                    if 5.0 < actor_dist_to_point < 50.0:
                        distance_feasible = True
                    else:
                        distance_feasible = False

                    if distance_feasible:
                        best_collision_score = 1.0 / (abs(time_diff) + 0.15)

            # 如果没找到合理的碰撞点，使用默认点
            if best_collision_point is None:
                best_collision_point = carla.Location(
                    trigger_ego_location.x + self.ego_forward[0] * 12.0 + left_vector[0] * 5.0,
                    trigger_ego_location.y + self.ego_forward[1] * 12.0 + left_vector[1] * 5.0,
                    trigger_ego_location.z
                )
                best_time_diff = 5.0
                best_collision_score = 0.1

            # === 综合评分 ===
            # 1. 时间同步评分（最重要）
            if abs(best_time_diff) <= 0.3:
                time_score = 35.0 - abs(best_time_diff) * 50.0
            elif abs(best_time_diff) <= 0.8:
                time_score = 25.0 - abs(best_time_diff) * 20.0
            elif abs(best_time_diff) <= 1.5:
                time_score = 15.0 - abs(best_time_diff) * 8.0
            else:
                time_score = max(0.1, 8.0 / (abs(best_time_diff) + 0.3))

            # 2. 速度评分：actor速度要足够快才能追上
            if 6.0 <= actor_speed <= 12.0:
                speed_score = 5.0
            elif actor_speed > 12.0:
                speed_score = 5.0 - (actor_speed - 12.0) * 0.3  # 太快也不好
            else:
                speed_score = actor_speed * 0.5  # 太慢给低分

            # 3. 触发距离评分
            actor_dist_to_best_point = self._calculate_distance(actor_start_pos, best_collision_point)
            if 15.0 <= trigger_distance_threshold <= 35.0:
                trigger_score = 3.0
            else:
                trigger_score = 1.0

            # 4. Actor位置合理性
            if 10.0 < actor_dist_to_best_point < 40.0:
                position_score = 3.0
            elif 5.0 < actor_dist_to_best_point < 55.0:
                position_score = 1.5
            else:
                position_score = 0.3

            # 5. 方向匹配评分（新增）
            # 检查actor方向是否有利于碰撞
            actor_to_collision = np.array([
                best_collision_point.x - actor_start_pos.x,
                best_collision_point.y - actor_start_pos.y
            ])
            actor_to_collision_norm = np.linalg.norm(actor_to_collision)
            if actor_to_collision_norm > 0.1:
                actor_to_collision = actor_to_collision / actor_to_collision_norm
                direction_alignment = np.dot(actor_direction, actor_to_collision)
                if direction_alignment > 0.7:  # 朝向碰撞点
                    direction_score = 4.0
                elif direction_alignment > 0.3:
                    direction_score = 2.0
                else:
                    direction_score = 0.5
            else:
                direction_score = 0.5

            total_score = time_score + speed_score + trigger_score + position_score + direction_score

            # 调试输出（可选）
            # if time_score > 25:
            #     print(f"   [左转优化] time_diff={best_time_diff:.2f}s, total={total_score:.2f}, "
            #           f"speed={actor_speed:.1f}, trigger={trigger_distance_threshold:.1f}, yaw={yaw_offset:.1f}")

            return -total_score

        except Exception as e:
            print(f">> Left turn fitness error: {e}")
            import traceback
            traceback.print_exc()
            return 1e6

    def genes_to_params(self, genes: np.ndarray) -> Dict[str, Any]:
        return {
            'actor_speed': float(genes[0]),
            'trigger_distance': float(genes[1]),
            'position_offset': {
                'x': float(genes[2]),
                'y': float(genes[3]),
                'yaw': float(genes[4])  # 新增yaw优化
            }
        }

class SignalizedJunctionRightTurnOptimizer(BaseScenarioOptimizer):
    """信号灯右转场景优化器"""

    # def get_bounds(self) -> List[Tuple[float, float]]:
    #     return [
    #         (2.0, 10.0),   # actor速度
    #         (5.0, 30.0),   # 触发距离阈值
    #         (-3.0, 3.0),   # 横向偏移
    #         (-5.0, 6.0)    # 纵向偏移
    #     ]

    def get_bounds(self) -> List[Tuple[float, float]]:
        return [
            (2.0, 10.0),  # actor速度 (提高下限，确保能追上ego)
            (5.0, 30.0),  # 触发距离阈值 (扩大范围)
            (-3.0, 3.0),  # 横向偏移 (扩大范围，允许更灵活的位置)
            (-5.0, 6.0),  # 纵向偏移 (扩大范围)
            (-10.0, 10.0)  # yaw偏移 (新增：允许actor调整冲击角度)
        ]

    def evaluate_fitness(self, genes: np.ndarray) -> float:
        """
        改进的左转场景评估：
        1. 使用多个候选碰撞点（左转路径上的不同位置）
        2. 考虑actor的实际行驶方向（加入yaw优化）
        3. 动态计算最佳碰撞时机
        """
        try:
            params = self.genes_to_params(genes)
            actor_speed = params['actor_speed']
            trigger_distance_threshold = params['trigger_distance']
            yaw_offset = params['position_offset']['yaw']

            # Actor的起始位置和实际行驶方向
            actor_start_pos = self._calculate_actor_position(genes)
            actor_direction = self._get_current_actor_direction(yaw_offset)

            # === 第一阶段：计算触发时刻 ===
            current_distance_to_actor = self._calculate_distance(self.ego_location, actor_start_pos)

            if current_distance_to_actor <= trigger_distance_threshold:
                time_to_trigger = 0.0
                trigger_ego_location = self.ego_location
            else:
                distance_to_reach_trigger = current_distance_to_actor - trigger_distance_threshold
                time_to_trigger = distance_to_reach_trigger / self.ego_speed
                trigger_ego_location = carla.Location(
                    self.ego_location.x + self.ego_forward[0] * distance_to_reach_trigger,
                    self.ego_location.y + self.ego_forward[1] * distance_to_reach_trigger,
                    self.ego_location.z
                )

            # === 第二阶段：在左转路径上寻找最佳碰撞点 ===
            ego_turn_speed = self.ego_speed * 0.7  # 左转减速

            # 定义左转路径上的多个候选点
            # 左转是逆时针90度转弯，分为3个阶段采样
            best_collision_score = -999
            best_time_diff = 999
            best_collision_point = None

            # 左转向量：ego_forward旋转-90度
            left_vector = np.array([-self.ego_forward[1], self.ego_forward[0]])

            # 在左转路径上采样5个候选碰撞点
            for i in range(5):
                # 前进距离：8-18米
                forward_dist = 8.0 + i * 2.5
                # 左偏距离：随前进距离增加而增加（模拟转弯）
                left_dist = (i / 4.0) * 8.0  # 0到8米的左偏

                # 计算候选碰撞点
                candidate_point = carla.Location(
                    trigger_ego_location.x + self.ego_forward[0] * forward_dist + left_vector[0] * left_dist,
                    trigger_ego_location.y + self.ego_forward[1] * forward_dist + left_vector[1] * left_dist,
                    trigger_ego_location.z
                )

                # 计算ego到这个点的路径长度（近似弧长）
                straight_part = forward_dist
                if left_dist > 0:
                    # 转弯部分用弧长公式：半径*角度(弧度)
                    turn_radius = forward_dist * 0.8  # 估算转弯半径
                    turn_angle_rad = left_dist / turn_radius if turn_radius > 0 else 0
                    arc_length = turn_radius * turn_angle_rad
                    ego_path_length = straight_part * 0.7 + arc_length
                else:
                    ego_path_length = straight_part

                time_ego_to_point = ego_path_length / ego_turn_speed
                ego_arrival_time = time_to_trigger + time_ego_to_point

                # 计算actor到这个点的距离和时间
                actor_dist_to_point = self._calculate_distance(actor_start_pos, candidate_point)

                # 检查actor是否能朝这个方向行驶（点积检查）
                actor_to_point = np.array([
                    candidate_point.x - actor_start_pos.x,
                    candidate_point.y - actor_start_pos.y
                ])
                actor_to_point_norm = np.linalg.norm(actor_to_point)
                if actor_to_point_norm > 0.1:
                    actor_to_point = actor_to_point / actor_to_point_norm
                    direction_match = np.dot(actor_direction, actor_to_point)

                    # 如果actor朝向与目标点夹角太大（>90度），跳过
                    if direction_match < -0.2:
                        continue

                    # 考虑方向偏差的实际行驶距离（夹角越大，实际距离越长）
                    angle_penalty = 1.0 / (0.5 + direction_match)  # direction_match越小，惩罚越大
                    effective_distance = actor_dist_to_point * angle_penalty
                else:
                    effective_distance = actor_dist_to_point

                actor_travel_time = effective_distance / actor_speed + 1.5  # 加1.5秒启动时间
                actor_arrival_time = time_to_trigger + actor_travel_time

                # 计算时间差
                time_diff = actor_arrival_time - ego_arrival_time

                # 评分：时间差越小越好
                if abs(time_diff) < abs(best_time_diff):
                    best_time_diff = time_diff
                    best_collision_point = candidate_point

                    # 距离合理性检查
                    if 5.0 < actor_dist_to_point < 50.0:
                        distance_feasible = True
                    else:
                        distance_feasible = False

                    if distance_feasible:
                        best_collision_score = 1.0 / (abs(time_diff) + 0.15)

            # 如果没找到合理的碰撞点，使用默认点
            if best_collision_point is None:
                best_collision_point = carla.Location(
                    trigger_ego_location.x + self.ego_forward[0] * 12.0 + left_vector[0] * 5.0,
                    trigger_ego_location.y + self.ego_forward[1] * 12.0 + left_vector[1] * 5.0,
                    trigger_ego_location.z
                )
                best_time_diff = 5.0
                best_collision_score = 0.1

            # === 综合评分 ===
            # 1. 时间同步评分（最重要）
            if abs(best_time_diff) <= 0.3:
                time_score = 35.0 - abs(best_time_diff) * 50.0
            elif abs(best_time_diff) <= 0.8:
                time_score = 25.0 - abs(best_time_diff) * 20.0
            elif abs(best_time_diff) <= 1.5:
                time_score = 15.0 - abs(best_time_diff) * 8.0
            else:
                time_score = max(0.1, 8.0 / (abs(best_time_diff) + 0.3))

            # 2. 速度评分：actor速度要足够快才能追上
            if 6.0 <= actor_speed <= 12.0:
                speed_score = 5.0
            elif actor_speed > 12.0:
                speed_score = 5.0 - (actor_speed - 12.0) * 0.3  # 太快也不好
            else:
                speed_score = actor_speed * 0.5  # 太慢给低分

            # 3. 触发距离评分
            actor_dist_to_best_point = self._calculate_distance(actor_start_pos, best_collision_point)
            if 15.0 <= trigger_distance_threshold <= 35.0:
                trigger_score = 3.0
            else:
                trigger_score = 1.0

            # 4. Actor位置合理性
            if 10.0 < actor_dist_to_best_point < 40.0:
                position_score = 3.0
            elif 5.0 < actor_dist_to_best_point < 55.0:
                position_score = 1.5
            else:
                position_score = 0.3

            # 5. 方向匹配评分（新增）
            # 检查actor方向是否有利于碰撞
            actor_to_collision = np.array([
                best_collision_point.x - actor_start_pos.x,
                best_collision_point.y - actor_start_pos.y
            ])
            actor_to_collision_norm = np.linalg.norm(actor_to_collision)
            if actor_to_collision_norm > 0.1:
                actor_to_collision = actor_to_collision / actor_to_collision_norm
                direction_alignment = np.dot(actor_direction, actor_to_collision)
                if direction_alignment > 0.7:  # 朝向碰撞点
                    direction_score = 4.0
                elif direction_alignment > 0.3:
                    direction_score = 2.0
                else:
                    direction_score = 0.5
            else:
                direction_score = 0.5

            total_score = time_score + speed_score + trigger_score + position_score + direction_score

            # 调试输出（可选）
            # if time_score > 25:
            #     print(f"   [左转优化] time_diff={best_time_diff:.2f}s, total={total_score:.2f}, "
            #           f"speed={actor_speed:.1f}, trigger={trigger_distance_threshold:.1f}, yaw={yaw_offset:.1f}")

            return -total_score

        except Exception as e:
            print(f">> Left turn fitness error: {e}")
            import traceback
            traceback.print_exc()
            return 1e6

    def genes_to_params(self, genes: np.ndarray) -> Dict[str, Any]:
        return {
            'actor_speed': float(genes[0]),
            'trigger_distance': float(genes[1]),
            'position_offset': {
                'x': float(genes[2]),
                'y': float(genes[3]),
                'yaw': float(genes[4])  # 新增yaw优化
            }
        }


class NoSignalJunctionOptimizer(BaseScenarioOptimizer):
    """无信号路口场景优化器 - 包含Yaw角优化，确保路径相交"""

    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        定义优化参数范围（5维）
        [0] actor_speed: 速度 (8-18 m/s)
        [1] trigger_distance: 触发距离 (10-40m)
        [2] lateral_offset: 横向偏移 (-4到4m)
        [3] longitudinal_offset: 纵向偏移 (-8到8m)
        [4] yaw_offset: 角度偏移 (-30到30度) - 扩大范围以找到交点
        """
        return [
            (7.0, 18.0),  # actor速度
            (9.0, 38.0),  # 触发距离阈值（扩大范围）
            (-4.0, 4.0),  # 横向偏移
            (-3.0, 3.0),  # 纵向偏移
            (-60.0, 60.0)  # yaw偏移（扩大到±30度）
        ]

    def genes_to_params(self, genes: np.ndarray) -> Dict[str, Any]:
        """将5维基因映射为参数"""
        return {
            'actor_speed': float(genes[0]),
            'trigger_distance': float(genes[1]),
            'position_offset': {
                'x': float(genes[2]),
                'y': float(genes[3]),
                'yaw': float(genes[4])
            }
        }

    def evaluate_fitness(self, genes: np.ndarray) -> float:
        """
        评估适应度函数
        关键：必须确保ego和actor的路径有交点！
        """
        try:
            params = self.genes_to_params(genes)
            actor_speed = params['actor_speed']
            trigger_distance_threshold = params['trigger_distance']
            yaw_offset = params['position_offset']['yaw']

            # 1. 计算actor的起始位置和实际行驶方向
            actor_start_pos = self._calculate_actor_position(genes)
            actor_direction = self._get_current_actor_direction(yaw_offset)

            # === 第一阶段：计算触发时刻 ===
            current_distance_to_actor = self._calculate_distance(self.ego_location, actor_start_pos)

            if current_distance_to_actor <= trigger_distance_threshold:
                time_to_trigger = 0.0
                trigger_ego_location = self.ego_location
            else:
                distance_to_reach_trigger = current_distance_to_actor - trigger_distance_threshold
                time_to_trigger = distance_to_reach_trigger / self.ego_speed
                trigger_ego_location = carla.Location(
                    self.ego_location.x + self.ego_forward[0] * distance_to_reach_trigger,
                    self.ego_location.y + self.ego_forward[1] * distance_to_reach_trigger,
                    self.ego_location.z
                )

            # === 第二阶段：找碰撞点（两条路径的交点）===
            # 这是关键：使用 _find_intersection_point 找到真正的路径交点
            collision_point, ego_dist_to_collision, actor_dist_to_collision = self._find_intersection_point(
                trigger_ego_location, self.ego_forward,
                actor_start_pos, actor_direction
            )

            # === 严格检查：如果没有有效交点，给予严厉惩罚 ===
            # ego_dist_to_collision 和 actor_dist_to_collision 必须都是正数且合理
            if ego_dist_to_collision <= 0 or ego_dist_to_collision > 80:
                # 没有前方交点，这个配置无效
                return 1e6

            if actor_dist_to_collision <= 0 or actor_dist_to_collision > 80:
                # actor无法到达交点，这个配置无效
                return 1e6

            # 额外检查：交点必须在合理的路口范围内
            # 从触发点到交点的距离应该在5-50米之间（路口大小）
            if ego_dist_to_collision < 5.0 or ego_dist_to_collision > 50.0:
                return 1e6

            if actor_dist_to_collision < 5.0 or actor_dist_to_collision > 50.0:
                return 1e6

            # === 验证交点的合理性：使用向量点积 ===
            # Actor必须朝向交点方向
            actor_to_collision = np.array([
                collision_point.x - actor_start_pos.x,
                collision_point.y - actor_start_pos.y
            ])
            actor_to_collision_norm = np.linalg.norm(actor_to_collision)
            if actor_to_collision_norm > 0.1:
                actor_to_collision_normalized = actor_to_collision / actor_to_collision_norm
                direction_alignment = np.dot(actor_direction, actor_to_collision_normalized)

                # 如果actor朝向与交点方向夹角太大（>60度，即cos<0.5），则无效
                if direction_alignment < 0.5:
                    return 1e6
            else:
                return 1e6

            # === 计算时间 ===
            time_ego_to_collision = ego_dist_to_collision / self.ego_speed
            ego_total_time = time_to_trigger + time_ego_to_collision

            time_actor_to_collision = actor_dist_to_collision / actor_speed + 1.5  # 加1.5秒启动时间
            actor_total_time = time_to_trigger + time_actor_to_collision

            time_diff = actor_total_time - ego_total_time

            # === 评分系统 ===
            # 1. 时间同步评分（权重最高）- 目标：完全同时到达交点
            # if abs(time_diff) <= 0.2:
            #     time_score = 40.0 - abs(time_diff) * 100.0
            # elif abs(time_diff) <= 0.5:
            #     time_score = 30.0 - abs(time_diff) * 40.0
            # elif abs(time_diff) <= 1.2:
            #     time_score = 20.0 - abs(time_diff) * 15.0
            # else:
            #     time_score = max(0.1, 12.0 / (abs(time_diff) + 0.5))

            # 1. 时间同步评分 - 偏向让actor稍早到达（-0.3到+0.1秒最佳）
            if -0.12 <= time_diff <= 0.2:
                # 最佳窗口：actor稍早到达，避免ego开过去
                time_score = 45.0 - abs(time_diff + 0.1) * 80.0
            elif -0.6 <= time_diff <= 0.4:
                # 次优窗口
                time_score = 35.0 - abs(time_diff) * 40.0
            elif -1.0 <= time_diff <= 1.0:
                # 可接受窗口
                time_score = 25.0 - abs(time_diff) * 20.0
            else:
                # 时间差太大
                time_score = max(0.1, 15.0 / (abs(time_diff) + 0.5))

            # 2. 交点位置评分 - 距离适中最好
            if 12.0 < ego_dist_to_collision < 25.0:
                intersection_score = 8.0
            elif 8.0 < ego_dist_to_collision < 35.0:
                intersection_score = 5.0
            else:
                intersection_score = 2.0

            # 3. 速度匹配评分
            speed_diff = abs(actor_speed - self.ego_speed)
            if speed_diff <= 3.0:
                speed_score = 5.0
            elif speed_diff <= 6.0:
                speed_score = 3.0
            else:
                speed_score = max(0.5, 3.0 - speed_diff * 0.3)

            # 4. Actor距离合理性
            if 12.0 < actor_dist_to_collision < 30.0:
                actor_dist_score = 4.0
            elif 8.0 < actor_dist_to_collision < 40.0:
                actor_dist_score = 2.0
            else:
                actor_dist_score = 0.5

            # 5. 触发距离评分
            if 15.0 <= trigger_distance_threshold <= 35.0:
                trigger_score = 3.0
            else:
                trigger_score = 1.0

            # 6. 方向对齐评分 - 已经在上面验证过，这里给额外奖励
            # direction_alignment 已经计算过了
            if actor_to_collision_norm > 0.1:
                if direction_alignment > 0.85:
                    direction_bonus = 5.0
                elif direction_alignment > 0.7:
                    direction_bonus = 3.0
                else:
                    direction_bonus = 1.0
            else:
                direction_bonus = 0.0

            # 7. 碰撞角度评分 - 垂直碰撞最危险
            dot_product = np.dot(self.ego_forward, actor_direction)
            angle_deg = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

            # 理想角度：60-120度（侧面碰撞）
            if 70.0 <= angle_deg <= 110.0:
                angle_score = 4.0
            elif 50.0 <= angle_deg <= 130.0:
                angle_score = 2.0
            else:
                angle_score = 0.5

            total_score = (time_score + intersection_score + speed_score +
                           actor_dist_score + trigger_score + direction_bonus + angle_score)

            # 调试输出（可选）
            if total_score > 50:
                print(f"   [良好配置] 总分={total_score:.2f}, 时间差={time_diff:.2f}s, "
                      f"交点距离(ego)={ego_dist_to_collision:.1f}m, "
                      f"交点距离(actor)={actor_dist_to_collision:.1f}m, "
                      f"方向对齐={direction_alignment:.2f}, 碰撞角度={angle_deg:.1f}°")

            return -total_score

        except Exception as e:
            # 如果计算出错（比如两条路径平行），返回极差的适应度
            return 1e6

    def _project_point_on_line(self, point, line_start, line_direction):
        """
        将点投影到直线上（辅助函数，可选）
        返回：投影点，以及从line_start到投影点的距离
        """
        point_vec = np.array([point.x - line_start.x, point.y - line_start.y])
        projection_length = np.dot(point_vec, line_direction)

        projection_point = carla.Location(
            line_start.x + line_direction[0] * projection_length,
            line_start.y + line_direction[1] * projection_length,
            line_start.z
        )

        return projection_point, projection_length


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