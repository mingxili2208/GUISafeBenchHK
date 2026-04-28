# safebench/scenario/tools/skopt_genetic_optimizer_VehicleTurningRoute_v4.py
"""
改进版 SkoptGeneticOptimizer for VehicleTurningRoute
- 增加安全距离保证自行车不会立即阻挡ego车辆
- 优化碰撞点计算逻辑
- 添加碰撞可行性验证
11月10 2025
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple
from sko.GA import GA
import carla


class SkoptGeneticOptimizer:
    """车辆转弯场景优化器（改进版）"""

    def __init__(self, scenario):
        self.scenario = scenario

        # ego 状态
        self.ego_transform = scenario.ego_vehicle.get_transform()
        self.ego_velocity = scenario.ego_vehicle.get_velocity()
        self.ego_speed = math.sqrt(self.ego_velocity.x ** 2 + self.ego_velocity.y ** 2)
        self.ego_heading = self.ego_transform.rotation.yaw

        if self.ego_speed < 1.0:
            self.ego_speed = 5.0  # 默认避免除零

        ego_heading_rad = math.radians(self.ego_heading)
        self.ego_forward = np.array([math.cos(ego_heading_rad), math.sin(ego_heading_rad)])
        self.ego_right = np.array([-math.sin(ego_heading_rad), math.cos(ego_heading_rad)])

        # GA 参数
        self.population_size = 50
        self.generations = 50
        self.mutation_rate = 0.15
        self.crossover_rate = 0.8

        self.best_params = None
        self.best_fitness = None
        self.ga = None

        # 安全距离参数
        self.MIN_SAFE_DISTANCE = 15.0  # 自行车与ego最小安全距离(米)
        self.MAX_SPAWN_DISTANCE = 35.0  # 最大生成距离(米)

    # ----------------- 对外优化接口 -----------------
    def optimize_initial_state(self) -> Dict[str, Any]:
        print(">> Starting VehicleTurningRoute optimization (improved version)")
        print(f">> Ego speed: {self.ego_speed:.2f} m/s | Ego heading: {self.ego_heading:.2f}°")
        print(f">> Safe distance range: {self.MIN_SAFE_DISTANCE:.1f}m - {self.MAX_SPAWN_DISTANCE:.1f}m")

        bounds = self._get_bounds()

        def fitness_func(x):
            return self._evaluate_fitness(x)

        self.ga = GA(
            func=fitness_func,
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

        params = self._genes_to_scene_params(best_x)
        actor_loc, heading_deg = self._calculate_actor_position_and_heading(params)

        # 返回 actor 初始 transform 供场景初始化使用
        initial_transform = carla.Transform(actor_loc, carla.Rotation(yaw=heading_deg))
        params["initial_transform"] = initial_transform
        params["computed_heading_deg"] = heading_deg

        # 打印最终结果
        ego_loc = self.ego_transform.location
        spawn_dist = math.hypot(actor_loc.x - ego_loc.x, actor_loc.y - ego_loc.y)
        print(f"\n>> Optimization completed!")
        print(f"   Best fitness: {self.best_fitness:.4f}")
        print(f"   Actor spawn distance: {spawn_dist:.2f}m")
        print(f"   Actor speed: {params['actor_speed']:.2f} m/s")
        print(f"   Trigger distance: {params['trigger_distance']:.2f}m")
        print(f"   Position: ({actor_loc.x:.2f}, {actor_loc.y:.2f})")
        print(f"   Heading: {heading_deg:.1f}°")

        return params

    # ----------------- 基因边界 -----------------
    def _get_bounds(self) -> List[Tuple[float, float]]:
        """
        优化参数边界:
        - actor_speed: 自行车速度 (3-8 m/s)
        - trigger_distance: 触发距离 (8-20m)
        - forward_offset: ego前方偏移 (15-30m)
        - lateral_offset: ego侧方偏移 (-3 to 3m)
        - yaw_offset: 朝向偏移 (-30° to 30°)
        """
        return [
            (3.0, 8.0),  # actor_speed
            (8.0, 20.0),  # trigger_distance (增加最小值)
            (15.0, 30.0),  # forward_offset (自行车在ego前方的距离)
            (-5.0, 5.0),  # lateral_offset (侧向偏移)
            (-30.0, 30.0)  # yaw_offset
        ]

    # ----------------- 适应度函数 -----------------
    def _evaluate_fitness(self, genes: np.ndarray) -> float:
        try:
            params = self._genes_to_scene_params(genes)
            actor_speed = params["actor_speed"]
            trigger_distance = params["trigger_distance"]

            actor_loc, actor_heading = self._calculate_actor_position_and_heading(params)
            ego_loc = self.ego_transform.location

            # 1. 安全距离检查 (硬约束)
            spawn_distance = math.hypot(actor_loc.x - ego_loc.x, actor_loc.y - ego_loc.y)
            if spawn_distance < self.MIN_SAFE_DISTANCE:
                # 严重惩罚距离太近的情况
                penalty = (self.MIN_SAFE_DISTANCE - spawn_distance) * 100
                return 1e5 + penalty

            if spawn_distance > self.MAX_SPAWN_DISTANCE:
                # 惩罚距离太远的情况
                penalty = (spawn_distance - self.MAX_SPAWN_DISTANCE) * 50
                return 1e5 + penalty

            # 2. 计算预期碰撞点
            # 碰撞应该发生在ego行驶路径上,距离ego前方一定距离
            collision_forward_dist = min(spawn_distance * 0.6, 15.0)  # 碰撞点在前方
            collision_point = carla.Location(
                ego_loc.x + self.ego_forward[0] * collision_forward_dist,
                ego_loc.y + self.ego_forward[1] * collision_forward_dist,
                ego_loc.z
            )

            # 3. 时间同步计算
            # Ego到达碰撞点的时间
            ego_to_collision = math.hypot(
                collision_point.x - ego_loc.x,
                collision_point.y - ego_loc.y
            )
            # 假设ego在转弯时速度降低到60%
            ego_effective_speed = self.ego_speed * 0.6

            # 触发前ego需要行驶的距离
            trigger_point_dist = max(spawn_distance - trigger_distance, 0.0)
            time_to_trigger = trigger_point_dist / max(self.ego_speed, 0.1)

            # 触发后ego到达碰撞点的时间
            time_trigger_to_collision = ego_to_collision / max(ego_effective_speed, 0.1)
            ego_total_time = time_to_trigger + time_trigger_to_collision

            # Actor到达碰撞点的时间
            actor_to_collision = math.hypot(
                collision_point.x - actor_loc.x,
                collision_point.y - actor_loc.y
            )
            actor_total_time = time_to_trigger + actor_to_collision / max(actor_speed, 0.1)

            # 4. 时间差评分 (核心目标: 双方同时到达碰撞点)
            time_diff = actor_total_time - ego_total_time

            if -0.5 <= time_diff <= 0.5:
                # 最优时间窗口: 几乎同时到达
                time_score = 40.0 - abs(time_diff) * 40.0
            elif -1.5 <= time_diff <= 1.5:
                # 次优时间窗口
                time_score = 25.0 - abs(time_diff) * 10.0
            else:
                # 时间差太大
                time_score = max(0.1, 10.0 / (abs(time_diff) + 0.5))

            # 5. 距离评分
            # 触发距离适中性 (不要太近也不要太远)
            ideal_trigger = 12.0
            trigger_score = 10.0 * math.exp(-abs(trigger_distance - ideal_trigger) / 5.0)

            # Actor到碰撞点的距离 (应该有足够距离但不要太远)
            distance_score = 10.0 * math.exp(-abs(actor_to_collision - 8.0) / 4.0)

            # 6. 速度评分 (偏好较高速度,更危险)
            speed_score = (actor_speed - 3.0) * 1.5

            # 7. 方向一致性评分 (actor应该朝向碰撞点)
            actor_to_collision_vec = np.array([
                collision_point.x - actor_loc.x,
                collision_point.y - actor_loc.y
            ])
            actor_to_collision_vec = actor_to_collision_vec / (np.linalg.norm(actor_to_collision_vec) + 1e-6)

            actor_heading_rad = math.radians(actor_heading)
            actor_forward = np.array([math.cos(actor_heading_rad), math.sin(actor_heading_rad)])

            direction_alignment = np.dot(actor_forward, actor_to_collision_vec)
            direction_score = max(0, direction_alignment) * 8.0

            # 8. 总分计算
            total_score = (
                    time_score * 1.5 +  # 时间同步最重要
                    trigger_score +
                    distance_score +
                    speed_score +
                    direction_score
            )

            # 打印高分候选
            if time_score > 30.0 or total_score > 60.0:
                print(f"   [Good candidate] time_diff={time_diff:.2f}s "
                      f"spawn={spawn_distance:.1f}m collision_dist={actor_to_collision:.1f}m "
                      f"speed={actor_speed:.1f}m/s score={total_score:.1f}")

            return -total_score

        except Exception as e:
            print(f">> [Fitness error] {e}")
            return 1e6

    # ----------------- 核心位置计算 -----------------
    def _calculate_actor_position_and_heading(self, params: Dict[str, Any]) -> Tuple[carla.Location, float]:
        """
        计算actor的生成位置和朝向
        策略: actor生成在ego前方偏侧方,朝向指向预期碰撞点
        """
        forward_offset = params["forward_offset"]
        lateral_offset = params["lateral_offset"]
        yaw_offset = params["yaw_offset"]

        ego_loc = self.ego_transform.location

        # 计算actor位置: ego前方 + 侧向偏移
        actor_x = ego_loc.x + self.ego_forward[0] * forward_offset + self.ego_right[0] * lateral_offset
        actor_y = ego_loc.y + self.ego_forward[1] * forward_offset + self.ego_right[1] * lateral_offset
        actor_z = ego_loc.z

        actor_loc = carla.Location(actor_x, actor_y, actor_z)

        # 计算碰撞点: ego前方适当距离
        spawn_dist = math.hypot(actor_x - ego_loc.x, actor_y - ego_loc.y)
        # collision_forward_dist = min(spawn_dist * 0.6, 15.0)
        collision_forward_dist=spawn_dist

        collision_point = carla.Location(
            ego_loc.x + self.ego_forward[0] * collision_forward_dist,
            ego_loc.y + self.ego_forward[1] * collision_forward_dist,
            actor_z
        )

        # 计算actor朝向: 从actor位置指向碰撞点
        vec_x = collision_point.x - actor_loc.x
        vec_y = collision_point.y - actor_loc.y

        if abs(vec_x) < 1e-6 and abs(vec_y) < 1e-6:
            heading_deg = self.ego_heading
        else:
            heading_deg = math.degrees(math.atan2(vec_y, vec_x))

        # 应用yaw偏移
        heading_deg = (heading_deg + yaw_offset) % 360.0

        return actor_loc, heading_deg

    # ----------------- 基因转换 -----------------
    def _genes_to_scene_params(self, genes: np.ndarray) -> Dict[str, Any]:
        return {
            "actor_speed": genes[0],
            "trigger_distance": genes[1],
            "forward_offset": genes[2],
            "lateral_offset": genes[3],
            "yaw_offset": genes[4],
            "position_offset": {
                "x": genes[3],  # lateral
                "y": genes[2],  # forward
                "yaw": genes[4]
            }
        }

    # ----------------- 信息输出 -----------------
    def get_optimization_info(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_fitness": self.best_fitness,
            "population_size": self.population_size,
            "generations": self.generations
        }