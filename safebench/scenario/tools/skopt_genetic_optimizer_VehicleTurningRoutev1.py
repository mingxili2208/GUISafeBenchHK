# safebench/scenario/tools/skopt_genetic_optimizer_VehicleTurningRoute.py
"""
SkoptGeneticOptimizer for VehicleTurningRoute - Immediate collision variant

目标：
 - 触发应在ego靠近自行车时发生（trigger_distance 很小）
 - 触发点靠近自行车
 - 触发后自行车几乎立刻冲入ego路径并与ego相撞

要点：
 - genes 包含 yaw_offset（允许微调朝向）
 - 返回结果包含 computed_heading_deg，初始化actor时请把spawn yaw设为该值
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple
from sko.GA import GA
import carla


class SkoptGeneticOptimizer:
    """通用车辆转弯场景优化器（立即撞击风格）"""

    def __init__(self, scenario):
        self.scenario = scenario

        # ego 状态
        self.ego_transform = scenario.ego_vehicle.get_transform()
        self.ego_velocity = scenario.ego_vehicle.get_velocity()
        self.ego_speed = math.sqrt(self.ego_velocity.x ** 2 + self.ego_velocity.y ** 2)
        self.ego_heading = self.ego_transform.rotation.yaw

        if self.ego_speed < 1.0:
            self.ego_speed = 5.0  # 默认以避免除零

        ego_heading_rad = math.radians(self.ego_heading)
        self.ego_forward = np.array([math.cos(ego_heading_rad), math.sin(ego_heading_rad)])

        # actor 基础位姿（配置中存在则使用）
        if hasattr(scenario.config, "other_actors") and len(scenario.config.other_actors) > 0:
            self.actor_base_transform = scenario.config.other_actors[0].transform
            self.actor_base_location = self.actor_base_transform.location
            self.actor_base_heading = self.actor_base_transform.rotation.yaw
        else:
            # 默认放在 ego 前方偏侧一点
            self.actor_base_location = carla.Location(
                self.ego_transform.location.x + 20.0,
                self.ego_transform.location.y + 6.0,
                self.ego_transform.location.z,
            )
            self.actor_base_heading = (self.ego_heading + 90.0) % 360.0
            self.actor_base_transform = carla.Transform(self.actor_base_location, carla.Rotation(yaw=self.actor_base_heading))

        # GA 参数
        self.population_size = 40
        self.generations = 40
        self.mutation_rate = 0.15
        self.crossover_rate = 0.8

        self.best_params = None
        self.best_fitness = None
        self.ga = None

    # ----------------- 对外优化接口 -----------------
    def optimize_initial_state(self) -> Dict[str, Any]:
        """运行优化并返回用于spawn actor的参数（包含 computed_heading_deg）"""
        print(">> Starting VehicleTurningRoute immediate-collision optimization")
        print(f">> Ego speed: {self.ego_speed:.2f} m/s | Ego heading: {self.ego_heading:.2f}°")

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
            precision=[1e-6] * len(bounds),
        )

        best_x, best_y = self.ga.run()
        self.best_params = best_x
        self.best_fitness = -best_y if not isinstance(best_y, np.ndarray) else -best_y.item()

        params = self._genes_to_scene_params(best_x)
        # 计算出最终的 computed_heading_deg（便于 spawn）
        actor_loc, heading_deg = self._calculate_actor_position_and_heading(params)
        params["computed_heading_deg"] = heading_deg

        print(f">> Optimization done - best_fitness: {self.best_fitness:.4f}")
        print(f">> Best params: {params}")
        return params

    # ----------------- 基因边界 -----------------
    def _get_bounds(self) -> List[Tuple[float, float]]:
        """
        genes:
         [0] actor_speed: 3.0 - 8.0 (m/s)
         [1] trigger_distance: 0.5 - 6.0 (m)  <-- 更偏好小触发距离
         [2] offset_x: -2.5 - 2.5 (m)
         [3] offset_y: -4.0 - 4.0 (m)
         [4] yaw_offset: -45 - 45 (deg)
        """
        return [
            (3.0, 8.0),
            (0.05, 1),
            (-0.5, 0.5),
            (-4.0, 4.0),
            (-45.0, 45.0),
        ]

    # ----------------- 适应度函数 -----------------
    def _evaluate_fitness(self, genes: np.ndarray) -> float:
        """
        使用 actor 实际 transform 计算触发距离和碰撞点时间同步
        """
        try:
            params = self._genes_to_scene_params(genes)
            actor_speed = params["actor_speed"]
            trigger_distance = params["trigger_distance"]

            # actor 起始位置与预设朝向
            actor_loc, actor_heading = self._calculate_actor_position_and_heading(params)

            # 使用实际 transform 计算 ego -> actor 距离
            ego_loc = self.ego_transform.location
            dist_ego_actor = ego_loc.distance(actor_loc)

            # 考虑触发距离
            if dist_ego_actor <= trigger_distance:
                time_to_trigger = 0.0
                trigger_loc = ego_loc
            else:
                # 使用实际 actor transform 计算触发点
                distance_to_trigger = max(dist_ego_actor - trigger_distance, 0.0)
                # 沿 ego -> actor 方向偏移
                vec_x = actor_loc.x - ego_loc.x
                vec_y = actor_loc.y - ego_loc.y
                norm = math.hypot(vec_x, vec_y)
                vec_x /= max(norm, 1e-6)
                vec_y /= max(norm, 1e-6)
                trigger_loc = carla.Location(
                    ego_loc.x + vec_x * distance_to_trigger,
                    ego_loc.y + vec_y * distance_to_trigger,
                    ego_loc.z,
                )
                time_to_trigger = distance_to_trigger / max(self.ego_speed, 0.1)

            # 碰撞点设为 actor -> ego 方向的一小段距离
            vec_to_ego_x = ego_loc.x - actor_loc.x
            vec_to_ego_y = ego_loc.y - actor_loc.y
            dist_to_ego = math.hypot(vec_to_ego_x, vec_to_ego_y)
            if dist_to_ego < 1e-3:
                dir_x, dir_y = self.ego_forward[0], self.ego_forward[1]
            else:
                dir_x, dir_y = vec_to_ego_x / dist_to_ego, vec_to_ego_y / dist_to_ego

            immediate_dist = min(max(1.0, dist_to_ego * 0.2), 3.0)
            collision_point = carla.Location(
                actor_loc.x + dir_x * immediate_dist,
                actor_loc.y + dir_y * immediate_dist,
                actor_loc.z,
            )

            # 时间同步评分
            ego_to_collision_dist = trigger_loc.distance(collision_point)
            ego_turn_speed = self.ego_speed * 0.4
            time_trigger_to_collision = ego_to_collision_dist / max(ego_turn_speed, 0.1)
            ego_total_time = time_to_trigger + time_trigger_to_collision

            actor_distance = actor_loc.distance(collision_point)
            actor_travel_time = actor_distance / max(actor_speed, 0.1)
            actor_total_time = time_to_trigger + actor_travel_time

            time_diff = actor_total_time - ego_total_time
            if -0.4 <= time_diff <= 0.4:
                time_score = 30.0 - abs(time_diff) * 30.0
            elif -1.0 <= time_diff <= 1.5:
                time_score = 18.0 - abs(time_diff) * 8.0
            else:
                time_score = max(0.1, 6.0 / (abs(time_diff) + 0.5))

            trigger_score = math.exp(-max(0.0, trigger_distance - 1.5) / 1.5) * 10.0
            immediate_score = math.exp(-actor_distance / 1.2) * 12.0
            speed_bonus = (actor_speed - 4.0) * 1.0
            lateral_penalty = abs(params["position_offset"]["x"]) * 0.5

            total_score = time_score + trigger_score + immediate_score + speed_bonus - lateral_penalty

            # debug
            if time_score > 25.0 or immediate_score > 10.0:
                print(f"   [candidate] time_diff={time_diff:.3f}s actor_dist={actor_distance:.2f}m "
                      f"trigger_d={trigger_distance:.2f}m speed={actor_speed:.2f} heading={actor_heading:.1f}")

            return -total_score

        except Exception as e:
            print(f">> [SkoptGeneticOptimizer] Fitness error: {e}")
            return 1e6

    # ----------------- 工具函数 -----------------
    def _calculate_distance(self, loc1: carla.Location, loc2: carla.Location) -> float:
        return loc1.distance(loc2)

    def _calculate_actor_position_and_heading(self, params: Dict[str, Any]) -> Tuple[carla.Location, float]:
        offset = params["position_offset"]
        actor_heading_rad = math.radians(self.actor_base_heading)
        base_forward = np.array([math.cos(actor_heading_rad), math.sin(actor_heading_rad)])
        base_right = np.array([-math.sin(actor_heading_rad), math.cos(actor_heading_rad)])

        offset_x = offset["x"] * base_right[0] + offset["y"] * base_forward[0]
        offset_y = offset["x"] * base_right[1] + offset["y"] * base_forward[1]

        actor_loc = carla.Location(
            self.actor_base_location.x + offset_x,
            self.actor_base_location.y + offset_y,
            self.actor_base_location.z,
        )

        heading_deg = (self.actor_base_heading + offset.get("yaw", 0.0)) % 360.0
        return actor_loc, heading_deg

    def _genes_to_scene_params(self, genes: np.ndarray) -> Dict[str, Any]:
        """
        将基因数组转换为场景参数
        genes:
            [0] actor_speed
            [1] trigger_distance
            [2] offset_x
            [3] offset_y
            [4] yaw_offset
        """
        params = {
            "actor_speed": genes[0],
            "trigger_distance": genes[1],
            "position_offset": {
                "x": genes[2],
                "y": genes[3],
                "yaw": genes[4],
            }
        }
        return params

    def get_optimization_info(self) -> Dict[str, Any]:
        """返回优化过程信息"""
        return {
            "best_params": self.best_params,
            "best_fitness": self.best_fitness,
            "generations": self.generations,
            "population_size": self.population_size
        }
