# safebench/scenario/tools/skopt_genetic_optimizer_pedestrian.py
"""
行人横穿道路场景优化器

场景设计:
1. 行人在ego车道的右侧路边等待
2. Ego直行接近
3. 当distance(ego, pedestrian) < trigger_distance时，行人突然横穿
4. 碰撞发生在ego车道上
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple
from sko.GA import GA
import carla


class SkoptGeneticOptimizer:
    """行人横穿道路场景优化器"""

    def __init__(self, scenario):
        self.scenario = scenario

        # Ego 状态
        self.ego_transform = scenario.ego_vehicle.get_transform()
        self.ego_velocity = scenario.ego_vehicle.get_velocity()
        self.ego_speed = math.sqrt(self.ego_velocity.x ** 2 + self.ego_velocity.y ** 2)
        self.ego_heading = self.ego_transform.rotation.yaw

        if self.ego_speed < 1.0:
            self.ego_speed = 5.0

        ego_heading_rad = math.radians(self.ego_heading)
        self.ego_forward = np.array([math.cos(ego_heading_rad), math.sin(ego_heading_rad)])
        self.ego_right = np.array([-math.sin(ego_heading_rad), math.cos(ego_heading_rad)])

        # GA 参数
        self.population_size = 50
        self.generations = 50
        self.mutation_rate = 0.2
        self.crossover_rate = 0.8

        self.best_params = None
        self.best_fitness = None
        self.ga = None

    def optimize_initial_state(self) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print(">> Pedestrian Crossing Optimization - COLLISION ON ROAD")
        print(">> Pedestrian waits on roadside, crosses when ego approaches")
        print("=" * 80)
        print(f">> Ego speed: {self.ego_speed:.2f} m/s")
        print(f">> Ego heading: {self.ego_heading:.2f}°")
        print(f">> Ego position: ({self.ego_transform.location.x:.2f}, {self.ego_transform.location.y:.2f})")

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

        initial_transform = carla.Transform(actor_loc, carla.Rotation(yaw=heading_deg))
        params["initial_transform"] = initial_transform
        params["computed_heading_deg"] = heading_deg

        # 详细输出
        ego_loc = self.ego_transform.location
        initial_distance = math.hypot(actor_loc.x - ego_loc.x, actor_loc.y - ego_loc.y)

        vec_to_actor = np.array([actor_loc.x - ego_loc.x, actor_loc.y - ego_loc.y])
        forward_component = np.dot(vec_to_actor, self.ego_forward)
        lateral_component = np.dot(vec_to_actor, self.ego_right)

        print("\n" + "=" * 80)
        print(">> OPTIMIZATION COMPLETED - PEDESTRIAN CROSSING SETUP")
        print("=" * 80)
        print(f"   Best fitness: {self.best_fitness:.4f}")
        print(f"   Pedestrian position: ({actor_loc.x:.2f}, {actor_loc.y:.2f})")
        print(f"   Pedestrian heading: {heading_deg:.1f}° (toward ego's lane)")
        print(f"   Pedestrian speed: {params['actor_speed']:.2f} m/s")
        print(f"   Initial ego-pedestrian distance: {initial_distance:.2f}m")
        print(f"   Trigger distance: {params['trigger_distance']:.2f}m")
        print(f"   Pedestrian relative position:")
        print(f"     - Ahead of ego: {forward_component:.2f}m")
        print(f"     - Right of ego (roadside): {abs(lateral_component):.2f}m")
        print(f"   >>> Scenario: Ego approaches, pedestrian crosses road <<<")
        print("=" * 80 + "\n")

        return params

    def _get_bounds(self) -> List[Tuple[float, float]]:
        """
        优化参数:
        0. actor_speed: 行人横穿速度 (0.8-2.5 m/s)
        1. trigger_distance: 触发距离 (8-20m)
        2. forward_distance: 行人在ego前方的距离 (10-25m)
        3. lateral_distance: 行人在路边的距离 (3-8m) - 车道宽度考虑
        4. cross_angle: 横穿角度偏移 (-20 to 20°) - 基本垂直穿越
        """
        return [
            (0.8, 2.5),  # actor_speed (行人速度)
            (14.0, 22.0),  # trigger_distance
            (10.0, 25.0),  # forward_distance
            (3.0, 8.0),  # lateral_distance (路边距离)
            (-20.0, 20.0),  # cross_angle (微调横穿角度)
        ]

    def _evaluate_fitness(self, genes: np.ndarray) -> float:
        try:
            params = self._genes_to_scene_params(genes)

            actor_speed = params["actor_speed"]
            trigger_distance = params["trigger_distance"]
            forward_distance = params["forward_distance"]
            lateral_distance = params["lateral_distance"]

            # 1. 计算行人位置
            actor_loc, actor_heading = self._calculate_actor_position_and_heading(params)
            ego_loc = self.ego_transform.location

            # 2. 初始距离
            initial_distance = math.hypot(actor_loc.x - ego_loc.x, actor_loc.y - ego_loc.y)

            # 3. 硬约束：初始距离必须大于触发距离
            if initial_distance <= trigger_distance:
                penalty = (trigger_distance - initial_distance + 3.0) * 300.0
                return 1e5 + penalty

            # 4. 硬约束：行人必须在前方（不能在后方或侧后方）
            vec_to_actor = np.array([actor_loc.x - ego_loc.x, actor_loc.y - ego_loc.y])
            forward_component = np.dot(vec_to_actor, self.ego_forward)
            lateral_component = np.dot(vec_to_actor, self.ego_right)

            if forward_component < 8.0:
                penalty = (8.0 - forward_component) * 200.0
                return 1e5 + penalty

            # 5. 硬约束：行人必须在侧方（路边）
            if abs(lateral_component) < 2.5:
                penalty = (2.5 - abs(lateral_component)) * 250.0
                return 1e5 + penalty

            # 6. 计算碰撞点（在ego车道上）
            # 碰撞应该发生在ego当前车道中心线上
            distance_to_trigger = initial_distance - trigger_distance
            collision_forward = distance_to_trigger + 2.0  # 触发后ego再前进2米

            collision_point = carla.Location(
                ego_loc.x + self.ego_forward[0] * collision_forward,
                ego_loc.y + self.ego_forward[1] * collision_forward,
                ego_loc.z
            )

            # 7. Ego时间线
            time_to_trigger = distance_to_trigger / max(self.ego_speed, 0.1)
            time_trigger_to_collision = 2.0 / max(self.ego_speed, 0.1)
            ego_total_time = time_to_trigger + time_trigger_to_collision

            # 8. 行人时间线（横穿道路）
            pedestrian_to_collision_dist = math.hypot(
                collision_point.x - actor_loc.x,
                collision_point.y - actor_loc.y
            )
            pedestrian_travel_time = pedestrian_to_collision_dist / max(actor_speed, 0.1)
            pedestrian_total_time = time_to_trigger + pedestrian_travel_time

            # 9. 时间差评分（核心目标）
            time_diff = pedestrian_total_time - ego_total_time

            if -0.3 <= time_diff <= 0.3:
                time_score = 120.0 - abs(time_diff) * 120.0
            elif -0.8 <= time_diff <= 0.8:
                time_score = 90.0 - abs(time_diff) * 60.0
            else:
                time_score = max(0.1, 40.0 / (abs(time_diff) + 0.5))

            # 10. 位置合理性评分
            ideal_forward = 16.0
            forward_score = 15.0 * math.exp(-abs(forward_distance - ideal_forward) / 5.0)

            ideal_lateral = 5.0  # 标准车道宽度边缘
            lateral_score = 12.0 * math.exp(-abs(lateral_distance - ideal_lateral) / 2.0)

            # 11. 触发距离合理性
            ideal_trigger = 12.0
            trigger_score = 10.0 * math.exp(-abs(trigger_distance - ideal_trigger) / 4.0)

            # 12. 速度评分（行人速度适中）
            ideal_speed = 1.5
            speed_score = 8.0 * math.exp(-abs(actor_speed - ideal_speed) / 0.8)

            # 13. 横穿方向评分（应该垂直穿越）
            pedestrian_to_collision_vec = np.array([
                collision_point.x - actor_loc.x,
                collision_point.y - actor_loc.y
            ])
            norm = np.linalg.norm(pedestrian_to_collision_vec)
            if norm > 1e-6:
                pedestrian_to_collision_vec = pedestrian_to_collision_vec / norm
                actor_heading_rad = math.radians(actor_heading)
                actor_forward = np.array([math.cos(actor_heading_rad), math.sin(actor_heading_rad)])
                direction_alignment = np.dot(actor_forward, pedestrian_to_collision_vec)
                direction_score = max(0, direction_alignment) * 20.0
            else:
                direction_score = 0.0

            # 14. 横穿距离评分（不能太远）
            cross_distance_score = 8.0 * math.exp(-pedestrian_to_collision_dist / 6.0)

            # 15. 总分
            total_score = (
                    time_score * 3.0 +  # 时间同步最重要
                    forward_score * 1.2 +
                    lateral_score * 1.5 +  # 位置必须在路边
                    trigger_score +
                    speed_score +
                    direction_score * 1.5 +
                    cross_distance_score
            )

            # 打印优秀候选
            if time_score > 100.0 or total_score > 250.0:
                print(f"   [★★★] time_diff={time_diff:.2f}s forward={forward_distance:.1f}m "
                      f"lateral={lateral_distance:.1f}m speed={actor_speed:.2f}m/s score={total_score:.1f}")

            return -total_score

        except Exception as e:
            print(f">> [Fitness error] {e}")
            import traceback
            traceback.print_exc()
            return 1e6

    def _calculate_actor_position_and_heading(self, params: Dict[str, Any]) -> Tuple[carla.Location, float]:
        """
        计算行人位置和朝向

        位置：ego前方 forward_distance 米 + 右侧路边 lateral_distance 米
        朝向：横向穿越道路（基本垂直于ego）
        """
        forward_distance = params["forward_distance"]
        lateral_distance = params["lateral_distance"]
        cross_angle = params["cross_angle"]

        ego_loc = self.ego_transform.location

        # 行人位置：ego前方 + 右侧路边
        actor_x = (ego_loc.x +
                   self.ego_forward[0] * forward_distance +  # 前方
                   self.ego_right[0] * (-lateral_distance))  # 右侧（负号）
        actor_y = (ego_loc.y +
                   self.ego_forward[1] * forward_distance +
                   self.ego_right[1] * (-lateral_distance))
        actor_z = ego_loc.z + 1.0  # 行人站立高度

        actor_loc = carla.Location(actor_x, actor_y, actor_z)

        # 行人朝向：从路边横穿到车道（基本垂直于ego方向）
        # 从右向左穿越
        base_heading = self.ego_heading + 90.0  # 垂直于ego

        # 应用横穿角度微调
        actor_heading = (base_heading + cross_angle) % 360.0

        return actor_loc, actor_heading

    def _genes_to_scene_params(self, genes: np.ndarray) -> Dict[str, Any]:
        return {
            "actor_speed": genes[0],
            "trigger_distance": genes[1],
            "forward_distance": genes[2],
            "lateral_distance": genes[3],
            "cross_angle": genes[4],
            "position_offset": {
                "x": -genes[3],  # 右侧路边
                "y": genes[2],  # 前方
                "yaw": genes[4]
            }
        }

    def get_optimization_info(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_fitness": self.best_fitness,
            "population_size": self.population_size,
            "generations": self.generations
        }