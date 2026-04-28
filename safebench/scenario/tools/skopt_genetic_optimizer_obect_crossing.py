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
        """优化初始状态 - 添加碰撞检测和重试"""
        print("\n" + "=" * 80)
        print(">> Pedestrian Crossing Optimization - COLLISION ON ROAD")
        print(">> Pedestrian waits on roadside, crosses when ego approaches")
        print("=" * 80)
        print(f">> Ego speed: {self.ego_speed:.2f} m/s")
        print(f">> Ego heading: {self.ego_heading:.2f}°")
        print(f">> Ego position: ({self.ego_transform.location.x:.2f}, "
              f"{self.ego_transform.location.y:.2f}, {self.ego_transform.location.z:.2f})")

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

        # ✅ 尝试多次寻找无碰撞位置
        max_retries = 5
        for retry in range(max_retries):
            params = self._genes_to_scene_params(best_x)
            actor_loc, heading_deg = self._calculate_actor_position_and_heading(params)

            # 检查碰撞
            if not self._check_spawn_collision(actor_loc):
                print(f"   [Spawn Check] Position OK (retry {retry + 1}/{max_retries})")
                break
            else:
                print(f"   [Spawn Check] Collision detected, adjusting parameters (retry {retry + 1}/{max_retries})")
                # 增加侧向偏移
                best_x[3] += 1.0  # lateral_distance
                if retry == max_retries - 1:
                    print(f"   [Warning] Could not find collision-free position after {max_retries} retries")

        # 重新计算最终参数
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
        print(f"   Pedestrian position: ({actor_loc.x:.2f}, {actor_loc.y:.2f}, {actor_loc.z:.2f})")
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
            (1.0, 5.0),  # lateral_distance (路边距离)
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

        ⭐ 完整修复方案:
        1. 沿ego行驶方向搜索路点
        2. 从路点偏移到路边(使用道路宽度)
        3. 获取正确的路面高度
        4. 检测碰撞风险
        """
        forward_distance = params["forward_distance"]
        lateral_distance = params["lateral_distance"]
        cross_angle = params["cross_angle"]

        carla_map = self.scenario._map
        ego_loc = self.ego_transform.location

        # ✅ Step 1: 获取ego当前路点
        ego_waypoint = carla_map.get_waypoint(
            ego_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if not ego_waypoint:
            print(f"   [Error] Cannot find ego waypoint!")
            # 使用简单计算作为后备
            actor_x = ego_loc.x + self.ego_forward[0] * forward_distance
            actor_y = ego_loc.y + self.ego_forward[1] * forward_distance
            actor_z = ego_loc.z + 1.0
            actor_loc = carla.Location(actor_x, actor_y, actor_z)
            base_heading = self.ego_heading + 90.0
            actor_heading = (base_heading + cross_angle) % 360.0
            return actor_loc, actor_heading

        # ✅ Step 2: 沿道路前进到目标位置
        forward_waypoint = ego_waypoint
        accumulated_distance = 0.0
        search_step = 1.0  # 减小步长以提高精度

        while accumulated_distance < forward_distance:
            next_waypoints = forward_waypoint.next(search_step)
            if not next_waypoints:
                print(f"   [Warning] Reached end of road at {accumulated_distance:.1f}m")
                break
            forward_waypoint = next_waypoints[0]
            accumulated_distance += search_step

        # ✅ Step 3: 获取路边位置
        waypoint_transform = forward_waypoint.transform
        waypoint_loc = waypoint_transform.location

        # 获取车道宽度
        lane_width = forward_waypoint.lane_width

        # 计算实际偏移距离（确保在路边但不会太远）
        # lateral_distance 是优化参数，但需要基于实际道路宽度调整
        actual_lateral_offset = min(lateral_distance, lane_width * 0.5 + 2.0)

        # 使用路点的右向量(已经考虑了道路曲率和坡度)
        waypoint_right = waypoint_transform.get_right_vector()

        # 计算行人位置(在路边)
        actor_x = waypoint_loc.x - waypoint_right.x * actual_lateral_offset
        actor_y = waypoint_loc.y - waypoint_right.y * actual_lateral_offset

        # ✅ Step 4: 获取该位置的精确路面高度
        temp_location = carla.Location(actor_x, actor_y, waypoint_loc.z + 50.0)

        try:
            # 尝试获取最近的路点(包括人行道)
            actor_waypoint = carla_map.get_waypoint(
                temp_location,
                project_to_road=True,
                lane_type=carla.LaneType.Any  # 包括人行道
            )

            if actor_waypoint:
                actor_z = actor_waypoint.transform.location.z + 1.0
            else:
                # 如果找不到，使用道路路点的高度
                actor_z = waypoint_loc.z + 1.0

        except Exception as e:
            print(f"   [Height Warning] {e}")
            actor_z = waypoint_loc.z + 1.0

        actor_loc = carla.Location(actor_x, actor_y, actor_z)

        # ✅ Step 5: 检查位置是否可行
        # 确保不在ego车道中心
        distance_to_road_center = math.hypot(
            actor_x - waypoint_loc.x,
            actor_y - waypoint_loc.y
        )

        if distance_to_road_center < lane_width * 0.3:
            # 太靠近车道中心，增加偏移
            actor_x = waypoint_loc.x - waypoint_right.x * (lane_width * 0.5 + 1.5)
            actor_y = waypoint_loc.y - waypoint_right.y * (lane_width * 0.5 + 1.5)
            actor_loc = carla.Location(actor_x, actor_y, actor_z)
            print(f"   [Position Adjust] Moved away from road center")

        # ✅ Step 6: 计算行人朝向(横穿道路)
        # 使用路点的朝向，而不是ego的朝向(处理弯道情况)
        road_heading = waypoint_transform.rotation.yaw
        base_heading = road_heading + 90.0  # 垂直于道路方向
        actor_heading = (base_heading + cross_angle) % 360.0

        # 调试输出
        height_diff = abs(actor_z - ego_loc.z)
        print(f"   [Position] Waypoint z={waypoint_loc.z:.2f}, Actor z={actor_z:.2f}, "
              f"Height diff from ego={height_diff:.2f}m")
        print(f"   [Position] Lane width={lane_width:.2f}m, Lateral offset={actual_lateral_offset:.2f}m")
        print(f"   [Position] Distance to road center={distance_to_road_center:.2f}m")

        return actor_loc, actor_heading

    def _check_spawn_collision(self, location: carla.Location, radius: float = 2.0) -> bool:
        """
        检查生成位置是否有碰撞

        Args:
            location: 要检查的位置
            radius: 检查半径

        Returns:
            True if collision detected, False otherwise
        """
        try:
            world = self.scenario.world

            # 获取附近的所有actor
            nearby_vehicles = world.get_actors().filter('vehicle.*')
            nearby_walkers = world.get_actors().filter('walker.*')

            for actor in list(nearby_vehicles) + list(nearby_walkers):
                actor_loc = actor.get_location()
                distance = math.hypot(
                    location.x - actor_loc.x,
                    location.y - actor_loc.y
                )

                if distance < radius:
                    print(f"   [Collision Check] Actor too close: distance={distance:.2f}m")
                    return True

            return False

        except Exception as e:
            print(f"   [Collision Check Error] {e}")
            return False

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