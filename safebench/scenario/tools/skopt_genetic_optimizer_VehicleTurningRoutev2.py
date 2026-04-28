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
            (0.5, 6.0),
            (-2.5, 2.5),
            (-4.0, 4.0),
            (-45.0, 45.0),
        ]

    # ----------------- 适应度函数 -----------------
    def _evaluate_fitness(self, genes: np.ndarray) -> float:
        """
        目标（优先级）：
         1) 触发距离越小越好（ego 必须靠近自行车时才触发）
         2) actor 到碰撞点距离越小越好（触发后自行车几乎立即撞上）
         3) actor 与 ego 在碰撞点时间越接近越好（时间同步）
        综合以上构造适应度（越小越好，因为返回负分）
        """
        try:
            params = self._genes_to_scene_params(genes)
            actor_speed = params["actor_speed"]
            trigger_distance = params["trigger_distance"]

            # actor 起始位置与预设朝向
            actor_loc, actor_heading = self._calculate_actor_position_and_heading(params)

            # 计算 ego 与 actor 的当前距离
            ego_loc = self.ego_transform.location
            dist_ego_actor = self._calculate_distance(ego_loc, actor_loc)

            # 计算触发点：我们要求触发点非常靠近actor -> 当 ego 距离 actor <= trigger_distance 时触发
            # 因为我们希望触发点靠近自行车本体，所以不再把触发点设在 ego 前方远处，而是直接基于 actor 位置
            # 触发时 ego 位置估计（approx）: ego 朝 forward 走 (dist_ego_actor - trigger_distance) 米
            if dist_ego_actor <= trigger_distance:
                time_to_trigger = 0.0
                trigger_loc = ego_loc
            else:
                distance_to_trigger = max(dist_ego_actor - trigger_distance, 0.0)
                time_to_trigger = distance_to_trigger / max(self.ego_speed, 0.1)
                trigger_loc = carla.Location(
                    ego_loc.x + self.ego_forward[0] * distance_to_trigger,
                    ego_loc.y + self.ego_forward[1] * distance_to_trigger,
                    ego_loc.z,
                )

            # ---- 关键：碰撞点设为 actor 向 ego 方向移动的一小段距离（立即撞）
            # 这样 actor 触发后仅需走一小段距离就能进入 ego 路径
            # 计算 actor -> ego 向量（取向 ego current loc）
            vec_to_ego_x = ego_loc.x - actor_loc.x
            vec_to_ego_y = ego_loc.y - actor_loc.y
            dist_to_ego = math.hypot(vec_to_ego_x, vec_to_ego_y)
            if dist_to_ego < 1e-3:
                # 极近时，把碰撞点设为 actor 当前点前方一点
                dir_x, dir_y = self.ego_forward[0], self.ego_forward[1]
            else:
                dir_x, dir_y = vec_to_ego_x / dist_to_ego, vec_to_ego_y / dist_to_ego

            # 小步长使自行车“立即”冲入：设置为 1.0 ~ 3.0 m（根据 actor 与 ego 距离缩放）
            immediate_dist = min(max(1.0, dist_to_ego * 0.2), 3.0)
            collision_point = carla.Location(
                actor_loc.x + dir_x * immediate_dist,
                actor_loc.y + dir_y * immediate_dist,
                actor_loc.z,
            )

            # 计算 ego 到碰撞点的时间（考虑转弯/减速，使用保守的右/左转时间估计）
            # 这里 ego 触发点离碰撞点的距离近（因为 collision_point 靠近 actor），但我们保守估计
            ego_to_collision_dist = self._calculate_distance(trigger_loc, collision_point)
            ego_turn_speed = max(self.ego_speed * 0.5, 2.5)
            time_trigger_to_collision = ego_to_collision_dist / ego_turn_speed
            ego_total_time = time_to_trigger + time_trigger_to_collision

            # actor 到碰撞点的距离通常很短（immediate_dist）
            actor_distance = self._calculate_distance(actor_loc, collision_point)
            actor_travel_time = actor_distance / max(actor_speed, 0.1)
            actor_total_time = time_to_trigger + actor_travel_time

            # 时间同步评分（越接近越好）
            time_diff = actor_total_time - ego_total_time
            if -0.4 <= time_diff <= 0.4:
                time_score = 30.0 - abs(time_diff) * 30.0
            elif -1.0 <= time_diff <= 1.5:
                time_score = 18.0 - abs(time_diff) * 8.0
            else:
                time_score = max(0.1, 6.0 / (abs(time_diff) + 0.5))

            # 触发距离评分：越小越好（0.5-2.5m 最佳）
            trigger_score = math.exp(-max(0.0, trigger_distance - 1.5) / 1.5)  # 1 when trigger_distance<=1.5
            trigger_score = trigger_score * 10.0  # scale

            # actor immediate 移动惩赏（走得越短越好）
            immediate_score = math.exp(-actor_distance / 1.2) * 12.0  # actor_distance small => big score

            # 速度加分（合适的速度更易撞上）
            speed_bonus = (actor_speed - 4.0) * 1.0

            # 位置偏差惩罚（横向越偏越不可靠）
            lateral_penalty = abs(params["position_offset"]["x"]) * 0.5

            # 总分（把即时性与触发接近放权重最高）
            total_score = time_score + trigger_score + immediate_score + speed_bonus - lateral_penalty

            # debug: 高风险候选输出
            # if time_score > 25.0 or immediate_score > 10.0:
            #     print(f"   [candidate] time_diff={time_diff:.3f}s actor_dist={actor_distance:.2f}m "
            #           f"trigger_d={trigger_distance:.2f}m speed={actor_speed:.2f} heading={actor_heading:.1f}")

            return -total_score

        except Exception as e:
            print(f">> [SkoptGeneticOptimizer] Fitness error: {e}")
            return 1e6

    # ----------------- 工具函数 -----------------
    def _calculate_distance(self, loc1: carla.Location, loc2: carla.Location) -> float:
        return math.hypot(loc1.x - loc2.x, loc1.y - loc2.y)

    def _calculate_actor_position_and_heading(self, params: Dict[str, Any]) -> Tuple[carla.Location, float]:
        """
        根据 base + offset 计算 actor 起点以及面向碰撞点的 heading（deg）。
        返回 (actor_loc, heading_deg)
        """
        offset = params["position_offset"]
        # base forward/right
        actor_heading_rad = math.radians(self.actor_base_heading)
        base_forward = np.array([math.cos(actor_heading_rad), math.sin(actor_heading_rad)])
        base_right = np.array([-math.sin(actor_heading_rad), math.cos(actor_heading_rad)])

        # 应用偏移
        offset_x = offset["x"] * base_right[0] + offset["y"] * base_forward[0]
        offset_y = offset["x"] * base_right[1] + offset["y"] * base_forward[1]

        actor_loc = carla.Location(
            self.actor_base_location.x + offset_x,
            self.actor_base_location.y + offset_y,
            self.actor_base_location.z,
        )

        # 估计碰撞点方向：朝向 ego（使自行车冲向 ego）
        ego_loc = self.ego_transform.location
        vec_x = ego_loc.x - actor_loc.x
        vec_y = ego_loc.y - actor_loc.y
        dist = math.hypot(vec_x, vec_y)
        if dist < 1e-3:
            # 若极近，按 actor_base_heading
            heading_rad = math.radians(self.actor_base_heading)
        else:
            heading_rad = math.atan2(vec_y, vec_x)

        heading_deg = math.degrees(heading_rad) % 360.0

        # 应用 yaw_offset 基因（微调）
        yaw_offset = params.get("yaw_offset", 0.0)
        heading_deg = (heading_deg + yaw_offset) % 360.0

        return actor_loc, heading_deg

    def _genes_to_scene_params(self, genes: np.ndarray) -> Dict[str, Any]:
        bounds = self._get_bounds()
        genes = np.clip(genes, [b[0] for b in bounds], [b[1] for b in bounds])
        return {
            "actor_speed": float(genes[0]),
            "trigger_distance": float(genes[1]),
            "position_offset": {"x": float(genes[2]), "y": float(genes[3]), "yaw": 0.0},
            "yaw_offset": float(genes[4]),
        }

    # ----------------- 返回优化信息 -----------------
    def get_optimization_info(self) -> Dict[str, Any]:
        if self.ga is None:
            return {}
        return {
            "best_fitness": self.best_fitness,
            "best_params": self.best_params.tolist() if self.best_params is not None else None,
            "generations": self.generations,
            "population_size": self.population_size,
        }


# ----------------- 单文件测试入口 -----------------
if __name__ == "__main__":
    class DummyVehicle:
        def get_transform(self):
            return carla.Transform(carla.Location(0, 0, 0), carla.Rotation(yaw=30))
        def get_velocity(self):
            return carla.Vector3D(8, 0, 0)

    class DummyScenario:
        name = "VehicleTurningRoute"
        ego_vehicle = DummyVehicle()
        config = type("C", (), {"other_actors": []})()

    scenario = DummyScenario()
    opt = SkoptGeneticOptimizer(scenario)
    res = opt.optimize_initial_state()
    print("优化结果:", res)
