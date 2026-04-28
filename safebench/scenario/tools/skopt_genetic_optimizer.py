


import numpy as np
from sko.GA import GA
from typing import List, Tuple, Dict, Any
import math
import carla


class SkoptGeneticOptimizer:
    """基于几何计算的遗传算法优化器，避免实时CARLA调用"""

    def __init__(self, scenario):
        self.scenario = scenario

        # 获取ego车辆信息
        self.ego_transform = scenario.ego_vehicle.get_transform()
        self.ego_velocity = scenario.ego_vehicle.get_velocity()
        self.ego_speed = math.sqrt(self.ego_velocity.x ** 2 + self.ego_velocity.y ** 2)
        self.ego_heading = self.ego_transform.rotation.yaw

        if self.ego_speed<1.0:
            self.ego_speed=5

        # 优化参数边界
        self.bounds = self._get_scenario_specific_bounds(scenario)

        # 遗传算法参数
        self.population_size = 30
        self.generations = 30
        self.mutation_rate = 0.13
        self.crossover_rate = 0.7

        # 优化结果
        self.best_params = None
        self.best_fitness = None
        self.ga = None

    def _get_scenario_specific_bounds(self, scenario):
        """根据场景类型获取合适的参数边界"""
        scenario_name = scenario.name.lower()

        if 'dynamicobjectcrossing' in scenario_name or 'pedestrian' in scenario_name:
            # 行人穿越场景
            return [
                (0.5, 3.0),  # [0] actor速度: 行人速度 0.5-3.0 m/s (1.8-10.8 km/h)
                (10.0, 35.0),  # [1] 触发距离: 10-35米
                (-8.0, 8.0),  # [2] X偏移: ±8米
                (-15.0, 15.0),  # [3] Y偏移: ±15米
                (-45.0, 45.0)  # [4] Yaw偏移: ±45度
            ]
        elif 'vehicleturningroute' in scenario_name or 'bicycle' in scenario_name:
            # 自行车/车辆转弯场景
            return [
                (2.0, 8.0),  # [0] actor速度: 自行车速度 2-8 m/s (7.2-28.8 km/h)
                (1.0, 5.0),  # [1] 触发距离:
                (-5.0, 5.0),  # [2] X偏移: ±10米
                (-20.0, 20.0),  # [3] Y偏移: ±20米
                (-60.0, 60.0)  # [4] Yaw偏移: ±60度
            ]
        elif 'otherleadingvehicle' in scenario_name or 'leading' in scenario_name:
            # 前车减速换道场景
            return [
                (1.0, 10.0),  # [0] actor速度: 车辆速度 8-18 m/s
                (10.0, 20.0),  # [1] 触发距离: 20-50米
                (-10.0, 10.0),  # [2] X偏移: ±6米（车道偏移）
                (-10.0, 10.0),  # [3] Y偏移: ±10米（纵向调整）
                (-20.0, 20.0)  # [4] Yaw偏移: ±30度
            ]
        elif 'maneuveroppositedirection' in scenario_name or 'opposite' in scenario_name:
            # 对向车辆超车场景
            return [
                (5.0, 20.0),  # [0] actor速度: 对向车辆速度 10-20 m/s
                (1.0, 30.0),  # [1] 触发距离: 30-60米
                (-0.0, 0.0),  # [2] X偏移: ±5米（车道偏移）
                (-15.0, 15.0),  # [3] Y偏移: ±15米（纵向调整）
                (-0.0, 0.0)  # [4] Yaw偏移: ±30度（角度调整）
            ]
        elif 'nosignaljunctioncrossing' in scenario_name or 'junction' in scenario_name:
            # 无信号路口穿越场景
            return [
                (2.0, 12.0),  # [0] actor速度: 车辆速度 5-15 m/s
                (5.0, 25.0),  # [1] 触发距离: 10-30米
                (-5.0, 5.0),  # [2] X偏移: ±10米
                (-5.0, 5.0),  # [3] Y偏移: ±20米
                (-10.0, 10.0)  # [4] Yaw偏移: ±45度
            ]
        else:
            # 默认设置（保守的车辆范围）
            return [
                (3.0, 15.0),  # [0] actor速度: 车辆速度 3-15 m/s (10.8-54 km/h)
                (5.0, 45.0),  # [1] 触发距离: 15-45米
                (-15.0, 15.0),  # [2] X偏移: ±15米
                (-25.0, 25.0),  # [3] Y偏移: ±25米
                (-90.0, 90.0)  # [4] Yaw偏移: ±90度
            ]

    def optimize_initial_state(self) -> Dict[str, Any]:
        """执行几何优化，返回最优参数"""
        print(">> Starting geometry-based genetic algorithm optimization")

        def fitness_func(x):
            return self._evaluate_geometric_fitness(x)

        # 创建遗传算法实例
        self.ga = GA(
            func=fitness_func,
            n_dim=len(self.bounds),
            size_pop=self.population_size,
            max_iter=self.generations,
            prob_mut=self.mutation_rate,
            lb=[bound[0] for bound in self.bounds],
            ub=[bound[1] for bound in self.bounds],
            precision=[1e-6] * len(self.bounds)
        )

        # 执行优化
        best_x, best_y = self.ga.run()

        self.best_params = best_x
        self.best_fitness = -best_y

        if isinstance(self.best_fitness, np.ndarray):
            self.best_fitness = self.best_fitness.item()

        print(f">> Optimization completed - Best fitness: {self.best_fitness:.4f}")

        return self._genes_to_scene_params(best_x)

    def _evaluate_geometric_fitness(self, genes: np.ndarray) -> float:
        """基于几何计算评估适应度 - 考虑触发距离"""
        scene_params = self._genes_to_scene_params(genes)

        # 计算几何指标
        collision_risk = self._calculate_collision_risk(scene_params)
        braking_intensity = self._calculate_braking_intensity(scene_params)
        surprise_factor = self._calculate_surprise_factor(scene_params)
        path_conflict = self._calculate_path_conflict(scene_params)

        # 新增：触发时机评估
        trigger_timing = self._calculate_trigger_timing(scene_params)
        # if (np.isnan(collision_risk) or np.isnan(braking_intensity) or
        #         np.isnan(surprise_factor) or np.isnan(path_conflict) or
        #         np.isnan(trigger_timing)):
        #     return -1e6  # 返回一个很大的负数作为惩罚
        # 综合评分（调整权重）
        total_fitness = (
                collision_risk * 0.45 +  # 碰撞风险
                braking_intensity * 0.15 +  # 制动强度
                trigger_timing * 0.20 +  # 触发时机
                surprise_factor * 0.10 +  # 突然性
                path_conflict * 0.10  # 路径冲突
        )

        return -total_fitness

    def _calculate_trigger_timing(self, params: Dict) -> float:
        """计算触发时机的好坏"""
        trigger_distance = params['trigger_distance']
        actor_speed = params['actor_speed']

        # 理想的触发距离应该让反应时间刚好处于临界点
        ego_speed = self.ego_speed

        # 计算ego到达actor位置的时间
        # 假设actor在ego前方大约20米处
        distance_to_actor = 10.0
        ego_arrival_time = distance_to_actor / max(ego_speed, 1.0)

        # 计算从触发到碰撞的时间
        reaction_time = trigger_distance / max(ego_speed, 1.0)

        # 理想的反应时间应该是2-3秒（人类反应时间的临界点）
        ideal_reaction_time = 2.5
        timing_diff = abs(reaction_time - ideal_reaction_time)

        # 时间差越小，触发时机越好
        timing_score = math.exp(-timing_diff / 2.0)

        return timing_score

    def _calculate_collision_risk(self, params: Dict) -> float:
        scenario_name = self.scenario.name.lower()
        """计算碰撞风险"""
        if 'maneuveroppositedirection' in scenario_name or 'opposite' in scenario_name:
            """计算碰撞风险 - 针对对向车辆场景"""
            actor_pos = self._predict_actor_position(params)

            ego_pos = self.ego_transform.location
            relative_x = actor_pos.x - ego_pos.x
            relative_y = actor_pos.y - ego_pos.y
            distance = math.sqrt(relative_x ** 2 + relative_y ** 2)

            actor_speed = params['actor_speed']
            ego_speed = self.ego_speed

            # 对向车辆的碰撞时间计算（相对速度相加）
            if distance > 0:
                relative_speed = ego_speed + actor_speed  # 对向行驶，速度相加
                time_to_collision = distance / max(relative_speed, 1.0)

                # 距离越近、时间越短，风险越高
                distance_risk = math.exp(-distance / 25.0)
                time_risk = math.exp(-time_to_collision / 2.0)

                risk = distance_risk * time_risk
            else:
                risk = 1.0
        else:
            # 通用碰撞风险计算

            # 获取actor的预估位置
            actor_pos = self._predict_actor_position(params)

            # 计算相对位置
            ego_pos = self.ego_transform.location
            relative_x = actor_pos.x - ego_pos.x
            relative_y = actor_pos.y - ego_pos.y
            distance = math.sqrt(relative_x ** 2 + relative_y ** 2)

            # 计算相对速度方向
            actor_speed = params['actor_speed']
            ego_speed = self.ego_speed

            # 碰撞时间估算
            if distance > 0:
                time_to_collision = distance / max(ego_speed + actor_speed, 1.0)
                # 距离越近、时间越短，风险越高
                risk = math.exp(-distance / 20.0) * math.exp(-time_to_collision / 3.0)
            else:
                risk = 1.0

        return min(risk, 1.0)

    def _calculate_braking_intensity(self, params: Dict) -> float:
        """计算制动强度"""
        trigger_distance = params['trigger_distance']
        actor_speed = params['actor_speed']

        # 触发距离越近、actor速度越快，制动要求越高
        distance_factor = max(0, (40 - trigger_distance) / 25.0)
        speed_factor = min(actor_speed / 20.0, 1.0)

        return distance_factor * speed_factor

    def _calculate_surprise_factor(self, params: Dict) -> float:
        """计算突然性因子"""
        offset = params['position_offset']

        # 位置偏移的幅度
        offset_magnitude = math.sqrt(offset['x'] ** 2 + offset['y'] ** 2)
        position_surprise = min(offset_magnitude / 30.0, 1.0)

        # 角度偏移的突然性
        yaw_surprise = abs(offset['yaw']) / 90.0

        return (position_surprise + yaw_surprise) / 2.0

    def _calculate_path_conflict(self, params: Dict) -> float:
        """计算路径冲突程度"""
        actor_pos = self._predict_actor_position(params)

        # 基于ego车辆前进方向计算冲突概率
        ego_heading_rad = math.radians(self.ego_heading)
        ego_forward = np.array([math.cos(ego_heading_rad), math.sin(ego_heading_rad)])

        ego_pos = self.ego_transform.location
        to_actor = np.array([actor_pos.x - ego_pos.x, actor_pos.y - ego_pos.y])

        # 计算actor在ego前进方向上的投影
        if np.linalg.norm(to_actor) > 0:
            to_actor_normalized = to_actor / np.linalg.norm(to_actor)
            alignment = abs(np.dot(ego_forward, to_actor_normalized))
            # 对齐度越高，冲突概率越大
            conflict = alignment
        else:
            conflict = 1.0

        return conflict

    def _predict_actor_position(self, params: Dict) -> carla.Location:
        """预测actor的位置"""
        # 获取场景的基础生成位置
        base_transform = self._get_base_transform()

        # 应用位置偏移
        offset = params['position_offset']
        position = base_transform.location

        # 计算偏移向量
        current_rotation = base_transform.rotation
        right_vector = current_rotation.get_right_vector()
        forward_vector = current_rotation.get_forward_vector()

        offset_vector = (right_vector * offset['x'] +
                         forward_vector * offset['y'])

        position = carla.Location(
            position.x + offset_vector.x,
            position.y + offset_vector.y,
            position.z
        )

        return position

    def _get_opposite_direction_base_transform(self) -> carla.Transform:
        """获取对向车辆超车场景的基础位置"""
        ego_transform = self.ego_transform

        # 对向车辆在ego前方50米处，但要反向行驶
        forward_vector = ego_transform.get_forward_vector()
        position = ego_transform.location + forward_vector * 50

        return carla.Transform(
            carla.Location(position.x, position.y, position.z),
            carla.Rotation(yaw=ego_transform.rotation.yaw + 180)  # 对向行驶，180度
        )

    # def _get_leading_vehicle_base_transform(self) -> carla.Transform:
    #     """获取前车场景的基础位置"""
    #     ego_transform = self.ego_transform
    #
    #     # 在ego车辆前方30-40米处（合理的前车位置）
    #     forward_vector = ego_transform.get_forward_vector()
    #     base_distance = 35  # 基础前车距离
    #
    #     position = ego_transform.location + forward_vector * base_distance
    #
    #     return carla.Transform(
    #         carla.Location(position.x, position.y, position.z),
    #         carla.Rotation(yaw=ego_transform.rotation.yaw)  # 与ego同向
    #     )

    def _get_leading_vehicle_base_transform(self) -> carla.Transform:
        """获取前车场景的基础位置 - 使用config中预设的位置"""
        # 使用config中第一辆actor的位置作为基础
        if hasattr(self.scenario.config, 'other_actors') and len(self.scenario.config.other_actors) > 0:
            return self.scenario.config.other_actors[0].transform
        else:
            # 回退方案：使用ego前方位置
            ego_transform = self.ego_transform
            forward_vector = ego_transform.get_forward_vector()
            base_distance = 35

            position = ego_transform.location + forward_vector * base_distance

            return carla.Transform(
                carla.Location(position.x, position.y, position.z),
                carla.Rotation(yaw=ego_transform.rotation.yaw)
            )

    def _get_base_transform(self) -> carla.Transform:
        """获取场景的基础变换"""
        # 根据场景类型返回不同的基础位置
        scenario_name = self.scenario.name.lower()

        if 'dynamicobjectcrossing' in scenario_name:
            return self._get_pedestrian_base_transform()
        elif 'vehicleturningroute' in scenario_name:
            return self._get_intersection_base_transform()
        elif 'otherleadingvehicle' in scenario_name:
            return self._get_leading_vehicle_base_transform()
        elif 'maneuveroppositedirection' in scenario_name:
            return self._get_opposite_direction_base_transform()

        else:
            # 默认基础位置
            return carla.Transform(
                carla.Location(self.ego_transform.location.x + 20,
                               self.ego_transform.location.y,
                               self.ego_transform.location.z),
                carla.Rotation(yaw=90)
            )

    def _get_pedestrian_base_transform(self) -> carla.Transform:
        """获取行人穿越场景的基础位置"""
        ego_transform = self.ego_transform
        # 在ego车辆右侧10米处
        right_vector = ego_transform.get_right_vector()
        position = ego_transform.location + right_vector * 10

        return carla.Transform(
            carla.Location(position.x, position.y, position.z + 0.6),
            carla.Rotation(yaw=ego_transform.rotation.yaw + 90)
        )

    def _get_intersection_base_transform(self) -> carla.Transform:
        """获取路口转弯场景的基础位置"""

        self.actor_base_transform = self.scenario.config.other_actors[0].transform
        # self.actor_base_location = self.actor_base_transform.location
        return self.actor_base_transform

        # ego_transform = self.ego_transform
        # # 在ego车辆前方15米，左侧5米处
        # forward_vector = ego_transform.get_forward_vector()
        # right_vector = ego_transform.get_right_vector()
        #
        # position = ego_transform.location + forward_vector * 15 - right_vector * 5
        #
        # return carla.Transform(
        #     carla.Location(position.x, position.y, position.z),
        #     carla.Rotation(yaw=ego_transform.rotation.yaw - 90)
        # )

    def _genes_to_scene_params(self, genes: np.ndarray) -> Dict[str, Any]:
        """将基因转换为场景参数"""

        # 确保基因值在有效范围内
        genes = np.clip(genes, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        return {
            'actor_speed': float(genes[0]),
            'trigger_distance': float(genes[1]),
            'position_offset': {
                'x': float(genes[2]),
                'y': float(genes[3]),
                'yaw': float(genes[4])
            }
        }

    def get_optimization_info(self) -> Dict[str, Any]:
        """获取优化信息"""
        if self.ga is None:
            return {}

        convergence_history = {}
        if hasattr(self.ga, 'generation_best_Y'):
            convergence_history['best_y_history'] = [-float(y) for y in self.ga.generation_best_Y]

        if hasattr(self.ga, 'all_history_Y'):
            convergence_history['mean_y_history'] = [-float(np.mean(y)) for y in self.ga.all_history_Y]

        return {
            'best_fitness': self.best_fitness,
            'best_params': self.best_params.tolist() if self.best_params is not None else None,
            'generations': self.generations,
            'population_size': self.population_size,
            'convergence_history': convergence_history
        }

