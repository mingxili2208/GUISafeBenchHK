# safebench/scenario/tools/geometry_genetic_optimizer.py
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

        if self.ego_speed < 1.0:
            self.ego_speed = 5.0

        # 优化参数边界
        self.bounds = self._get_scenario_specific_bounds(scenario)

        # 遗传算法参数 - 针对自行车场景调整
        self.population_size = 40  # 增加种群规模
        self.generations = 40  # 增加迭代次数
        self.mutation_rate = 0.15  # 稍微增加变异率
        self.crossover_rate = 0.7

        # 优化结果
        self.best_params = None
        self.best_fitness = None
        self.ga = None

    def _get_scenario_specific_bounds(self, scenario):
        """根据场景类型获取合适的参数边界"""
        scenario_name = scenario.name.lower()

        if 'vehicleturningroute' in scenario_name or 'bicycle' in scenario_name:
            # 自行车/车辆转弯场景 - 调整更合理的范围
            return [
                (3.0, 8.0),  # [0] actor速度: 自行车速度 3-10 m/s (10.8-36 km/h)
                (5.0, 15.0),  # [1] 触发距离: 15-35米
                (-1, 1),  # [2] X偏移: ±8米（横向）
                (-10.0, 10.0),  # [3] Y偏移: ±15米（纵向）
                (-50.0, 50.0)  # [4] Yaw偏移: ±45度
            ]
        elif 'dynamicobjectcrossing' in scenario_name or 'pedestrian' in scenario_name:
            # 行人穿越场景
            return [
                (0.5, 3.0),
                (10.0, 35.0),
                (-8.0, 8.0),
                (-15.0, 15.0),
                (-45.0, 45.0)
            ]
        elif 'otherleadingvehicle' in scenario_name or 'leading' in scenario_name:
            # 前车减速换道场景
            return [
                (1.0, 10.0),
                (10.0, 20.0),
                (-10.0, 10.0),
                (-10.0, 10.0),
                (-30.0, 30.0)
            ]
        elif 'maneuveroppositedirection' in scenario_name or 'opposite' in scenario_name:
            # 对向车辆超车场景
            return [
                (5.0, 20.0),
                (1.0, 30.0),
                (-0.0, 0.0),
                (-15.0, 15.0),
                (-0.0, 0.0)
            ]
        else:
            # 默认设置
            return [
                (3.0, 15.0),
                (5.0, 45.0),
                (-15.0, 15.0),
                (-25.0, 25.0),
                (-90.0, 90.0)
            ]

    def optimize_initial_state(self) -> Dict[str, Any]:
        """执行几何优化，返回最优参数"""
        print(">> Starting geometry-based genetic algorithm optimization")
        print(f">> Ego vehicle speed: {self.ego_speed:.2f} m/s")
        print(f">> Ego vehicle heading: {self.ego_heading:.2f}°")

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
        print(f">> Optimized actor speed: {best_x[0]:.2f} m/s")
        print(f">> Optimized trigger distance: {best_x[1]:.2f} m")

        return self._genes_to_scene_params(best_x)

    def _evaluate_geometric_fitness(self, genes: np.ndarray) -> float:
        """基于几何计算评估适应度 - 针对自行车转弯场景优化"""
        scene_params = self._genes_to_scene_params(genes)

        # 计算各项指标
        collision_risk = self._calculate_collision_risk_bicycle(scene_params)
        braking_intensity = self._calculate_braking_intensity(scene_params)
        surprise_factor = self._calculate_surprise_factor(scene_params)
        path_conflict = self._calculate_path_conflict_bicycle(scene_params)
        trigger_timing = self._calculate_trigger_timing(scene_params)

        # 综合评分 - 针对自行车场景调整权重
        total_fitness = (
                collision_risk * 0.40 +  # 碰撞风险最重要
                path_conflict * 0.25 +  # 路径冲突很重要
                trigger_timing * 0.20 +  # 触发时机
                braking_intensity * 0.10 +  # 制动强度
                surprise_factor * 0.05  # 突然性
        )

        return -total_fitness

    def _calculate_collision_risk_bicycle(self, params: Dict) -> float:
        """计算自行车场景的碰撞风险"""
        actor_pos = self._predict_actor_position(params)
        ego_pos = self.ego_transform.location

        # 计算相对距离
        relative_x = actor_pos.x - ego_pos.x
        relative_y = actor_pos.y - ego_pos.y
        distance = math.sqrt(relative_x ** 2 + relative_y ** 2)

        actor_speed = params['actor_speed']
        ego_speed = self.ego_speed

        # 预测碰撞时间
        if distance > 0.1 and (ego_speed + actor_speed) > 0.1:
            # 计算相对速度（考虑方向）
            ego_heading_rad = math.radians(self.ego_heading)
            ego_vel_x = ego_speed * math.cos(ego_heading_rad)
            ego_vel_y = ego_speed * math.sin(ego_heading_rad)

            # 简化的碰撞时间估算
            relative_speed = math.sqrt((ego_vel_x) ** 2 + (ego_vel_y) ** 2) + actor_speed
            time_to_collision = distance / max(relative_speed, 1.0)

            # 距离风险（距离越近风险越高）
            distance_risk = math.exp(-distance / 20.0)

            # 时间风险（2-4秒内碰撞风险最高）
            ideal_ttc = 3.0  # 理想的碰撞时间
            time_risk = math.exp(-abs(time_to_collision - ideal_ttc) / 2.0)

            # 综合风险
            risk = distance_risk * 0.6 + time_risk * 0.4
        else:
            risk = 0.0

        return min(risk, 1.0)

    def _calculate_path_conflict_bicycle(self, params: Dict) -> float:
        """计算自行车的路径冲突程度"""
        actor_pos = self._predict_actor_position(params)
        ego_pos = self.ego_transform.location

        # 计算ego到actor的向量
        to_actor = np.array([actor_pos.x - ego_pos.x, actor_pos.y - ego_pos.y])
        distance = np.linalg.norm(to_actor)

        if distance < 0.1:
            return 1.0

        # ego的前进方向
        ego_heading_rad = math.radians(self.ego_heading)
        ego_forward = np.array([math.cos(ego_heading_rad), math.sin(ego_heading_rad)])

        # 计算方向一致性
        to_actor_normalized = to_actor / distance
        alignment = np.dot(ego_forward, to_actor_normalized)

        # 如果自行车在ego前进方向上，冲突更高
        if alignment > 0:
            conflict = alignment * math.exp(-distance / 25.0)
        else:
            conflict = 0.1

        return min(conflict, 1.0)

    def _calculate_trigger_timing(self, params: Dict) -> float:
        """计算触发时机的好坏"""
        trigger_distance = params['trigger_distance']
        actor_speed = params['actor_speed']
        ego_speed = self.ego_speed

        # 计算反应时间
        if ego_speed > 0.1:
            reaction_time = trigger_distance / ego_speed
        else:
            reaction_time = trigger_distance / 5.0

        # 理想反应时间应该是2-3.5秒
        ideal_reaction_time = 2.8
        timing_diff = abs(reaction_time - ideal_reaction_time)

        # 时间差越小，触发时机越好
        timing_score = math.exp(-timing_diff / 1.5)

        return timing_score

    def _calculate_braking_intensity(self, params: Dict) -> float:
        """计算制动强度"""
        trigger_distance = params['trigger_distance']
        actor_speed = params['actor_speed']

        # 触发距离越近、actor速度越快，制动要求越高
        distance_factor = max(0, (35 - trigger_distance) / 20.0)
        speed_factor = min(actor_speed / 15.0, 1.0)

        return distance_factor * speed_factor

    def _calculate_surprise_factor(self, params: Dict) -> float:
        """计算突然性因子"""
        offset = params['position_offset']

        # 位置偏移的幅度
        offset_magnitude = math.sqrt(offset['x'] ** 2 + offset['y'] ** 2)
        position_surprise = min(offset_magnitude / 25.0, 1.0)

        # 角度偏移的突然性
        yaw_surprise = abs(offset['yaw']) / 90.0

        return (position_surprise + yaw_surprise) / 2.0

    def _calculate_collision_risk(self, params: Dict) -> float:
        """通用碰撞风险计算"""
        return self._calculate_collision_risk_bicycle(params)

    def _calculate_path_conflict(self, params: Dict) -> float:
        """通用路径冲突计算"""
        return self._calculate_path_conflict_bicycle(params)

    def _predict_actor_position(self, params: Dict) -> carla.Location:
        """预测actor的位置"""
        base_transform = self._get_base_transform()
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

    def _get_base_transform(self) -> carla.Transform:
        """获取场景的基础变换"""
        scenario_name = self.scenario.name.lower()

        if 'vehicleturningroute' in scenario_name or 'bicycle' in scenario_name:
            return self._get_intersection_base_transform()
        elif 'dynamicobjectcrossing' in scenario_name:
            return self._get_pedestrian_base_transform()
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

    def _get_intersection_base_transform(self) -> carla.Transform:
        """获取路口转弯场景的基础位置"""
        # 使用配置中的actor位置
        if hasattr(self.scenario.config, 'other_actors') and len(self.scenario.config.other_actors) > 0:
            return self.scenario.config.other_actors[0].transform
        else:
            # 备用：在ego前方20米，左侧5米
            ego_transform = self.ego_transform
            forward_vector = ego_transform.get_forward_vector()
            right_vector = ego_transform.get_right_vector()

            position = ego_transform.location + forward_vector * 20 - right_vector * 5

            return carla.Transform(
                carla.Location(position.x, position.y, position.z),
                carla.Rotation(yaw=ego_transform.rotation.yaw - 90)
            )

    def _get_pedestrian_base_transform(self) -> carla.Transform:
        """获取行人穿越场景的基础位置"""
        ego_transform = self.ego_transform
        right_vector = ego_transform.get_right_vector()
        position = ego_transform.location + right_vector * 10

        return carla.Transform(
            carla.Location(position.x, position.y, position.z + 0.6),
            carla.Rotation(yaw=ego_transform.rotation.yaw + 90)
        )

    def _get_leading_vehicle_base_transform(self) -> carla.Transform:
        """获取前车场景的基础位置"""
        ego_transform = self.ego_transform
        forward_vector = ego_transform.get_forward_vector()
        base_distance = 35

        position = ego_transform.location + forward_vector * base_distance

        return carla.Transform(
            carla.Location(position.x, position.y, position.z),
            carla.Rotation(yaw=ego_transform.rotation.yaw)
        )

    def _get_opposite_direction_base_transform(self) -> carla.Transform:
        """获取对向车辆超车场景的基础位置"""
        ego_transform = self.ego_transform
        forward_vector = ego_transform.get_forward_vector()
        position = ego_transform.location + forward_vector * 50

        return carla.Transform(
            carla.Location(position.x, position.y, position.z),
            carla.Rotation(yaw=ego_transform.rotation.yaw + 180)
        )

    def _genes_to_scene_params(self, genes: np.ndarray) -> Dict[str, Any]:
        """将基因转换为场景参数"""
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