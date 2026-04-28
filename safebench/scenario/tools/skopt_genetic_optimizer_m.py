import numpy as np
import time
from typing import Dict, List, Any, Tuple

from sko.GA import GA

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
        if self.ego_speed < 1.0:  # 小于1m/s认为是静止或刚起步
            self.ego_speed = 7.0  # 设置为10 m/s (36 km/h) 的合理城市速度
            print(
                f">> Ego speed is {self.ego_velocity.x:.2f}, {self.ego_velocity.y:.2f}, using default speed: {self.ego_speed} m/s")
        else:
            print(f">> Ego speed detected: {self.ego_speed:.2f} m/s")
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
                (10.0, 40.0),  # [1] 触发距离: 10-40米
                (-10.0, 10.0),  # [2] X偏移: ±10米
                (-20.0, 20.0),  # [3] Y偏移: ±20米
                (-60.0, 60.0)  # [4] Yaw偏移: ±60度
            ]
        elif 'maneuveroppositedirection' in scenario_name:
            # ego撞上第二辆车追尾场景 - 这是主要的优化目标
            return [
                (3.0,5.0),  # [0] 第二辆车初始速度: 18-25 m/s (65-90 km/h)
                (7, 10.0),  # [1] 停车触发距离: 20-45米
                (-4, 4),  # [2] X偏移: ±2米（车道内横向偏移）
                (-12.0, 5.0),  # [3] Y偏移: ±15米（纵向距离调整）
                (-5.0, 5.0)  # [4] Yaw偏移: ±15度（轻微角度调整）
            ]
        elif 'otherleadingvehicle' in scenario_name or 'leading' in scenario_name:
            # 前车减速换道场景
            return [
                (1.0, 10.0),  # [0] actor速度: 车辆速度 8-18 m/s
                (10.0, 20.0),  # [1] 触发距离: 20-50米
                (-10.0, 10.0),  # [2] X偏移: ±6米（车道偏移）
                (-10.0, 10.0),  # [3] Y偏移: ±10米（纵向调整）
                (-30.0, 30.0)  # [4] Yaw偏移: ±30度
            ]
        else:
            # 默认设置（保守的车辆范围）
            return [
                (3.0, 6.0),  # [0] actor速度: 车辆速度 3-15 m/s (10.8-54 km/h)
                (8.0, 45.0),  # [1] 触发距离: 15-45米
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
        """基于几何计算评估适应度 - 针对ego撞上第二辆车追尾场景优化"""
        scene_params = self._genes_to_scene_params(genes)
        scenario_name = self.scenario.name.lower()

        if 'maneuveroppositedirection' in scenario_name:
            # ego撞上第二辆车追尾场景的专门适应度计算
            collision_risk = self._calculate_ego_second_vehicle_collision_risk(scene_params)
            emergency_braking_intensity = self._calculate_emergency_braking_intensity(scene_params)
            surprise_timing = self._calculate_sudden_stop_surprise_timing(scene_params)
            lane_alignment = self._calculate_lane_alignment_factor(scene_params)
            spacing_safety = self._calculate_vehicles_spacing_safety(scene_params)

            # ego撞上第二辆车追尾场景的权重分配
            total_fitness = (
                    collision_risk * 0.50 +  # 碰撞风险（最重要）
                    emergency_braking_intensity * 0.15 +  # 紧急制动强度
                    surprise_timing * 0.25 +  # 停车突然性
                    lane_alignment * 0.05 +  # 车道对齐度
                    spacing_safety * 0.05  # 车辆间距安全性
            )
        else:
            # 其他场景使用原有逻辑
            collision_risk = self._calculate_collision_risk(scene_params)
            braking_intensity = self._calculate_braking_intensity(scene_params)
            surprise_factor = self._calculate_surprise_factor(scene_params)
            path_conflict = self._calculate_path_conflict(scene_params)

            total_fitness = (
                    collision_risk * 0.45 +
                    braking_intensity * 0.15 +
                    surprise_factor * 0.20 +
                    path_conflict * 0.10
            )

        return -total_fitness

    def _calculate_ego_second_vehicle_collision_risk(self, params: Dict) -> float:
        """计算ego与第二辆车的追尾碰撞风险"""
        trigger_distance = params['trigger_distance']
        second_vehicle_speed = params['actor_speed']
        ego_speed = self.ego_speed

        # ✅ 添加：使用预测位置计算实际距离
        predicted_second_vehicle_pos = self._predict_actor_position(params)
        ego_pos = self.ego_transform.location
        actual_distance_to_second_vehicle = math.sqrt(
            (predicted_second_vehicle_pos.x - ego_pos.x) ** 2 +
            (predicted_second_vehicle_pos.y - ego_pos.y) ** 2
        )

        print(
            f">> [Fitness Debug] Predicted second vehicle distance from ego: {actual_distance_to_second_vehicle:.2f}m")
        print(f">> [Fitness Debug] Trigger distance: {trigger_distance:.2f}m")
        print(f">> [Fitness Debug] Second vehicle speed: {second_vehicle_speed:.2f}m/s, Ego speed: {ego_speed:.2f}m/s")

        # 计算ego的反应距离
        reaction_time = 1.0  # 1秒反应时间

        # ego的制动距离（假设制动减速度为6 m/s²）
        if ego_speed > 0:
            ego_braking_distance = (ego_speed ** 2) / (2 * 7)
        else:
            ego_braking_distance = 0

        # 总的停车距离
        total_stopping_distance = ego_speed * reaction_time + ego_braking_distance

        # ✅ 修改：同时考虑触发距离和实际距离
        # 如果第二辆车太远（>30米），降低碰撞风险
        if actual_distance_to_second_vehicle > 30.0:
            distance_penalty = math.exp(-(actual_distance_to_second_vehicle - 30.0) / 10.0)
        else:
            distance_penalty = 1.0

        # 碰撞风险：触发距离越接近总停车距离，风险越高
        distance_ratio = trigger_distance / max(total_stopping_distance, 1.0)

        # ✅ 修改：距离比例在0.8-1.2范围内风险最高（更容易撞上）
        if 0.8 <= distance_ratio <= 1.2:
            collision_risk = 1.0
        else:
            collision_risk = math.exp(-abs(distance_ratio - 1.0) * 1.5)  # ✅ 从2.0改为1.5，扩大高风险区间

        # 速度差异影响风险
        speed_diff = abs(second_vehicle_speed - ego_speed)
        # 速度差越小越危险（第二辆车和ego速度接近时更容易追尾）
        if speed_diff < 5.0:  # 速度差小于5 m/s
            speed_factor = 1.0
        else:
            speed_factor = 0.7 + 0.3 * min(speed_diff / 15.0, 1.0)

        final_risk = collision_risk * speed_factor * distance_penalty

        print(f">> [Fitness Debug] Collision risk components:")
        print(f"   - Distance ratio: {distance_ratio:.2f}")
        print(f"   - Collision risk: {collision_risk:.2f}")
        print(f"   - Speed factor: {speed_factor:.2f}")
        print(f"   - Distance penalty: {distance_penalty:.2f}")
        print(f"   - Final risk: {final_risk:.2f}")

        return final_risk

    # def _calculate_vehicles_spacing_safety(self, params: Dict) -> float:
    #     """计算两车之间的间距安全性"""
    #     offset = params['position_offset']
    #
    #     # 从偏移参数估算第二辆车与第一辆车的距离
    #     # 第二辆车在第一辆车后面15-25米是合理的
    #     # 这里用y偏移来模拟间距调整
    #     spacing_estimate = max(15, min(25, abs(offset.get('y', 0.0)) + 20))
    #
    #     # 理想间距是20米左右
    #     ideal_spacing = 20.0
    #     spacing_diff = abs(spacing_estimate - ideal_spacing)
    #
    #     # 间距越接近理想值，安全性越高
    #     spacing_safety = math.exp(-spacing_diff / 5.0)
    #
    #     return spacing_safety

    def _calculate_vehicles_spacing_safety(self, params: Dict) -> float:
        """计算两车之间的间距安全性"""
        offset = params['position_offset']

        # 使用预测位置计算第一辆车和第二辆车的实际间距
        # 假设第一辆车在第二辆车前方约8-12米
        predicted_spacing = 10.0 + offset.get('y', 0.0) * 0.5

        # 理想间距缩小到8米（原来20米太远）
        ideal_spacing = 8.0
        spacing_diff = abs(predicted_spacing - ideal_spacing)

        # 间距越接近理想值，安全性越高
        spacing_safety = math.exp(-spacing_diff / 3.0)  # ✅ 从5.0改为3.0，更敏感

        print(
            f">> [Fitness Debug] Spacing safety: predicted={predicted_spacing:.2f}m, ideal={ideal_spacing:.2f}m, safety={spacing_safety:.2f}")

        return spacing_safety

    def _calculate_emergency_braking_intensity(self, params: Dict) -> float:
        """计算紧急制动强度"""
        trigger_distance = params['trigger_distance']
        second_vehicle_speed = params['actor_speed']

        # 触发距离越近，第二辆车速度越快，要求的制动强度越高
        distance_intensity = max(0, (40 - trigger_distance) / 30.0)
        speed_intensity = second_vehicle_speed / 25.0  # 归一化到0-1

        return min(distance_intensity * speed_intensity, 1.0)

    def _calculate_sudden_stop_surprise_timing(self, params: Dict) -> float:
        """计算突然停车的时机突然性"""
        trigger_distance = params['trigger_distance']
        ego_speed = self.ego_speed

        # 理想的突然停车距离：让ego刚好来不及反应
        ideal_trigger_distance = ego_speed * 1.0  # 1.5秒的反应时间

        # 距离差越小，突然性越强
        distance_diff = abs(trigger_distance - ideal_trigger_distance)
        surprise_factor = math.exp(-distance_diff / 3.0)

        return surprise_factor

    def _calculate_lane_alignment_factor(self, params: Dict) -> float:
        """计算车道对齐因子 - 确保在同一车道内"""
        offset = params['position_offset']

        # 横向偏移应该较小（同一车道内）
        lateral_offset = abs(offset['x'])
        alignment_factor = math.exp(-lateral_offset / 3.0)  # 3米是车道宽度的一半

        # 角度偏移也应该较小
        yaw_alignment = math.exp(-abs(offset['yaw']) / 30.0)

        return (alignment_factor + yaw_alignment) / 2.0

    def _calculate_collision_risk(self, params: Dict) -> float:
        """通用碰撞风险计算"""
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
        """预测第二辆车（会停车）的位置"""
        # 获取场景的基础生成位置 - 使用第二辆车的基础位置
        base_transform = self._get_second_vehicle_base_transform()

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

    def _get_second_vehicle_base_transform(self) -> carla.Transform:
        """获取第二辆车（会停车）的基础位置"""
        # 使用scenario的reference_waypoint而不是ego_transform
        if hasattr(self.scenario, '_reference_waypoint'):
            reference_waypoint = self.scenario._reference_waypoint

            # 使用与initialize_actorsHK相同的逻辑
            # 第二辆车的基础距离（这个值应该与场景中_second_vehicle_location的平均值一致）
            base_distance = 12.0  # 第二辆车在reference点后约15米处

            # 沿着车道获取waypoint
            current_waypoint = reference_waypoint
            traveled_distance = 0.0

            while traveled_distance < base_distance:
                next_waypoints = current_waypoint.next(1.0)
                if not next_waypoints:
                    break
                current_waypoint = next_waypoints[0]
                traveled_distance += 1.0

            return carla.Transform(
                current_waypoint.transform.location,
                current_waypoint.transform.rotation
            )
        else:
            # 降级方案：使用ego前方位置
            ego_transform = self.ego_transform
            forward_vector = ego_transform.get_forward_vector()
            position = ego_transform.location + forward_vector * 10

            return carla.Transform(
                carla.Location(position.x, position.y, position.z),
                carla.Rotation(yaw=ego_transform.rotation.yaw)
            )

    def _get_leading_vehicle_base_transform(self) -> carla.Transform:
        """获取前车场景的基础位置"""
        ego_transform = self.ego_transform

        # 在ego车辆前方30-40米处（合理的前车位置）
        forward_vector = ego_transform.get_forward_vector()
        base_distance = 35  # 基础前车距离

        position = ego_transform.location + forward_vector * base_distance

        return carla.Transform(
            carla.Location(position.x, position.y, position.z),
            carla.Rotation(yaw=ego_transform.rotation.yaw)  # 与ego同向
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
            return self._get_second_vehicle_base_transform()  # ✅ 第二辆车基础位置
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
        ego_transform = self.ego_transform
        # 在ego车辆前方15米，左侧5米处
        forward_vector = ego_transform.get_forward_vector()
        right_vector = ego_transform.get_right_vector()

        position = ego_transform.location + forward_vector * 15 - right_vector * 5

        return carla.Transform(
            carla.Location(position.x, position.y, position.z),
            carla.Rotation(yaw=ego_transform.rotation.yaw - 90)
        )

    def _genes_to_scene_params(self, genes: np.ndarray) -> Dict[str, Any]:
        """将基因转换为场景参数"""
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