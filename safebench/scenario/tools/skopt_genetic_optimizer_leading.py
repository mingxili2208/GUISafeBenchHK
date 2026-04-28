import numpy as np
from sko.GA import GA
from typing import List, Tuple, Dict, Any
import math
import carla


class SkoptGeneticOptimizer:
    """基于几何计算的遗传算法优化器 - 优化碰撞概率"""

    def __init__(self, scenario):
        self.scenario = scenario

        # 获取ego车辆信息
        self.ego_transform = scenario.ego_vehicle.get_transform()
        self.ego_velocity = scenario.ego_vehicle.get_velocity()
        self.ego_speed = math.sqrt(self.ego_velocity.x ** 2 + self.ego_velocity.y ** 2)
        self.ego_heading = self.ego_transform.rotation.yaw

        if self.ego_speed < 1.0:
            self.ego_speed = 6

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

        if 'otherleadingvehicle' in scenario_name or 'leading' in scenario_name:
            # 前车减速场景 - 优化碰撞
            # [0] actor速度, [1] 场景触发距离, [2] 减速触发距离, [3] X偏移, [4] Y偏移, [5] Yaw偏移
            return [
                (4.6, 9.0),  # [0] actor速度: 车辆速度 8-18 m/s (29-65 km/h)
                (30.0, 60.0),  # [1] 场景触发距离: ego多远时车辆开始运动 (30-60米)
                (12.9, 17),  # [2] 减速触发距离: 运动后ego多近时减速 (8-20米)
                (-1, 1),  # [3] X偏移: ±10米(车道偏移)
                (-15.0, 10.0),  # [4] Y偏移: ±10米(纵向调整)
                (-1, 1)  # [5] Yaw偏移: ±20度
            ]
        elif 'dynamicobjectcrossing' in scenario_name or 'pedestrian' in scenario_name:
            # 行人穿越场景
            return [
                (0.5, 3.0),
                (10.0, 35.0),
                (10.0, 35.0),  # 这里也加一个占位
                (-8.0, 8.0),
                (-15.0, 15.0),
                (-45.0, 45.0)
            ]
        elif 'vehicleturningroute' in scenario_name or 'bicycle' in scenario_name:
            # 自行车/车辆转弯场景
            return [
                (2.0, 8.0),
                (1.0, 5.0),
                (1.0, 5.0),
                (-5.0, 5.0),
                (-20.0, 20.0),
                (-60.0, 60.0)
            ]
        elif 'maneuveroppositedirection' in scenario_name or 'opposite' in scenario_name:
            # 对向车辆超车场景
            return [
                (5.0, 20.0),
                (1.0, 30.0),
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
                (5.0, 45.0),
                (-15.0, 15.0),
                (-25.0, 25.0),
                (-90.0, 90.0)
            ]

    def optimize_initial_state(self) -> Dict[str, Any]:
        """执行几何优化,返回最优参数"""
        print(">> Starting genetic algorithm optimization for COLLISION")

        def fitness_func(x):
            return self._evaluate_collision_fitness(x)

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

        print(f">> Optimization completed - Best collision fitness: {self.best_fitness:.4f}")

        return self._genes_to_scene_params(best_x)

    def _evaluate_collision_fitness(self, genes: np.ndarray) -> float:
        """
        评估碰撞适应度 - 目标是最大化碰撞概率

        核心思想:
        1. 场景触发距离要合适 - 给ego足够反应时间但又不能太远
        2. 减速触发距离要危险 - 在ego很难刹停的距离突然减速
        3. 速度差要大 - 第一辆车减速后速度很慢,ego速度快
        4. 位置要在ego正前方 - 避免ego有躲避空间
        """
        scene_params = self._genes_to_scene_params(genes)

        # 1. 碰撞风险评估 (最重要,权重最大)
        collision_risk = self._calculate_collision_risk_for_leading(scene_params)

        # 2. 刹车难度评估
        braking_difficulty = self._calculate_braking_difficulty(scene_params)

        # 3. 减速时机危险度
        deceleration_danger = self._calculate_deceleration_danger(scene_params)

        # 4. 位置对齐度(在ego正前方)
        alignment_score = self._calculate_path_alignment(scene_params)

        # 5. 速度差评估
        speed_differential = self._calculate_speed_differential(scene_params)

        # [新增] 逃逸惩罚：如果前车速度显著快于Ego，给予重罚
        # 因为如果前车比Ego快，距离会拉大，永远无法触发减速逻辑
        actor_speed = scene_params['actor_speed']
        # 获取ego当前速度，如果没有动则假设一个典型速度
        current_ego_speed = self.ego_speed if self.ego_speed > 1.0 else 10.0

        # 如果前车速度 > Ego速度，极大概率会导致距离拉大而不是缩小
        escape_penalty = 0.0
        if actor_speed > current_ego_speed * 1.07:  # 前车速度超过Ego的90%就开始惩罚
            escape_penalty = 0.25  # 扣除大量分数

        # 综合评分 - 针对碰撞优化
        total_fitness = (
                collision_risk * 0.35 +  # 碰撞风险
                braking_difficulty * 0.25 +  # 刹车难度
                deceleration_danger * 0.20 +  # 减速时机
                alignment_score * 0.10 +  # 位置对齐
                speed_differential * 0.10  # 速度差
        )

        total_fitness = total_fitness - escape_penalty

        return -total_fitness  # GA最小化,所以取负

    def _calculate_collision_risk_for_leading(self, params: Dict) -> float:
        """计算前车场景的碰撞风险 - 越高越好"""
        scenario_trigger_dist = params['scenario_trigger_distance']
        decel_trigger_dist = params['deceleration_trigger_distance']
        actor_speed = params['actor_speed']

        # 1. 场景触发距离评估
        # 太远(>50m): ego可能还没加速,碰撞风险低
        # 太近(<30m): 可能来不及触发场景
        # 最优: 40-45m
        ideal_scenario_trigger = 28.0
        scenario_trigger_score = math.exp(-abs(scenario_trigger_dist - ideal_scenario_trigger) / 10.0)

        # 2. 减速触发距离评估
        # 太远(>18m): ego有足够时间刹车
        # 太近(<8m): 可能ego已经很慢或已经碰了
        # 最优: 10-15m (根据ego速度,这是人类反应时间的临界点)
        ego_speed = self.ego_speed

        # 根据ego速度计算危险距离
        # 假设人类反应时间1.5s + 刹车距离
        reaction_distance = ego_speed * 1.5
        braking_distance = (ego_speed ** 2) / (2 * 5.0)  # 假设减速度5m/s²
        safe_distance = reaction_distance + braking_distance

        # 理想的减速触发距离应该略小于安全距离
        ideal_decel_trigger = safe_distance * 0.9
        decel_trigger_score = math.exp(-abs(decel_trigger_dist - ideal_decel_trigger) / 5.0)

        # 3. 两个触发距离的关系
        # 场景触发距离必须 > 减速触发距离
        # 距离差应该给ego足够时间加速接近,但又不能太多
        distance_gap = scenario_trigger_dist - decel_trigger_dist
        ideal_gap = 20.0  # 理想差距20米
        gap_score = math.exp(-abs(distance_gap - ideal_gap) / 10.0)

        # 4. 速度因素
        # 速度越快,碰撞风险越高
        speed_risk = min(actor_speed / 15.0, 1.0)

        # 综合评分
        risk = (
                scenario_trigger_score * 0.25 +
                decel_trigger_score * 0.35 +
                gap_score * 0.25 +
                speed_risk * 0.15
        )

        return risk

    def _calculate_braking_difficulty(self, params: Dict) -> float:
        """计算刹车难度 - 越难越好(越容易碰撞)"""
        decel_trigger_dist = params['deceleration_trigger_distance']
        actor_speed = params['actor_speed']
        ego_speed = self.ego_speed

        # 计算ego需要的刹车距离
        # v² = u² + 2as -> s = (v² - u²) / (2a)
        # 假设ego能提供的最大减速度为7m/s²
        max_deceleration = 6.0

        # ego需要的刹车距离
        required_braking_distance = (ego_speed ** 2) / (2 * max_deceleration)

        # 实际可用距离(减速触发距离 - 一些缓冲)
        available_distance = decel_trigger_dist - 2.0  # 2米缓冲

        # 如果可用距离 < 需要距离,则很难刹住
        if available_distance < required_braking_distance:
            difficulty = 1.0 - (available_distance / required_braking_distance)
        else:
            # 可用距离越接近需要距离,难度越高
            difficulty = math.exp(-(available_distance - required_braking_distance) / 10.0)

        return min(difficulty, 1.0)

    def _calculate_deceleration_danger(self, params: Dict) -> float:
        """计算减速时机的危险度 - 越危险越好"""
        decel_trigger_dist = params['deceleration_trigger_distance']
        actor_speed = params['actor_speed']
        ego_speed = self.ego_speed

        # 1. 时间紧迫度
        # 在减速触发距离时,ego还有多少时间
        time_available = decel_trigger_dist / max(ego_speed, 1.0)

        # 理想的时间应该是1.5-2.5秒(人类反应时间的边缘)
        ideal_time = 2.3
        time_danger = math.exp(-abs(time_available - ideal_time) / 1.5)

        # 2. 速度差因素
        # 假设actor减速到原速度的15-25%
        decel_speed = actor_speed * 0.20
        speed_diff = ego_speed - decel_speed

        # 速度差越大,危险度越高
        speed_danger = min(speed_diff / 15.0, 1.0)

        # 3. 距离危险度
        # 距离越近,危险度越高
        distance_danger = math.exp(-decel_trigger_dist / 12.0)

        danger = (
                time_danger * 0.4 +
                speed_danger * 0.3 +
                distance_danger * 0.3
        )

        return danger

    def _calculate_path_alignment(self, params: Dict) -> float:
        """计算路径对齐度 - 越对齐越好(正前方,无法躲避)"""
        offset = params['position_offset']

        # X偏移(横向)应该尽可能小
        lateral_offset = abs(offset['x'])
        lateral_score = math.exp(-lateral_offset / 3.0)

        # Yaw偏移应该尽可能小
        yaw_offset = abs(offset['yaw'])
        yaw_score = math.exp(-yaw_offset / 15.0)

        alignment = (lateral_score * 0.6 + yaw_score * 0.4)

        return alignment

    def _calculate_speed_differential(self, params: Dict) -> float:
        """计算速度差 - 差距越大越好"""
        actor_speed = params['actor_speed']
        ego_speed = self.ego_speed

        # 假设actor减速到原速度的15-25%
        decel_speed = actor_speed * 0.20

        # ego与减速后actor的速度差
        speed_diff = ego_speed - decel_speed

        # 速度差越大,碰撞动能越大
        differential_score = min(speed_diff / 20.0, 1.0)

        return differential_score

    def _calculate_collision_risk(self, params: Dict) -> float:
        """通用碰撞风险计算(保留用于其他场景)"""
        actor_pos = self._predict_actor_position(params)

        ego_pos = self.ego_transform.location
        relative_x = actor_pos.x - ego_pos.x
        relative_y = actor_pos.y - ego_pos.y
        distance = math.sqrt(relative_x ** 2 + relative_y ** 2)

        actor_speed = params['actor_speed']
        ego_speed = self.ego_speed

        if distance > 0:
            time_to_collision = distance / max(ego_speed + actor_speed, 1.0)
            risk = math.exp(-distance / 20.0) * math.exp(-time_to_collision / 3.0)
        else:
            risk = 1.0

        return min(risk, 1.0)

    def _predict_actor_position(self, params: Dict) -> carla.Location:
        """预测actor的位置"""
        base_transform = self._get_base_transform()

        offset = params['position_offset']
        position = base_transform.location

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

    def _get_leading_vehicle_base_transform(self) -> carla.Transform:
        """获取前车场景的基础位置 - 使用config中预设的位置"""
        if hasattr(self.scenario.config, 'other_actors') and len(self.scenario.config.other_actors) > 0:
            return self.scenario.config.other_actors[0].transform
        else:
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
        scenario_name = self.scenario.name.lower()

        if 'otherleadingvehicle' in scenario_name:
            return self._get_leading_vehicle_base_transform()
        else:
            return carla.Transform(
                carla.Location(self.ego_transform.location.x + 20,
                               self.ego_transform.location.y,
                               self.ego_transform.location.z),
                carla.Rotation(yaw=90)
            )

    def _genes_to_scene_params(self, genes: np.ndarray) -> Dict[str, Any]:
        """将基因转换为场景参数"""
        genes = np.clip(genes, [b[0] for b in self.bounds], [b[1] for b in self.bounds])

        return {
            'actor_speed': float(genes[0]),
            'scenario_trigger_distance': float(genes[1]),  # 新增
            'deceleration_trigger_distance': float(genes[2]),  # 新增
            'position_offset': {
                'x': float(genes[3]),
                'y': float(genes[4]),
                'yaw': float(genes[5])
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