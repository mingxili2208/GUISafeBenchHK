''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-30 12:19:20
Description: 
    Copyright (c) 2022-2023 SafeBench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla
from safebench.scenario.tools.skopt_genetic_optimizer import SkoptGeneticOptimizer
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.tools.scenario_helper import get_waypoint_in_distance
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_operation import ScenarioOperation


class ManeuverOppositeDirection(BasicScenario):
    """
    对向车辆超车场景 - 使用几何遗传算法优化初始状态
    Vehicle is passing another vehicle in a rural area, in daylight, under clear weather conditions,
    at a non-junction and encroaches into another vehicle traveling in the opposite direction.
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(ManeuverOppositeDirection, self).__init__("ManeuverOppositeDirection-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)

        # 遗传算法相关
        self.genetic_optimizer = None
        self.optimized_params = None

        # 基础参数（遗传算法会优化这些值）
        self.actor_speed = 15.0  # 对向车辆速度
        self.trigger_distance_threshold = 40.0
        self.position_offset = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.trigger_location = config.trigger_points[0].location

        self.scenario_operation = ScenarioOperation()
        self.ego_max_driven_distance = 200
        self._actor_distance = 120
        self._opposite_speed = 15.0  # 对向车辆速度

    def create_behavior(self, scenario_init_action):
        """创建场景行为 - 使用几何遗传算法优化"""
        # 检查是否使用遗传算法
        if self._should_use_genetic_algorithm(scenario_init_action):
            print(">> Using geometry-based genetic algorithm for opposite direction maneuver optimization")

            # 创建优化器并执行优化
            self.genetic_optimizer = SkoptGeneticOptimizer(self)
            self.optimized_params = self.genetic_optimizer.optimize_initial_state()

            # 应用优化后的参数
            self.actor_speed = self.optimized_params['actor_speed']
            self.trigger_distance_threshold = self.optimized_params['trigger_distance']
            self.position_offset = self.optimized_params['position_offset']
            self._opposite_speed = self.actor_speed

            # 计算基于优化参数的车辆位置
            self._calculate_vehicle_positions()

            opt_info = self.genetic_optimizer.get_optimization_info()
            print(f">> Opposite direction maneuver optimization completed - Best fitness: {opt_info['best_fitness']:.4f}")

        else:
            # 传统方式
            print(">> Using default parameters for opposite direction maneuver scenario")
            actions = self.convert_actions(scenario_init_action)
            x1, x2, v2 = actions
            self._first_vehicle_location = x1
            self._second_vehicle_location = x1 + x2
            self._opposite_speed = v2

    def _calculate_vehicle_positions(self):
        """根据优化参数计算车辆位置"""
        base_distance = 30  # 基础距离

        # 从位置偏移中提取参数
        offset = self.position_offset
        distance_offset = offset.get('y', 0.0) * 2  # 纵向偏移影响距离
        lateral_offset = offset.get('x', 0.0) * 3  # 横向偏移影响超车起点

        # 计算两车位置
        self._first_vehicle_location = max(40, min(70, base_distance + distance_offset))
        self._second_vehicle_location = self._first_vehicle_location + max(15, min(25, abs(lateral_offset) + 20))

    def _should_use_genetic_algorithm(self, scenario_init_action) -> bool:
        """
        判断是否应该使用遗传算法
        """
        if scenario_init_action is None:
            return False
        # 检查config中是否启用遗传算法参数
        if (hasattr(self.config, 'parameters') and
                self.config.parameters[0] == 'genetic_algorithm' and
                self.config.parameters[1] is True):
            return True

        return False

    def convert_actions(self, actions):
        """转换动作（保持原有逻辑）"""
        x_min = 40
        x_max = 60
        x_scale = (x_max-x_min)/2

        y_min = 20
        y_max = 40
        y_scale = (y_max-y_min)/2

        yaw_min = 6
        yaw_max = 10
        yaw_scale = (yaw_max-yaw_min)/2

        x_mean = (x_max + x_min)/2
        y_mean = (y_max + y_min)/2
        yaw_mean = (yaw_max + yaw_min)/2

        x = actions[0] * x_scale + x_mean
        y = actions[1] * y_scale + y_mean
        yaw = actions[2] * yaw_scale + yaw_mean
        return [x, y, yaw]

    def initialize_actorsHK(self):
        """初始化场景参与者 - 对向车辆超车场景"""
        # 获取基础的航点位置
        first_actor_waypoint, _ = self._reference_waypoint
        second_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, 30)
        # tempsecond_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, 20)


        # 创建第一个车辆的基础变换（被超车辆，慢车）
        first_actor_transform = carla.Transform(
            first_actor_waypoint.transform.location,
            first_actor_waypoint.transform.rotation
        )
        first_actor_transform.rotation.yaw = (first_actor_transform.rotation.yaw) % 360

        # 创建第二个车辆的基础变换
        second_actor_transform = second_actor_waypoint.transform

        # 应用遗传算法优化的位置偏移（主要对第一个车辆，即对向车辆）
        if hasattr(self, 'position_offset') and self.position_offset:
            # 获取第2个车辆的方向向量
            current_rotation = second_actor_transform.rotation
            right_vector = current_rotation.get_right_vector()
            forward_vector_current = current_rotation.get_forward_vector()

            # 获取遗传算法优化的偏移量
            x_offset = self.position_offset.get('x', 0.0)  # 横向偏移（右为正）
            y_offset = self.position_offset.get('y', 0.0)  # 纵向偏移（前为正）
            yaw_offset = self.position_offset.get('yaw', 0.0)  # 角度偏移

            # 应用位置偏移到第二个车辆（对向车辆）
            offset_vector = right_vector * x_offset + forward_vector_current * y_offset
            second_actor_transform.location += offset_vector

            # 应用角度偏移
            second_actor_transform.rotation.yaw += yaw_offset
            second_actor_transform.rotation.yaw = second_actor_transform.rotation.yaw % 360

            # 记录应用的遗传算法优化
            print(f">> [ManeuverOppositeDirection] Applied genetic algorithm optimizations:")
            print(f"   - X offset (lateral): {x_offset:.2f}m")
            print(f"   - Y offset (longitudinal): {y_offset:.2f}m")
            print(f"   - Yaw offset: {yaw_offset:.2f}°")
            print(f"   - Opposite vehicle final position: ({second_actor_transform.location.x:.2f}, {second_actor_transform.location.y:.2f})")
            print(f"   - Opposite vehicle final yaw: {second_actor_transform.rotation.yaw:.2f}°")

        # 记录遗传算法优化的其他参数
        if hasattr(self, 'actor_speed') and hasattr(self, 'trigger_distance_threshold'):
            print(f">> [ManeuverOppositeDirection] Applied optimized opposite vehicle speed: {self.actor_speed:.2f} m/s")
            print(f">> [ManeuverOppositeDirection] Applied optimized trigger distance: {self.trigger_distance_threshold:.2f} m")

        # 使用有效的CARLA车辆模型
        self.actor_type_list = ['vehicle.toyota.prius', 'vehicle.audi.a2']
        self.actor_transform_list = [first_actor_transform, second_actor_transform]

        try:
            self.other_actors = self.scenario_operation.initialize_vehicle_actors(
                self.actor_transform_list, self.actor_type_list
            )
            self.reference_actor = self.other_actors[1]  # used for triggering this scenario
            print(f">> [ManeuverOppositeDirection] Successfully initialized two vehicles for opposite direction scenario")
        except Exception as e:
            print(f">> [ManeuverOppositeDirection] Failed to initialize vehicles: {e}")
            self.reference_actor = None

    def update_behavior(self, scenario_action):
        """更新场景行为"""
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        for i in range(len(self.other_actors)):
            if i == 0:
                # 被超车辆，慢速行驶
                self.scenario_operation.go_straight(8, i)  # 慢车速度 8 m/s
            elif i == 1:
                # 对向车辆，以优化速度行驶
                self.scenario_operation.go_straight(self._opposite_speed, i)

    def check_stop_condition(self):
        """检查停止条件"""
        if not self.other_actors or len(self.other_actors) < 2:
            return True

        # 可以根据具体需求实现停止条件
        return False