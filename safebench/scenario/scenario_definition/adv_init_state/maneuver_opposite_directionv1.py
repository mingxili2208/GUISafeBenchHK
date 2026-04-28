''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-30 12:19:20
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.tools.scenario_helper import get_waypoint_in_distance
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_operation import ScenarioOperation

from typing import Dict
class ManeuverOppositeDirection(BasicScenario):
    """
        Vehicle is passing another vehicle in a rural area, in daylight, under clear weather conditions, 
        at a non-junction and encroaches into another vehicle traveling in the opposite direction.
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(ManeuverOppositeDirection, self).__init__("ManeuverOppositeDirection-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        # self._source_transform = None
        # self._sink_location = None
        self._first_actor_transform = None
        self._second_actor_transform = None
        # self._third_actor_transform = None

        # 基础参数（遗传算法会优化这些值）
        self.actor_speed = 12.0
        self.trigger_distance_threshold = 40.0
        self.position_offset = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.x = 0.0
        self.trigger = False
        self.trigger_location = config.trigger_points[0].location if hasattr(config,
                                                                             'trigger_points') and config.trigger_points else None

        self.scenario_operation = ScenarioOperation()
        # self.trigger_distance_threshold = 45
        self.ego_max_driven_distance = 200
        self._actor_distance = 120

    def convert_actions(self, actions):
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

    def initialize_actorsHK_(self):
        first_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        second_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
        second_actor_waypoint = second_actor_waypoint.get_left_lane()
        first_actor_transform = carla.Transform(first_actor_waypoint.transform.location, first_actor_waypoint.transform.rotation)
        first_actor_transform.rotation.yaw = (first_actor_transform.rotation.yaw) % 360
        second_actor_transform = second_actor_waypoint.transform

        self.actor_type_list = ['vehicle.nissan.micra', 'vehicle.nissan.micra']
        self.actor_transform_list = [first_actor_transform, second_actor_transform]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario

    def initialize_actorsHK(self):
        """
        初始化场景参与者 - ManeuverOppositeDirection场景
        集成遗传算法优化的位置偏移
        """
        # 获取基础的航点位置
        first_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        second_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
        second_actor_waypoint = second_actor_waypoint.get_left_lane()

        # 创建第一个车辆的基础变换（对向车辆）
        first_actor_transform = carla.Transform(
            first_actor_waypoint.transform.location,
            first_actor_waypoint.transform.rotation
        )
        first_actor_transform.rotation.yaw = (first_actor_transform.rotation.yaw) % 360

        # 创建第二个车辆的基础变换
        second_actor_transform = second_actor_waypoint.transform

        # 应用遗传算法优化的位置偏移（主要对第一个车辆，即对向车辆）
        if hasattr(self, 'position_offset') and self.position_offset:
            # 获取第一个车辆掉头后的方向向量
            current_rotation = first_actor_transform.rotation
            right_vector = current_rotation.get_right_vector()
            forward_vector_current = current_rotation.get_forward_vector()

            # 获取遗传算法优化的偏移量
            x_offset = self.position_offset.get('x', 0.0)  # 横向偏移（右为正）
            y_offset = self.position_offset.get('y', 0.0)  # 纵向偏移（前为正）
            yaw_offset = self.position_offset.get('yaw', 0.0)  # 角度偏移

            # 应用位置偏移到第一个车辆（对向车辆）
            offset_vector = right_vector * x_offset + forward_vector_current * y_offset
            first_actor_transform.location += offset_vector

            # 应用角度偏移
            first_actor_transform.rotation.yaw += yaw_offset
            first_actor_transform.rotation.yaw = first_actor_transform.rotation.yaw % 360

            # 记录应用的遗传算法优化
            print(f">> [ManeuverOppositeDirection] Applied genetic algorithm optimizations:")
            print(f"   - X offset (lateral): {x_offset:.2f}m")
            print(f"   - Y offset (longitudinal): {y_offset:.2f}m")
            print(f"   - Yaw offset: {yaw_offset:.2f}°")
            print(
                f"   - First actor final position: ({first_actor_transform.location.x:.2f}, {first_actor_transform.location.y:.2f})")
            print(f"   - First actor final yaw: {first_actor_transform.rotation.yaw:.2f}°")

        # 记录遗传算法优化的其他参数
        if hasattr(self, 'actor_speed') and hasattr(self, 'trigger_distance_threshold'):
            print(f">> [ManeuverOppositeDirection] Applied optimized speed: {self.actor_speed:.2f} m/s")
            print(
                f">> [ManeuverOppositeDirection] Applied optimized trigger distance: {self.trigger_distance_threshold:.2f} m")

        # 设置参与者类型和变换列表
        self.actor_type_list = ['vehicle.nissan.micra', 'vehicle.nissan.micra']
        self.actor_transform_list = [first_actor_transform, second_actor_transform]

        # 初始化参与者
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(
            self.actor_transform_list, self.actor_type_list
        )
        self.reference_actor = self.other_actors[0]  # used for triggering this scenario
        
    def create_behavior(self, scenario_init_action):
        if self._should_use_genetic_algorithm(scenario_init_action):
            print(">> Using genetic algorithm for opposite direction maneuver optimization")

            # 导入优化器（避免循环导入）
            from safebench.scenario.tools.skopt_genetic_optimizer import SkoptGeneticOptimizer

            # 创建遗传算法优化器
            self.genetic_optimizer = SkoptGeneticOptimizer(self)

            # 执行优化
            self.optimized_params = self.genetic_optimizer.optimize_initial_state()

            # 应用优化后的参数
            self.actor_speed = self.optimized_params['actor_speed']
            self.trigger_distance_threshold = self.optimized_params['trigger_distance']
            self.position_offset = self.optimized_params['position_offset']

            # 设置self.x
            self.x = 0.0

            opt_info = self.genetic_optimizer.get_optimization_info()
            print(
                f">> Opposite direction maneuver optimization completed - Best fitness: {opt_info['best_fitness']:.4f}")

        else:
            actions = self.convert_actions(scenario_init_action)
            x1, x2, v2 = actions
            self._first_vehicle_location = x1
            self._second_vehicle_location = self._first_vehicle_location + x2
            self._opposite_speed = v2  # m/s


    def _should_use_genetic_algorithm(self, scenario_init_action) -> bool:
        """
        判断是否应该使用遗传算法
        """
        if scenario_init_action is None:
            return False

        # # 如果scenario_init_action是列表且包含特定标识,则使用遗传算法。
        # 下述代码未启用，改为通过config参数控制
        # if hasattr(scenario_init_action, '__getitem__') and len(scenario_init_action) > 0:
        #     first_element = scenario_init_action[0]
        #     if isinstance(first_element, str) and first_element == "genetic_algorithm":
        #         return True

        # 检查config中是否启用遗传算法参数
        if (hasattr(self.config, 'parameters') and
                self.config.parameters[0] == 'genetic_algorithm' and
                self.config.parameters[1] is True):
            return True

        return False

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        # first actor run in low speed, second actor run in normal speed from oncoming route
        self.scenario_operation.go_straight(self._opposite_speed, 1)

    def check_stop_condition(self):
        pass
