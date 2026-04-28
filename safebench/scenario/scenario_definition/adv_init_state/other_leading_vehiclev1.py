''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-30 12:19:37
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla
import numpy as np

from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.tools.scenario_utils import calculate_distance_transforms
from safebench.scenario.tools.scenario_helper import get_waypoint_in_distance

from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.tools.skopt_genetic_optimizer import SkoptGeneticOptimizer
from typing import Dict

class OtherLeadingVehicle(BasicScenario):
    """
    前车减速换道场景 - 使用遗传算法优化初始状态
    The user-controlled ego vehicle follows a leading car driving down a given road.
    At some point the leading car has to decelerate. The ego vehicle has to react accordingly by changing lane
    to avoid a collision and follow the leading car in other lane. The scenario ends either via a timeout,
    or if the ego vehicle drives some distance. (Traffic Scenario 05)
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(OtherLeadingVehicle, self).__init__("OtherLeadingVehicle-GA", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._other_actor_max_brake = 1.0
        self._first_actor_transform = None
        self._second_actor_transform = None

        # 遗传算法相关
        self.genetic_optimizer = None
        self.use_genetic_algorithm = False
        self.optimized_params = None

        # 基础参数（遗传算法会优化这些值）
        self.position_offset = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.trigger_location = config.trigger_points[0].location if hasattr(config,
                                                                             'trigger_points') and config.trigger_points else None

        # 场景参数
        self.dece_distance = 5
        self.dece_target_speed = 2  # 3 will be safe
        self.need_decelerate = False
        self._first_vehicle_location = 35
        self._second_vehicle_location = 36
        self._first_vehicle_speed = 12
        self._second_vehicle_speed = 12
        self.other_actor_speed = [12, 12]

        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = 35
        self.actions = None

    def create_behavior(self, scenario_init_action):
        """
        创建场景行为
        如果使用遗传算法，则在这里执行优化
        """
        # 检查是否使用遗传算法
        if self._should_use_genetic_algorithm(scenario_init_action):
            print(">> Using genetic algorithm for leading vehicle scenario optimization")




            # 创建遗传算法优化器
            self.genetic_optimizer = SkoptGeneticOptimizer(self)

            # 执行优化
            self.optimized_params = self.genetic_optimizer.optimize_initial_state()

            # 应用优化后的参数
            # 将遗传算法优化的参数映射到场景参数
            self.trigger_distance_threshold = self.optimized_params['trigger_distance']
            self.position_offset = self.optimized_params['position_offset']

            # 从position_offset中提取车辆位置和速度参数
            # 这里可以根据需要设计映射关系
            base_distance = 35
            distance_offset = self.position_offset.get('y', 0.0) * 5  # 纵向偏移影响车距
            speed_factor = (self.optimized_params['actor_speed'] - 10) / 5  # 速度因子

            # 计算优化后的参数
            x1 = max(25, min(45, base_distance + distance_offset))  # 第一车距离
            x2 = max(0.5, min(8, abs(self.position_offset.get('x', 2.0))))  # 车间距
            v1 = max(8, min(18, self.optimized_params['actor_speed']))  # 第一车速度
            v2 = max(8, min(18, self.optimized_params['actor_speed'] + speed_factor))  # 第二车速度

            self.actions = [x1, x2, v1, v2]

            opt_info = self.genetic_optimizer.get_optimization_info()
            print(f">> Leading vehicle scenario optimization completed - Best fitness: {opt_info['best_fitness']:.4f}")

        else:
            # 传统方式：从scenario_init_action获取参数
            if scenario_init_action is not None:
                try:
                    self.actions = self.convert_actions(scenario_init_action)
                except Exception as e:
                    print(f">> Error processing scenario_init_action: {e}")
                    # 使用默认值
                    self.actions = [35.0, 1.0, 12.0, 12.0]
            else:
                # 完全默认情况
                self.actions = [35.0, 1.0, 12.0, 12.0]

        # 应用actions到场景参数
        x1, x2, v1, v2 = self.actions  # [35, 1, 12, 12]
        self._first_vehicle_location = x1
        self._second_vehicle_location = self._first_vehicle_location + x2
        self._first_vehicle_speed = v1
        self._second_vehicle_speed = v2
        self.other_actor_speed = [v1, v2]

        print(f">> [OtherLeadingVehicle] Applied parameters:")
        print(f"   - First vehicle distance: {self._first_vehicle_location:.2f}m")
        print(f"   - Second vehicle distance: {self._second_vehicle_location:.2f}m")
        print(f"   - First vehicle speed: {self._first_vehicle_speed:.2f} m/s")
        print(f"   - Second vehicle speed: {self._second_vehicle_speed:.2f} m/s")

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
        x_min = 30
        x_max = 40
        x_scale = (x_max-x_min)/2

        y_min = 0
        y_max = 5
        y_scale = (y_max-y_min)/2

        yaw_min = 8
        yaw_max = 16
        yaw_scale = (yaw_max-yaw_min)/2

        d_min = 8
        d_max = 16
        d_scale = (d_max-d_min)/2

        x_mean = (x_max + x_min)/2
        y_mean = (y_max + y_min)/2
        yaw_mean = (yaw_max + yaw_min)/2
        dist_mean = (d_max + d_min)/2

        x = actions[0] * x_scale + x_mean
        y = actions[1] * y_scale + y_mean
        yaw = actions[2] * yaw_scale + yaw_mean
        dist = actions[3] * d_scale + dist_mean
        return [x, y, yaw, dist]

    def initialize_actorsHK(self):
        """
        初始化场景参与者 - OtherLeadingVehicle场景
        集成遗传算法优化的位置偏移
        """
        # 确保actions已经设置
        if self.actions is None:
            self.actions = [35.0, 1.0, 12.0, 12.0]
            x1, x2, v1, v2 = self.actions
            self._first_vehicle_location = x1
            self._second_vehicle_location = self._first_vehicle_location + x2
            self._first_vehicle_speed = v1
            self._second_vehicle_speed = v2
            self.other_actor_speed = [v1, v2]

        # 获取车辆航点
        first_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        second_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
        second_vehicle_waypoint = second_vehicle_waypoint.get_left_lane()

        # 创建基础变换
        first_vehicle_transform = carla.Transform(
            first_vehicle_waypoint.transform.location,
            first_vehicle_waypoint.transform.rotation
        )
        first_vehicle_transform.rotation.yaw = (first_vehicle_transform.rotation.yaw) % 360

        second_vehicle_transform = carla.Transform(
            second_vehicle_waypoint.transform.location,
            second_vehicle_waypoint.transform.rotation
        )

        # 应用遗传算法优化的位置偏移（主要对第一个车辆）
        if hasattr(self, 'position_offset') and self.position_offset:
            # 获取第一个车辆的方向向量
            current_rotation = first_vehicle_transform.rotation
            right_vector = current_rotation.get_right_vector()
            forward_vector_current = current_rotation.get_forward_vector()

            # 获取遗传算法优化的偏移量
            x_offset = self.position_offset.get('x', 0.0)  # 横向偏移（右为正）
            y_offset = self.position_offset.get('y', 0.0)  # 纵向偏移（前为正）
            yaw_offset = self.position_offset.get('yaw', 0.0)  # 角度偏移

            # 应用位置偏移到第一个车辆
            offset_vector = right_vector * x_offset + forward_vector_current * y_offset
            first_vehicle_transform.location += offset_vector

            # 应用角度偏移
            first_vehicle_transform.rotation.yaw += yaw_offset
            first_vehicle_transform.rotation.yaw = first_vehicle_transform.rotation.yaw % 360

            # 对第二个车辆应用较小的偏移（保持相对位置关系）
            second_offset_vector = right_vector * (x_offset * 0.3) + forward_vector_current * (y_offset * 0.3)
            second_vehicle_transform.location += second_offset_vector

            # 记录应用的遗传算法优化
            print(f">> [OtherLeadingVehicle] Applied genetic algorithm optimizations:")
            print(f"   - First vehicle X offset: {x_offset:.2f}m")
            print(f"   - First vehicle Y offset: {y_offset:.2f}m")
            print(f"   - First vehicle Yaw offset: {yaw_offset:.2f}°")
            print(f"   - First vehicle final position: ({first_vehicle_transform.location.x:.2f}, {first_vehicle_transform.location.y:.2f})")
            print(f"   - Second vehicle final position: ({second_vehicle_transform.location.x:.2f}, {second_vehicle_transform.location.y:.2f})")

        # 设置参与者类型和变换
        self.actor_type_list = ['vehicle.nissan.patrol', 'vehicle.audi.tt']
        self.actor_transform_list = [first_vehicle_transform, second_vehicle_transform]

        try:
            self.other_actors = self.scenario_operation.initialize_vehicle_actors(
                self.actor_transform_list, self.actor_type_list
            )
            self.reference_actor = self.other_actors[0]  # used for triggering this scenario
            print(f">> [OtherLeadingVehicle] Successfully initialized two vehicles")
        except Exception as e:
            print(f">> [OtherLeadingVehicle] Failed to initialize vehicles: {e}")
            self.reference_actor = None



    def update_behavior(self, scenario_action):
        """更新场景行为（保持原有逻辑）"""
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        # At specific point, vehicle in front of ego will decelerate other_actors[0] is the vehicle before the ego
        cur_distance = calculate_distance_transforms(
            self.actor_transform_list[0],
            CarlaDataProvider.get_transform(self.other_actors[0])
        )
        if cur_distance > self.dece_distance:
            self.need_decelerate = True

        for i in range(len(self.other_actors)):
            if i == 0 and self.need_decelerate:
                self.scenario_operation.go_straight(self.dece_target_speed, i)
            else:
                self.scenario_operation.go_straight(self.other_actor_speed[i], i)

    def check_stop_condition(self):
        """检查停止条件"""
        # 可以根据需要添加停止条件
        if not self.other_actors or len(self.other_actors) < 2:
            return True

        # 例如：当ego车辆行驶了足够距离后停止
        # 这里可以根据具体需求实现
        return False