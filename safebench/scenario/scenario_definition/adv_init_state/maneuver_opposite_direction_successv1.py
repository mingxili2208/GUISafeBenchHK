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
from safebench.scenario.tools.skopt_genetic_optimizer_m import SkoptGeneticOptimizer
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.tools.scenario_helper import get_waypoint_in_distance
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_operation import ScenarioOperation


class ManeuverOppositeDirection(BasicScenario):
    """
    追尾场景 - 三车跟车，中间车辆突然停车，导致ego追尾
    布局：第一辆车（前面，正常）→ 第二辆车（中间，停车）→ ego（后面，追尾）
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(ManeuverOppositeDirection, self).__init__("ManeuverOppositeDirection-Adv", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)

        # ✅ 添加：保存ego车道信息，供遗传算法使用
        ego_location = ego_vehicle.get_location()
        self._ego_waypoint = self._map.get_waypoint(ego_location)

        # 遗传算法相关
        self.genetic_optimizer = None
        self.optimized_params = None

        # 基础参数（遗传算法会优化这些值）
        self.initial_speed = 1.0
        self.trigger_distance_threshold = 35.0
        self.position_offset = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.trigger_location = config.trigger_points[0].location

        self.scenario_operation = ScenarioOperation()
        self.ego_max_driven_distance = 200
        self._stop_executed = False

    def create_behavior(self, scenario_init_action):
        """创建场景行为 - 使用几何遗传算法优化追尾场景"""
        # 检查是否使用遗传算法
        if self._should_use_genetic_algorithm(scenario_init_action):
            print(">> Using geometry-based genetic algorithm for middle-vehicle rear-end optimization")

            # 创建优化器并执行优化
            self.genetic_optimizer = SkoptGeneticOptimizer(self)
            self.optimized_params = self.genetic_optimizer.optimize_initial_state()

            # 应用优化后的参数
            self.initial_speed = self.optimized_params['actor_speed']
            self.trigger_distance_threshold = self.optimized_params['trigger_distance']
            self.position_offset = self.optimized_params['position_offset']

            # 计算基于优化参数的车辆位置
            self._calculate_vehicle_positions()

            opt_info = self.genetic_optimizer.get_optimization_info()
            print(f">> Middle-vehicle rear-end optimization completed - Best fitness: {opt_info['best_fitness']:.4f}")

        else:
            # 传统方式
            print(">> Using default parameters for middle-vehicle rear-end scenario")
            actions = self.convert_actions(scenario_init_action)
            x1, x2 = actions
            self._first_vehicle_location = x1  # 第一辆车位置（最前面，正常行驶）
            self._second_vehicle_location = x1 - x2  # 第二辆车位置（中间，会停车）

    def _calculate_vehicle_positions(self):
        """根据优化参数计算两车位置 - 第二辆车会突然停车"""
        base_distance = 30  # 基础距离

        # 从位置偏移中提取参数
        offset = self.position_offset
        distance_offset = offset.get('y', 0.0) * 3  # 纵向偏移影响距离
        lateral_offset = offset.get('x', 0.0) * 1  # 横向偏移（保持车道内）

        # 计算第一辆车位置（前面，正常行驶）
        self._first_vehicle_location = max(35, min(60, base_distance + distance_offset))

        # 第二辆车位置（在第一辆车后面，会突然停车）
        # 两车之间应该保持5-15米的固定距离
        vehicle_spacing = 10.0  # 固定间距20米
        # 可以根据遗传算法的纵向偏移稍微调整间距(±5米)
        spacing_adjustment = max(-5.0, min(4.0, distance_offset * 0.4))

        self._second_vehicle_location = self._first_vehicle_location - (vehicle_spacing + spacing_adjustment)



        # 确保第二辆车在ego后面，不要太近也不要太远
        self._second_vehicle_location = max(10, min(25, self._second_vehicle_location))


        print(f">> [_calculate_vehicle_positions] First vehicle: {self._first_vehicle_location:.2f}m from ego")
        print(f">> [_calculate_vehicle_positions] Second vehicle: {self._second_vehicle_location:.2f}m from ego")
        print(
            f">> [_calculate_vehicle_positions] Spacing between vehicles: {self._first_vehicle_location - self._second_vehicle_location:.2f}m")

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
        x_max = 70
        x_scale = (x_max-x_min)/2
        x_mean = (x_max + x_min)/2

        y_min = 20
        y_max = 30
        y_scale = (y_max-y_min)/2
        y_mean = (y_max + y_min)/2

        x = actions[0] * x_scale + x_mean
        y = actions[1] * y_scale + y_mean
        return [x, y]

    def initialize_actorsHK(self):
        """初始化场景参与者 - 同车道两辆车,第二辆会突然停车"""

        # 使用ego车辆的当前位置和车道，而不是reference_waypoint
        ego_location = self.ego_vehicle.get_location()
        ego_waypoint = self._map.get_waypoint(ego_location)

        print(f">> [initialize_actorsHK] Ego position: ({ego_location.x:.2f}, {ego_location.y:.2f})")
        print(f">> [initialize_actorsHK] Ego lane_id: {ego_waypoint.lane_id}")

        # 获取第一辆车(前面,正常行驶)的航点位置 - 在ego前方
        first_actor_waypoint = ego_waypoint
        traveled_distance = 0.0
        target_distance = self._first_vehicle_location

        while traveled_distance < target_distance:
            next_waypoints = first_actor_waypoint.next(1.0)
            if not next_waypoints:
                break
            first_actor_waypoint = next_waypoints[0]
            traveled_distance += 1.0

        # 创建第一辆车的变换
        first_actor_transform = carla.Transform(
            first_actor_waypoint.transform.location,
            first_actor_waypoint.transform.rotation
        )
        first_actor_transform.rotation.yaw = (first_actor_transform.rotation.yaw) % 360

        print(
            f">> [initialize_actorsHK] First vehicle position: ({first_actor_transform.location.x:.2f}, {first_actor_transform.location.y:.2f})")
        print(f">> [initialize_actorsHK] First vehicle lane_id: {first_actor_waypoint.lane_id}")

        # 第二辆车基于ego waypoint向前，距离为_second_vehicle_location
        second_actor_waypoint = ego_waypoint
        traveled_distance = 0.0
        target_distance = self._second_vehicle_location

        while traveled_distance < target_distance:
            next_waypoints = second_actor_waypoint.next(1.0)
            if not next_waypoints:
                break
            second_actor_waypoint = next_waypoints[0]
            traveled_distance += 1.0

        # 创建第二辆车的变换
        second_actor_transform = carla.Transform(
            second_actor_waypoint.transform.location,
            second_actor_waypoint.transform.rotation
        )
        second_actor_transform.rotation.yaw = (second_actor_transform.rotation.yaw) % 360

        print(
            f">> [initialize_actorsHK] Second vehicle position: ({second_actor_transform.location.x:.2f}, {second_actor_transform.location.y:.2f})")
        print(f">> [initialize_actorsHK] Second vehicle lane_id: {second_actor_waypoint.lane_id}")

        # ✅ 只对第二辆车应用遗传算法优化的偏移
        if hasattr(self, 'position_offset') and self.position_offset:
            current_rotation = second_actor_transform.rotation
            right_vector = current_rotation.get_right_vector()
            forward_vector_current = current_rotation.get_forward_vector()

            # 限制偏移范围，避免偏离车道
            x_offset = max(-1.5, min(1.5, self.position_offset.get('x', 0.0)))  # 限制横向±1.5米
            y_offset = max(-8.0, min(3.0, self.position_offset.get('y', 0.0)))  # 限制纵向±3米
            yaw_offset = max(-10.0, min(10.0, self.position_offset.get('yaw', 0.0)))  # 限制角度±10度

            offset_vector = right_vector * x_offset + forward_vector_current * y_offset
            second_actor_transform.location += offset_vector
            second_actor_transform.rotation.yaw += yaw_offset
            second_actor_transform.rotation.yaw = second_actor_transform.rotation.yaw % 360

        self.trigger_location = second_actor_transform.location

        # 计算实际距离
        actual_distance = first_actor_transform.location.distance(second_actor_transform.location)

        print(f">> [ManeuverOppositeDirection] Applied genetic algorithm optimizations:")
        print(f"   - Actual distance between vehicles: {actual_distance:.2f}m")
        print(f"   - Applied X offset: {x_offset:.2f}m")
        print(f"   - Applied Y offset: {y_offset:.2f}m")
        print(f"   - Applied Yaw offset: {yaw_offset:.2f}°")



        # 安全检查
        if actual_distance < 5.0:
            print(f">> WARNING: Vehicles too close ({actual_distance:.2f}m), this may cause collision!")

        # 使用有效的CARLA车辆模型
        self.actor_type_list = ['vehicle.toyota.prius', 'vehicle.audi.a2']
        self.actor_transform_list = [first_actor_transform, second_actor_transform]

        try:
            self.other_actors = self.scenario_operation.initialize_vehicle_actors(
                self.actor_transform_list, self.actor_type_list
            )
            self.reference_actor = self.other_actors[1] if len(self.other_actors) > 1 else None
            print(f">> [ManeuverOppositeDirection] Successfully initialized {len(self.other_actors)} vehicles")
        except Exception as e:
            print(f">> [ManeuverOppositeDirection] Failed to initialize vehicles: {e}")
            import traceback
            traceback.print_exc()
            self.reference_actor = None
            self.other_actors = []

    def update_behavior(self, scenario_action):
        """更新场景行为 - 中间车辆停车追尾场景"""
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        if not self.other_actors or len(self.other_actors) < 2:
            return

        # 获取ego车辆和第二辆车（中间，会停车）的距离
        ego_location = self.ego_vehicle.get_location()
        second_vehicle_location = self.other_actors[1].get_location()
        distance = ego_location.distance(second_vehicle_location)
        print(f">> trigger_distance_threshold: {self.trigger_distance_threshold}")

        if not self._stop_executed and distance <= self.trigger_distance_threshold:
            # 触发第二辆车突然停车
            print(f">> trigger_distance_threshold: {self.trigger_distance_threshold}")
            print(f">> [ManeuverOppositeDirection] Triggering middle vehicle sudden stop at distance: {distance:.2f}m")
            self.scenario_operation.brake(self.other_actors[1])  # 第二辆车（中间）停车
            self._stop_executed = True
        elif not self._stop_executed:
            # 两辆车都继续以初始速度行驶
            self.scenario_operation.go_straight(4, 0)  # 第一辆车（前面）正常行驶
            self.scenario_operation.go_straight(3, 1)  # 第二辆车（中间）正常行驶
        else:
            # 第二辆车已经停车，保持停止状态
            self.scenario_operation.go_straight(self.initial_speed, 0)  # 第一辆车（前面）继续正常行驶
            self.scenario_operation.brake(self.other_actors[1])  # 第二辆车（中间）保持停车

    def check_stop_condition(self):
        """检查停止条件"""
        if not self.other_actors or len(self.other_actors) < 2:
            return True

        # 检查是否发生碰撞（ego与第二辆车）
        ego_location = self.ego_vehicle.get_location()
        second_vehicle_location = self.other_actors[1].get_location()
        distance = ego_location.distance(second_vehicle_location)

        # 如果距离小于2米，认为发生碰撞
        if distance < 2.0:
            print(">> [ManeuverOppositeDirection] Collision detected between ego and middle vehicle!")
            return True
        #
        # # 检查ego车辆是否行驶超过最大距离
        # if hasattr(self, '_ego_distance_driven'):
        #     if self._ego_distance_driven > self.ego_max_driven_distance:
        #         return True

        return False