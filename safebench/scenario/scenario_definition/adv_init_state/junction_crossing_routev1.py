''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-30 12:19:04
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla
from typing import Dict
from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_utils import calculate_distance_transforms
from safebench.scenario.tools.skopt_genetic_optimizer_junction import create_optimizer


class OppositeVehicleRunningRedLight(BasicScenario):
    """
        This class holds everything required for a scenario, in which an other vehicle takes priority from the ego vehicle, 
        by running a red traffic light (while the ego vehicle has green).
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(OppositeVehicleRunningRedLight, self).__init__("OppositeVehicleRunningRedLight-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout


        # 遗传算法相关
        self.genetic_optimizer = None
        self.use_genetic_algorithm = False
        self.optimized_params = None

        # 基础参数（遗传算法会优化这些值）
        self.actor_speed = 10.0
        self.trigger_distance_threshold = 35.0
        self.position_offset = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.trigger = False

        self.trigger_location = config.trigger_points[0].location if hasattr(config,
                                                                             'trigger_points') and config.trigger_points else None

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light is None:
            print(">> No traffic light for the given location of the ego vehicle found")
        else:
            self._traffic_light.set_state(carla.TrafficLightState.Green)
            self._traffic_light.set_green_time(self.timeout)

        self.scenario_operation = ScenarioOperation()
        self._actor_distance = 110
        self.ego_max_driven_distance = 150

    def convert_actions(self, actions):
        """ Process the action from model. action is assumed in [-1, 1] """
        y_scale = 5
        yaw_scale = 5
        d_scale = 5
        y_mean = yaw_mean = dist_mean = 0

        y = actions[0] * y_scale + y_mean
        yaw = actions[1] * yaw_scale + yaw_mean
        dist = actions[2] * d_scale + dist_mean
        return [y, yaw, dist]

    # def initialize_actorsHK(self):
    #     other_actor_transform = self.config.other_actors[0].transform
    #     forward_vector = other_actor_transform.rotation.get_forward_vector() * self.x
    #     other_actor_transform.location += forward_vector
    #     first_vehicle_transform = carla.Transform(
    #         carla.Location(other_actor_transform.location.x, other_actor_transform.location.y, other_actor_transform.location.z),
    #         other_actor_transform.rotation
    #     )
    #
    #     # 香港靠左行驶,初始化车辆掉头行驶，否则逆行 TODO: 行驶方向也可由路径导出的时候统一处理
    #     first_vehicle_transform.rotation.yaw = (first_vehicle_transform.rotation.yaw - 180) % 360
    #     # 设置参与者变换和类型
    #     self.actor_transform_list = [first_vehicle_transform]
    #     self.actor_type_list = ["vehicle.audi.tt"]
    #     # 初始化参与者
    #     self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
    #     self.reference_actor = self.other_actors[0] # used for triggering this scenario
    #
    #     # other vehicle's traffic light
    #     traffic_light_other = CarlaDataProvider.get_next_traffic_light(other_actor_transform, False, True)
    #     if traffic_light_other is None:
    #         print(">> No traffic light for the given location of the other vehicle found")
    #     else:
    #         traffic_light_other.set_state(carla.TrafficLightState.Red)
    #         traffic_light_other.set_red_time(self.timeout)

    def initialize_actorsHK(self):
        """
        初始化场景参与者 - OppositeVehicleRunningRedLight场景
        集成遗传算法优化的位置偏移
        """
        other_actor_transform = self.config.other_actors[0].transform

        # 应用原有的x偏移逻辑（这个self.x来自原有的convert_actions或遗传算法优化）
        forward_vector = other_actor_transform.rotation.get_forward_vector() * self.x
        other_actor_transform.location += forward_vector

        # 创建基础变换
        first_vehicle_transform = carla.Transform(
            carla.Location(
                other_actor_transform.location.x,
                other_actor_transform.location.y,
                other_actor_transform.location.z
            ),
            other_actor_transform.rotation
        )

        # 香港靠左行驶,初始化车辆掉头行驶，否则逆行
        # first_vehicle_transform.rotation.yaw = (first_vehicle_transform.rotation.yaw) % 360

        # 应用遗传算法优化的位置偏移
        if hasattr(self, 'position_offset') and self.position_offset:
            # 获取当前变换后的方向向量
            current_rotation = first_vehicle_transform.rotation
            right_vector = current_rotation.get_right_vector()
            forward_vector_current = current_rotation.get_forward_vector()

            # 获取遗传算法优化的偏移量
            x_offset = self.position_offset.get('x', 0.0)  # 横向偏移（右为正）
            y_offset = self.position_offset.get('y', 0.0)  # 纵向偏移（前为正）
            yaw_offset = self.position_offset.get('yaw', 0.0)  # 角度偏移

            # yaw_offset=0

            # 应用位置偏移（在掉头后的坐标系下）
            offset_vector = right_vector * x_offset + forward_vector_current * y_offset
            first_vehicle_transform.location += offset_vector

            # 应用角度偏移
            first_vehicle_transform.rotation.yaw += yaw_offset
            first_vehicle_transform.rotation.yaw = first_vehicle_transform.rotation.yaw % 360

            # 记录应用的遗传算法优化
            print(f">> Applied genetic algorithm optimizations:")
            print(f"   - X offset (lateral): {x_offset:.2f}m")
            print(f"   - Y offset (longitudinal): {y_offset:.2f}m")
            print(f"   - Yaw offset: {yaw_offset:.2f}°")
            print(
                f"   - Final position: ({first_vehicle_transform.location.x:.2f}, {first_vehicle_transform.location.y:.2f})")
            print(f"   - Final yaw: {first_vehicle_transform.rotation.yaw:.2f}°")

        # 记录遗传算法优化的其他参数
        if hasattr(self, 'actor_speed') and hasattr(self, 'trigger_distance_threshold'):
            print(f">> Applied optimized speed: {self.actor_speed:.2f} m/s")
            print(f">> Applied optimized trigger distance: {self.trigger_distance_threshold:.2f} m")

        # 设置参与者变换和类型
        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]

        # 初始化参与者
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(
            self.actor_transform_list, self.actor_type_list
        )
        self.reference_actor = self.other_actors[0]  # used for triggering this scenario

        # 设置其他车辆的交通灯
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(other_actor_transform, False, True)
        if traffic_light_other is None:
            print(">> No traffic light for the given location of the other vehicle found")
        else:
            traffic_light_other.set_state(carla.TrafficLightState.Red)
            traffic_light_other.set_red_time(self.timeout)


    def create_behavior(self, scenario_init_action):
        # 检查是否使用遗传算法
        if self._should_use_genetic_algorithm(scenario_init_action):

            print(">> Using genetic algorithm for initial state optimization")

            # 创建遗传算法优化器
            self.genetic_optimizer = create_optimizer(self)

            # 执行优化
            self.optimized_params = self.genetic_optimizer.optimize()

            # 应用优化后的参数
            self.actor_speed = self.optimized_params['actor_speed']
            self.trigger_distance_threshold = self.optimized_params['trigger_distance']
            self.position_offset = self.optimized_params['position_offset']
            # actions = self.convert_actions(scenario_init_action)
            # self.x, delta_v, delta_dist = actions
            self.x = 0.0
            print(f">> Optimized params:")
            print(f"   - Actor speed: {self.actor_speed:.2f} m/s")
            print(f"   - Trigger distance: {self.trigger_distance_threshold:.2f} m")
            print(f"   - Position offset: x={self.position_offset['x']:.2f}, y={self.position_offset['y']:.2f}")

            # opt_info = self.genetic_optimizer.get_optimization_info()
            # print(f">> Optimization completed - Best fitness: {opt_info['best_fitness']:.4f}")

        else:
            # 传统方式：从scenario_init_action获取参数
            if scenario_init_action is not None:
                try:
                    actions = self.convert_actions(scenario_init_action)
                    self.x, delta_v, delta_dist = actions
                    self.actor_speed = 10 + delta_v
                    self.trigger_distance_threshold = 35 + delta_dist
                except Exception as e:
                    print(f">> Error processing scenario_init_action: {e}")
                    # 使用默认值
                    self.actor_speed = 10.0
                    self.trigger_distance_threshold = 35.0


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

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        cur_ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicle)
        if cur_ego_speed and cur_ego_speed > 0.5:
            self.trigger = True
        if self.trigger:
            for i in range(len(self.other_actors)):
                self.scenario_operation.go_straight(self.actor_speed, i)

    def check_stop_condition(self):
        # stop when actor runs a specific distance
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class SignalizedJunctionLeftTurn(BasicScenario):
    """
        Vehicle turning left at signalized junction scenario. 
        An actor has higher priority, ego needs to yield to oncoming actor.
    """

    # def __init__(self, world, ego_vehicle, config, timeout=60):
    #     super(SignalizedJunctionLeftTurn, self).__init__("SignalizedJunctionLeftTurn-Init-State", config, world)
    #     self.ego_vehicle = ego_vehicle
    #     self._map = CarlaDataProvider.get_map()
    #     self.timeout = timeout
    #
    #     # self._brake_value = 0.5
    #     # self._ego_distance = 110
    #     self._actor_distance = 100
    #     self._traffic_light = None
    #     self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
    #     if self._traffic_light is None:
    #         print(">> No traffic light for the given location found")
    #     else:
    #         self._traffic_light.set_state(carla.TrafficLightState.Green)
    #         self._traffic_light.set_green_time(self.timeout)
    #
    #     # other vehicle's traffic light
    #     self.scenario_operation = ScenarioOperation()
    #     self.reference_actor = None
    #     self.ego_max_driven_distance = 150
    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(SignalizedJunctionLeftTurn, self).__init__("SignalizedJunctionLeftTurn", config, world)
        self.ego_vehicle = ego_vehicle
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        # 遗传算法相关
        self.genetic_optimizer = None
        self.use_genetic_algorithm = False
        self.optimized_params = None

        # 基础参数
        self._target_vel = 12.0
        self._actor_distance = 100
        self.trigger_distance_threshold = 45
        self.position_offset = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.trigger_location = config.trigger_points[0].location

        # 交通灯设置
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        # if self._traffic_light is None:
        #     raise RuntimeError("No traffic light for the given location found")
        # self._traffic_light.set_state(carla.TrafficLightState.Green)
        # self._traffic_light.set_green_time(self.timeout)

        if self._traffic_light is not None:
            self._traffic_light.set_state(carla.TrafficLightState.Green)
            self._traffic_light.set_green_time(self.timeout)
        else:
            print("No traffic light for the given location found")

        # 场景操作
        self.scenario_operation = ScenarioOperation()
        self.ego_max_driven_distance = 150

    def convert_actions(self, actions):
        y_scale = 5
        yaw_scale = 5
        d_scale = 5
        y_mean = yaw_mean = dist_mean = 0

        y = actions[0] * y_scale + y_mean
        yaw = actions[1] * yaw_scale + yaw_mean
        dist = actions[2] * d_scale + dist_mean
        return [y, yaw, dist]

    def initialize_actorsHK_(self):
        other_actor_transform = self.config.other_actors[0].transform
        forward_vector = other_actor_transform.rotation.get_forward_vector() * self.x
        other_actor_transform.location += forward_vector
        first_vehicle_transform = carla.Transform(
            carla.Location(other_actor_transform.location.x, other_actor_transform.location.y, other_actor_transform.location.z),
            other_actor_transform.rotation
        )



        # 香港靠左行驶,初始化车辆掉头行驶，否则逆行 TODO: 行驶方向也可由路径导出的时候统一处理
        first_vehicle_transform.rotation.yaw = (first_vehicle_transform.rotation.yaw) % 360

        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario

        traffic_light_other = CarlaDataProvider.get_next_traffic_light(other_actor_transform, False, True)
        if traffic_light_other is None:
            print(">> No traffic light for the given location found")
        else:
            traffic_light_other.set_state(carla.TrafficLightState.Green)
            traffic_light_other.set_green_time(self.timeout)

    def initialize_actorsHK(self):
        """
        初始化场景参与者 - SignalizedJunctionLeftTurn场景
        集成遗传算法优化的位置偏移
        """
        other_actor_transform = self.config.other_actors[0].transform

        # 应用原有的x偏移逻辑
        forward_vector = other_actor_transform.rotation.get_forward_vector() * self.x
        other_actor_transform.location += forward_vector

        # 创建基础变换
        first_vehicle_transform = carla.Transform(
            carla.Location(
                other_actor_transform.location.x,
                other_actor_transform.location.y,
                other_actor_transform.location.z
            ),
            other_actor_transform.rotation
        )

        # 香港靠左行驶,初始化车辆掉头行驶，否则逆行
        first_vehicle_transform.rotation.yaw = (first_vehicle_transform.rotation.yaw) % 360

        # 应用遗传算法优化的位置偏移
        if hasattr(self, 'position_offset') and self.position_offset:
            # 获取掉头后的方向向量
            current_rotation = first_vehicle_transform.rotation
            right_vector = current_rotation.get_right_vector()
            forward_vector_current = current_rotation.get_forward_vector()

            # 获取遗传算法优化的偏移量
            x_offset = self.position_offset.get('x', 0.0)  # 横向偏移（右为正）
            y_offset = self.position_offset.get('y', 0.0)  # 纵向偏移（前为正）
            yaw_offset = self.position_offset.get('yaw', 0.0)  # 角度偏移

            # 应用位置偏移（在掉头后的坐标系下）
            offset_vector = right_vector * x_offset + forward_vector_current * y_offset
            first_vehicle_transform.location += offset_vector

            # 应用角度偏移
            first_vehicle_transform.rotation.yaw += yaw_offset
            first_vehicle_transform.rotation.yaw = first_vehicle_transform.rotation.yaw % 360

            # 记录应用的遗传算法优化
            print(f">> [SignalizedJunctionLeftTurn] Applied genetic algorithm optimizations:")
            print(f"   - X offset (lateral): {x_offset:.2f}m")
            print(f"   - Y offset (longitudinal): {y_offset:.2f}m")
            print(f"   - Yaw offset: {yaw_offset:.2f}°")
            print(
                f"   - Final position: ({first_vehicle_transform.location.x:.2f}, {first_vehicle_transform.location.y:.2f})")
            print(f"   - Final yaw: {first_vehicle_transform.rotation.yaw:.2f}°")

        # 记录遗传算法优化的其他参数
        if hasattr(self, '_target_vel') and hasattr(self, 'trigger_distance_threshold'):
            print(f">> [SignalizedJunctionLeftTurn] Applied optimized speed: {self._target_vel:.2f} m/s")
            print(
                f">> [SignalizedJunctionLeftTurn] Applied optimized trigger distance: {self.trigger_distance_threshold:.2f} m")

        # 设置参与者变换和类型
        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]

        # 初始化参与者
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(
            self.actor_transform_list, self.actor_type_list
        )
        self.reference_actor = self.other_actors[0]  # used for triggering this scenario

        # 设置交通灯（注意：这里是Green，与OppositeVehicleRunningRedLight不同）
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(other_actor_transform, False, True)
        if traffic_light_other is None:
            print(">> No traffic light for the given location found")
        else:
            traffic_light_other.set_state(carla.TrafficLightState.Green)
            traffic_light_other.set_green_time(self.timeout)

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        for i in range(len(self.other_actors)):
            self.scenario_operation.go_straight(self._target_vel, i)

    def create_behavior_(self, scenario_init_action):
        actions = self.convert_actions(scenario_init_action)
        self.x, delta_v, delta_dist = actions  
        self._target_vel = 12.0 + delta_v
        self.trigger_distance_threshold = 45 + delta_dist

    def create_behavior(self, scenario_init_action):
        """创建场景行为"""
        # 检查是否使用遗传算法
        if self._should_use_genetic_algorithm(scenario_init_action):
            print(">> Using genetic algorithm for left turn scenario optimization")

            # 创建遗传算法优化器
            self.genetic_optimizer = create_optimizer(self)

            # 执行优化
            self.optimized_params = self.genetic_optimizer.optimize()

            # 应用优化后的参数
            self._target_vel = self.optimized_params['actor_speed']
            self.trigger_distance_threshold = self.optimized_params['trigger_distance']
            self.position_offset = self.optimized_params['position_offset']

            self.x = 0.0
            # self.y = 0.0

            # opt_info = self.genetic_optimizer.get_optimization_info()
            # print(f">> Left turn optimization completed - Best fitness: {opt_info['best_fitness']:.4f}")

            print(f">> Optimized params:")
            print(f"   - Actor speed: {self._target_vel:.2f} m/s")
            print(f"   - Trigger distance: {self.trigger_distance_threshold:.2f} m")
            print(f"   - Position offset: x={self.position_offset['x']:.2f}, y={self.position_offset['y']:.2f}")
        else:
            # 传统方式
            if scenario_init_action is not None:
                try:
                    actions = self.convert_actions(scenario_init_action)
                    self.x, delta_v, delta_dist = actions
                    self._target_vel = 12 + delta_v
                    self.trigger_distance_threshold = 45 + delta_dist
                except Exception as e:
                    print(f">> Error processing scenario_init_action: {e}")

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

    def check_stop_condition(self):
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False

# TODO: 右转场景也集成遗传算法
class SignalizedJunctionRightTurn_(BasicScenario):
    """
        Vehicle turning right at signalized junction scenario an actor has higher priority, ego needs to yield to oncoming actor
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(SignalizedJunctionRightTurn_, self).__init__("SignalizedJunctionRightTurn-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        # self._brake_value = 0.5
        # self._ego_distance = 110
        self._actor_distance = 100
        self._traffic_light = None
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light is None:
            print(">> No traffic light for the given location found")
        else:
            self._traffic_light.set_state(carla.TrafficLightState.Red)
            self._traffic_light.set_green_time(self.timeout)

        self.scenario_operation = ScenarioOperation()
        self.trigger = False
        self.ego_max_driven_distance = 150

    def convert_actions(self, actions):
        y_scale = 5
        yaw_scale = 5
        d_scale = 5
        y_mean = yaw_mean = dist_mean = 0

        y = actions[0] * y_scale + y_mean
        yaw = actions[1] * yaw_scale + yaw_mean
        dist = actions[2] * d_scale + dist_mean
        return [y, yaw, dist]

    def initialize_actorsHK(self):
        other_actor_transform = self.config.other_actors[0].transform
        forward_vector = other_actor_transform.rotation.get_forward_vector() * self.x
        other_actor_transform.location += forward_vector
        first_vehicle_transform = carla.Transform(
            carla.Location(other_actor_transform.location.x, other_actor_transform.location.y, other_actor_transform.location.z),
            other_actor_transform.rotation
        )
        # 香港靠左行驶,初始化车辆掉头行驶，否则逆行 TODO: 行驶方向也可由路径导出的时候统一处理
        first_vehicle_transform.rotation.yaw = (first_vehicle_transform.rotation.yaw) % 360

        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario

        traffic_light_other = CarlaDataProvider.get_next_traffic_light(other_actor_transform, False, True)
        if traffic_light_other is None:
            print(">> No traffic light for the given location found")
        else:
            traffic_light_other.set_state(carla.TrafficLightState.Green)
            traffic_light_other.set_green_time(self.timeout)

    def create_behavior(self, scenario_init_action):
        actions = self.convert_actions(scenario_init_action)
        self.x, delta_v, delta_dist = actions  
        self._target_vel = 12 + delta_v
        self.trigger_distance_threshold = 35 + delta_dist

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        cur_ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicle)
        if cur_ego_speed and cur_ego_speed > 0.5:
            self.trigger = True
        if self.trigger:
            for i in range(len(self.other_actors)):
                self.scenario_operation.go_straight(self._target_vel, i)

    def check_stop_condition(self):
        # stop when actor runs a specific distance
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class SignalizedJunctionRightTurn(BasicScenario):
    """
    Vehicle turning right at signalized junction scenario - 使用遗传算法优化初始状态
    An actor has higher priority, ego needs to yield to oncoming actor
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(SignalizedJunctionRightTurn, self).__init__("SignalizedJunctionRightTurn-GA", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        # 遗传算法相关
        self.genetic_optimizer = None
        self.use_genetic_algorithm = False
        self.optimized_params = None

        # 基础参数（遗传算法会优化这些值）
        self._target_vel = 12.0
        self.trigger_distance_threshold = 35.0
        self.position_offset = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.x = 0.0
        self.trigger = False

        # 场景参数
        self._actor_distance = 100
        self.ego_max_driven_distance = 150
        self.trigger_location = config.trigger_points[0].location

        # 交通灯设置
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light is None:
            print(">> No traffic light for the given location found")
        else:
            self._traffic_light.set_state(carla.TrafficLightState.Red)
            self._traffic_light.set_green_time(self.timeout)

        # 场景操作
        self.scenario_operation = ScenarioOperation()

    def create_behavior(self, scenario_init_action):
        """
        创建场景行为
        如果使用遗传算法，则在这里执行优化
        """
        # 检查是否使用遗传算法
        if self._should_use_genetic_algorithm(scenario_init_action):
            print(">> Using genetic algorithm for right turn scenario optimization")



            # 创建遗传算法优化器
            self.genetic_optimizer = create_optimizer(self)

            # 执行优化
            self.optimized_params = self.genetic_optimizer.optimize()

            # 应用优化后的参数
            self._target_vel = self.optimized_params['actor_speed']
            self.trigger_distance_threshold = self.optimized_params['trigger_distance']
            self.position_offset = self.optimized_params['position_offset']

            # 设置self.x
            self.x = 0.0

            print(f">> Optimized params:")
            print(f"   - Actor speed: {self._target_vel:.2f} m/s")
            print(f"   - Trigger distance: {self.trigger_distance_threshold:.2f} m")
            print(f"   - Position offset: x={self.position_offset['x']:.2f}, y={self.position_offset['y']:.2f}")

        else:
            # 传统方式：从scenario_init_action获取参数
            if scenario_init_action is not None:
                try:
                    actions = self.convert_actions(scenario_init_action)
                    self.x, delta_v, delta_dist = actions
                    self._target_vel = 12 + delta_v
                    self.trigger_distance_threshold = 35 + delta_dist
                except Exception as e:
                    print(f">> Error processing scenario_init_action: {e}")
                    # 使用默认值
                    self.x = 0.0
                    self._target_vel = 12.0
                    self.trigger_distance_threshold = 35.0
            else:
                # 完全默认情况
                self.x = 0.0
                self._target_vel = 12.0
                self.trigger_distance_threshold = 35.0


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

    def initialize_actorsHK(self):
        """
        初始化场景参与者 - SignalizedJunctionRightTurn场景
        集成遗传算法优化的位置偏移
        """
        other_actor_transform = self.config.other_actors[0].transform

        # 应用原有的x偏移逻辑
        forward_vector = other_actor_transform.rotation.get_forward_vector() * self.x
        other_actor_transform.location += forward_vector

        # 创建基础变换
        first_vehicle_transform = carla.Transform(
            carla.Location(
                other_actor_transform.location.x,
                other_actor_transform.location.y,
                other_actor_transform.location.z
            ),
            other_actor_transform.rotation
        )

        # 香港靠左行驶,初始化车辆掉头行驶，否则逆行
        first_vehicle_transform.rotation.yaw = (first_vehicle_transform.rotation.yaw) % 360

        # 应用遗传算法优化的位置偏移
        if hasattr(self, 'position_offset') and self.position_offset:
            # 获取掉头后的方向向量
            current_rotation = first_vehicle_transform.rotation
            right_vector = current_rotation.get_right_vector()
            forward_vector_current = current_rotation.get_forward_vector()

            # 获取遗传算法优化的偏移量
            x_offset = self.position_offset.get('x', 0.0)  # 横向偏移（右为正）
            y_offset = self.position_offset.get('y', 0.0)  # 纵向偏移（前为正）
            yaw_offset = self.position_offset.get('yaw', 0.0)  # 角度偏移

            # 应用位置偏移（在掉头后的坐标系下）
            offset_vector = right_vector * x_offset + forward_vector_current * y_offset
            first_vehicle_transform.location += offset_vector

            # 应用角度偏移
            first_vehicle_transform.rotation.yaw += yaw_offset
            first_vehicle_transform.rotation.yaw = first_vehicle_transform.rotation.yaw % 360

            # 记录应用的遗传算法优化
            print(f">> [SignalizedJunctionRightTurn] Applied genetic algorithm optimizations:")
            print(f"   - X offset (lateral): {x_offset:.2f}m")
            print(f"   - Y offset (longitudinal): {y_offset:.2f}m")
            print(f"   - Yaw offset: {yaw_offset:.2f}°")
            print(
                f"   - Final position: ({first_vehicle_transform.location.x:.2f}, {first_vehicle_transform.location.y:.2f})")
            print(f"   - Final yaw: {first_vehicle_transform.rotation.yaw:.2f}°")

        # 记录遗传算法优化的其他参数
        if hasattr(self, '_target_vel') and hasattr(self, 'trigger_distance_threshold'):
            print(f">> [SignalizedJunctionRightTurn] Applied optimized speed: {self._target_vel:.2f} m/s")
            print(
                f">> [SignalizedJunctionRightTurn] Applied optimized trigger distance: {self.trigger_distance_threshold:.2f} m")

        # 设置参与者变换和类型
        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]

        # 初始化参与者
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(
            self.actor_transform_list, self.actor_type_list
        )
        self.reference_actor = self.other_actors[0]  # used for triggering this scenario

        # 设置交通灯（右转场景中其他车辆为Green）
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(other_actor_transform, False, True)
        if traffic_light_other is None:
            print(">> No traffic light for the given location found")
        else:
            traffic_light_other.set_state(carla.TrafficLightState.Green)
            traffic_light_other.set_green_time(self.timeout)

    def convert_actions(self, actions):
        """转换动作（保持原有逻辑）"""
        y_scale = 5
        yaw_scale = 5
        d_scale = 5
        y_mean = yaw_mean = dist_mean = 0

        y = actions[0] * y_scale + y_mean
        yaw = actions[1] * yaw_scale + yaw_mean
        dist = actions[2] * d_scale + dist_mean
        return [y, yaw, dist]

    def update_behavior(self, scenario_action):
        """更新场景行为（保持原有逻辑）"""
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        cur_ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicle)
        if cur_ego_speed and cur_ego_speed > 0.5:
            self.trigger = True
        if self.trigger:
            for i in range(len(self.other_actors)):
                self.scenario_operation.go_straight(self._target_vel, i)

    def check_stop_condition(self):
        """检查停止条件（保持原有逻辑）"""
        # stop when actor runs a specific distance
        cur_distance = calculate_distance_transforms(
            CarlaDataProvider.get_transform(self.other_actors[0]),
            self.actor_transform_list[0]
        )
        if cur_distance >= self._actor_distance:
            return True
        return False


class NoSignalJunctionCrossingRoute_(BasicScenario):
    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(NoSignalJunctionCrossingRoute_, self).__init__("NoSignalJunctionCrossingRoute-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self.scenario_operation = ScenarioOperation()
        self.reference_actor = None
        
        self.trigger = False
        self._actor_distance = 110
        self.ego_max_driven_distance = 150

    def convert_actions(self, actions):
        y_scale = 5
        yaw_scale = 5
        d_scale = 5
        y_mean = yaw_mean = dist_mean = 0

        y = actions[0] * y_scale + y_mean
        yaw = actions[1] * yaw_scale + yaw_mean
        dist = actions[2] * d_scale + dist_mean
        return [y, yaw, dist]

    def initialize_actorsHK(self):
        other_actor_transform = self.config.other_actors[0].transform
        forward_vector = other_actor_transform.rotation.get_forward_vector() * self.x
        other_actor_transform.location += forward_vector
        first_vehicle_transform = carla.Transform(
            carla.Location(other_actor_transform.location.x, other_actor_transform.location.y, other_actor_transform.location.z),
            other_actor_transform.rotation
        )

        # 香港靠左行驶,初始化车辆掉头行驶，否则逆行 TODO: 行驶方向也可由路径导出的时候统一处理
        first_vehicle_transform.rotation.yaw = (first_vehicle_transform.rotation.yaw) % 360

        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario
        
    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        cur_ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicle)
        if cur_ego_speed and cur_ego_speed > 0.5:
            self.trigger = True
        if self.trigger:
            for i in range(len(self.other_actors)):
                self.scenario_operation.go_straight(self.actor_speed, i)

    def create_behavior(self, scenario_init_action):
        actions = self.convert_actions(scenario_init_action)
        self.x, self.y, delta_v, delta_dist = actions  
        self.actor_speed = 10 + delta_v
        self.trigger_distance_threshold = 35 + delta_dist

    def check_stop_condition(self):
        # stop when actor runs a specific distance
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class NoSignalJunctionCrossingRoute(BasicScenario):
    """
    无信号路口交叉场景 - 使用遗传算法优化初始状态
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(NoSignalJunctionCrossingRoute, self).__init__("NoSignalJunctionCrossingRoute-GA", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        # 遗传算法相关
        self.genetic_optimizer = None
        self.use_genetic_algorithm = False
        self.optimized_params = None

        # 基础参数（遗传算法会优化这些值）
        self.actor_speed = 10.0
        self.trigger_distance_threshold = 35.0
        self.position_offset = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.x = 0.0
        self.y = 0.0  # 注意：这个场景多了一个y参数
        self.trigger = False

        # 场景参数
        self._actor_distance = 110
        self.ego_max_driven_distance = 150

        # 场景操作
        self.scenario_operation = ScenarioOperation()
        self.reference_actor = None

    def create_behavior(self, scenario_init_action):
        """
        创建场景行为
        如果使用遗传算法，则在这里执行优化
        """
        # 检查是否使用遗传算法
        if self._should_use_genetic_algorithm(scenario_init_action):
            print(">> Using genetic algorithm for no-signal junction crossing optimization")

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

            # 设置位置参数
            self.x = 0.0  # 前向偏移
            self.y = 0.0  # 横向偏移（这个场景特有）

            opt_info = self.genetic_optimizer.get_optimization_info()
            print(f">> No-signal junction optimization completed - Best fitness: {opt_info['best_fitness']:.4f}")

        else:
            # 传统方式：从scenario_init_action获取参数
            if scenario_init_action is not None:
                try:
                    actions = self.convert_actions(scenario_init_action)
                    self.x, self.y, delta_v, delta_dist = actions  # 注意：这里有4个参数
                    self.actor_speed = 10 + delta_v
                    self.trigger_distance_threshold = 35 + delta_dist
                except Exception as e:
                    print(f">> Error processing scenario_init_action: {e}")
                    # 使用默认值
                    self.x = 0.0
                    self.y = 0.0
                    self.actor_speed = 10.0
                    self.trigger_distance_threshold = 35.0
            else:
                # 完全默认情况
                self.x = 0.0
                self.y = 0.0
                self.actor_speed = 10.0
                self.trigger_distance_threshold = 35.0

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

    def initialize_actorsHK(self):
        """
        初始化场景参与者 - NoSignalJunctionCrossingRoute场景
        集成遗传算法优化的位置偏移
        """
        other_actor_transform = self.config.other_actors[0].transform

        # 应用原有的x偏移逻辑（前向偏移）
        forward_vector = other_actor_transform.rotation.get_forward_vector() * self.x
        other_actor_transform.location += forward_vector

        # 创建基础变换
        first_vehicle_transform = carla.Transform(
            carla.Location(
                other_actor_transform.location.x,
                other_actor_transform.location.y,
                other_actor_transform.location.z
            ),
            other_actor_transform.rotation
        )

        # 香港靠左行驶,初始化车辆掉头行驶，否则逆行
        first_vehicle_transform.rotation.yaw = (first_vehicle_transform.rotation.yaw) % 360

        # 应用原有的y偏移逻辑（如果有的话）
        if hasattr(self, 'y') and self.y != 0.0:
            # y偏移通常是横向偏移
            right_vector = first_vehicle_transform.rotation.get_right_vector()
            y_offset_vector = right_vector * self.y
            first_vehicle_transform.location += y_offset_vector
            print(f">> Applied original Y offset: {self.y:.2f}m")

        # 应用遗传算法优化的位置偏移
        if hasattr(self, 'position_offset') and self.position_offset:
            # 获取掉头后的方向向量
            current_rotation = first_vehicle_transform.rotation
            right_vector = current_rotation.get_right_vector()
            forward_vector_current = current_rotation.get_forward_vector()

            # 获取遗传算法优化的偏移量
            x_offset = self.position_offset.get('x', 0.0)  # 横向偏移（右为正）
            y_offset = self.position_offset.get('y', 0.0)  # 纵向偏移（前为正）
            yaw_offset = self.position_offset.get('yaw', 0.0)  # 角度偏移

            # 应用位置偏移（在掉头后的坐标系下）
            offset_vector = right_vector * x_offset + forward_vector_current * y_offset
            first_vehicle_transform.location += offset_vector

            # 应用角度偏移
            first_vehicle_transform.rotation.yaw += yaw_offset
            first_vehicle_transform.rotation.yaw = first_vehicle_transform.rotation.yaw % 360

            # 记录应用的遗传算法优化
            print(f">> [NoSignalJunctionCrossing] Applied genetic algorithm optimizations:")
            print(f"   - X offset (lateral): {x_offset:.2f}m")
            print(f"   - Y offset (longitudinal): {y_offset:.2f}m")
            print(f"   - Yaw offset: {yaw_offset:.2f}°")
            print(
                f"   - Final position: ({first_vehicle_transform.location.x:.2f}, {first_vehicle_transform.location.y:.2f})")
            print(f"   - Final yaw: {first_vehicle_transform.rotation.yaw:.2f}°")

        # 记录遗传算法优化的其他参数
        if hasattr(self, 'actor_speed') and hasattr(self, 'trigger_distance_threshold'):
            print(f">> [NoSignalJunctionCrossing] Applied optimized speed: {self.actor_speed:.2f} m/s")
            print(
                f">> [NoSignalJunctionCrossing] Applied optimized trigger distance: {self.trigger_distance_threshold:.2f} m")

        # 设置参与者变换和类型
        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]

        # 初始化参与者
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(
            self.actor_transform_list, self.actor_type_list
        )
        self.reference_actor = self.other_actors[0]  # used for triggering this scenario

    def convert_actions(self, actions):
        """转换动作（保持原有逻辑）"""
        y_scale = 5
        yaw_scale = 5
        d_scale = 5
        y_mean = yaw_mean = dist_mean = 0

        y = actions[0] * y_scale + y_mean
        yaw = actions[1] * yaw_scale + yaw_mean
        dist = actions[2] * d_scale + dist_mean
        return [y, yaw, dist]

    def update_behavior(self, scenario_action):
        """更新场景行为（保持原有逻辑）"""
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        cur_ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicle)
        print(cur_ego_speed)
        if cur_ego_speed and cur_ego_speed > 0.5:
            self.trigger = True
        if self.trigger:
            for i in range(len(self.other_actors)):
                self.scenario_operation.go_straight(self.actor_speed, i)

    def check_stop_condition(self):
        """检查停止条件（保持原有逻辑）"""
        # stop when actor runs a specific distance
        cur_distance = calculate_distance_transforms(
            CarlaDataProvider.get_transform(self.other_actors[0]),
            self.actor_transform_list[0]
        )
        if cur_distance >= self._actor_distance:
            return True
        return False
