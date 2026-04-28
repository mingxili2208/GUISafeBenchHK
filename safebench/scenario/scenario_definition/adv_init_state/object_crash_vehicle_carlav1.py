''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-30 12:19:31
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import math

import carla
from typing import Dict,Tuple,Any
import time

from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.tools.scenario_utils import calculate_distance_transforms
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_helper import get_location_in_distance_from_wp


class DynamicObjectCrossing(BasicScenario):
    """
        Without prior vehicle action involving a vehicle and a cyclist/pedestrian, the ego vehicle is passing through a road,
        and encounters a cyclist/pedestrian crossing the road.
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(DynamicObjectCrossing, self).__init__("DynamicObjectCrossing-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout
        
        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)

        # other vehicle parameters
        self._other_actor_target_velocity = 2.5
        self._num_lane_changes = 1

        # 遗传算法相关
        self.genetic_optimizer = None
        self.use_genetic_algorithm = False
        self.optimized_params = None

        # 基础参数（遗传算法会优化这些值）
        self.actor_speed = 3.0  # 行人/骑行者速度
        self.position_offset = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.x = 0.0
        self.y = 0.0
        self.trigger = False

        # Note: transforms for walker and blocker
        self.transform = None
        self.transform2 = None
        self._trigger_location = config.trigger_points[0].location
        self._number_of_attempts = 20  # Total Number of attempts to relocate a vehicle before spawning
        self._spawn_attempted = 0  # Number of attempts made so far

        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = 20
        self.ego_max_driven_distance = 150

    def convert_actions(self, actions):
        yaw_scale = 60
        yaw_mean = 0

        d_min = 15
        d_max = 50
        d_scale = (d_max - d_min) / 2
        dist_mean = (d_max + d_min)/2

        y = actions[0] / 2 + 0.5
        yaw = actions[1] * yaw_scale + yaw_mean
        dist = actions[2] * d_scale + dist_mean
        return [y, yaw, dist]

    def _calculate_base_transform(self, _start_distance, waypoint):
        lane_width = waypoint.lane_width

        # Patches false junctions
        if self._reference_waypoint.is_junction:
            stop_at_junction = False
        else:
            stop_at_junction = True

        location, _ = get_location_in_distance_from_wp(waypoint, _start_distance, stop_at_junction)
        waypoint = self._map.get_waypoint(location)
        offset = {"orientation": 270, "position": 90, "z": 0.6, "k": 1.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw))
        )
        location += offset_location
        location.z = self._trigger_location.z + offset['z']
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _spawn_blocker(self, transform, orientation_yaw):
        """
            Spawn the blocker prop that blocks the vision from the egovehicle of the jaywalker
        """
        # static object transform
        shift = 0.9
        x_ego = self._reference_waypoint.transform.location.x
        y_ego = self._reference_waypoint.transform.location.y
        x_cycle = transform.location.x
        y_cycle = transform.location.y
        x_static = x_ego + shift * (x_cycle - x_ego)
        y_static = y_ego + shift * (y_cycle - y_ego)
        spawn_point_wp = self.ego_vehicle.get_world().get_map().get_waypoint(transform.location)
        self.transform2 = carla.Transform(
            carla.Location(x_static, y_static, spawn_point_wp.transform.location.z + 0.3),
            carla.Rotation(yaw=orientation_yaw + 180)
        )

    def initialize_actorsHK(self):
        """
        初始化场景参与者 - DynamicObjectCrossing场景
        集成遗传算法优化的位置偏移
        """
        # 确保actions已经设置
        if self.actions is None:
            self.actions = [0.5, 0.0, 20.0]  # 默认值

        y, yaw, self.trigger_distance_threshold = self.actions  # [0, 1], [-60, 60], [15, 50]

        # cyclist transform
        _start_distance = 45
        # we start by getting and waypoint in the closest sidewalk.
        waypoint = self._reference_waypoint

        while True:
            wp_next = waypoint.get_right_lane()
            self._num_lane_changes += 1
            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder:
                # Filter Parkings considered as Shoulders
                if wp_next.lane_width > 2:
                    _start_distance += 1.5
                    waypoint = wp_next
                break
            else:
                _start_distance += 1.5
                waypoint = wp_next

        # we keep trying to spawn avoiding props
        while True:
            try:
                self.transform, orientation_yaw = self._calculate_base_transform(_start_distance, waypoint)
                self._spawn_blocker(self.transform, orientation_yaw)
                forward_vector = self.transform.rotation.get_forward_vector() * y * self._reference_waypoint.lane_width
                self.transform.location += forward_vector
                yaw = self.transform.rotation.yaw + yaw
                if yaw < 0:
                    yaw += 360
                if yaw > 360:
                    yaw -= 360
                self.transform = carla.Transform(
                    self.transform.location,
                    carla.Rotation(self.transform.rotation.pitch, yaw, self.transform.rotation.roll)
                )
                break
            except RuntimeError as r:
                print("Base transform is blocking objects ", self.transform)
                _start_distance += 0.4
                self._spawn_attempted += 1
                if self._spawn_attempted >= self._number_of_attempts:
                    raise r

        # 应用遗传算法优化的额外位置偏移
        if hasattr(self, 'position_offset') and self.position_offset:
            # 获取当前的方向向量
            current_rotation = self.transform.rotation
            right_vector = current_rotation.get_right_vector()
            forward_vector_current = current_rotation.get_forward_vector()

            # 获取遗传算法优化的偏移量
            x_offset = self.position_offset.get('x', 0.0)  # 横向偏移（右为正）
            y_offset = self.position_offset.get('y', 0.0)  # 纵向偏移（前为正）
            # yaw偏移已经在上面应用了

            # 应用额外的位置偏移
            offset_vector = right_vector * x_offset + forward_vector_current * y_offset
            self.transform.location += offset_vector

            # 记录应用的遗传算法优化
            print(f">> [DynamicObjectCrossing] Applied genetic algorithm optimizations:")
            print(f"   - Y position factor: {y:.3f}")
            print(f"   - Yaw angle: {yaw:.2f}°")
            print(f"   - Additional X offset: {x_offset:.2f}m")
            print(f"   - Additional Y offset: {y_offset:.2f}m")
            print(f"   - Final position: ({self.transform.location.x:.2f}, {self.transform.location.y:.2f})")
            print(f"   - Trigger distance: {self.trigger_distance_threshold:.2f}m")

        # 记录遗传算法优化的速度参数
        if hasattr(self, 'actor_speed'):
            self._other_actor_target_velocity = self.actor_speed
            print(f">> [DynamicObjectCrossing] Applied optimized walker speed: {self.actor_speed:.2f} m/s")

        # Now that we found a possible position we just put the vehicle to the underground
        disp_transform = carla.Transform(
            carla.Location(self.transform.location.x, self.transform.location.y, self.transform.location.z),
            self.transform.rotation
        )
        prop_disp_transform = carla.Transform(
            carla.Location(self.transform2.location.x, self.transform2.location.y, self.transform2.location.z),
            self.transform2.rotation
        )

        self.actor_type_list = ['walker.*', 'static.prop.vendingmachine']
        self.actor_transform_list = [disp_transform, prop_disp_transform]

        try:
            self.other_actors = self.scenario_operation.initialize_vehicle_actors(
                self.actor_transform_list, self.actor_type_list
            )
            self.reference_actor = self.other_actors[0]  # used for triggering this scenario
            print(f">> [DynamicObjectCrossing] Successfully initialized walker and blocker")
        except Exception as e:
            print(f">> [DynamicObjectCrossing] Failed to initialize actors: {e}")
            self.reference_actor = None


    def initialize_actorsHK_(self):
        """
            Set a blocker that blocks ego's view on the walker
            Request a walker walk through the street when ego come
        """
        y, yaw, self.trigger_distance_threshold = self.actions  # [0, 1], [-60, 60], [15, 50]

        # cyclist transform
        _start_distance = 45
        # we start by getting and waypoint in the closest sidewalk.
        waypoint = self._reference_waypoint

        while True:
            wp_next = waypoint.get_right_lane()
            self._num_lane_changes += 1
            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder:
                # Filter Parkings considered as Shoulders
                if wp_next.lane_width > 2:
                    _start_distance += 1.5
                    waypoint = wp_next
                break
            else:
                _start_distance += 1.5
                waypoint = wp_next

        # we keep trying to spawn avoiding props
        while True:  
            try:
                self.transform, orientation_yaw = self._calculate_base_transform(_start_distance, waypoint)
                self._spawn_blocker(self.transform, orientation_yaw)
                forward_vector = self.transform.rotation.get_forward_vector() * y * self._reference_waypoint.lane_width
                self.transform.location += forward_vector
                yaw = self.transform.rotation.yaw + yaw
                if yaw < 0:
                    yaw += 360
                if yaw > 360:
                    yaw -= 360
                self.transform = carla.Transform(
                    self.transform.location,
                    carla.Rotation(self.transform.rotation.pitch, yaw, self.transform.rotation.roll)
                )
                break
            except RuntimeError as r:
                print("Base transform is blocking objects ", self.transform)
                _start_distance += 0.4
                self._spawn_attempted += 1
                if self._spawn_attempted >= self._number_of_attempts:
                    raise r

        # Now that we found a possible position we just put the vehicle to the underground
        disp_transform = carla.Transform(
            carla.Location(self.transform.location.x, self.transform.location.y, self.transform.location.z),
            self.transform.rotation
        )
        prop_disp_transform = carla.Transform(
            carla.Location(self.transform2.location.x, self.transform2.location.y, self.transform2.location.z),
            self.transform2.rotation
        )

        self.actor_type_list = ['walker.*', 'static.prop.vendingmachine']
        self.actor_transform_list = [disp_transform, prop_disp_transform]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario

    def create_behavior_(self, scenario_init_action):

        self.actions = self.convert_actions(scenario_init_action)

    def create_behavior(self, scenario_init_action):
        """
        创建场景行为
        如果使用遗传算法，则在这里执行优化
        """
        # 检查是否使用遗传算法
        if self._should_use_genetic_algorithm(scenario_init_action):
            print(">> Using genetic algorithm for dynamic object crossing optimization")

            carla_env = self._get_carla_env()
            if carla_env is None:
                print(">> [ERROR] Cannot get CARLA environment, using default parameters")

                return

            # 创建并运行优化器
            from safebench.scenario.tools.skopt_genetic_optimizer import SkoptGeneticOptimizer
            optimizer = SkoptGeneticOptimizer(self, carla_env)  # ⭐ 传入CARLA环境

            # 执行优化
            self.optimized_initial_params = optimizer.optimize_initial_state()

            # 获取优化信息
            opt_info = optimizer.get_optimization_info()
            print(f">> Optimization complete:")
            print(f"   Best fitness: {opt_info.get('best_fitness', 0):.3f}")
            print(f"   Collisions achieved: {opt_info.get('collision_count', 0)}")
            print(f"   Success rate: {opt_info.get('success_rate', 0):.2%}")

            # 应用优化参数
            self.actions=self._apply_optimized_params()


        else:
            print('>> Not using genetic algorithm for dynamic object crossing optimization')
            # 传统方式：从scenario_init_action获取参数
            if scenario_init_action is not None:
                try:
                    self.actions = self.convert_actions(scenario_init_action)
                except Exception as e:
                    print(f">> Error processing scenario_init_action: {e}")
                    # 使用默认值
                    self.actions = [0.5, 0.0, 20.0]
            else:
                # 完全默认情况
                self.actions = [0.5, 0.0, 20.0]

    def _apply_optimized_params(self):
        """应用优化后的参数"""
        if self.optimized_initial_params:
            self.actor_speed = self.optimized_initial_params['actor_speed']
            self.trigger_distance_threshold = self.optimized_initial_params['trigger_distance']
            self.position_offset = self.optimized_initial_params['position_offset']

            # 如果场景有其他相关参数，也一并更新
            if hasattr(self, '_other_actor_target_velocity'):
                self._other_actor_target_velocity = self.actor_speed

            print(f">> Applied optimized parameters:")
            print(f"   Speed: {self.actor_speed:.2f} m/s")
            print(f"   Trigger distance: {self.trigger_distance_threshold:.2f} m")
            print(f"   Position offset: ({self.position_offset['x']:.1f}, "
                  f"{self.position_offset['y']:.1f}, {self.position_offset['yaw']:.1f})")
        return [self.position_offset['y'],self.position_offset['yaw'],self.trigger_distance_threshold]


    # 在场景类中
    def _get_carla_env(self):
        """
        获取CARLA环境实例 - 通过全局引用
        """
        try:
            import safebench.scenario.scenario_definition.adv_init_state as adv_init_state_module
            if hasattr(adv_init_state_module, 'GLOBAL_CARLA_ENV'):
                return adv_init_state_module.GLOBAL_CARLA_ENV
            else:
                print(">> [WARNING] No global CARLA environment reference available")
                return None
        except Exception as e:
            print(f">> [ERROR] Failed to get CARLA environment: {e}")
            return None

    def lightweight_reset(self):
        """
        轻量级场景重置 - 供遗传算法使用
        """
        try:
            # 重新初始化场景的关键组件，但不改变整体结构
            if hasattr(self, 'other_actors') and self.other_actors:
                # 重新设置其他参与者的参数
                for actor in self.other_actors:
                    if hasattr(actor, 'set_target_velocity'):
                        actor.set_target_velocity(carla.Vector3D(x=self.actor_speed, y=0, z=0))

            # 重置触发器等
            self.trigger = False

        except Exception as e:
            print(f">> [WARNING] Lightweight reset failed: {e}")

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
        
        # the walker starts crossing the road
        self.scenario_operation.walker_go_straight(self._other_actor_target_velocity, 0)

    def check_stop_condition(self):
        lane_width = self._reference_waypoint.lane_width
        lane_width = lane_width + (1.25 * lane_width * self._num_lane_changes)
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.transform)
        if cur_distance > 0.6 * lane_width:
            return True
        return False
