''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-30 12:19:25
Description: 
    Copyright (c) 2022-2023 Safebench Team
    Fixed version for genetic algorithm optimization

    这个版本是返回偏移量，然后进行计算
'''

import math
from typing import Dict
import carla

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario

from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.tools.scenario_helper import get_crossing_point, get_junction_topology
from safebench.scenario.tools.skopt_genetic_optimizer_VehicleTurningRoute import SkoptGeneticOptimizer

def get_opponent_transform(added_dist, waypoint, trigger_location):
    """
        Calculate the transform of the adversary
    """
    lane_width = waypoint.lane_width

    offset = {"orientation": 270, "position": 90, "k": 1.0}
    _wp = waypoint.next(added_dist)
    if _wp:
        _wp = _wp[-1]
    else:
        raise RuntimeError("Cannot get next waypoint !")

    location = _wp.transform.location
    orientation_yaw = _wp.transform.rotation.yaw + offset["orientation"]
    position_yaw = _wp.transform.rotation.yaw + offset["position"]

    offset_location = carla.Location(
        offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
        offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
    location += offset_location
    location.x = trigger_location.x + 20
    location.z = trigger_location.z
    transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))

    return transform


def get_right_driving_lane(waypoint):
    """
        Gets the driving / parking lane that is most to the right of the waypoint as well as the number of lane changes done
    """
    lane_changes = 0
    while True:
        wp_next = waypoint.get_right_lane()
        lane_changes += 1

        if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
            break
        elif wp_next.lane_type == carla.LaneType.Shoulder:
            # Filter Parkings considered as Shoulders
            if is_lane_a_parking(wp_next):
                lane_changes += 1
                waypoint = wp_next
            break
        else:
            waypoint = wp_next
    return waypoint, lane_changes


def is_lane_a_parking(waypoint):
    """
        This function filters false negative Shoulder which are in reality Parking lanes.
        These are differentiated from the others because, similar to the driving lanes,
        they have, on the right, a small Shoulder followed by a Sidewalk.
    """

    # Parking are wide lanes
    if waypoint.lane_width > 2:
        wp_next = waypoint.get_right_lane()

        # That are next to a mini-Shoulder
        if wp_next is not None and wp_next.lane_type == carla.LaneType.Shoulder:
            wp_next_next = wp_next.get_right_lane()

            # Followed by a Sidewalk
            if wp_next_next is not None and wp_next_next.lane_type == carla.LaneType.Sidewalk:
                return True
    return False


class VehicleTurningRoute(BasicScenario):
    """
        The ego vehicle is passing through a road and encounters a cyclist after taking a turn.
    """

    def __init__(self, world, ego_vehicle, config, timeout=240):
        super(VehicleTurningRoute, self).__init__("VehicleTurningRoute-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self.running_distance = 10
        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = 20
        self.ego_max_driven_distance = 180

        self.trigger_location = config.trigger_points[0].location if hasattr(config,
                                                                             'trigger_points') and config.trigger_points else None

        # 遗传算法相关
        self.genetic_optimizer = None
        self.use_genetic_algorithm = False
        self.optimized_params = None

        # 基础参数（遗传算法会优化这些值）
        self.actor_speed = 4.0  # 骑行者速度较低
        self.trigger_distance_threshold = 25.0
        self.position_offset = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.x = 0.0
        self.y = 0.0
        self.trigger = False

        # Added:直接转向,不再等待红灯
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light is None:
            print(">> No traffic light for the given location of the ego vehicle found")
        else:
            self._traffic_light.set_state(carla.TrafficLightState.Green)
            self._traffic_light.set_green_time(self.timeout)

    def convert_actions(self, actions, x_scale, y_scale, x_mean, y_mean):
        yaw_min = 0
        yaw_max = 360
        yaw_scale = (yaw_max - yaw_min) / 2
        yaw_mean = (yaw_max + yaw_min) / 2

        d_min = 10
        d_max = 50
        d_scale = (d_max - d_min) / 2
        dist_mean = (d_max + d_min) / 2

        x = actions[0] * x_scale + x_mean
        y = actions[1] * y_scale + y_mean
        yaw = actions[2] * yaw_scale + yaw_mean
        dist = actions[3] * d_scale + dist_mean
        return [x, y, yaw, dist]

    def initialize_actorsHK(self):
        """
        初始化场景参与者 - VehicleTurningRoute场景
        集成遗传算法优化的位置偏移
        """
        # 获取路口信息和基础计算
        cross_location = get_crossing_point(self.ego_vehicle)
        cross_waypoint = CarlaDataProvider.get_map().get_waypoint(cross_location)
        entry_wps, exit_wps = get_junction_topology(cross_waypoint.get_junction())

        if len(entry_wps) != len(exit_wps):
            print(
                f"[WARNING] Waypoint mismatch: {len(entry_wps)} entries, {len(exit_wps)} exits. Using {min(len(entry_wps), len(exit_wps))} pairs.")
            min_len = min(len(entry_wps), len(exit_wps))
            entry_wps = entry_wps[:min_len]
            exit_wps = exit_wps[:min_len]

        # 计算路口中心和缩放参数
        x_mean = y_mean = 0
        max_x_scale = max_y_scale = 0
        for i in range(len(entry_wps)):
            x_mean += entry_wps[i].transform.location.x + exit_wps[i].transform.location.x
            y_mean += entry_wps[i].transform.location.y + exit_wps[i].transform.location.y
        x_mean /= len(entry_wps) * 2
        y_mean /= len(entry_wps) * 2

        for i in range(len(entry_wps)):
            max_x_scale = max(max_x_scale,
                              abs(entry_wps[i].transform.location.x - x_mean),
                              abs(exit_wps[i].transform.location.x - x_mean))
            max_y_scale = max(max_y_scale,
                              abs(entry_wps[i].transform.location.y - y_mean),
                              abs(exit_wps[i].transform.location.y - y_mean))
        max_x_scale *= 0.8
        max_y_scale *= 0.8

        # 获取基础位置和参数
        if hasattr(self, 'actions') and self.actions is not None:
            # 传统方式：使用convert_actions处理
            x, y, yaw, trigger_dist = self.convert_actions(self.actions, max_x_scale, max_y_scale, x_mean, y_mean)
            if hasattr(self, 'trigger_distance_threshold'):
                self.trigger_distance_threshold = trigger_dist
        else:
            # 使用配置中的actor位置作为基础位置
            if hasattr(self.config, 'other_actors') and len(self.config.other_actors) > 0:
                base_transform = self.config.other_actors[0].transform
                x = base_transform.location.x
                y = base_transform.location.y
                yaw = base_transform.rotation.yaw
            else:
                # 如果没有配置，使用路口中心
                x, y, yaw = x_mean, y_mean, 0.0

            if not hasattr(self, 'trigger_distance_threshold'):
                self.trigger_distance_threshold = 25.0

        # 创建基础变换
        other_actor_transform = carla.Transform(
            carla.Location(x, y, 0.5),  # 稍微抬高避免穿模
            carla.Rotation(yaw=yaw)
        )

        # 应用遗传算法优化的位置偏移
        if hasattr(self, 'position_offset') and self.position_offset:
            # 获取当前的方向向量
            current_rotation = other_actor_transform.rotation
            right_vector = current_rotation.get_right_vector()
            forward_vector_current = current_rotation.get_forward_vector()

            # 获取遗传算法优化的偏移量
            x_offset = self.position_offset.get('x', 0.0)  # 横向偏移（右为正）
            y_offset = self.position_offset.get('y', 0.0)  # 纵向偏移（前为正）
            yaw_offset = self.position_offset.get('yaw', 0.0)  # 角度偏移

            # 应用位置偏移
            offset_vector = right_vector * x_offset + forward_vector_current * y_offset
            other_actor_transform.location += offset_vector

            # 应用角度偏移
            other_actor_transform.rotation.yaw += yaw_offset
            other_actor_transform.rotation.yaw = other_actor_transform.rotation.yaw % 360

            # 记录应用的遗传算法优化
            print(f">> [VehicleTurningRoute] Applied genetic algorithm optimizations:")
            print(f"   - Base position: ({x:.2f}, {y:.2f})")
            print(f"   - X offset (lateral): {x_offset:.2f}m")
            print(f"   - Y offset (longitudinal): {y_offset:.2f}m")
            print(f"   - Yaw offset: {yaw_offset:.2f}°")
            print(
                f"   - Final position: ({other_actor_transform.location.x:.2f}, {other_actor_transform.location.y:.2f})")
            print(f"   - Final yaw: {other_actor_transform.rotation.yaw:.2f}°")

        # 记录遗传算法优化的其他参数
        if hasattr(self, 'actor_speed') and hasattr(self, 'trigger_distance_threshold'):
            print(f">> [VehicleTurningRoute] Applied optimized cyclist speed: {self.actor_speed:.2f} m/s")
            print(
                f">> [VehicleTurningRoute] Applied optimized trigger distance: {self.trigger_distance_threshold:.2f} m")

        # 设置参与者变换和类型（骑行者）
        self.actor_transform_list = [other_actor_transform]
        self.actor_type_list = ['vehicle.diamondback.century']  # 自行车类型

        # 增加逻辑判断：如果尝试失败则没有reference actor,场景实际上不被触发
        try:
            self.other_actors = self.scenario_operation.initialize_vehicle_actors(
                self.actor_transform_list, self.actor_type_list
            )
            self.reference_actor = self.other_actors[0]  # used for triggering this scenario
            print(
                f">> [VehicleTurningRoute] Successfully initialized cyclist at position ({other_actor_transform.location.x:.2f}, {other_actor_transform.location.y:.2f})")
        except Exception as e:
            self.reference_actor = None
            print(f">> [VehicleTurningRoute] Failed to initialize cyclist: {e}")
            print(f">> [VehicleTurningRoute] Scenario will not be triggered")


    def create_behavior(self, scenario_init_action):
        # 检查是否使用遗传算法
        if self._should_use_genetic_algorithm(scenario_init_action):
            print(">> Using genetic algorithm for vehicle turning route optimization")

            # 创建遗传算法优化器
            self.genetic_optimizer = SkoptGeneticOptimizer(self)

            # 执行优化
            self.optimized_params = self.genetic_optimizer.optimize_initial_state()

            # 应用优化后的参数
            self.actor_speed = self.optimized_params['actor_speed']
            self.trigger_distance_threshold = self.optimized_params['trigger_distance']
            self.position_offset = self.optimized_params['position_offset']

            # 设置位置参数
            self.x = 0.0
            self.y = 0.0

            opt_info = self.genetic_optimizer.get_optimization_info()
            print(f">> Vehicle turning route optimization completed - Best fitness: {opt_info['best_fitness']:.4f}")

        else:
            self.actions = scenario_init_action

    def _should_use_genetic_algorithm(self, scenario_init_action) -> bool:
        """
        判断是否应该使用遗传算法
        """
        if scenario_init_action is None:
            return False

        # 检查config中是否启用遗传算法参数
        if (hasattr(self.config, 'parameters') and
                len(self.config.parameters) >= 2 and
                self.config.parameters[0] == 'genetic_algorithm' and
                self.config.parameters[1] is True):
            return True

        return False

    # def update_behavior(self, scenario_action):
    #     """
    #     更新自行车行为 - 关键修复：使用优化后的速度
    #     """
    #     assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'
    #
    #     # 使用遗传算法优化后的速度，而不是固定值
    #     cur_actor_target_speed = self.actor_speed if hasattr(self, 'actor_speed') else 4.0
    #
    #     for i in range(len(self.other_actors)):
    #         self.scenario_operation.go_straight(cur_actor_target_speed, i)
    #
    #     # 调试输出
    #     # if not hasattr(self, '_speed_printed'):
    #     print(f">> [VehicleTurningRoute] Cyclist moving at speed: {cur_actor_target_speed:.2f} m/s")
    #     self._speed_printed = True

    def update_behavior(self, scenario_action):
        """
        更新自行车行为 - 关键修复：使用优化后的速度
        """
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        # 使用遗传算法优化后的速度
        cur_actor_target_speed = self.actor_speed if hasattr(self, 'actor_speed') else 4.0

        # self._update_count += 1

        for i in range(len(self.other_actors)):
            actor = self.other_actors[i]

            # ⭐⭐⭐ 直接控制，不用go_straight
            current_velocity = actor.get_velocity()
            current_speed = math.sqrt(current_velocity.x ** 2 + current_velocity.y ** 2 + current_velocity.z ** 2)

            control = carla.VehicleControl()

            # 根据当前速度决定油门或刹车
            if current_speed < cur_actor_target_speed:
                control.throttle = 1  # 加速
                control.brake = 0.0
            elif current_speed > cur_actor_target_speed + 2.0:
                control.throttle = 0.0
                control.brake = 0.3  # 轻微刹车
            else:
                control.throttle = 0.5  # 维持速度
                control.brake = 0.0

            control.steer = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False
            actor.apply_control(control)

            # 同时设置目标速度作为备用
            transform = actor.get_transform()
            forward_vector = transform.rotation.get_forward_vector()
            target_velocity = carla.Vector3D(
                forward_vector.x * cur_actor_target_speed,
                forward_vector.y * cur_actor_target_speed,
                0
            )
            actor.set_target_velocity(target_velocity)

    def check_stop_condition(self):
        return False