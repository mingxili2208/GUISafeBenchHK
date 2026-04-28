''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-30 12:19:25
Description:
    Copyright (c) 2022-2023 Safebench Team
    Fixed version for genetic algorithm optimization
    这个版本是返回偏移量
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
        super(VehicleTurningRoute, self).__init__("VehicleTurningRoute-Adv", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self.scenario_operation = ScenarioOperation()
        self.running_distance = 10

        self.trigger_location = config.trigger_points[0].location if hasattr(config,
                                                                             'trigger_points') and config.trigger_points else None

        # 遗传算法相关
        self.genetic_optimizer = None
        self.use_genetic_algorithm = False
        self.optimized_params = None

        # 基础参数
        self.actor_speed = 4.0
        self.trigger_distance_threshold = 25.0
        self.position_offset = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.x = 0.0
        self.y = 0.0
        self.trigger = False

        # 直接绿灯
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light:
            self._traffic_light.set_state(carla.TrafficLightState.Green)
            self._traffic_light.set_green_time(self.timeout)

    # ---------------------------------------------------------
    def initialize_actorsHK(self):
        """
        初始化场景参与者，直接使用遗传算法优化后的 initial_transform
        """
        if hasattr(self, 'other_actors') and self.other_actors:
            print(f">> [VehicleTurningRoute] Cleaning up existing actors: {len(self.other_actors)}")
            for actor in self.other_actors:
                if actor and actor.is_alive:
                    CarlaDataProvider.remove_actor_by_id(actor.id)
            self.other_actors = []
            self.reference_actor = None


        if not self.optimized_params:
            # 未调用遗传算法，先创建优化器
            self.genetic_optimizer = SkoptGeneticOptimizer(self)
            self.optimized_params = self.genetic_optimizer.optimize_initial_state()

        initial_transform = self.optimized_params.get("initial_transform", None)
        if initial_transform is None:
            raise RuntimeError("Optimized initial_transform not found!")

        self.actor_speed = self.optimized_params["actor_speed"]
        self.trigger_distance_threshold = self.optimized_params["trigger_distance"]
        self.position_offset = self.optimized_params["position_offset"]
        # 用优化后的actor位置重置trigger_location
        self.trigger_location = initial_transform.location
        print(
            f">> trigger_location reset to actor position: ({self.trigger_location.x:.2f}, {self.trigger_location.y:.2f})")

        # 生成 actor
        # blueprint = self.world.get_blueprint_library().find('vehicle.diamondback.century')
        # blueprint.set_attribute('role_name', 'adversary')

        try:
            # actor = self.world.spawn_actor(blueprint, initial_transform)
            # self.other_actors = [actor]
            # self.reference_actor = actor
            # print(f">> [VehicleTurningRoute] Spawned actor at ({initial_transform.location.x:.2f}, "
            #       f"{initial_transform.location.y:.2f}) yaw {initial_transform.rotation.yaw:.2f}")

            self.actor_type_list = ['vehicle.diamondback.century']
            self.actor_transform_list = [initial_transform]
            self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list,
                                                                                  self.actor_type_list)
            self.reference_actor = self.other_actors[0]
        except Exception as e:
            # self.reference_actor = None
            # self.other_actors = []
            other_actor_transform = self.config.other_actors[0].transform
            self.actor_type_list = ['vehicle.diamondback.century']
            self.actor_transform_list = [other_actor_transform]
            self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list,
                                                                                  self.actor_type_list)
            self.reference_actor = self.other_actors[0]
            print(f">> [VehicleTurningRoute] Failed to spawn actor: {e}. Use default position at ({other_actor_transform.location.x:.2f}, ")

    # ---------------------------------------------------------
    def create_behavior(self, scenario_init_action):
        """
        创建 actor 行为，保留遗传算法逻辑
        """
        if self._should_use_genetic_algorithm(scenario_init_action):
            print(">> Using genetic algorithm for vehicle turning route optimization")

            if not self.genetic_optimizer:
                self.genetic_optimizer = SkoptGeneticOptimizer(self)

            # 执行优化
            self.optimized_params = self.genetic_optimizer.optimize_initial_state()

            # 更新参数
            self.actor_speed = self.optimized_params['actor_speed']
            self.trigger_distance_threshold = self.optimized_params['trigger_distance']
            self.position_offset = self.optimized_params['position_offset']

            opt_info = self.genetic_optimizer.get_optimization_info()
            print(f">> Vehicle turning route optimization completed - Best fitness: {opt_info['best_fitness']:.4f}")
        else:
            self.actions = scenario_init_action

    # ---------------------------------------------------------
    def update_behavior(self, scenario_action):
        """
        更新自行车行为，使用优化后的速度
        """
        assert scenario_action is None, f'{self.name} should receive [None] action.'

        cur_actor_target_speed = getattr(self, 'actor_speed', 4.0)

        for actor in self.other_actors:
            current_velocity = actor.get_velocity()
            current_speed = math.sqrt(current_velocity.x**2 + current_velocity.y**2 + current_velocity.z**2)

            control = carla.VehicleControl()

            if current_speed < cur_actor_target_speed:
                control.throttle = 1.0
                control.brake = 0.0
            elif current_speed > cur_actor_target_speed + 2.0:
                control.throttle = 0.0
                control.brake = 0.3
            else:
                control.throttle = 0.5
                control.brake = 0.0

            control.steer = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False

            actor.apply_control(control)

            # 设置目标速度向量
            transform = actor.get_transform()
            forward_vec = transform.rotation.get_forward_vector()
            actor.set_target_velocity(carla.Vector3D(
                forward_vec.x * cur_actor_target_speed,
                forward_vec.y * cur_actor_target_speed,
                0
            ))

    # ---------------------------------------------------------
    def check_stop_condition(self):
        if not self.other_actors or len(self.other_actors) == 0:
            return True

    # ---------------------------------------------------------
    def _should_use_genetic_algorithm(self, scenario_init_action) -> bool:
        """
        判断是否应该使用遗传算法
        """
        if scenario_init_action is None:
            return False

        if (hasattr(self.config, 'parameters') and
                len(self.config.parameters) >= 2 and
                self.config.parameters[0] == 'genetic_algorithm' and
                self.config.parameters[1] is True):
            return True

        return False
