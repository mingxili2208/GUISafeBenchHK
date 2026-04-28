"""
    主车沿着一辆前车行驶在指定道路上。在某个时刻，前车突然减速。
    Ego vehicle follows a leading car driving down a given road. At some point the leading car has to decelerate.
"""

import carla
from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.tools.scenario_utils import calculate_distance_transforms
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.tools.scenario_helper import get_waypoint_in_distance
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario


class OtherLeadingVehicle(BasicScenario):
    """
        Ego vehicle follows a leading car driving down a given road. At some point the leading car has to decelerate.
        The ego vehicle has to react accordingly by changing lane to avoid a collision and follow the leading car in other lane. 
        The scenario ends either via a timeout, or if the ego vehicle drives some distance. (Traffic Scenario 05)
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(OtherLeadingVehicle, self).__init__("OtherLeadingVehicle", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout
        self._map = CarlaDataProvider.get_map()

        self.trigger_location = config.trigger_points[0].location
        self.dece_distance = 5  # 前车离开其出生点该段距离之后开始减速
        self.dece_target_speed = 2  # m/s
        self.need_decelerate = False
        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = 6
        self.ego_max_driven_distance = 1000

    # 直接在给定点生成车辆
    def initialize_actorsHK(self):
        # 获取前车的路点
        first_vehicle_transform = self.config.other_actors[0].transform
        first_vehicle_waypoint = self._map.get_waypoint(first_vehicle_transform.location)
        temp_second_vehicle_waypoint, _ = get_waypoint_in_distance(first_vehicle_waypoint, 1)
        # 第二辆车不是在左侧车道就是在右侧车道
        try:
            second_vehicle_waypoint = temp_second_vehicle_waypoint.get_right_lane()
        except:
            second_vehicle_waypoint = temp_second_vehicle_waypoint.get_left_lane()

        second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location, second_vehicle_waypoint.transform.rotation)
        self.actor_type_list = ['vehicle.nissan.patrol', 'vehicle.audi.tt']
        self.actor_transform_list = [first_vehicle_transform, second_vehicle_transform]
        self.actor_speed_list = [10, 10]  # m/s
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario
        
    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

    def update_behavior(self, scenario_action):
        """
            Just make two vehicles move forward with specific speed
            At specific point, vehicle in front of ego will decelerate other_actors[0] is the vehicle before the ego
        """
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'
        
        cur_distance = calculate_distance_transforms(self.actor_transform_list[0], CarlaDataProvider.get_transform(self.other_actors[0]))
        if cur_distance > self.dece_distance:
            self.need_decelerate = True
        for i in range(len(self.other_actors)):
            if i == 0 and self.need_decelerate:
                self.scenario_operation.go_straight(self.dece_target_speed, i)
            else:
                self.scenario_operation.go_straight(self.actor_speed_list[i], i)

    def check_stop_condition(self):
        pass
