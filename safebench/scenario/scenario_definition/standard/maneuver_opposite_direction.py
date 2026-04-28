"""
本车前面车辆缓慢行驶,借道行驶
"""
import carla
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.tools.scenario_helper import get_waypoint_in_distance
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_operation import ScenarioOperation


class ManeuverOppositeDirection(BasicScenario):
    """
        Vehicle Maneuvering In Opposite Direction.
        Vehicle is passing another vehicle in a rural area, in daylight, under clear weather conditions, 
        at a non-junction and encroaches into another vehicle traveling in the opposite direction.
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(ManeuverOppositeDirection, self).__init__("ManeuverOppositeDirection", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout
        self._map = CarlaDataProvider.get_map()
        self.trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(self.trigger_location)
        self._opposite_speed = 8 # m/s
        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = 10
        self.ego_max_driven_distance = 1000

    def initialize_actorsHK(self):
        first_actor_waypoint = self._reference_waypoint
        temp_second_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, 30)

        # 第二辆车不是在左侧车道就是在右侧车道
        try:
            second_actor_waypoint = temp_second_vehicle_waypoint.get_right_lane()
        except:
            second_actor_waypoint = temp_second_vehicle_waypoint.get_left_lane()

        first_actor_transform = carla.Transform(first_actor_waypoint.transform.location, first_actor_waypoint.transform.rotation)
        second_actor_transform = second_actor_waypoint.transform

        # create two other actors
        self.actor_transform_list = [first_actor_transform, second_actor_transform]
        self.actor_type_list = ['vehicle.nissan.patrol', 'vehicle.audi.tt']
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario
        
    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'
        self.scenario_operation.go_straight(self._opposite_speed, 1)

    def check_stop_condition(self):
        pass
