"""
主车直行过程中遇到行人横穿马路的场景定义"""

from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario


class DynamicObjectCrossing(BasicScenario):
    """
        A simple object crash without prior vehicle action involving a vehicle and a cyclist/pedestrian.
        The ego vehicle is passing through a road, and encounters a cyclist/pedestrian crossing the road.
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(DynamicObjectCrossing, self).__init__("DynamicObjectCrossing", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()
        self._map = CarlaDataProvider.get_map()
        # self._ego_vehicle_distance_driven = 40
        # other vehicle parameters
        self._other_actor_target_velocity = 8
        # self._other_actor_max_brake = 1.0
        # self._time_to_reach = 10
        # self._walker_yaw = 0
        self._num_lane_changes = 1
        # Note: transforms for walker and blocker
        self.transform = None
        self.trigger_location = config.trigger_points[0].location
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 20
        # Number of attempts made so far
        self._spawn_attempted = 0

        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = 6  # 距离场景触发点位置小于该阈值的时候触发场景
        self.ego_max_driven_distance = 500

    def initialize_actorsHK(self):
        """
        简化版本,不必生成vending machine,直接在触发点位置生成具体场景的walker
        """
        # 直接提取行人的生成位置
        actor_spawn_transform = self.config.other_actors[0].transform
        # 定义DynamicObjectCrossing场景中需要生成的对抗actor的类型
        self.actor_type_list = ['walker.*']
        self.actor_transform_list = [actor_spawn_transform]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        if len(self.other_actors) > 0:
            self.reference_actor = self.other_actors[0]

        # self.actor_type_list = ['walker.*']
        # self.actor_transform_list = [actor_spawn_transform]
        # autopilot_list = [False]
        # simulate_list = [True]
        # amount = 5  # 生成背景actor的数量
        # for _ in range(amount):
        #     location = self.ego_vehicle.get_transform().location
        #     rotation = self.ego_vehicle.get_transform().rotation
        #     location.x -= np.random.random() * 60
        #     location.y += np.random.random() * 30
        #
        #     # 保证背景车辆生成在道路上
        #     waypoints_dense = np.load(self.config.waypoint_dense_path)
        #     distances = np.linalg.norm(waypoints_dense[:, :2] - [location.x, location.y], axis=1)
        #     waypoint_list = self.world.get_map().generate_waypoints(1.0)
        #     location = waypoint_list[np.argmin(distances)].transform.location
        #
        #     new_ego_transform = carla.Transform(location, rotation)
        #     self.actor_transform_list.append(new_ego_transform)
        #     self.actor_type_list.append('vehicle.*')
        #     autopilot_list.append(True)
        #     simulate_list.append(True)
        #
        # self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list,
        #                                                                       self.actor_type_list,
        #                                                                       autopilot_list=autopilot_list,
        #                                                                       simulate_list=simulate_list)
        # self.reference_actor = self.other_actors[0]  # used for triggering this scenario

    def create_behavior(self, scenario_init_action):
        # 确保scenario的初始行为是None
        assert scenario_init_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        # the walker starts crossing the road
        self.scenario_operation.walker_go_straight(self._other_actor_target_velocity, 0)