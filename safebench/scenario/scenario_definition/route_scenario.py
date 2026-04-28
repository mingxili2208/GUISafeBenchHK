"""
处理主车生成、构建场景实例、测试评价
"""
import copy
import traceback
import numpy as np
import carla
from safebench.util.run_util import class_from_path
from safebench.scenario.scenario_manager.timer import GameTime
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_manager.scenario_config import RouteScenarioConfig
from safebench.scenario.tools.route_parser import RouteParser
from safebench.carla_agents.navigation.global_route_planner import GlobalRoutePlanner
from safebench.scenario.tools.scenario_utils import (
    convert_json_to_transform,
    convert_json_to_actor, 
    convert_transform_to_location
)

from safebench.scenario.scenario_definition.atomic_criteria import (
    Status,
    CollisionTest,
    DrivenDistanceTest,
    AverageVelocityTest,
    OffRoadTest,
    KeepLaneTest,
    InRouteTest,
    RouteCompletionTest,
    RunningRedLightTest,
    RunningStopTest,
)


class RouteScenario:
    """
        Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
        along which several smaller scenarios are triggered
    """

    def __init__(self, world, config, ego_id, logger, max_running_step):
        self.world = world
        self.logger = logger
        self.config = config
        self.ego_id = ego_id
        self.max_running_step = max_running_step
        self.timeout = 60  # 默认设置的仿真超时时间
        # 这里的scenario_definition指的是经过route_scenario.py解析之后的具体场景定义,比如要执行的DynamicObjectCrossing名称,触发位置等
        self.route, self.ego_vehicle, scenario_definitions = self._update_route_and_ego(timeout=self.timeout)
        self.background_actors = []
        self.list_scenarios = self._build_scenario_instances(scenario_definitions)  # 根据场景定义来创建控制场景动作的实例
        self.criteria = self._create_criteria()
        self._stuck_step_count = 0  # ego卡死连续低速步数计数器

    def _update_route_and_ego(self, timeout=None):
        # transform the scenario file into a dictionary
        if self.config.scenario_file is not None:
            # scenario_file形如scenario_01.json
            world_annotations = RouteParser.parse_annotations_file(self.config.scenario_file)
        else:
            world_annotations = self.config.scenario_config

        # 使用CARLA的路径搜索功能,获取插值轨迹
        wmap = self.world.get_map()
        global_planner = GlobalRoutePlanner(wmap, sampling_resolution=2.0)
        start_location = self.config.trajectory[0]
        end_location = self.config.trajectory[-1]
        route = global_planner.trace_route(start_location, end_location)

        # 创建一个新的route列表
        updated_route = []
        for i in range(len(route)):
            waypoint = route[i][0]
            road_option = route[i][1]
            new_transform = carla.Transform(
                location=waypoint.transform.location,
                rotation=carla.Rotation(
                    pitch=waypoint.transform.rotation.pitch,
                    yaw=waypoint.transform.rotation.yaw,
                    roll=waypoint.transform.rotation.roll
                )
            )
            updated_route.append((new_transform, road_option))
        route = updated_route

        ego_vehicle = self._spawn_ego_vehicle(self.config.initial_transform, self.config.auto_ego)

        # possible_scenarios是一个字典,键值为数字,值为场景定义的列表,包含‘match_position’,'name', 'trigger_position'等字段
        possible_scenarios, _ = RouteParser.scan_route_for_scenariosHK(
            self.config.town,
            self.config.route_id,  # route_id是一个字符串,表示当前测试场景的道路编号
            world_annotations  # route是一个列表,列表中的每个元素都是一个长度为2的tuple,包含有Transform和RoadOption
        )

        # 将所有可能的场景定义从possible_scenarios字典中提取出来,此时scenarios_definitions存储着场景的名称、触发位置等信息
        scenarios_definitions = []
        for trigger in possible_scenarios.keys():
            scenarios_definitions.extend(possible_scenarios[trigger])

        assert len(scenarios_definitions) >= 1, f"There should be at least 1 scenario definition in the route"
        scenarios_definitions = [scenarios_definitions[0]]

        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(route))
        CarlaDataProvider.set_scenario_config(self.config)

        # Timeout of scenario in seconds
        self.timeout = self._estimate_route_timeout(route) if timeout is None else timeout
        return route, ego_vehicle, scenarios_definitions

    def _estimate_route_timeout(self, route):
        route_length = 0.0  # in meters
        min_length = 1000.0
        SECONDS_GIVEN_PER_METERS = 1

        if len(route) == 1:
            return int(SECONDS_GIVEN_PER_METERS * min_length)

        prev_point = route[0][0]
        for current_point, _ in route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point
        return int(SECONDS_GIVEN_PER_METERS * route_length)

    def _spawn_ego_vehicle(self, elevate_transform, autopilot=False):
        role_name = 'ego_vehicle' + str(self.ego_id)  # self.ego_id的取值与当前执行的是第几个场景有关

        success = False
        ego_vehicle = None
        while not success:
            try: # 这里指定了ego vehicle的类型为'vehicle.lincoln.mkz_2017'
                ego_vehicle = CarlaDataProvider.request_new_actor(
                    'vehicle.lincoln.mkz_2017',
                    elevate_transform,
                    rolename=role_name, 
                    autopilot=autopilot
                )
                ego_vehicle.set_autopilot(autopilot, CarlaDataProvider.get_traffic_manager_port())

                # 缩短ego的跟车距离
                if autopilot:
                    tm = CarlaDataProvider.get_client().get_trafficmanager(CarlaDataProvider.get_traffic_manager_port())
                    tm.distance_to_leading_vehicle(ego_vehicle, 5)  # 8米
                    # tm.vehicle_percentage_speed_difference(ego_vehicle, -30.0)  # 超速30%

                success = True
            except RuntimeError:
                elevate_transform.location.z += 0.1
        return ego_vehicle

    def _build_scenario_instances(self, scenario_definitions):
        """
        构建所有场景类的实例
        """
        scenario_instance_list = []
        for _, definition in enumerate(scenario_definitions):
            # get the class of the scenario
            scenario_path = [
                'safebench.scenario.scenario_definition',  # 场景定义的基础路径
                self.config.scenario_folder,  # 配置中指定的场景文件夹
                definition['name'],  # 场景定义的名称
            ]
            scenario_class = class_from_path('.'.join(scenario_path))

            # 如果场景定义中有其他参与者,则创建这些参与者的实例
            if definition['other_actors'] is not None:
                list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
            else:
                list_of_actor_conf_instances = []

            # 创建场景运行的触发位置
            egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
            route_config = RouteScenarioConfig()
            route_config.other_actors = list_of_actor_conf_instances
            route_config.trigger_points = [egoactor_trigger_position]
            route_config.parameters = self.config.parameters
            route_config.num_scenario = self.config.num_scenario
            route_config.route_id = self.config.route_id
            if self.config.weather is not None:
                route_config.weather = self.config.weather

            try:
                # 实例化场景类
                scenario_instance = scenario_class(self.world, self.ego_vehicle, route_config, timeout=self.timeout)
            except Exception as e:   
                traceback.print_exc()
                print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                continue

            scenario_instance_list.append(scenario_instance)
        return scenario_instance_list

    def _get_actors_instances(self, list_of_antagonist_actors):
        def get_actors_from_list(list_of_actor_def):
            # receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            sublist_of_actors = []
            for actor_def in list_of_actor_def:
                sublist_of_actors.append(convert_json_to_actor(actor_def))
            return sublist_of_actors

        list_of_actors = []
        if 'left' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['left'])
        if 'right' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['right'])
        if 'center' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['center'])
        return list_of_actors

    # TODO: 在0.9.13-Safebench地图上面，修改amount为非零数值，即可增加背景车流，但是在自行编译的Carla上面运行地图，则会出现通信卡死，中断的情况
    def initialize_actors(self):
        # amount = 0  # 生成背景actor的数量
        # new_actors = CarlaDataProvider.request_new_batch_actors(
        #     'vehicle.*',
        #     amount,
        #     carla.Transform(),
        #     autopilot=True,
        #     random_location=True,
        #     rolename='background'
        # )
        amount = 10  # 生成背景车流,指定生成的数量
        new_actors = CarlaDataProvider.request_new_batch_actors(
            'vehicle.*',
            self.ego_vehicle,
            amount,
            autopilot=True,
            random_location=False,
            rolename='background'
        )

        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")
        successful_actors = [actor for actor in new_actors if actor is not None]
        if len(successful_actors) < amount:
            self.logger.log(
                f'>> Warning: only spawned {len(successful_actors)}/{amount} background actors',
                'yellow',
            )
        for actor in successful_actors:
            self.background_actors.append(actor)

    _STUCK_SPEED_THRESHOLD = 0.1   # m/s，低于此速度视为卡死
    _STUCK_STEP_LIMIT = 200        # 连续卡死超过此步数则提前终止（约20秒@10Hz）

    def get_running_status(self, running_record):
        running_status = {
            'ego_velocity': CarlaDataProvider.get_velocity(self.ego_vehicle),
            'ego_acceleration_x': self.ego_vehicle.get_acceleration().x,
            'ego_acceleration_y': self.ego_vehicle.get_acceleration().y,
            'ego_acceleration_z': self.ego_vehicle.get_acceleration().z,
            'ego_x': CarlaDataProvider.get_transform(self.ego_vehicle).location.x,
            'ego_y': CarlaDataProvider.get_transform(self.ego_vehicle).location.y,
            'ego_z': CarlaDataProvider.get_transform(self.ego_vehicle).location.z,
            'ego_roll': CarlaDataProvider.get_transform(self.ego_vehicle).rotation.roll,
            'ego_pitch': CarlaDataProvider.get_transform(self.ego_vehicle).rotation.pitch,
            'ego_yaw': CarlaDataProvider.get_transform(self.ego_vehicle).rotation.yaw,
            'current_game_time': GameTime.get_time()
        }
        # running_status首先获取其主车状态,其次获得各个预定义的atomic criteria的状态
        for criterion_name, criterion in self.criteria.items():
            running_status[criterion_name] = criterion.update()

        stop = False
        # collision with other objects
        if running_status['collision'] == Status.FAILURE:
            stop = True
            self.logger.log('>> Scenario stops due to collision', color='yellow')

        # out of the road detection
        if running_status['off_road'] == Status.FAILURE:
            stop = True
            self.logger.log('>> Scenario stops due to off road', color='yellow')

        # only check when evaluating
        if self.config.scenario_id != 0:  
            # route completed
            if running_status['route_complete'] == 100:
                stop = True
                self.logger.log('>> Scenario stops due to route completion', color='yellow')

        # stop at max step
        if len(running_record) >= self.max_running_step: 
            stop = True
            self.logger.log('>> Scenario stops due to max steps', color='yellow')

        # ego卡死检测：连续低速（NPC堵车）超过阈值时提前终止，避免等到 timeout 才清理
        if running_status['ego_velocity'] < self._STUCK_SPEED_THRESHOLD:
            self._stuck_step_count += 1
        else:
            self._stuck_step_count = 0
        if self._stuck_step_count >= self._STUCK_STEP_LIMIT:
            stop = True
            self._stuck_step_count = 0
            self.logger.log(
                f'>> Scenario stops due to ego stuck (speed < {self._STUCK_SPEED_THRESHOLD} m/s '
                f'for {self._STUCK_STEP_LIMIT} consecutive steps)',
                color='yellow',
            )

        for scenario in self.list_scenarios:
            # only check when evaluating
            if self.config.scenario_id != 0:  
                if running_status['driven_distance'] >= scenario.ego_max_driven_distance:
                    stop = True
                    self.logger.log('>> Scenario stops due to max driven distance', color='yellow')
                    break
            if running_status['current_game_time'] >= scenario.timeout:
                stop = True
                self.logger.log('>> Scenario stops due to timeout', color='yellow') 
                break

        return running_status, stop

    def _create_criteria(self):
        criteria = {}
        route = convert_transform_to_location(self.route)

        criteria['driven_distance'] = DrivenDistanceTest(actor=self.ego_vehicle, distance_success=1e4, distance_acceptable=1e4, optional=True)
        criteria['average_velocity'] = AverageVelocityTest(actor=self.ego_vehicle, avg_velocity_success=1e4, avg_velocity_acceptable=1e4, optional=True)
        criteria['lane_invasion'] = KeepLaneTest(actor=self.ego_vehicle, optional=True)
        criteria['off_road'] = OffRoadTest(actor=self.ego_vehicle, optional=True)
        criteria['collision'] = CollisionTest(actor=self.ego_vehicle, terminate_on_failure=True)
        criteria['run_red_light'] = RunningRedLightTest(actor=self.ego_vehicle)
        criteria['run_stop'] = RunningStopTest(actor=self.ego_vehicle)
        if self.config.scenario_id != 0:  # only check when evaluating
            criteria['distance_to_route'] = InRouteTest(self.ego_vehicle, route=route, offroad_max=30)
            criteria['route_complete'] = RouteCompletionTest(self.ego_vehicle, route=route)
        return criteria

    @staticmethod
    def _get_actor_state(actor):
        actor_trans = actor.get_transform()
        actor_x = actor_trans.location.x
        actor_y = actor_trans.location.y
        actor_yaw = actor_trans.rotation.yaw / 180 * np.pi
        yaw = np.array([np.cos(actor_yaw), np.sin(actor_yaw)])
        velocity = actor.get_velocity()
        acc = actor.get_acceleration()
        return [actor_x, actor_y, actor_yaw, yaw[0], yaw[1], velocity.x, velocity.y, acc.x, acc.y]

    def update_info(self):
        ego_state = self._get_actor_state(self.ego_vehicle)
        actor_info = [ego_state]
        for s_i in self.list_scenarios:
            for a_i in s_i.other_actors:
                actor_state = self._get_actor_state(a_i)
                actor_info.append(actor_state)

        actor_info = np.array(actor_info)
        # get the info of the ego vehicle and the other actors
        return {
            'actor_info': actor_info
        }

    def _flush_cleanup_world(self, ticks=1):
        """
            Give CARLA one or more world steps to fully apply pending destroy
            operations before the next cleanup stage starts.
        """
        if self.world is None or ticks <= 0:
            return

        for _ in range(ticks):
            try:
                if self.world.get_settings().synchronous_mode:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()
                CarlaDataProvider.on_carla_tick()
            except Exception as world_error:
                self.logger.log(f'>> RouteScenario cleanup world flush failed: {world_error}', 'yellow')
                break

    def clean_up(self):
        import time as _time
        def _dbg(msg):
            ts = _time.strftime('%H:%M:%S')
            print(f'[ROUTE-CLEANUP-DBG {ts}] {msg}', flush=True)

        _dbg('--- RouteScenario.clean_up START ---')

        # First stop all criterion sensor listeners, then destroy them together.
        _dbg('R1: stopping criterion sensors')
        for cname, criterion in self.criteria.items():
            try:
                _dbg(f'R1:   criterion.stop_sensor [{cname}]')
                criterion.stop_sensor()
            except Exception as criterion_error:
                _dbg(f'R1:   [{cname}] FAILED: {criterion_error}')
                self.logger.log(f'>> Criterion sensor stop failed: {criterion_error}', 'yellow')
        _dbg('R1: criterion sensors stopped')

        _dbg('R2: world flush after criterion sensor stop')
        self._flush_cleanup_world()
        _dbg('R2: flush OK')

        _dbg('R3: terminating criteria')
        for cname, criterion in self.criteria.items():
            try:
                _dbg(f'R3:   criterion.terminate [{cname}]')
                criterion.terminate()
            except Exception as criterion_error:
                _dbg(f'R3:   [{cname}] FAILED: {criterion_error}')
                self.logger.log(f'>> Criterion cleanup failed: {criterion_error}', 'yellow')
        _dbg('R3: criteria terminated')

        _dbg('R4: world flush after criteria terminate')
        self._flush_cleanup_world()
        _dbg('R4: flush OK')

        # each scenario remove its own actors
        _dbg(f'R5: cleaning up {len(self.list_scenarios)} scenario instance(s) (scenario actors)')
        for scenario in self.list_scenarios:
            try:
                _dbg(f'R5:   scenario.clean_up [{scenario.name}]')
                scenario.clean_up()
                _dbg(f'R5:   [{scenario.name}] OK')
            except Exception as scenario_error:
                _dbg(f'R5:   [{scenario.name}] FAILED: {scenario_error}')
                self.logger.log(f'>> Scenario cleanup failed: {scenario_error}', 'yellow')

        _dbg('R6: world flush after scenario actors cleanup')
        self._flush_cleanup_world()
        _dbg('R6: flush OK')

        _dbg('R7: inspecting world actors for background vehicle cleanup')
        world_actor_ids = set()
        try:
            world = CarlaDataProvider.get_world()
            if world is not None:
                world_actor_ids = {actor.id for actor in world.get_actors()}
                _dbg(f'R7: world has {len(world_actor_ids)} actors')
        except Exception as world_error:
            _dbg(f'R7: world.get_actors FAILED: {world_error}')
            self.logger.log(f'>> Failed to inspect world actors during cleanup: {world_error}', 'yellow')

        # remove background vehicles
        valid_background_actor_ids = []
        for actor in self.background_actors:
            if actor is None:
                continue
            actor_id = getattr(actor, 'id', None)
            if actor_id is None:
                continue
            if world_actor_ids and actor_id not in world_actor_ids:
                continue
            if not CarlaDataProvider.actor_id_exists(actor_id):
                continue
            valid_background_actor_ids.append(actor_id)

        skipped_actors = len(self.background_actors) - len(valid_background_actor_ids)
        if skipped_actors > 0:
            self.logger.log(
                f'>> Skipping {skipped_actors} stale background actors during cleanup',
                'yellow',
            )

        # 关键修复：先关闭 autopilot，让 Traffic Manager 停止控制这些车辆，
        # 再销毁 actor。否则 TM 会继续向已销毁的 actor 发送 apply_control_to_vehicle，
        # 导致 RPC 连接状态损坏，下一个场景 spawn 时 server_session::close() 触发
        # "Bad file descriptor" → SIGSEGV 服务端崩溃。
        _dbg(f'R8a: disabling autopilot for {len(valid_background_actor_ids)} background vehicles before destroy')
        tm_port = CarlaDataProvider.get_traffic_manager_port()
        for actor_id in valid_background_actor_ids:
            actor = CarlaDataProvider._carla_actor_pool.get(actor_id)
            if actor is None:
                continue
            try:
                actor.set_autopilot(False, tm_port)
            except Exception as ap_error:
                _dbg(f'R8a: set_autopilot(False) failed for {actor_id}: {ap_error}')
        _dbg('R8a: autopilot disabled for all background vehicles')

        # 给 TM 一点时间处理 autopilot 关闭，再执行 destroy
        _dbg('R8b: world flush to let TM process autopilot-off')
        self._flush_cleanup_world()
        _dbg('R8b: flush OK')

        _dbg(f'R8c: destroying {len(valid_background_actor_ids)} background vehicles (apply_batch_sync RPC)')
        if valid_background_actor_ids:
            try:
                CarlaDataProvider.remove_actors_by_ids(valid_background_actor_ids)
                _dbg('R8c: background vehicles destroyed OK')
            except Exception as actor_error:
                _dbg(f'R8c: FAILED: {actor_error}')
                self.logger.log(f'>> Background actor cleanup failed: {actor_error}', 'yellow')

        _dbg('R9: world flush after background vehicles cleanup')
        self._flush_cleanup_world()
        _dbg('R9: flush OK')
        self.background_actors = []
        _dbg('--- RouteScenario.clean_up DONE ---')
