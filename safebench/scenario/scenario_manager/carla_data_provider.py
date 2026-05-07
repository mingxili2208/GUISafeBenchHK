"""
1、提供了从CARLA Server中获取和管理各种数据的方法,包括actor的速度、位置、变换、交通信号灯的位置和状态等
2、提供了注册、更新和获取actor的方法,可以批量创建、请求和删除actor
3、提供了获取和更新交通信号等状态的方法
4、设置和获取CARLA客户端和世界
5、获取天气,清理预设等
"""

import math
import re

import numpy as np
from numpy import random
from six import iteritems
import carla


def calculate_velocity(actor):
    """m/s"""
    velocity_squared = actor.get_velocity().x**2
    velocity_squared += actor.get_velocity().y**2
    return math.sqrt(velocity_squared)


class CarlaDataProvider(object):
    """定义了一些类的静态变量，用于存储CARLA仿真器中的各种数据和状态"""
    _actor_velocity_map = {}  # 存储actor的速度
    _actor_location_map = {}  # 存储actor的位置
    _actor_transform_map = {}  # 存储actor的变换(位置和方向)
    _traffic_light_map = {}  # 存储交通信号灯的位置和状态
    _carla_actor_pool = {}  # 存储所有actor的池
    _global_osc_parameters = {}  # 存储全局的OSC参数, OpenSCENARIO是一个描述驾驶场景的标准格式
    _client = None
    _world = None
    _map = None
    _sync_flag = False  # 是否使用同步模式
    _spawn_points = None  # 存储生成点
    _spawn_index = 0  # 生成点的索引
    _blueprint_library = None
    _ego_vehicle_route = None  # 自车的规划路线
    _traffic_manager_port = 8000
    _random_seed = 2000
    _rng = random.RandomState(_random_seed)

    @staticmethod
    def register_actor(actor):
        """
        一般意味着同一个actor,不能被注册两次
        """
        if actor in CarlaDataProvider._actor_velocity_map:
            raise KeyError("Vehicle '{}' already registered. Cannot register twice!".format(actor.id))
        else:
            CarlaDataProvider._actor_velocity_map[actor] = 0.0

        if actor in CarlaDataProvider._actor_location_map:
            raise KeyError("Vehicle '{}' already registered. Cannot register twice!".format(actor.id))
        else:
            CarlaDataProvider._actor_location_map[actor] = None

        if actor in CarlaDataProvider._actor_transform_map:
            raise KeyError("Vehicle '{}' already registered. Cannot register twice!".format(actor.id))
        else:
            CarlaDataProvider._actor_transform_map[actor] = None


    @staticmethod
    def register_actors(actors):
        """
            Add new set of actors to dictionaries
        """
        for actor in actors:
            CarlaDataProvider.register_actor(actor)

    @staticmethod
    def on_carla_tick():
        """
            Callback from CARLA
        """
        dead_actors = [
            actor for actor in list(CarlaDataProvider._actor_velocity_map.keys())
            if actor is None or not actor.is_alive
        ]
        for actor in dead_actors:
            CarlaDataProvider._actor_velocity_map.pop(actor, None)
            CarlaDataProvider._actor_location_map.pop(actor, None)
            CarlaDataProvider._actor_transform_map.pop(actor, None)

        for actor in CarlaDataProvider._actor_velocity_map:
            if actor is not None and actor.is_alive:
                CarlaDataProvider._actor_velocity_map[actor] = calculate_velocity(actor)

        for actor in CarlaDataProvider._actor_location_map:
            if actor is not None and actor.is_alive:
                CarlaDataProvider._actor_location_map[actor] = actor.get_location()

        for actor in CarlaDataProvider._actor_transform_map:
            if actor is not None and actor.is_alive:
                CarlaDataProvider._actor_transform_map[actor] = actor.get_transform()

        world = CarlaDataProvider._world
        if world is None:
            print("WARNING: CarlaDataProvider couldn't find the world")

    @staticmethod
    def get_velocity(actor):
        """
            returns the absolute velocity for the given actor
        """
        for key in CarlaDataProvider._actor_velocity_map:
            if key.id == actor.id:
                return CarlaDataProvider._actor_velocity_map[key]

        # We are intentionally not throwing here
        # __name__是当前模块的名字, actor是传入的actor对象
        print('{}.get_velocity: {} not found!' .format(__name__, actor))
        return 0.0

    @staticmethod
    def get_location(actor):
        """
            returns the location for the given actor
        """
        for key in CarlaDataProvider._actor_location_map:
            if key.id == actor.id:
                return CarlaDataProvider._actor_location_map[key]

        # We are intentionally not throwing here
        print('{}.get_location: {} not found!' .format(__name__, actor))
        return None

    @staticmethod
    def get_transform(actor):
        """
            returns the transform for the given actor
        """
        for key in CarlaDataProvider._actor_transform_map:
            if key.id == actor.id:
                return CarlaDataProvider._actor_transform_map[key]

        # We are intentionally not throwing here
        print('{}.get_transform: {} not found!' .format(__name__, actor))
        return None

    @staticmethod
    def set_client(client):
        """
            Set the CARLA client
        """
        CarlaDataProvider._client = client

    @staticmethod
    def get_client():
        """
            Get the CARLA client
        """
        return CarlaDataProvider._client

    @staticmethod
    def set_world(world):
        """
            Set the world and world settings
        """
        CarlaDataProvider._world = world
        CarlaDataProvider._sync_flag = world.get_settings().synchronous_mode
        CarlaDataProvider._map = world.get_map()
        CarlaDataProvider._blueprint_library = world.get_blueprint_library()
        CarlaDataProvider.generate_spawn_points()
        # 设置当前地图并加载该地图的所有交通信号灯到 _traffic_light_map 字典中
        CarlaDataProvider.prepare_map()

    @staticmethod
    def get_world():
        """
            Return world
        """
        return CarlaDataProvider._world

    @staticmethod
    def get_map(world=None):
        """
            Get the current map
        """
        if CarlaDataProvider._map is None:
            if world is None:
                if CarlaDataProvider._world is None:
                    raise ValueError("class member \'world'\' not initialized yet")
                else:
                    CarlaDataProvider._map = CarlaDataProvider._world.get_map()
            else:
                CarlaDataProvider._map = world.get_map()

        return CarlaDataProvider._map

    @staticmethod
    def get_random_seed():
        """
            return true if synchronous mode is used
        """
        return CarlaDataProvider._rng

    @staticmethod
    def is_sync_mode():
        """
            return true if synchronous mode is used
        """
        return CarlaDataProvider._sync_flag

    @staticmethod
    def find_weather_presets():
        """Get weather presets from CARLA"""
        rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
        name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
        presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

    @staticmethod
    def prepare_map():
        """
            This function set the current map and loads all traffic lights for this map to _traffic_light_map
        """
        if CarlaDataProvider._map is None:
            CarlaDataProvider._map = CarlaDataProvider._world.get_map()

        # Parse all traffic lights
        CarlaDataProvider._traffic_light_map.clear()
        for traffic_light in CarlaDataProvider._world.get_actors().filter('*traffic_light*'):
            if traffic_light not in CarlaDataProvider._traffic_light_map.keys():
                CarlaDataProvider._traffic_light_map[traffic_light] = traffic_light.get_transform()
            else:
                raise KeyError("Traffic light '{}' already registered. Cannot register twice!".format(traffic_light.id))

    @staticmethod
    def annotate_trafficlight_in_group(traffic_light):
        """
        获取给定交通信号灯的字典
        """
        dict_annotations = {'ref': [], 'opposite': [], 'left': [], 'right': []}

        # Get the waypoints
        ref_location = CarlaDataProvider.get_trafficlight_trigger_location(traffic_light)
        ref_waypoint = CarlaDataProvider.get_map().get_waypoint(ref_location)
        ref_yaw = ref_waypoint.transform.rotation.yaw

        group_tl = traffic_light.get_group_traffic_lights()
        for target_tl in group_tl:
            if traffic_light.id == target_tl.id:
                dict_annotations['ref'].append(target_tl)
            else:
                # Get the angle between yaws
                target_location = CarlaDataProvider.get_trafficlight_trigger_location(target_tl)
                target_waypoint = CarlaDataProvider.get_map().get_waypoint(target_location)
                target_yaw = target_waypoint.transform.rotation.yaw

                diff = (target_yaw - ref_yaw) % 360
                if diff > 330:
                    continue
                elif diff > 225:
                    dict_annotations['right'].append(target_tl)
                elif diff > 135.0:
                    dict_annotations['opposite'].append(target_tl)
                elif diff > 30:
                    dict_annotations['left'].append(target_tl)

        return dict_annotations

    @staticmethod
    def get_trafficlight_trigger_location(traffic_light):  # pylint: disable=invalid-name (忽略变量、函数或类的名称不符合规范)
        """
            Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """
        def rotate_point(point, angle):
            """计算交通信号灯触发区域的一个特定点的位置，并返回"""
            x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
            y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
            return carla.Vector3D(x_, y_, point.z)

        # 获取交通信号灯的位置和角度
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw

        # 计算触发区域的位置和范围
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), base_rot)
        point_location = area_loc + carla.Location(x=point.x, y=point.y)
        return carla.Location(point_location.x, point_location.y, point_location.z)

    @staticmethod
    def update_light_states(ego_light, annotations, states, freeze=False, timeout=1000000000):
        """
        用于存储每个交通信号灯的原始状态和时间
        """
        reset_params = []

        for state in states:
            relevant_lights = []
            if state == 'ego':
                relevant_lights = [ego_light]
            else:
                relevant_lights = annotations[state]
            # 遍历每个交通信号灯
            for light in relevant_lights:
                # 获取并存储交通信号灯的当前状态、绿灯时间、红灯时间和黄灯时间
                prev_state = light.get_state()
                prev_green_time = light.get_green_time()
                prev_red_time = light.get_red_time()
                prev_yellow_time = light.get_yellow_time()
                reset_params.append({
                    'light': light,
                    'state': prev_state,
                    'green_time': prev_green_time,
                    'red_time': prev_red_time,
                    'yellow_time': prev_yellow_time
                })
                # 设置交通信号灯的新状态
                light.set_state(states[state])
                # 如果freeze为True，则将交通信号灯的绿灯时间、红灯时间和黄灯时间设为一个相当大的值，从而冻结交通信号灯的状态
                if freeze:
                    light.set_green_time(timeout)
                    light.set_red_time(timeout)
                    light.set_yellow_time(timeout)

        return reset_params

    @staticmethod
    def reset_lights(reset_params):
        """
            Reset traffic lights
        """
        for param in reset_params:
            param['light'].set_state(param['state'])
            param['light'].set_green_time(param['green_time'])
            param['light'].set_red_time(param['red_time'])
            param['light'].set_yellow_time(param['yellow_time'])

    @staticmethod
    def get_next_traffic_light(actor, use_cached_location=True, use_transform=False):
        """
            returns the next relevant traffic light for the provided actor
        """

        if use_transform:
            location = actor.location
        else:
            if not use_cached_location:
                location = actor.get_transform().location
            else:
                location = CarlaDataProvider.get_location(actor)

        waypoint = CarlaDataProvider.get_map().get_waypoint(location)
        # Create list of all waypoints until next intersection
        list_of_waypoints = []
        while waypoint and not waypoint.is_intersection:
            list_of_waypoints.append(waypoint)
            waypoint = waypoint.next(2.0)[0]

        # If the list is empty, the actor is in an intersection
        if not list_of_waypoints:
            return None

        relevant_traffic_light = None
        distance_to_relevant_traffic_light = float("inf")

        for traffic_light in CarlaDataProvider._traffic_light_map:
            if hasattr(traffic_light, 'trigger_volume'):
                tl_t = CarlaDataProvider._traffic_light_map[traffic_light]
                transformed_tv = tl_t.transform(traffic_light.trigger_volume.location)
                # 用于确定哪个交通信号灯是车辆路径中下一个交叉口的交通信号灯
                distance = carla.Location(transformed_tv).distance(list_of_waypoints[-1].transform.location)

                if distance < distance_to_relevant_traffic_light:
                    relevant_traffic_light = traffic_light
                    distance_to_relevant_traffic_light = distance

        return relevant_traffic_light

    @staticmethod
    def set_ego_vehicle_route(route):
        """
            Set the route of the ego vehicle
        """
        CarlaDataProvider._ego_vehicle_route = route

    @staticmethod
    def get_ego_vehicle_route():
        """
            returns the currently set route of the ego vehicle
            Note: Can be None
        """
        return CarlaDataProvider._ego_vehicle_route

    @staticmethod
    def generate_spawn_points():
        """
            Generate spawn points for the current map
        """
        # 生成并随机排列地图中的所有可行生成点
        spawn_points = list(CarlaDataProvider.get_map(CarlaDataProvider._world).get_spawn_points())
        CarlaDataProvider._rng.shuffle(spawn_points)
        CarlaDataProvider._spawn_points = spawn_points
        CarlaDataProvider._spawn_index = 0

    @staticmethod
    def create_blueprint(model, rolename='scenario', color=None, actor_category="car", safe=False):
        """
            Function to setup the blueprint of an actor given its model and other relevant parameters
        """

        _actor_blueprint_categories = {
            'car': 'vehicle.tesla.model3',
            'van': 'vehicle.volkswagen.t2',
            'truck': 'vehicle.carlamotors.carlacola',
            'trailer': '',
            'semitrailer': '',
            'bus': 'vehicle.volkswagen.t2',
            'motorbike': 'vehicle.kawasaki.ninja',
            'bicycle': 'vehicle.diamondback.century',
            'train': '',
            'tram': '',
            'pedestrian': 'walker.pedestrian.0001',
        }

        # Set the model
        try:
            blueprints = CarlaDataProvider._blueprint_library.filter(model)
            blueprints_ = []  # 初始化一个空列表，用于存储筛选后的蓝图
            if safe:
                for bp in blueprints:
                    if bp.id.endswith('firetruck') or bp.id.endswith('ambulance') or int(bp.get_attribute('number_of_wheels')) != 4:
                        # Two wheeled vehicles take much longer to render + bicicles shouldn't behave like cars
                        continue
                    blueprints_.append(bp)
            else:
                blueprints_ = blueprints
            # 从筛选后的蓝图中随机选择一个蓝图
            blueprint = CarlaDataProvider._rng.choice(blueprints_)
        except ValueError:
            # The model is not part of the blueprint library. Let's take a default one for the given category
            bp_filter = "vehicle.*"
            new_model = _actor_blueprint_categories[actor_category]
            if new_model != '':
                bp_filter = new_model
            print("WARNING: Actor model {} not available. Using instead {}".format(model, new_model))
            blueprint = CarlaDataProvider._rng.choice(CarlaDataProvider._blueprint_library.filter(bp_filter))

        # Set the color
        if color:
            if not blueprint.has_attribute('color'):
                print("WARNING: Cannot set Color ({}) for actor {} due to missing blueprint attribute".format(color, blueprint.id))
            else:
                default_color_rgba = blueprint.get_attribute('color').as_color()
                default_color = '({}, {}, {})'.format(default_color_rgba.r, default_color_rgba.g, default_color_rgba.b)
                try:
                    blueprint.set_attribute('color', color)
                except ValueError:
                    # Color can't be set for this vehicle
                    print("WARNING: Color ({}) cannot be set for actor {}. Using instead: ({})".format(color, blueprint.id, default_color))
                    blueprint.set_attribute('color', default_color)
        else:
            if blueprint.has_attribute('color') and rolename != 'hero':
                color = CarlaDataProvider._rng.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)

        # Keep pedestrians invincible so background NPC vehicles cannot kill them
        # mid-scenario; a dead-walker apply_control_to_walker RPC corrupts the
        # CARLA ASIO session state and causes a server SIGSEGV.
        # (原来设置 is_invincible='false' 导致背景NPC车辆撞死行人，
        #  之后 walker_go_straight 继续调用 apply_control_to_walker 打到已销毁
        #  actor，服务端 ASIO 状态损坏引发 SIGSEGV)
        # if blueprint.has_attribute('is_invincible'):
        #     blueprint.set_attribute('is_invincible', 'false')

        # Set the rolename
        if blueprint.has_attribute('role_name'):
            blueprint.set_attribute('role_name', rolename)

        return blueprint

    @staticmethod
    def handle_actor_batch(batch, tick=True):
        """
            Forward a CARLA command batch to spawn actors to CARLA, and gather the responses.
            Returns list of actors on success, none otherwise
        """
        sync_mode = CarlaDataProvider.is_sync_mode()
        actors = []

        if CarlaDataProvider._client:
            # Always pass due_tick_cue=False so apply_batch_sync does NOT tick internally.
            # We perform the tick explicitly below to avoid a double-tick in sync mode
            # (apply_batch_sync(batch, True) + world.tick() would advance the sim twice).
            responses = CarlaDataProvider._client.apply_batch_sync(batch, False)
        else:
            raise ValueError("class member \'client'\' not initialized yet")

        # Wait (or not) for the actors to be spawned properly before we do anything
        if not tick:
            pass
        elif sync_mode:
            # 在同步模式下, 每次调用world.tick()都会使仿真器前进一帧
            CarlaDataProvider._world.tick()
        else:
            # 在异步模式下, 调用这个函数会阻塞当前线程,直到仿真器完成一个时间步长
            CarlaDataProvider._world.wait_for_tick()

        actor_ids = [r.actor_id for r in responses if not r.error]
        failed_responses = [r for r in responses if r.error]
        if failed_responses:
            print(f"WARNING: {len(failed_responses)}/{len(responses)} actors failed to spawn")
            for r in failed_responses:
                print(f"  spawn error: {r.error}")
            # In sync mode, an extra tick flushes Unreal's pending destroy commands for
            # partially-allocated physics bodies that result from failed SpawnActor calls.
            # Without this flush, ghost physics bodies can stall subsequent world.tick() calls.
            if tick and sync_mode and CarlaDataProvider._world is not None:
                try:
                    CarlaDataProvider._world.tick()
                except Exception as _extra_err:
                    print(f"WARNING: Extra cleanup tick after spawn failure: {_extra_err}")
        actors = list(CarlaDataProvider._world.get_actors(actor_ids))
        return actors

    @staticmethod
    def request_new_actor(
        model,
        spawn_point,
        rolename='scenario',
        autopilot=False,
        random_location=False,
        color=None,
        actor_category="car",
        safe_blueprint=False,
        tick=True
    ):
        """
            This method tries to create a new actor, returning it if successful (None otherwise).
        """
        blueprint = CarlaDataProvider.create_blueprint(model, rolename, color, actor_category, safe_blueprint)

        if random_location:
            actor = None
            while not actor:
                spawn_point = CarlaDataProvider._rng.choice(CarlaDataProvider._spawn_points)
                actor = CarlaDataProvider._world.try_spawn_actor(blueprint, spawn_point)
        else:
            # slightly lift the actor to avoid collisions with ground when spawning the actor
            # DO NOT USE spawn_point directly, as this will modify spawn_point permanently
            _spawn_point = carla.Transform(carla.Location(), spawn_point.rotation)
            _spawn_point.location.x = spawn_point.location.x
            _spawn_point.location.y = spawn_point.location.y
            _spawn_point.location.z = spawn_point.location.z + 0.2
            actor = CarlaDataProvider._world.try_spawn_actor(blueprint, _spawn_point)

        if actor is None:
            raise RuntimeError("Error: Cannot spawn actor {} at position {}".format(model, spawn_point.location))

        # De/activate the autopilot of the actor if it belongs to vehicle
        if autopilot:
            if actor.type_id.startswith('vehicle.'):
                actor.set_autopilot(autopilot, CarlaDataProvider._traffic_manager_port)
            else:
                print("WARNING: Tried to set the autopilot of a non vehicle actor")

        # Wait for the actor to be spawned properly before we do anything
        if not tick:
            pass
        elif CarlaDataProvider.is_sync_mode():
            CarlaDataProvider._world.tick()
        else:
            CarlaDataProvider._world.wait_for_tick()

        CarlaDataProvider._carla_actor_pool[actor.id] = actor
        CarlaDataProvider.register_actor(actor)

        return actor

    @staticmethod
    def request_new_actors(actor_list, safe_blueprint=False, tick=True):
        """
            This method tries to series of actor in batch. If this was successful, the new actors are returned, None otherwise.
                actor_list: list of ActorConfigurationData
        """
        SpawnActor = carla.command.SpawnActor
        PhysicsCommand = carla.command.SetSimulatePhysics
        FutureActor = carla.command.FutureActor
        ApplyTransform = carla.command.ApplyTransform
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState

        batch = []
        CarlaDataProvider.generate_spawn_points()
        for actor in actor_list:
            # Get the blueprint
            blueprint = CarlaDataProvider.create_blueprint(actor.model, actor.rolename, actor.color, actor.category, safe_blueprint)

            # Get the spawn point
            transform = actor.transform
            if actor.random_location:
                if CarlaDataProvider._spawn_index >= len(CarlaDataProvider._spawn_points):
                    print("No more spawn points to use")
                    break
                else:
                    _spawn_point = CarlaDataProvider._spawn_points[CarlaDataProvider._spawn_index]
                    CarlaDataProvider._spawn_index += 1
            else:
                _spawn_point = carla.Transform()
                _spawn_point.rotation = transform.rotation
                _spawn_point.location.x = transform.location.x
                _spawn_point.location.y = transform.location.y
                if blueprint.has_tag('walker'):
                    # 在导入的OpenDRIVE地图上，生成行人可能会失败, 过增加z值可以提高成功率。
                    map_name = CarlaDataProvider._map.name.split("/")[-1]
                    if not map_name.startswith('OpenDrive'):
                        _spawn_point.location.z = transform.location.z + 0.2
                    else:
                        _spawn_point.location.z = transform.location.z + 0.8
                else:
                    _spawn_point.location.z = transform.location.z + 0.2

            # 获取命令
            command = SpawnActor(blueprint, _spawn_point)
            command.then(SetAutopilot(FutureActor, actor.autopilot, CarlaDataProvider._traffic_manager_port))

            if actor.args is not None and 'physics' in actor.args and actor.args['physics'] == "off":
                command.then(ApplyTransform(FutureActor, _spawn_point)).then(PhysicsCommand(FutureActor, False))
            elif actor.category == 'misc':
                command.then(PhysicsCommand(FutureActor, True))
            if actor.args is not None and 'lights' in actor.args and actor.args['lights'] == "on":
                command.then(SetVehicleLightState(FutureActor, carla.VehicleLightState.All))

            batch.append(command)

        actors = CarlaDataProvider.handle_actor_batch(batch, tick)
        for actor in actors:
            if actor is None:
                continue
            CarlaDataProvider._carla_actor_pool[actor.id] = actor
            CarlaDataProvider.register_actor(actor)

        return actors

    @staticmethod
    def request_new_batch_actors(
            model,
            ego_vehicle,
            amount,
            autopilot=False,
            random_location=False,
            rolename='scenario',
            safe_blueprint=False,
            tick=True
    ):
        """
            Simplified version of "request_new_actors". This method also create several actors in batch.
            Instead of needing a list of ActorConfigurationData, an "amount" parameter is used.
            This makes actor spawning easier but reduces the amount of configurability.
            Some parameters are the same for all actors (rolename, autopilot and random location) while others are randomized (color)
        """

        # 首先简化定义一些CARLA命令
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        CarlaDataProvider.generate_spawn_points()

        # If ego_vehicle provided, prefer waypoints within 200m of ego as spawn points
        waypoints = CarlaDataProvider._spawn_points
        nearby_transforms = []
        if ego_vehicle is not None:
            try:
                ego_loc = ego_vehicle.get_location()
                for wp in waypoints:
                    distance = math.sqrt(
                        (wp.location.x - ego_loc.x) ** 2 +
                        (wp.location.y - ego_loc.y) ** 2 +
                        (wp.location.z - ego_loc.z) ** 2
                    )
                    if distance <= 150.0:
                        nearby_transforms.append(wp)

                if nearby_transforms:
                    spawn_points_local = nearby_transforms
                else:
                    # fallback to map spawn points if no nearby waypoints found
                    spawn_points_local = waypoints
            except Exception:
                # keep given spawn_points on any error
                spawn_points_local = waypoints

        # Pre-filter: remove spawn points that already have a vehicle within 5 m.
        # This avoids certain-collision spawns that can leave ghost physics bodies
        # in Unreal when SpawnActor's FindTeleportSpot fails to place the actor.
        try:
            existing_vehicles = CarlaDataProvider._world.get_actors().filter('vehicle.*')
            occupied_locs = [v.get_location() for v in existing_vehicles]
            _min_dist_sq = 5.0 * 5.0

            def _spawn_point_free(sp):
                for _loc in occupied_locs:
                    dx = sp.location.x - _loc.x
                    dy = sp.location.y - _loc.y
                    if dx * dx + dy * dy < _min_dist_sq:
                        return False
                return True

            filtered = [sp for sp in spawn_points_local if _spawn_point_free(sp)]
            if filtered:
                spawn_points_local = filtered
            # else: all points appear occupied — keep original list as last resort
        except Exception:
            pass  # keep original list if the pre-check itself fails

        batch = []
        for idx in range(amount):
            blueprint = CarlaDataProvider.create_blueprint(model, rolename, safe=safe_blueprint)
            if random_location:
                if CarlaDataProvider._spawn_index >= len(CarlaDataProvider._spawn_points):
                    print("No more spawn points to use. Spawned {} actors out of {}".format(idx + 1, amount))
                    break
                else:
                    spawn_point = CarlaDataProvider._spawn_points[CarlaDataProvider._spawn_index]
                    CarlaDataProvider._spawn_index += 1
            else:
                if CarlaDataProvider._spawn_index >= len(spawn_points_local):
                    print("No more spawn points to use. Spawned {} actors out of {}".format(idx + 1, amount))
                    break
                else:
                    spawn_point = spawn_points_local[CarlaDataProvider._spawn_index]
                    CarlaDataProvider._spawn_index += 1

            if spawn_point:
                batch.append(SpawnActor(blueprint, spawn_point).then(
                    SetAutopilot(FutureActor, autopilot, CarlaDataProvider._traffic_manager_port)))

        actors = CarlaDataProvider.handle_actor_batch(batch, tick)
        for actor in actors:
            if actor is None:
                continue
            CarlaDataProvider._carla_actor_pool[actor.id] = actor
            CarlaDataProvider.register_actor(actor)
        return actors

    @staticmethod
    def get_actors():
        """
            Return list of actors and their ids
            Note: iteritems from six is used to allow compatibility with Python 2 and 3
        """
        return iteritems(CarlaDataProvider._carla_actor_pool)

    @staticmethod
    def actor_id_exists(actor_id):
        """
            Check if a certain id is still at the simulation
        """
        if actor_id in CarlaDataProvider._carla_actor_pool:
            return True

        return False

    @staticmethod
    def _get_world_actor_ids():
        """
            Best-effort snapshot of actor ids that still exist in the current CARLA world.
        """
        world = CarlaDataProvider._world
        if world is None:
            return None

        try:
            return {actor.id for actor in world.get_actors()}
        except Exception:
            return None

    @staticmethod
    def _drop_actor_tracking(actor_id, actor=None):
        """
            Remove stale local references for a single actor without touching the world.
        """
        if actor is None:
            actor = CarlaDataProvider._carla_actor_pool.get(actor_id)

        if actor is not None:
            CarlaDataProvider._actor_velocity_map.pop(actor, None)
            CarlaDataProvider._actor_location_map.pop(actor, None)
            CarlaDataProvider._actor_transform_map.pop(actor, None)

        CarlaDataProvider._carla_actor_pool.pop(actor_id, None)

    @staticmethod
    def remove_actors_by_ids(actor_ids):
        """
            Remove a list of actors from the pool using a single CARLA batch call.

            Using a background Python thread to call actor.destroy() can leave dangling
            RPC requests alive across scenario transitions. That pattern is a strong
            candidate for destabilizing CARLA's server_session teardown. We therefore
            keep destroy calls on the main thread and prefer a single batch request.
        """
        unique_actor_ids = []
        seen_ids = set()
        for actor_id in actor_ids:
            if actor_id is None or actor_id in seen_ids:
                continue
            seen_ids.add(actor_id)
            unique_actor_ids.append(actor_id)

        if not unique_actor_ids:
            return []

        world_actor_ids = CarlaDataProvider._get_world_actor_ids()
        valid_actor_ids = []
        for actor_id in unique_actor_ids:
            actor = CarlaDataProvider._carla_actor_pool.get(actor_id)
            if actor is None:
                CarlaDataProvider._drop_actor_tracking(actor_id, actor=None)
                continue
            if world_actor_ids is not None and actor_id not in world_actor_ids:
                CarlaDataProvider._drop_actor_tracking(actor_id, actor=actor)
                continue
            valid_actor_ids.append(actor_id)

        if not valid_actor_ids:
            return []

        destroyed_actor_ids = []

        if CarlaDataProvider._client is not None:
            batch = [carla.command.DestroyActor(actor_id) for actor_id in valid_actor_ids]
            try:
                responses = CarlaDataProvider._client.apply_batch_sync(batch)
            except RuntimeError as error:
                if "time-out" in str(error):
                    print(
                        "WARNING: CARLA timed out while destroying actor batch {}".format(
                            valid_actor_ids
                        )
                    )
                    return []
                raise

            for actor_id, response in zip(valid_actor_ids, responses):
                actor = CarlaDataProvider._carla_actor_pool.get(actor_id)
                if response.error:
                    error_text = response.error.lower()
                    print(f"WARNING: Actor destroy failed for ID {actor_id}: {response.error}")
                    if "not found" in error_text or "invalid" in error_text:
                        CarlaDataProvider._drop_actor_tracking(actor_id, actor=actor)
                    continue

                CarlaDataProvider._drop_actor_tracking(actor_id, actor=actor)
                destroyed_actor_ids.append(actor_id)

            return destroyed_actor_ids

        for actor_id in valid_actor_ids:
            actor = CarlaDataProvider._carla_actor_pool.get(actor_id)
            if actor is None:
                CarlaDataProvider._drop_actor_tracking(actor_id, actor=None)
                continue
            try:
                actor.destroy()
            except RuntimeError as error:
                if "time-out" in str(error):
                    print(f"WARNING: Actor destroy timed out for ID {actor_id}")
                    continue
                print(f"WARNING: Actor destroy failed for ID {actor_id}: {error}")
                continue
            except Exception as error:
                print(f"WARNING: Actor destroy failed for ID {actor_id}: {error}")
                continue

            CarlaDataProvider._drop_actor_tracking(actor_id, actor=actor)
            destroyed_actor_ids.append(actor_id)

        return destroyed_actor_ids

    @staticmethod
    def get_hero_actor():
        """
            Get the actor object of the hero actor if it exists, returns none otherwise.
        """
        for actor_id in CarlaDataProvider._carla_actor_pool:
            if CarlaDataProvider._carla_actor_pool[actor_id].attributes['role_name'] == 'hero':
                return CarlaDataProvider._carla_actor_pool[actor_id]
        return None

    @staticmethod
    def get_actor_by_id(actor_id):
        """
            Get an actor from the pool by using its ID. If the actor does not exist, None is returned.
        """
        if actor_id in CarlaDataProvider._carla_actor_pool:
            return CarlaDataProvider._carla_actor_pool[actor_id]
        print("Non-existing actor id {}".format(actor_id))
        return None

    @staticmethod
    def remove_actor_by_id(actor_id):
        """
            Remove an actor from the pool using its ID
        """
        if actor_id in CarlaDataProvider._carla_actor_pool:
            CarlaDataProvider.remove_actors_by_ids([actor_id])
        else:
            print("Trying to remove a non-existing actor id {}".format(actor_id))

    @staticmethod
    def remove_actors_in_surrounding(location, distance):
        """
            Remove all actors from the pool that are closer than distance to the provided location
        """
        actor_ids_to_remove = []
        for actor_id, actor in list(CarlaDataProvider._carla_actor_pool.items()):
            try:
                if actor is not None and actor.get_location().distance(location) < distance:
                    actor_ids_to_remove.append(actor_id)
            except Exception:
                continue

        CarlaDataProvider.remove_actors_by_ids(actor_ids_to_remove)

    @staticmethod
    def get_traffic_manager_port():
        """
            Get the port of the traffic manager.
        """
        return CarlaDataProvider._traffic_manager_port

    @staticmethod
    def set_scenario_config(config):
        CarlaDataProvider._scenario_config = config

    @staticmethod
    def set_traffic_manager_port(tm_port):
        """
            Set the port to use for the traffic manager.
        """
        CarlaDataProvider._traffic_manager_port = tm_port

    @staticmethod
    def cleanup():
        """
            Cleanup and remove all entries from all dictionaries
        """
        DestroyActor = carla.command.DestroyActor       # pylint: disable=invalid-name
        batch = []

        for actor_id in CarlaDataProvider._carla_actor_pool.copy():
            actor = CarlaDataProvider._carla_actor_pool[actor_id]
            if actor is not None and actor.is_alive:
                batch.append(DestroyActor(actor))

        if CarlaDataProvider._client:
            try:
                CarlaDataProvider._client.apply_batch_sync(batch)
            except RuntimeError as e:
                if "time-out" in str(e):
                    pass
                else:
                    raise e

        CarlaDataProvider._actor_velocity_map.clear()
        CarlaDataProvider._actor_location_map.clear()
        CarlaDataProvider._actor_transform_map.clear()
        CarlaDataProvider._traffic_light_map.clear()
        CarlaDataProvider._map = None
        CarlaDataProvider._world = None
        CarlaDataProvider._sync_flag = False
        CarlaDataProvider._ego_vehicle_route = None
        CarlaDataProvider._carla_actor_pool = {}
        CarlaDataProvider._client = None
        CarlaDataProvider._spawn_points = None
        CarlaDataProvider._spawn_index = 0
        CarlaDataProvider._rng = random.RandomState(CarlaDataProvider._random_seed)

    @staticmethod
    def clear_actor_tracking_maps():
        """
            Remove dead or orphaned actor references from local tracking structures.
            This is lighter than a full cleanup and is safe to call between scenarios.
        """
        world_actor_ids = None
        if CarlaDataProvider._world is not None:
            try:
                world_actor_ids = {actor.id for actor in CarlaDataProvider._world.get_actors()}
            except Exception:
                world_actor_ids = None

        tracked_actors = list(CarlaDataProvider._actor_velocity_map.keys())
        for actor in tracked_actors:
            actor_id = getattr(actor, 'id', None)
            should_remove = (
                actor is None
                or not getattr(actor, 'is_alive', False)
                or (world_actor_ids is not None and actor_id not in world_actor_ids)
            )
            if should_remove:
                CarlaDataProvider._actor_velocity_map.pop(actor, None)
                CarlaDataProvider._actor_location_map.pop(actor, None)
                CarlaDataProvider._actor_transform_map.pop(actor, None)

        for actor_id, actor in list(CarlaDataProvider._carla_actor_pool.items()):
            should_remove = (
                actor is None
                or not getattr(actor, 'is_alive', False)
                or (world_actor_ids is not None and actor_id not in world_actor_ids)
            )
            if should_remove:
                CarlaDataProvider._carla_actor_pool.pop(actor_id, None)
