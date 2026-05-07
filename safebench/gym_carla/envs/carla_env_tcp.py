"""
为TCP自动驾驶算法定制的CarlaEnv环境
基于原始carla_env.py，添加了TCP所需的传感器配置

主要变更:
1. 相机配置改为TCP期望的尺寸 (900x256, FOV=100)
2. 相机位置调整为 (x=-1.5, z=2.0)
3. 添加了GPS和IMU传感器（可选，因为可以从ego_vehicle直接获取）
4. 扩展obs以包含原始尺寸相机图像
"""

import random
import numpy as np
import pygame
from skimage.transform import resize
import gym
from gym import spaces
import carla
from safebench.gym_carla.envs.route_planner import RoutePlanner
from safebench.gym_carla.envs.misc import (
    display_to_rgb, 
    rgb_to_display_surface, 
    get_lane_dis, 
    get_pos, 
    get_preview_lane_dis
)
from safebench.scenario.scenario_definition.route_scenario import RouteScenario
from safebench.scenario.scenario_manager.scenario_manager import ScenarioManager
from safebench.scenario.tools.route_manipulation import interpolate_trajectory
from safebench.scenario.scenario_definition.atomic_criteria import Status
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


class CarlaEnvTCP(gym.Env):
    """ 
        An OpenAI-gym style interface for CARLA simulator, customized for TCP.
    """
    def __init__(self, env_params, birdeye_render=None, display=None, world=None, logger=None):
        assert world is not None, "the world passed into CarlaEnvTCP is None"

        self.config = None
        self.world = world
        self.display = display
        self.logger = logger
        self.birdeye_render = birdeye_render

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0
        self.is_running = True
        self.env_id = None
        self.ego_vehicle = None
        self.auto_ego = env_params['auto_ego']

        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None
        self.lidar_data = None
        self.lidar_height = 2.1

        # Save sync settings at init so reset() can restore them after cleanup
        # (cleanup disables sync mode temporarily; without restoration the simulation
        # runs in async mode for the new episode, making world.tick() non-blocking)
        try:
            _init_settings = self.world.get_settings()
            self._restore_sync_mode = _init_settings.synchronous_mode
            self._restore_fixed_delta = _init_settings.fixed_delta_seconds
        except Exception:
            self._restore_sync_mode = False
            self._restore_fixed_delta = None
        
        # TCP相机图像（原始尺寸）
        self.camera_img_raw = None
        
        # scenario manager
        use_scenic = True if env_params['scenario_category'] == 'scenic' else False
        # self.scenario_manager = ScenarioManager(self.logger, use_scenic=use_scenic)
        self.scenario_manager = ScenarioManager(self.logger)

        # for birdeye view and front view visualization
        self.display_size = env_params['display_size']
        self.obs_range = env_params['obs_range']
        self.d_behind = env_params['d_behind']
        self.disable_lidar = env_params['disable_lidar']

        # for env wrapper
        self.max_past_step = env_params['max_past_step']
        self.max_episode_step = env_params['max_episode_step']
        self.max_waypt = env_params['max_waypt']
        self.lidar_bin = env_params['lidar_bin']
        self.out_lane_thres = env_params['out_lane_thres']
        self.desired_speed = env_params['desired_speed']
        self.acc_max = env_params['continuous_accel_range'][1]
        self.steering_max = env_params['continuous_steer_range'][1]

        # for scenario
        self.ROOT_DIR = env_params['ROOT_DIR']
        self.scenario_category = env_params['scenario_category']
        self.warm_up_steps = env_params['warm_up_steps']

        # TCP相机配置
        self.tcp_camera_width = 900
        self.tcp_camera_height = 256
        self.tcp_camera_fov = 100
        self.running_results = {}
        if self.scenario_category in ['planning', 'scenic']:
            self.obs_size = int(self.obs_range/self.lidar_bin)
            observation_space_dict = {
                'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'camera_raw': spaces.Box(low=0, high=255, shape=(self.tcp_camera_height, self.tcp_camera_width, 3), dtype=np.uint8),
                'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'state': spaces.Box(np.array([-2, -1, -5, 0], dtype=np.float32), np.array([2, 1, 30, 1], dtype=np.float32), dtype=np.float32)
            }
        elif self.scenario_category == 'perception':
            self.obs_size = env_params['image_sz']
            observation_space_dict = {
                'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'camera_raw': spaces.Box(low=0, high=255, shape=(self.tcp_camera_height, self.tcp_camera_width, 3), dtype=np.uint8),
            }
        else:
            raise ValueError(f'Unknown scenario category: {self.scenario_category}')

        # define obs space
        self.observation_space = spaces.Dict(observation_space_dict)

        # action and observation spaces
        self.discrete = env_params['discrete']
        self.discrete_act = [env_params['discrete_acc'], env_params['discrete_steer']]
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        if self.discrete:
            self.action_space = spaces.Discrete(self.n_acc * self.n_steer)
        else:
            self.action_space = spaces.Box(np.array([-1, -1], dtype=np.float32), np.array([1, 1], dtype=np.float32), dtype=np.float32)

    def _create_sensors(self):
        """创建传感器，使用TCP的相机配置"""
        # collision sensor
        self.collision_hist_l = 1
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        
        if self.scenario_category != 'perception':
            # lidar sensor
            self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
            self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            self.lidar_bp.set_attribute('channels', '16')
            self.lidar_bp.set_attribute('range', '1000')
        
        # TCP相机传感器 - 使用TCP的配置
        self.camera_img = np.zeros((self.tcp_camera_height, self.tcp_camera_width, 3), dtype=np.uint8)
        self.camera_img_raw = np.zeros((self.tcp_camera_height, self.tcp_camera_width, 3), dtype=np.uint8)
        
        # TCP相机位置 (x=-1.5, z=2.0)
        self.camera_trans = carla.Transform(carla.Location(x=-1.5, z=2.0))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        
        # TCP相机参数 (900x256, FOV=100)
        self.camera_bp.set_attribute('image_size_x', str(self.tcp_camera_width))
        self.camera_bp.set_attribute('image_size_y', str(self.tcp_camera_height))
        self.camera_bp.set_attribute('fov', str(self.tcp_camera_fov))
        self.camera_bp.set_attribute('sensor_tick', '0.02')

    def _create_scenario(self, config, env_id):
        self.logger.log(f">> Loading scenario data id: {config.data_id}")

        scenario = RouteScenario(
            world=self.world,
            config=config,
            ego_id=env_id,
            max_running_step=self.max_episode_step,
            logger=self.logger
        )

        self.ego_vehicle = scenario.ego_vehicle
        self.scenario_manager.load_scenario(scenario)

    def _run_scenario(self, scenario_init_action):
        self.scenario_manager.run_scenario(scenario_init_action)

    def _parse_route(self, config):
        route = interpolate_trajectory(self.world, config.trajectory)
        waypoints_list = []
        carla_map = self.world.get_map()
        for node in route:
            loc = node[0].location
            waypoint = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            waypoints_list.append(waypoint)
        return waypoints_list

    def get_static_obs(self, config):
        route = interpolate_trajectory(self.world, config.trajectory, 5.0)

        waypoint_xy = []
        for transform_tuple in route:
            waypoint_xy.append([transform_tuple[0].location.x, transform_tuple[0].location.y])

        if config.parameters is not None and len(config.parameters) > 2:
            self.desired_speed = config.parameters[2]

        state = {
            'route': np.array(waypoint_xy),
            'target_speed': self.desired_speed,
        }

        return state

    def reset(self, config, env_id, scenario_init_action):
        self.config = config
        self.env_id = env_id

        # Snapshot sync settings RIGHT BEFORE cleanup so we can restore them
        # after cleanup disables sync mode.  _restore_sync_mode captured at
        # __init__ is unreliable because sync mode is typically enabled AFTER
        # the env object is constructed, so it was always False there.
        _was_sync = False
        _was_delta = None
        try:
            _pre_settings = self.world.get_settings()
            _was_sync = _pre_settings.synchronous_mode
            _was_delta = _pre_settings.fixed_delta_seconds
        except Exception:
            pass

        self._cleanup_runtime_actors(prefix='Reset pre-cleanup')

        # Re-enable synchronous mode that cleanup just disabled.  In async mode
        # world.tick() returns immediately without waiting for the server, so the
        # inter-sensor tick in _attach_sensor() would not advance the frame and
        # the collision + camera sensors would land in the same server frame,
        # causing the streaming fd race → close(EBADF) → SIGSEGV.
        if _was_sync:
            try:
                _settings = self.world.get_settings()
                if not _settings.synchronous_mode:
                    _settings.synchronous_mode = True
                    _settings.fixed_delta_seconds = _was_delta
                    self.world.apply_settings(_settings)
                    CarlaDataProvider._sync_flag = True
            except Exception as _sync_err:
                if self.logger is not None:
                    self.logger.log(f'>> reset: failed to restore sync mode: {_sync_err}', color='yellow')

        self._create_sensors()
        self._create_scenario(config, env_id)
        self._run_scenario(scenario_init_action)
        self._attach_sensor()

        self.route_waypoints = self._parse_route(config)
        self.routeplanner = RoutePlanner(self.ego_vehicle, self.max_waypt, self.route_waypoints)
        self.waypoints, _, _, _, _, self.vehicle_front, = self.routeplanner.run_step()

        self.vehicle_polygons = [self._get_actor_polygons('vehicle.*')]
        self.walker_polygons = [self._get_actor_polygons('walker.*')]

        vehicle_info_dict_list = self._get_actor_info('vehicle.*')
        self.vehicle_trajectories = [vehicle_info_dict_list[0]]
        self.vehicle_accelerations = [vehicle_info_dict_list[1]]
        self.vehicle_angular_velocities = [vehicle_info_dict_list[2]]
        self.vehicle_velocities = [vehicle_info_dict_list[3]]

        self.time_step = 0
        self.reset_step += 1

        self.settings = self.world.get_settings()
        self.world.apply_settings(self.settings)

        for _ in range(self.warm_up_steps):
            self.world.tick()
        return self._get_obs(), self._get_info()

    def _wait_sensor_first_data(self, attr_name, timeout_ticks=10):
        """Tick world until self.<attr_name> is not None.

        Each CARLA sensor's streaming socket is established asynchronously after
        spawn_actor() returns.  Ticking until the first data packet arrives
        confirms the socket is fully set up end-to-end, making it safe to spawn
        the next sensor without risking a streaming-fd double-close race.
        """
        import time as _time_mod
        for _ in range(timeout_ticks):
            try:
                self.world.tick()
            except Exception as _tick_err:
                if self.logger is not None:
                    self.logger.log(f'>> _wait_sensor_first_data tick failed: {_tick_err}', color='yellow')
                _time_mod.sleep(0.1)
            if getattr(self, attr_name, None) is not None:
                return

    def _attach_sensor(self):
        # ── Collision sensor ──────────────────────────────────────────────
        # Define callback before listen() so it is bound in the local scope
        # before any tick can deliver data on a background thread.
        self.collision_hist = []

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)

        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(get_collision_hist)
        # Collision sensor only fires on collision events, so we cannot wait for
        # data to confirm readiness.  A single tick gives CARLA's async
        # streaming-socket setup time to start before the next sensor is spawned.
        self.world.tick()

        # ── Lidar sensor ──────────────────────────────────────────────────
        if self.scenario_category != 'perception' and not self.disable_lidar:
            self.lidar_data = None  # reset sentinel so _wait_sensor_first_data can detect arrival

            def get_lidar_data(data):
                self.lidar_data = data

            self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego_vehicle)
            self.lidar_sensor.listen(get_lidar_data)
            # Tick until the first lidar scan arrives.  Receiving actual data
            # proves the streaming socket is fully established end-to-end, so
            # spawning the camera sensor next cannot race on the same fd.
            self._wait_sensor_first_data('lidar_data')

        # ── Camera sensor ─────────────────────────────────────────────────
        # Reset to None so _wait_sensor_first_data can detect first-frame arrival.
        self.camera_img = None

        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array
            self.camera_img_raw = array.copy()

        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego_vehicle)
        self.camera_sensor.listen(get_camera_img)
        # Tick until the first camera frame arrives — streaming socket confirmed ready.
        self._wait_sensor_first_data('camera_img')

    def step_before_tick(self, ego_action, scenario_action):
        if self.world:
            snapshot = self.world.get_snapshot()
            if snapshot:
                timestamp = snapshot.timestamp
                
                if self.scenario_category in ['perception']:
                    assert isinstance(ego_action, dict), 'ego action in ObjectDetectionScenario should be a dict'
                    world_2_camera = np.array(self.camera_sensor.get_transform().get_inverse_matrix())
                    fov = self.camera_bp.get_attribute('fov').as_float()
                    image_w, image_h = self.tcp_camera_width, self.tcp_camera_height
                    self.scenario_manager.background_scenario.evaluate(ego_action, world_2_camera, image_w, image_h, fov, self.camera_img)
                    ego_action = ego_action['ego_action']

                self.scenario_manager.get_update(timestamp, scenario_action)
                self.is_running = self.scenario_manager._running

                if not self.auto_ego:
                    if self.discrete:
                        acc = self.discrete_act[0][ego_action // self.n_steer]
                        steer = self.discrete_act[1][ego_action % self.n_steer]
                    else:
                        acc = ego_action[0]
                        steer = ego_action[1]

                    acc = acc * self.acc_max
                    steer = steer * self.steering_max
                    acc = max(min(self.acc_max, acc), -self.acc_max)
                    steer = max(min(self.steering_max, steer), -self.steering_max)

                    if acc > 0:
                        throttle = np.clip(acc / 3, 0, 1)
                        brake = 0
                    else:
                        throttle = 0
                        brake = np.clip(-acc / 8, 0, 1)

                    act = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
                    self.ego_vehicle.apply_control(act)
            else:
                self.logger.log('>> Can not get snapshot!', color='red')
                raise Exception()
        else:
            self.logger.log('>> Please specify a Carla world!', color='red')
            raise Exception()

    def step_after_tick(self):
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
            self.vehicle_polygons.pop(0)
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > self.max_past_step:
            self.walker_polygons.pop(0)

        vehicle_info_dict_list = self._get_actor_info('vehicle.*')
        self.vehicle_trajectories.append(vehicle_info_dict_list[0])
        while len(self.vehicle_trajectories) > self.max_past_step:
            self.vehicle_trajectories.pop(0)
        self.vehicle_accelerations.append(vehicle_info_dict_list[1])
        while len(self.vehicle_accelerations) > self.max_past_step:
            self.vehicle_accelerations.pop(0)
        self.vehicle_angular_velocities.append(vehicle_info_dict_list[2])
        while len(self.vehicle_angular_velocities) > self.max_past_step:
            self.vehicle_angular_velocities.pop(0)
        self.vehicle_velocities.append(vehicle_info_dict_list[3])
        while len(self.vehicle_velocities) > self.max_past_step:
            self.vehicle_velocities.pop(0)

        self.waypoints, _, _, _, _, self.vehicle_front, = self.routeplanner.run_step()

        self.time_step += 1
        self.total_step += 1
        return (self._get_obs(), self._get_scenario_reward(), self._terminal(), self._get_info())

    def _get_info(self):
        info = {
            'waypoints': self.waypoints,
            'route_waypoints': self.route_waypoints,
            'vehicle_front': self.vehicle_front,
            'cost': self._get_cost()
        }
        info.update(self.scenario_manager.background_scenario.update_info())
        return info

    def _init_traffic_light(self):
        actor_list = self.world.get_actors()
        for actor in actor_list:
            if isinstance(actor, carla.TrafficLight):
                actor.set_red_time(3)
                actor.set_green_time(3)
                actor.set_yellow_time(1)

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _get_actor_polygons(self, filt):
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _get_actor_info(self, filt):
        actor_trajectory_dict = {}
        actor_acceleration_dict = {}
        actor_angular_velocity_dict = {}
        actor_velocity_dict = {}

        for actor in self.world.get_actors().filter(filt):
            actor_trajectory_dict[actor.id] = actor.get_transform()
            actor_acceleration_dict[actor.id] = actor.get_acceleration()
            actor_angular_velocity_dict[actor.id] = actor.get_angular_velocity()
            actor_velocity_dict[actor.id] = actor.get_velocity()
        return actor_trajectory_dict, actor_acceleration_dict, actor_angular_velocity_dict, actor_velocity_dict

    def _get_obs(self):
        """获取观测，包含TCP所需的原始尺寸相机图像"""
        ego_trans = self.ego_vehicle.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        yaw = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
        delta_yaw = np.arcsin(np.cross(w, yaw))

        v = self.ego_vehicle.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        acc = self.ego_vehicle.get_acceleration()
        state = np.array([lateral_dis, -delta_yaw, speed, self.vehicle_front])

        if self.scenario_category != 'perception': 
            self.birdeye_render.set_hero(self.ego_vehicle, self.ego_vehicle.id)
            self.birdeye_render.vehicle_polygons = self.vehicle_polygons
            self.birdeye_render.walker_polygons = self.walker_polygons
            self.birdeye_render.waypoints = self.waypoints

            birdeye_render_types = ['roadmap', 'actors', 'waypoints']
            birdeye_surface = self.birdeye_render.render(birdeye_render_types)
            birdeye_surface = pygame.surfarray.array3d(birdeye_surface)
            center = (int(birdeye_surface.shape[0]/2), int(birdeye_surface.shape[1]/2))
            width = height = int(self.display_size/2)
            birdeye = birdeye_surface[center[0]-width:center[0]+width, center[1]-height:center[1]+height]
            birdeye = display_to_rgb(birdeye, self.obs_size)

            if not self.disable_lidar:
                point_cloud = np.copy(np.frombuffer(self.lidar_data.raw_data, dtype=np.dtype('f4')))
                point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))
                x = point_cloud[:, 0:1]
                y = point_cloud[:, 1:2]
                z = point_cloud[:, 2:3]
                intensity = point_cloud[:, 3:4]
                point_cloud = np.concatenate([y, -x, z], axis=1)
                y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind + self.lidar_bin, self.lidar_bin)
                x_bins = np.arange(-self.obs_range / 2, self.obs_range / 2 + self.lidar_bin, self.lidar_bin)
                z_bins = [-self.lidar_height - 1, -self.lidar_height + 0.25, 1]
                lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
                lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
                lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
                wayptimg = birdeye[:, :, 0] < 0
                wayptimg = np.expand_dims(wayptimg, axis=2)
                wayptimg = np.fliplr(np.rot90(wayptimg, 3))
                lidar = np.concatenate((lidar, wayptimg), axis=2)
                lidar = np.flip(lidar, axis=1)
                lidar = np.rot90(lidar, 1) * 255

                birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
                self.display.blit(birdeye_surface, (0, self.env_id*self.display_size))

                lidar_surface = rgb_to_display_surface(lidar, self.display_size)
                self.display.blit(lidar_surface, (self.display_size, self.env_id*self.display_size))

                camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
                camera_surface = rgb_to_display_surface(camera, self.display_size)
                self.display.blit(camera_surface, (self.display_size*2, self.env_id*self.display_size))
            else:
                birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
                self.display.blit(birdeye_surface, (0, self.env_id*self.display_size))

                camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
                camera_surface = rgb_to_display_surface(camera, self.display_size)
                self.display.blit(camera_surface, (self.display_size, self.env_id*self.display_size))

            obs = {
                'camera': camera.astype(np.uint8),
                'camera_raw': self.camera_img_raw.astype(np.uint8),  # TCP原始尺寸图像
                'lidar': None if self.disable_lidar else lidar.astype(np.uint8),
                'birdeye': birdeye.astype(np.uint8),
                'state': state.astype(np.float32),
            }
        else:
            camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
            camera_surface = rgb_to_display_surface(camera, self.display_size)
            self.display.blit(camera_surface, (0, self.env_id*self.display_size))

            obs = {
                'camera': camera.astype(np.uint8),
                'camera_raw': self.camera_img_raw.astype(np.uint8),
                'state': state.astype(np.float32),
            }
        return obs

    def _get_scenario_reward(self):
        last_record = self.scenario_manager.running_record[-1] if self.scenario_manager.running_record else None
        is_collision = last_record is not None and last_record['collision'] == Status.FAILURE
        r_collision = 25.0 if is_collision else 0.0

        ego_loc = self.ego_vehicle.get_location()
        adv_veh = self.scenario_manager.scenario_list[0].reference_actor

        if adv_veh is not None:
            dist = ego_loc.distance(adv_veh.get_location())
        else:
            dist = 100.0

        k_dist = 0.2
        r_distance = np.exp(-k_dist * dist) * 5.0

        r_penalty = 0.0

        total_reward = r_collision + r_distance - r_penalty

        return total_reward

    def _get_ego_reward(self):
        r_collision = -1 if len(self.collision_hist) > 0 else 0
        r_steer = -self.ego_vehicle.get_control().steer ** 2

        ego_x, ego_y = get_pos(self.ego_vehicle)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = -1 if abs(dis) > self.out_lane_thres else 0

        v = self.ego_vehicle.get_velocity()
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)
        r_fast = -1 if lspeed_lon > self.desired_speed / 3.6 else 0

        r_lat = -abs(self.ego_vehicle.get_control().steer) * lspeed_lon**2

        r = 1 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat
        return r

    def _get_cost(self):
        r_collision = 0
        if len(self.collision_hist) > 0:
            r_collision = -1
        return r_collision

    def _terminal(self):
        return not self.scenario_manager._running 

    def _iter_env_sensors(self):
        return (
            'collision_sensor',
            'lidar_sensor',
            'camera_sensor',
        )

    def _stop_sensor(self):
        for sensor_name in self._iter_env_sensors():
            sensor = getattr(self, sensor_name, None)
            if sensor is None:
                continue
            try:
                if sensor.is_alive:
                    sensor.stop()
            except Exception:
                pass

    def _destroy_sensor(self):
        for sensor_name in self._iter_env_sensors():
            sensor = getattr(self, sensor_name, None)
            if sensor is None:
                continue
            try:
                if sensor.is_alive:
                    sensor.destroy()
            except Exception:
                pass
            setattr(self, sensor_name, None)

    def _remove_sensor(self):
        self._stop_sensor()
        self._destroy_sensor()

    def _remove_ego(self):
        if self.ego_vehicle is not None:
            try:
                if self.ego_vehicle.is_alive:
                    self.ego_vehicle.destroy()
            except Exception:
                pass
            self.ego_vehicle = None

    def _flush_cleanup_world(self, ticks=1):
        if self.world is None:
            return

        for _ in range(max(0, ticks)):
            try:
                if self.world.get_settings().synchronous_mode:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()
                CarlaDataProvider.on_carla_tick()
            except Exception as world_error:
                if self.logger is not None:
                    self.logger.log(f'>> Cleanup world flush failed: {world_error}', color='yellow')
                break

    def _clear_runtime_caches(self):
        self.vehicle_polygons = []
        self.walker_polygons = []
        self.vehicle_trajectories = []
        self.vehicle_accelerations = []
        self.vehicle_angular_velocities = []
        self.vehicle_velocities = []
        self.collision_hist = []
        self.lidar_data = None
        self.camera_img_raw = None

    def _cleanup_runtime_actors(self, prefix='Cleanup'):
        import time as _time
        def _dbg(msg):
            ts = _time.strftime('%H:%M:%S')
            print(f'[CLEANUP-DBG {ts}] {msg}', flush=True)

        _dbg(f'=== {prefix} START ===')

        # 清理前先切换为异步模式，避免同步 tick/apply_batch_sync 在堵车场景后阻塞60秒导致 C++ TimeoutException
        _dbg('Step 1: disabling synchronous mode')
        try:
            if self.world is not None:
                settings = self.world.get_settings()
                if settings.synchronous_mode:
                    settings.synchronous_mode = False
                    settings.fixed_delta_seconds = None
                    self.world.apply_settings(settings)
                    _dbg('Step 1: synchronous mode disabled OK')
                else:
                    _dbg('Step 1: already async, skip')
        except Exception as settings_error:
            _dbg(f'Step 1 FAILED: {settings_error}')
            if self.logger is not None:
                self.logger.log(f'>> {prefix} failed to disable synchronous mode: {settings_error}', color='yellow')

        _dbg('Step 2: stopping sensors (sensor.stop RPC)')
        try:
            self._stop_sensor()
            _dbg('Step 2: sensors stopped OK')
        except Exception as sensor_error:
            _dbg(f'Step 2 FAILED: {sensor_error}')
            if self.logger is not None:
                self.logger.log(f'>> {prefix} sensor stop failed: {sensor_error}', color='yellow')

        _dbg('Step 3: world.tick/wait_for_tick after sensor stop (2 ticks)')
        self._flush_cleanup_world(2)
        _dbg('Step 3: flush OK')

        _dbg('Step 4: destroying sensors (sensor.destroy RPC)')
        try:
            self._destroy_sensor()
            _dbg('Step 4: sensors destroyed OK')
        except Exception as sensor_error:
            _dbg(f'Step 4 FAILED: {sensor_error}')
            if self.logger is not None:
                self.logger.log(f'>> {prefix} sensor destroy failed: {sensor_error}', color='yellow')

        _dbg('Step 5: world.tick/wait_for_tick after sensor destroy (3 ticks + 200ms sleep)')
        # sensor.destroy() is a synchronous RPC (actor removed from simulation), but
        # CARLA\'s streaming server closes the sensor\'s ASIO socket asynchronously on
        # a background thread.  Without enough wall-clock time here, the socket fd can
        # still be half-open when the next episode spawns sensors, leading to fd reuse
        # races in the epoll reactor that accumulate across episodes and eventually
        # corrupt descriptor_state* → SIGSEGV (address 0x3 or 0x90).
        self._flush_cleanup_world(3)
        import time as _time_cleanup
        _time_cleanup.sleep(0.2)
        _dbg('Step 5: flush OK')

        _dbg('Step 6: scenario_manager.clean_up() -> criteria sensors, scenario actors, background vehicles')
        try:
            self.scenario_manager.clean_up()
            _dbg('Step 6: scenario_manager.clean_up OK')
        except Exception as scenario_error:
            _dbg(f'Step 6 FAILED: {scenario_error}')
            if self.logger is not None:
                self.logger.log(f'>> {prefix} scenario cleanup failed: {scenario_error}', color='yellow')

        _dbg('Step 7: world.tick/wait_for_tick after scenario cleanup')
        self._flush_cleanup_world()
        _dbg('Step 7: flush OK')

        if self.scenario_category != 'scenic':
            _dbg('Step 8: removing ego vehicle (ego.is_alive + ego.destroy RPC)')
            try:
                self._remove_ego()
                _dbg('Step 8: ego removed OK')
            except Exception as ego_error:
                _dbg(f'Step 8 FAILED: {ego_error}')
                if self.logger is not None:
                    self.logger.log(f'>> {prefix} ego cleanup failed: {ego_error}', color='yellow')

            _dbg('Step 9: world.tick/wait_for_tick after ego removal')
            self._flush_cleanup_world()
            _dbg('Step 9: flush OK')

        _dbg('Step 10: clearing actor tracking maps')
        CarlaDataProvider.clear_actor_tracking_maps()
        _dbg('Step 10: OK')

        _dbg('Step 11: clearing runtime caches')
        self._clear_runtime_caches()

        # -----------------------------------------------------------------------
        # PROBE A: 检查 CARLA world 里是否还有 sensor/vehicle actor 残留
        # -----------------------------------------------------------------------
        _dbg('Step 12 [PROBE-A]: residual actors in CARLA world after cleanup')
        try:
            all_actors = self.world.get_actors()
            sensors  = [a for a in all_actors if a.type_id.startswith('sensor.')]
            vehicles = [a for a in all_actors if a.type_id.startswith('vehicle.')]
            walkers  = [a for a in all_actors if a.type_id.startswith('walker.')]
            _dbg(f'  sensors={len(sensors)} vehicles={len(vehicles)} walkers={len(walkers)}')
            for s in sensors:
                _dbg(f'  RESIDUAL SENSOR: id={s.id} type={s.type_id} alive={s.is_alive}')
            if sensors:
                _dbg(f'  WARNING: {len(sensors)} sensors still alive → possible actor leak')
        except Exception as _probe_err:
            _dbg(f'  PROBE-A FAILED: {_probe_err}')

        # -----------------------------------------------------------------------
        # PROBE B: 统计本进程 TCP socket fd 数量
        # CARLA sensor streaming 用 TCP。如果每轮 cleanup 后 TCP fd 数量递增，
        # 说明 ASIO 后台线程还没把旧 socket 关完 → 下一轮 epoll_reactor 遇到半关闭的 fd → SIGSEGV
        # -----------------------------------------------------------------------
        _dbg('Step 12 [PROBE-B]: TCP socket fd count in this process (sensor streaming leak indicator)')
        try:
            import os as _os
            import socket as _socket
            tcp_fds = []
            fd_dir = f'/proc/{_os.getpid()}/fd'
            for _fd_name in _os.listdir(fd_dir):
                try:
                    _fd_path = _os.readlink(f'{fd_dir}/{_fd_name}')
                    if 'socket' in _fd_path:
                        _fd_num = int(_fd_name)
                        try:
                            _sock = _socket.fromfd(_fd_num, _socket.AF_INET, _socket.SOCK_STREAM)
                            _peer = _sock.getpeername()
                            _sock.detach()  # don't close original fd
                            tcp_fds.append((_fd_num, _fd_path, str(_peer)))
                        except Exception:
                            tcp_fds.append((_fd_num, _fd_path, '(not TCP or not connected)'))
                except (OSError, ValueError):
                    continue
            _dbg(f'  total socket fds: {len(tcp_fds)}')
            for _fd_num, _fd_path, _peer in tcp_fds:
                _dbg(f'    fd={_fd_num}  {_fd_path}  peer={_peer}')
            if len(tcp_fds) > 10:
                _dbg(f'  WARNING: {len(tcp_fds)} socket fds — may indicate streaming socket leak')
        except Exception as _probe_b_err:
            _dbg(f'  PROBE-B FAILED: {_probe_b_err}')

        # -----------------------------------------------------------------------
        # PROBE C: 检查 CARLA server 端 TCP 连接状态（ss/netstat）
        # 看是否有 CLOSE_WAIT / TIME_WAIT 状态的连接在 2000/2001 端口上堆积
        # -----------------------------------------------------------------------
        _dbg('Step 12 [PROBE-C]: CARLA RPC/streaming port connection states (port 2000-2002)')
        try:
            import subprocess as _sp
            _r = _sp.run(
                ['ss', '-tnp', 'sport', '=', ':2000', 'or', 'sport', '=', ':2001', 'or',
                 'dport', '=', ':2000', 'or', 'dport', '=', ':2001'],
                capture_output=True, text=True, timeout=3
            )
            _lines = [l for l in _r.stdout.splitlines() if l.strip()]
            _dbg(f'  ss output ({len(_lines)} lines):')
            for _l in _lines[:20]:
                _dbg(f'    {_l}')
            # 统计各状态
            import re as _re
            _states = {}
            for _l in _lines:
                _m = _re.match(r'(\S+)\s', _l)
                if _m:
                    _s = _m.group(1)
                    _states[_s] = _states.get(_s, 0) + 1
            _dbg(f'  state summary: {_states}')
            if _states.get('CLOSE-WAIT', 0) + _states.get('TIME-WAIT', 0) > 5:
                _dbg(f'  WARNING: many half-closed connections → streaming teardown incomplete')
        except Exception as _probe_c_err:
            _dbg(f'  PROBE-C FAILED: {_probe_c_err}')
        # -----------------------------------------------------------------------

        _dbg(f'=== {prefix} DONE ===')

    def clean_up(self):
        self._cleanup_runtime_actors(prefix='Cleanup')
