"""
CarlaEnv 类是一个基于 OpenAI Gym 接口的 CARLA 模拟器环境。其设计逻辑和功能可以概括如下:
(1)初始化：
    类的初始化方法 __init__ 接收环境参数 env_params 以及其他可选参数（如 birdeye_render、display、world 和 logger）
    初始化各种传感器（如碰撞传感器、激光雷达传感器和摄像头传感器）和场景管理器 ScenarioManager
    根据不同的场景类别（如 planning 和 perception）设置观测空间和动作空间
(2)传感器创建：
    方法 _create_sensors 创建并配置碰撞传感器、激光雷达传感器和摄像头传感器
(3)场景创建和运行：
    方法 _create_scenario 根据不同的场景类别创建相应的场景（如 PerceptionScenario、RouteScenario）
    方法 _run_scenario 运行场景管理器中的场景
(4)重置环境：
    方法 reset 重置环境，包括创建传感器、加载和运行场景、附加传感器、解析路线等
(5)步骤执行：
    方法 step_before_tick 在每个carla world tick之前执行动作，包括计算加速度和转向角度，并应用于车辆控制
    方法 step_after_tick 在每个carla world tick之后更新车辆和行人的多边形信息、车辆状态信息等
(6)观测和奖励：
    方法 _get_obs 获取当前的观测信息，包括状态观测、激光雷达图像、鸟瞰图像和摄像头图像
    方法 _get_reward 计算每个时间步的奖励，奖励函数考虑了碰撞、转向、偏离车道、速度跟踪等因素
(7)其他辅助方法：
    方法 _get_info 获取当前状态的信息
    方法 _init_traffic_light 初始化交通信号灯
    方法 _create_vehicle_blueprint 创建车辆蓝图
    方法 _get_actor_polygons 获取演员的多边形信息
    方法 _get_actor_info 获取演员的状态信息
    方法 _get_cost 计算碰撞的代价
    方法 _terminal 判断场景是否结束
    方法 _remove_sensor 和 _remove_ego 移除传感器和自车
    方法 clean_up 清理环境
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
from safebench.carla_agents.navigation.global_route_planner import GlobalRoutePlanner


class CarlaEnv(gym.Env):
    """ 
        An OpenAI-gym style interface for CARLA simulator. 
    """
    def __init__(self, env_params, birdeye_render=None, display=None, world=None, logger=None):
        assert world is not None, "the world passed into CarlaEnv is None"

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
        
        # scenario manager
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
        self.scenario_category = env_params['scenario_category']  # 指出是进行感知类测试还是规划类测试
        self.warm_up_steps = env_params['warm_up_steps']
        self.running_results = {}

        if self.scenario_category in ['planning']:
            self.obs_size = int(self.obs_range/self.lidar_bin)
            """
            字典 observation_space_dict，用于描述观测空间的不同部分。具体来说：
            camera: 定义了一个形状为 (self.obs_size, self.obs_size, 3) 的 Box 空间，表示相机图像的观测空间。图像的每个像素值在 0 到 255 之间，数据类型为 np.uint8。
            lidar: 定义了一个形状为 (self.obs_size, self.obs_size, 3) 的 Box 空间，表示激光雷达图像的观测空间。图像的每个像素值在 0 到 255 之间，数据类型为 np.uint8。
            birdeye: 定义了一个形状为 (self.obs_size, self.obs_size, 3) 的 Box 空间，表示鸟瞰图的观测空间。图像的每个像素值在 0 到 255 之间，数据类型为 np.uint8。
            state: 定义了一个 Box 空间，表示状态信息的观测空间。状态信息的每个值在 [-2, -1, -5, 0] 到 [2, 1, 30, 1] 之间，数据类型为 np.float32。
            """
            observation_space_dict = {
                'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'state': spaces.Box(np.array([-2, -1, -5, 0], dtype=np.float32), np.array([2, 1, 30, 1], dtype=np.float32), dtype=np.float32)
            }
        elif self.scenario_category == 'perception':
            self.obs_size = env_params['image_sz']
            observation_space_dict = {
                'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            }
        else:
            raise ValueError(f'Unknown scenario category: {self.scenario_category}')

        # define obs space
        self.observation_space = spaces.Dict(observation_space_dict)

        # action and observation spaces
        self.discrete = env_params['discrete']
        self.discrete_act = [env_params['discrete_acc'], env_params['discrete_steer']]  # acc, steer
        self.n_acc = len(self.discrete_act[0])-233.83
        self.n_steer = len(self.discrete_act[1])
        if self.discrete:
            self.action_space = spaces.Discrete(self.n_acc * self.n_steer)
        else:
            # assume the output of NN is from -1 to 1
            self.action_space = spaces.Box(np.array([-1, -1], dtype=np.float32), np.array([1, 1], dtype=np.float32), dtype=np.float32)  # acc, steer

    def _create_sensors(self): # 这个函数主要用于配置传感器信息
        # collision sensor
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        if self.scenario_category != 'perception':
            # lidar sensor
            self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
            self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            self.lidar_bp.set_attribute('channels', '16')
            self.lidar_bp.set_attribute('range', '1000')
        
        # camera sensor
        self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8) 
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')

    def _create_scenario(self, config, env_id):
        self.logger.log(f">> Loading scenario data id: {config.data_id}")

        if self.scenario_category == 'planning':
            scenario = RouteScenario(
                world=self.world, 
                config=config, 
                ego_id=env_id, 
                max_running_step=self.max_episode_step, 
                logger=self.logger
            )  # 在这里生成主车、传感器、评价准则、构建测试场景实例
        else:
            raise ValueError(f'Unknown scenario category: {self.scenario_category}')

        # init scenario
        self.ego_vehicle = scenario.ego_vehicle
        self.scenario_manager.load_scenario(scenario)

    def _run_scenario(self, scenario_init_action):
        self.scenario_manager.run_scenario(scenario_init_action)

    def _parse_route(self, config):
        # 20250730修改：使用carla自带的路径搜索功能,直接获取路由点
        wmap = self.world.get_map()
        global_planner = GlobalRoutePlanner(wmap, sampling_resolution=2.0)
        start_location = config.trajectory[0]
        end_location = config.trajectory[-1]
        route = global_planner.trace_route(start_location, end_location)
        # 遍历route中的每个transform_tuple,提取waypoint构成waypoint_list
        waypoints_list = []
        for transform_tuple in route:
            waypoints_list.append(transform_tuple[0])

        return waypoints_list

    def get_static_obs(self, config):
        """
            This function returns static observation used for static scenario generation
        """
        wmap = self.world.get_map()
        global_planner = GlobalRoutePlanner(wmap, sampling_resolution=5.0)
        start_location = config.trajectory[0]
        end_location = config.trajectory[-1]
        route = global_planner.trace_route(start_location, end_location)

        # get [x, y] along the route
        waypoint_xy = []
        for transform_tuple in route:
            waypoint_xy.append([transform_tuple[0].transform.location.x, transform_tuple[0].transform.location.y])

        # combine state obs    
        state = {
            'route': np.array(waypoint_xy),   # [n, 2]
            'target_speed': self.desired_speed,
        }
        return state

    def reset(self, config, env_id, scenario_init_action):
        self.config = config
        self.env_id = env_id

        self._cleanup_runtime_actors(prefix='Reset pre-cleanup')

        # create sensors, load and run scenarios
        self._create_sensors()  # 只是指明了传感器的蓝图、生成位置等
        self._create_scenario(config, env_id)  # 创建主车、创建具体场景实例、创建评分准则、创建传感器
        self._run_scenario(scenario_init_action)  # 创建具体场景实例中规定的actor
        self._attach_sensor()  # 在这里才真正生成了传感器

        # 解析出来的waypoints路径用于BEV可视化+评分计算
        self.route_waypoints = self._parse_route(config)
        self.routeplanner = RoutePlanner(self.ego_vehicle, self.max_waypt, self.route_waypoints)
        # self.waypoints用来渲染可视化路径点;self.vehicle_front用来可视化交通体
        self.waypoints, _, _, _, _, self.vehicle_front, = self.routeplanner.run_step()
    
        # Get actors polygon list (for visualization)
        self.vehicle_polygons = [self._get_actor_polygons('vehicle.*')]
        self.walker_polygons = [self._get_actor_polygons('walker.*')]

        # Get actors info list
        vehicle_info_dict_list = self._get_actor_info('vehicle.*')
        self.vehicle_trajectories = [vehicle_info_dict_list[0]]
        self.vehicle_accelerations = [vehicle_info_dict_list[1]]
        self.vehicle_angular_velocities = [vehicle_info_dict_list[2]]
        self.vehicle_velocities = [vehicle_info_dict_list[3]]

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # applying setting can tick the world and get data from sensros
        self.settings = self.world.get_settings()
        self.world.apply_settings(self.settings)

        for _ in range(self.warm_up_steps):
            self.world.tick()
        return self._get_obs(), self._get_info()

    def _attach_sensor(self):
        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)
        self.collision_hist = []

        # Add lidar sensor
        if self.scenario_category != 'perception' and not self.disable_lidar:
            self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego_vehicle)
            self.lidar_sensor.listen(lambda data: get_lidar_data(data))

        def get_lidar_data(data):
            self.lidar_data = data

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego_vehicle)
        self.camera_sensor.listen(lambda data: get_camera_img(data))

        def get_camera_img(data):            
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array

    def step_before_tick(self, ego_action, scenario_action):
        if self.world:
            snapshot = self.world.get_snapshot()
            if snapshot:
                timestamp = snapshot.timestamp
                # get update on evaluation results before getting update of running status
                if self.scenario_category in ['perception']:
                    assert isinstance(ego_action, dict), 'ego action in ObjectDetectionScenario should be a dict'
                    world_2_camera = np.array(self.camera_sensor.get_transform().get_inverse_matrix())
                    fov = self.camera_bp.get_attribute('fov').as_float()
                    image_w, image_h = self.obs_size, self.obs_size
                    self.scenario_manager.background_scenario.evaluate(ego_action, world_2_camera, image_w, image_h, fov, self.camera_img)
                    ego_action = ego_action['ego_action']

                # 每一次在tick carla world之前,都会通过scenario_manager检查场景状态
                self.scenario_manager.get_update(timestamp, scenario_action)
                self.is_running = self.scenario_manager._running

                # Calculate acceleration and steering
                if not self.auto_ego:
                    if self.discrete:
                        acc = self.discrete_act[0][ego_action // self.n_steer]
                        steer = self.discrete_act[1][ego_action % self.n_steer]
                    else:
                        acc = ego_action[0]
                        steer = ego_action[1]

                    # normalize and clip the action
                    acc = acc * self.acc_max
                    steer = steer * self.steering_max
                    acc = max(min(self.acc_max, acc), -self.acc_max)
                    steer = max(min(self.steering_max, steer), -self.steering_max)

                    # Convert acceleration to throttle and brake
                    if acc > 0:
                        throttle = np.clip(acc / 3, 0, 1)
                        brake = 0
                    else:
                        throttle = 0
                        brake = np.clip(-acc / 8, 0, 1)

                    # apply control
                    act = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
                    self.ego_vehicle.apply_control(act)
                    # print(self.ego_vehicle.get_location(), self.ego_vehicle.get_velocity(), self.ego_vehicle.get_acceleration())
            else:
                self.logger.log('>> Can not get snapshot!', color='red')
                raise Exception()
        else:
            self.logger.log('>> Please specify a Carla world!', color='red')
            raise Exception()

    def step_after_tick(self):
        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
            self.vehicle_polygons.pop(0)
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > self.max_past_step:
            self.walker_polygons.pop(0)

        # Append actors info list
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

        # route planner
        self.waypoints, _, _, _, _, self.vehicle_front, = self.routeplanner.run_step()

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        v = self.ego_vehicle.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)  # 计算速度大小
        print(f"[CarlaEnv-{self.env_id}] Ego speed: {speed:.2f} m/s ({speed * 3.6:.2f} km/h)")

        return (self._get_obs(), self._get_reward(), self._terminal(), self._get_info())
    
    def _get_info(self):
        # state information
        info = {
            'waypoints': self.waypoints,
            'route_waypoints': self.route_waypoints,
            'vehicle_front': self.vehicle_front,
            'cost': self._get_cost()
        }

        # info from scenarios
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
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
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
        # State observation
        ego_trans = self.ego_vehicle.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)  # 车辆当前位置与规划轨迹中线之间的横向距离。这个值用于评估车辆是否偏离车道中心
        yaw = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
        delta_yaw = np.arcsin(np.cross(w, yaw)) # 表示车辆当前朝向与车道方向之间的偏差角度。这个值用于评估车辆是否偏离车道方向

        v = self.ego_vehicle.get_velocity()  # m/s
        speed = np.sqrt(v.x**2 + v.y**2) # 表示车辆的当前速度。这个值用于评估车辆的行驶速度
        acc = self.ego_vehicle.get_acceleration()
        state = np.array([lateral_dis, -delta_yaw, speed, self.vehicle_front])  # self.vehicle_front表示车辆前方的距离或状态信息。这个值用于评估车辆前方的情况，例如前方是否有障碍物或其他车辆

        if self.scenario_category != 'perception': 
            # set ego information for birdeye_render
            self.birdeye_render.set_hero(self.ego_vehicle, self.ego_vehicle.id)
            self.birdeye_render.vehicle_polygons = self.vehicle_polygons
            self.birdeye_render.walker_polygons = self.walker_polygons
            self.birdeye_render.waypoints = self.waypoints

            # render birdeye image with the birdeye_render
            birdeye_render_types = ['roadmap', 'actors', 'waypoints']
            birdeye_surface = self.birdeye_render.render(birdeye_render_types)
            birdeye_surface = pygame.surfarray.array3d(birdeye_surface)
            center = (int(birdeye_surface.shape[0]/2), int(birdeye_surface.shape[1]/2))
            width = height = int(self.display_size/2)
            birdeye = birdeye_surface[center[0]-width:center[0]+width, center[1]-height:center[1]+height]
            birdeye = display_to_rgb(birdeye, self.obs_size)

            if not self.disable_lidar:
                # get Lidar image
                point_cloud = np.copy(np.frombuffer(self.lidar_data.raw_data, dtype=np.dtype('f4')))
                point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))
                x = point_cloud[:, 0:1]
                y = point_cloud[:, 1:2]
                z = point_cloud[:, 2:3]
                intensity = point_cloud[:, 3:4]
                point_cloud = np.concatenate([y, -x, z], axis=1)
                # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin, and z is set to be two bins.
                y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind + self.lidar_bin, self.lidar_bin)
                x_bins = np.arange(-self.obs_range / 2, self.obs_range / 2 + self.lidar_bin, self.lidar_bin)
                z_bins = [-self.lidar_height - 1, -self.lidar_height + 0.25, 1]
                # Get lidar image according to the bins
                lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
                lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
                lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
                wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
                wayptimg = np.expand_dims(wayptimg, axis=2)
                wayptimg = np.fliplr(np.rot90(wayptimg, 3))
                # Get the final lidar image
                lidar = np.concatenate((lidar, wayptimg), axis=2)
                lidar = np.flip(lidar, axis=1)
                lidar = np.rot90(lidar, 1) * 255

                # display birdeye image
                birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
                self.display.blit(birdeye_surface, (0, self.env_id*self.display_size))

                # display lidar image
                lidar_surface = rgb_to_display_surface(lidar, self.display_size)
                self.display.blit(lidar_surface, (self.display_size, self.env_id*self.display_size))

                # display camera image
                camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
                camera_surface = rgb_to_display_surface(camera, self.display_size)
                self.display.blit(camera_surface, (self.display_size*2, self.env_id*self.display_size))
            else:
                # display birdeye image
                birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
                self.display.blit(birdeye_surface, (0, self.env_id*self.display_size))

                # display camera image
                camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
                camera_surface = rgb_to_display_surface(camera, self.display_size)
                self.display.blit(camera_surface, (self.display_size, self.env_id*self.display_size))

            obs = {
                'camera': camera.astype(np.uint8),
                'lidar': None if self.disable_lidar else lidar.astype(np.uint8),
                'birdeye': birdeye.astype(np.uint8),
                'state': state.astype(np.float32),
            }
        else:
            """ Get the observations for object detection. """
            camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
            camera_surface = rgb_to_display_surface(camera, self.display_size)
            self.display.blit(camera_surface, (0, self.env_id*self.display_size))

            obs = {
                'camera': camera.astype(np.uint8),
                'state': state.astype(np.float32),
            }
        return obs

    def _get_reward(self):
        """ Calculate the step reward.
        It penalizes the agent for collision, out of lane, too fast, and too much steering.
        The final reward is a weighted sum of these components, encouraging smooth, safe, and lane-compliant driving.
        """
        r_collision = -1 if len(self.collision_hist) > 0 else 0

        # reward for steering:
        r_steer = -self.ego_vehicle.get_control().steer ** 2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego_vehicle)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)  # w 是车道方向向量，用于将当前速度投影到车道方向
        r_out = -1 if abs(dis) > self.out_lane_thres else 0

        # reward for speed tracking
        v = self.ego_vehicle.get_velocity()

        # cost for too fast
        lspeed = np.array([v.x, v.y])

        # Longitudinal speed
        lspeed_lon = np.dot(lspeed, w)
        r_fast = -1 if lspeed_lon > self.desired_speed/3.6 else 0

        # cost for lateral acceleration
        r_lat = -abs(self.ego_vehicle.get_control().steer) * lspeed_lon**2

        # combine all rewards
        r = 1 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat
        return r

    def _get_cost(self):
        # cost for collision
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
        """
            Flush one or more CARLA world steps after cleanup operations so the
            server fully applies pending removals before the next scenario starts
            spawning actors.
        """
        if self.world is None:
            return

        from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider

        for _ in range(max(0, ticks)):
            try:
                if self.world.get_settings().synchronous_mode:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()
                CarlaDataProvider.on_carla_tick()
            except Exception as world_error:
                self.logger.log(f'>> Cleanup world flush failed: {world_error}', 'yellow')
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

    def _cleanup_runtime_actors(self, prefix='Cleanup'):
        from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider

        try:
            self._stop_sensor()
        except Exception as sensor_error:
            self.logger.log(f'>> {prefix} sensor stop failed: {sensor_error}', 'yellow')
        self._flush_cleanup_world()

        try:
            self._destroy_sensor()
        except Exception as sensor_error:
            self.logger.log(f'>> {prefix} sensor destroy failed: {sensor_error}', 'yellow')
        self._flush_cleanup_world()

        try:
            self.scenario_manager.clean_up()
        except Exception as scenario_error:
            self.logger.log(f'>> {prefix} scenario cleanup failed: {scenario_error}', 'yellow')
        self._flush_cleanup_world()

        try:
            self._remove_ego()
        except Exception as ego_error:
            self.logger.log(f'>> {prefix} ego cleanup failed: {ego_error}', 'yellow')
        self._flush_cleanup_world()

        CarlaDataProvider.clear_actor_tracking_maps()
        self._clear_runtime_caches()

    def clean_up(self):
        self._cleanup_runtime_actors(prefix='Cleanup')
