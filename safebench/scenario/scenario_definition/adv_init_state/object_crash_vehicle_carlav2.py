''' 

'''

import math

import carla
from typing import Dict, Tuple, Any
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

        # ⭐ 重要：初始化 actions 为默认值，避免 AttributeError
        self.actions = [0.5, 0.0, 20.0]  # [y, yaw, trigger_distance]

        # 遗传算法相关标志
        self.use_genetic_algorithm = False
        self.genetic_optimization_completed = False
        self.optimized_params = None

        # 基础参数（遗传算法会优化这些值）
        self.actor_speed = 3.0  # 行人/骑行者速度
        self.position_offset = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.trigger_distance_threshold = 20.0

        # Note: transforms for walker and blocker
        self.transform = None
        self.transform2 = None
        self._trigger_location = config.trigger_points[0].location
        self._number_of_attempts = 20
        self._spawn_attempted = 0
        self.trigger_location = self._trigger_location

        self.scenario_operation = ScenarioOperation()
        self.ego_max_driven_distance = 150

        print(f">> [DynamicObjectCrossing] Initialized with default parameters")

    def convert_actions(self, actions):
        """将动作转换为场景参数"""
        yaw_scale = 60
        yaw_mean = 0

        d_min = 15
        d_max = 50
        d_scale = (d_max - d_min) / 2
        dist_mean = (d_max + d_min) / 2

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

    def create_behavior(self, scenario_init_action):
        """
        创建场景行为 - 方案3：只设置标志，不执行优化
        """
        print(f">> [DynamicObjectCrossing] create_behavior() called")

        # ⭐ 首先确保 actions 有值（从 scenario_init_action 获取或使用默认值）
        if scenario_init_action is not None:
            try:
                self.actions = self.convert_actions(scenario_init_action)
                print(f"   Converted actions from scenario_init_action: {self.actions}")
            except Exception as e:
                print(f"   [WARN] Failed to convert scenario_init_action: {e}")
                print(f"   Using default actions: {self.actions}")
        else:
            print(f"   scenario_init_action is None, using default actions: {self.actions}")

        # ⭐ 检查是否应该使用遗传算法（只设置标志，不执行优化）
        if self._should_use_genetic_algorithm(scenario_init_action):
            self.use_genetic_algorithm = True
            print(">> [DynamicObjectCrossing] Genetic algorithm mode ENABLED")
            print("   Optimization will be performed after actors initialization")
        else:
            self.use_genetic_algorithm = False
            print(">> [DynamicObjectCrossing] Using traditional parameter setting")

    def initialize_actorsHK(self):
        """
        初始化场景参与者 - 方案3：初始化后执行优化
        """
        print(f">> [DynamicObjectCrossing] initialize_actorsHK() called")

        # ⭐ 双重保险：确保 actions 存在
        if not hasattr(self, 'actions') or self.actions is None:
            print("   [WARN] actions not set, using default")
            self.actions = [0.5, 0.0, 20.0]

        # ⭐ 第一阶段：使用当前参数初始化基础场景
        self._initialize_base_scenario()

        # ⭐ 第二阶段：如果启用遗传算法，执行优化并重新初始化
        if self.use_genetic_algorithm and not self.genetic_optimization_completed:
            print("\n>> [DynamicObjectCrossing] Starting genetic algorithm optimization...")
            self._run_genetic_optimization()

            # 优化完成后，使用优化参数重新初始化场景
            if self.optimized_params:
                print(">> [DynamicObjectCrossing] Re-initializing scenario with optimized parameters...")
                self._apply_optimized_params()
                self._reinitialize_scenario_with_optimized_params()

            self.genetic_optimization_completed = True
            print(">> [DynamicObjectCrossing] Optimization and re-initialization completed\n")

    def _initialize_base_scenario(self):
        """
        初始化基础场景（第一次初始化或重新初始化）
        """
        print(f">> [DynamicObjectCrossing] Initializing base scenario...")

        # 从 actions 获取参数
        y, yaw, self.trigger_distance_threshold = self.actions

        # cyclist transform
        _start_distance = 45
        waypoint = self._reference_waypoint

        while True:
            wp_next = waypoint.get_right_lane()
            self._num_lane_changes += 1
            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder:
                if wp_next.lane_width > 2:
                    _start_distance += 1.5
                    waypoint = wp_next
                break
            else:
                _start_distance += 1.5
                waypoint = wp_next

        # 计算基础变换
        while True:
            try:
                self.transform, orientation_yaw = self._calculate_base_transform(_start_distance, waypoint)
                self._spawn_blocker(self.transform, orientation_yaw)

                # 应用 y 和 yaw 参数
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
                print(f"   Base transform is blocking objects {self.transform}")
                _start_distance += 0.4
                self._spawn_attempted += 1
                if self._spawn_attempted >= self._number_of_attempts:
                    raise r

        # ⭐ 应用遗传算法的额外位置偏移（如果有）
        if hasattr(self, 'position_offset') and self.position_offset:
            current_rotation = self.transform.rotation
            right_vector = current_rotation.get_right_vector()
            forward_vector_current = current_rotation.get_forward_vector()

            x_offset = self.position_offset.get('x', 0.0)
            y_offset = self.position_offset.get('y', 0.0)

            offset_vector = right_vector * x_offset + forward_vector_current * y_offset
            self.transform.location += offset_vector

            print(f"   Applied position offset: X={x_offset:.2f}m, Y={y_offset:.2f}m")

        # ⭐ 应用遗传算法优化的速度（如果有）
        if hasattr(self, 'actor_speed'):
            self._other_actor_target_velocity = self.actor_speed
            print(f"   Applied walker speed: {self.actor_speed:.2f} m/s")

        # 创建变换
        disp_transform = carla.Transform(
            carla.Location(self.transform.location.x, self.transform.location.y, self.transform.location.z),
            self.transform.rotation
        )
        prop_disp_transform = carla.Transform(
            carla.Location(self.transform2.location.x, self.transform2.location.y, self.transform2.location.z),
            self.transform2.rotation
        )

        # 初始化actors
        self.actor_type_list = ['walker.*', 'static.prop.vendingmachine']
        self.actor_transform_list = [disp_transform, prop_disp_transform]

        try:
            # ⭐ 如果是重新初始化，先清理旧的actors
            if hasattr(self, 'other_actors') and self.other_actors:
                print("   Cleaning up old actors before re-initialization...")
                for actor in self.other_actors:
                    if actor is not None and actor.is_alive:
                        actor.destroy()
                self.other_actors = []

            self.other_actors = self.scenario_operation.initialize_vehicle_actors(
                self.actor_transform_list, self.actor_type_list
            )
            self.reference_actor = self.other_actors[0]
            self.trigger_location = self.transform.location
            print(f"   Successfully initialized walker and blocker")
            print(f"   Final position: ({self.transform.location.x:.2f}, {self.transform.location.y:.2f})")
            print(f"   Trigger distance: {self.trigger_distance_threshold:.2f}m")
        except Exception as e:
            print(f"   [ERROR] Failed to initialize actors: {e}")
            self.reference_actor = None

    def _run_genetic_optimization(self):
        """
        执行遗传算法优化 - 在场景初始化后运行
        """
        print(">> [DynamicObjectCrossing] Executing genetic algorithm optimization...")

        try:
            carla_env = self._get_carla_env()
            if carla_env is None:
                print("   [ERROR] Cannot get CARLA environment, skipping optimization")
                return

            # 创建并运行优化器
            from safebench.scenario.tools.skopt_genetic_optimizer import SkoptGeneticOptimizer
            optimizer = SkoptGeneticOptimizer(self, carla_env)

            # 执行优化
            self.optimized_params = optimizer.optimize_initial_state()

            # 获取优化信息
            opt_info = optimizer.get_optimization_info()
            print(f">> Optimization complete:")
            print(f"   Best fitness: {opt_info.get('best_fitness', 0):.3f}")
            print(f"   Collisions achieved: {opt_info.get('collision_count', 0)}")
            print(f"   Success rate: {opt_info.get('success_rate', 0):.2%}")
            print(f"   Total evaluations: {opt_info.get('total_evaluations', 0)}")
            print(f"   Optimization time: {opt_info.get('optimization_time', 0):.2f}s")

        except Exception as e:
            print(f"   [ERROR] Genetic optimization failed: {e}")
            import traceback
            traceback.print_exc()

    def _apply_optimized_params(self):
        """
        应用优化后的参数到场景
        """
        if not self.optimized_params:
            print("   [WARN] No optimized parameters to apply")
            return

        print(">> [DynamicObjectCrossing] Applying optimized parameters...")

        # 更新场景参数
        self.actor_speed = self.optimized_params['actor_speed']
        self.trigger_distance_threshold = self.optimized_params['trigger_distance']
        self.position_offset = self.optimized_params['position_offset']

        # 更新相关速度参数
        self._other_actor_target_velocity = self.actor_speed

        print(f"   Speed: {self.actor_speed:.2f} m/s")
        print(f"   Trigger distance: {self.trigger_distance_threshold:.2f} m")
        print(f"   Position offset: X={self.position_offset['x']:.2f}m, "
              f"Y={self.position_offset['y']:.2f}m, Yaw={self.position_offset['yaw']:.2f}°")

    def _reinitialize_scenario_with_optimized_params(self):
        """
        使用优化参数重新初始化场景
        """
        print(">> [DynamicObjectCrossing] Re-initializing scenario with optimized parameters...")

        # 重置spawn计数器
        self._spawn_attempted = 0
        self._num_lane_changes = 1

        # 重新初始化场景
        self._initialize_base_scenario()

    def _get_carla_env(self):
        """
        获取CARLA环境实例 - 通过全局引用
        """
        try:
            import safebench.scenario.scenario_definition.adv_init_state as adv_init_state_module
            if hasattr(adv_init_state_module, 'GLOBAL_CARLA_ENV'):
                env = adv_init_state_module.GLOBAL_CARLA_ENV
                print(f"   Retrieved CARLA environment: {type(env).__name__}")
                return env
            else:
                print("   [WARNING] No global CARLA environment reference available")
                return None
        except Exception as e:
            print(f"   [ERROR] Failed to get CARLA environment: {e}")
            return None

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

    def update_behavior(self, scenario_action):
        """
        更新场景行为
        """
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

        # the walker starts crossing the road
        self.scenario_operation.walker_go_straight(self._other_actor_target_velocity, 0)

    def check_stop_condition(self):
        """
        检查场景停止条件
        """
        lane_width = self._reference_waypoint.lane_width
        lane_width = lane_width + (1.25 * lane_width * self._num_lane_changes)
        cur_distance = calculate_distance_transforms(
            CarlaDataProvider.get_transform(self.other_actors[0]),
            self.transform
        )
        if cur_distance > 0.6 * lane_width:
            return True
        return False