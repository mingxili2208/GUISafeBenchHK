''' 
VehicleTurningRoute - 完整版本
直接优化初始Transform，确保碰撞
'''

import math
from typing import Dict
import carla

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.tools.scenario_helper import get_crossing_point, get_junction_topology
from safebench.scenario.tools.skopt_genetic_optimizer_VehicleTurningRoute import SkoptGeneticOptimizer


class VehicleTurningRoute(BasicScenario):
    """
    The ego vehicle is passing through a road and encounters a cyclist after taking a turn.
    """

    def __init__(self, world, ego_vehicle, config, timeout=240):
        super(VehicleTurningRoute, self).__init__("VehicleTurningRoute-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self.scenario_operation = ScenarioOperation()
        self.running_distance = 10
        self.ego_max_driven_distance = 180

        self.trigger_location = config.trigger_points[0].location if hasattr(
            config, 'trigger_points') and config.trigger_points else None

        # 遗传算法相关
        self.genetic_optimizer = None
        self.use_genetic_algorithm = False
        self.optimized_params = None

        # 基础参数
        self.actor_speed = 4.0
        self.trigger_distance_threshold = 25.0
        self.position_offset = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}

        # 调试
        self._update_count = 0
        self._last_position = None

        # 绿灯
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light:
            self._traffic_light.set_state(carla.TrafficLightState.Green)
            self._traffic_light.set_green_time(self.timeout)

    def create_behavior(self, scenario_init_action):
        """
        在初始化actors之前先执行优化
        """
        if self._should_use_genetic_algorithm(scenario_init_action):
            print("\n" + "="*60)
            print("🧬 GENETIC ALGORITHM OPTIMIZATION")
            print("="*60)

            # 创建优化器并执行优化
            self.genetic_optimizer = SkoptGeneticOptimizer(self)
            self.optimized_params = self.genetic_optimizer.optimize_initial_state()

            # 提取优化结果
            self.actor_speed = self.optimized_params['actor_speed']
            self.trigger_distance_threshold = self.optimized_params['trigger_distance']
            self.position_offset = self.optimized_params['position_offset']

            opt_info = self.genetic_optimizer.get_optimization_info()
            print(f"✅ Optimization completed - Best fitness: {opt_info['best_fitness']:.4f}")
            print(f"✅ Optimized speed: {self.actor_speed:.2f} m/s")
            print(f"✅ Optimized trigger distance: {self.trigger_distance_threshold:.2f} m")
            print("="*60 + "\n")
        else:
            self.actions = scenario_init_action

    def initialize_actorsHK(self):
        """
        ⭐⭐⭐ 关键：函数名必须是 initialize_actors
        使用优化后的initial_transform初始化actor
        """
        print("\n" + "="*60)
        print("🚴 INITIALIZING CYCLIST WITH OPTIMIZED POSITION")
        print("="*60)

        # 确保已经优化过
        if not self.optimized_params or 'initial_transform' not in self.optimized_params:
            print("⚠️ No optimized params found, running optimization now...")
            self.genetic_optimizer = SkoptGeneticOptimizer(self)
            self.optimized_params = self.genetic_optimizer.optimize_initial_state()
            self.actor_speed = self.optimized_params['actor_speed']
            self.trigger_distance_threshold = self.optimized_params['trigger_distance']

        initial_transform = self.optimized_params['initial_transform']

        # ⭐ 关键：确保位置在路上
        world_map = CarlaDataProvider.get_map()
        waypoint = world_map.get_waypoint(
            initial_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if waypoint:
            # 使用路上的位置，但保持优化的朝向
            initial_transform.location = waypoint.transform.location
            initial_transform.location.z += 0.3  # 稍微抬高避免穿模
            print(f"✅ Projected to road waypoint")
            print(f"   Distance to original: {waypoint.transform.location.distance(self.optimized_params['initial_transform'].location):.2f}m")

        print(f"📍 Final position: ({initial_transform.location.x:.2f}, {initial_transform.location.y:.2f})")
        print(f"📍 Final heading: {initial_transform.rotation.yaw:.2f}°")
        print(f"🚴 Actor speed: {self.actor_speed:.2f} m/s ({self.actor_speed * 3.6:.2f} km/h)")
        print(f"🎯 Trigger distance: {self.trigger_distance_threshold:.2f} m")

        # 使用scenario_operation来生成actor（正确的方式）
        self.actor_transform_list = [initial_transform]
        self.actor_type_list = ['vehicle.diamondback.century']

        try:
            # ⭐ 使用scenario_operation的方法，确保正确注册
            self.other_actors = self.scenario_operation.initialize_vehicle_actors(
                self.actor_transform_list,
                self.actor_type_list
            )

            if len(self.other_actors) == 0 or self.other_actors[0] is None:
                raise RuntimeError("Failed to spawn actor")

            self.reference_actor = self.other_actors[0]

            # ⭐ 立即设置物理和控制
            actor = self.reference_actor

            # 1. 禁用自动驾驶
            try:
                actor.set_autopilot(False)
            except:
                pass

            # 2. 设置物理控制
            try:
                physics = actor.get_physics_control()
                physics.use_gear_autobox = True
                actor.apply_physics_control(physics)
            except Exception as e:
                print(f"⚠️ Physics control warning: {e}")

            # 3. 立即应用初始控制
            control = carla.VehicleControl()
            control.throttle = 0.8
            control.steer = 0.0
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False
            actor.apply_control(control)

            # 4. 设置目标速度
            forward_vector = initial_transform.rotation.get_forward_vector()
            target_velocity = carla.Vector3D(
                forward_vector.x * self.actor_speed,
                forward_vector.y * self.actor_speed,
                0
            )
            actor.set_target_velocity(target_velocity)

            # 5. 记录初始位置
            self._last_position = actor.get_location()

            print("="*60)
            print("✅ CYCLIST INITIALIZATION COMPLETE")
            print("="*60 + "\n")

        except Exception as e:
            self.reference_actor = None
            self.other_actors = []
            print(f"❌ Failed to initialize cyclist: {e}")
            print(f"   Transform: {initial_transform}")
            import traceback
            traceback.print_exc()

    def update_behavior(self, scenario_action):
        """
        更新自行车行为 - 强制控制
        """
        assert scenario_action is None, f'{self.name} should receive [None] action.'

        if len(self.other_actors) == 0:
            return

        cur_actor_target_speed = self.actor_speed
        self._update_count += 1

        for actor in self.other_actors:
            # 获取当前速度
            current_velocity = actor.get_velocity()
            current_speed = math.sqrt(
                current_velocity.x**2 +
                current_velocity.y**2 +
                current_velocity.z**2
            )

            # 根据速度调整控制
            control = carla.VehicleControl()

            if current_speed < cur_actor_target_speed - 1.0:
                control.throttle = 1.0  # 全油门加速
                control.brake = 0.0
            elif current_speed > cur_actor_target_speed + 2.0:
                control.throttle = 0.0
                control.brake = 0.3  # 轻微刹车
            else:
                control.throttle = 0.6  # 维持速度
                control.brake = 0.0

            control.steer = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False
            actor.apply_control(control)

            # 同时设置目标速度
            transform = actor.get_transform()
            forward_vec = transform.rotation.get_forward_vector()
            actor.set_target_velocity(carla.Vector3D(
                forward_vec.x * cur_actor_target_speed,
                forward_vec.y * cur_actor_target_speed,
                0
            ))

            # 每10帧输出调试信息
            if self._update_count % 10 == 0:
                current_pos = actor.get_location()

                print(f"🚴 [Frame {self._update_count}]")
                print(f"   Position: ({current_pos.x:.2f}, {current_pos.y:.2f})")
                print(f"   Speed: {current_speed:.2f} m/s | Target: {cur_actor_target_speed:.2f} m/s")
                print(f"   Control: throttle={control.throttle:.2f} brake={control.brake:.2f}")

                if self._last_position:
                    moved = math.sqrt(
                        (current_pos.x - self._last_position.x)**2 +
                        (current_pos.y - self._last_position.y)**2
                    )
                    print(f"   Moved: {moved:.2f}m in 10 frames")

                    if moved < 0.5:
                        print(f"   ⚠️⚠️⚠️ Barely moving! Applying emergency control!")
                        emergency = carla.VehicleControl()
                        emergency.throttle = 1.0
                        emergency.steer = 0.0
                        emergency.brake = 0.0
                        emergency.hand_brake = False
                        actor.apply_control(emergency)

                self._last_position = current_pos

    def check_stop_condition(self):
        return False

    def _should_use_genetic_algorithm(self, scenario_init_action) -> bool:
        """判断是否使用遗传算法"""
        if scenario_init_action is None:
            return False

        if (hasattr(self.config, 'parameters') and
                len(self.config.parameters) >= 2 and
                self.config.parameters[0] == 'genetic_algorithm' and
                self.config.parameters[1] is True):
            return True

        return False