# safebench/scenario/scenario_definition/adv_init_state/object_crash_vehicle.py
"""
行人横穿道路场景 - 基于距离触发

场景流程:
1. 行人在ego前方路边等待（静止）
2. Ego直行接近
3. 当distance(ego, pedestrian) < trigger_distance时，行人开始横穿
4. 碰撞发生在ego车道上
"""

import math
import carla
from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.tools.scenario_utils import calculate_distance_transforms
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.skopt_genetic_optimizer_obect_crossing import SkoptGeneticOptimizer



import json
import os
from glob import glob

# def save_dynamic_object_crossing_json_incremental(optimized_params, save_dir="output"):
#     """
#     将 DynamicObjectCrossing 场景增量保存为 JSON 文件
#     文件名格式: adv_scn_<序号>.json
#     """
#     transform = optimized_params.get("initial_transform", None)
#
#     if transform is None:
#         raise ValueError("optimized_params 中没有找到 initial_transform！")
#
#     route_id=config.route_id
#     # 构建单个事件配置
#     event_configuration = {
#         "transform": {
#             "x": f"{transform.location.x:.2f}",
#             "y": f"{transform.location.y:.2f}",
#             "z": f"{transform.location.z:.2f}",
#             "pitch": f"{transform.rotation.pitch:.2f}",
#             "yaw": f"{transform.rotation.yaw:.2f}"
#         },
#         "route_id": route_id
#     }
#
#     # 构建完整场景
#     scenario_data = {
#         "available_scenarios": [
#             {
#                 "center": [
#                     {
#                         "available_event_configurations": [event_configuration],
#                         "scenario_name": "DynamicObjectCrossing"
#                     }
#                 ]
#             }
#         ]
#     }
#
#     # 确保保存目录存在
#     os.makedirs(save_dir, exist_ok=True)
#
#     # 获取当前目录下已有 adv_scn_*.json 文件数，生成下一个序号
#     existing_files = glob(os.path.join(save_dir, "adv_scn_*.json"))
#     next_index = len(existing_files) + 1
#     save_path = os.path.join(save_dir, f"adv_scn_{next_index}.json")
#
#     # 写入文件
#     with open(save_path, "w", encoding="utf-8") as f:
#         json.dump(scenario_data, f, ensure_ascii=False, indent=2)
#
#     print(f">> 场景初始状态已保存至: {save_path}（route_id: {route_id}）")


class DynamicObjectCrossing(BasicScenario):
    """行人横穿场景 - 支持遗传算法优化"""

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(DynamicObjectCrossing, self).__init__("DynamicObjectCrossing-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self._map = CarlaDataProvider.get_map()

        # 基础参数
        self.actor_speed = 1.5  # 行人速度
        self.trigger_distance_threshold = 12.0
        self.position_offset = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}

        # 遗传算法相关
        self.genetic_optimizer = None
        self.optimized_params = None

        # 状态标志
        self.config=config
        self.scenario_triggered = False
        self.tick_count = 0
        self.last_pedestrian_location = None
        self.initial_distance = 999.0
        self.trigger_location = None  # 将在initialize_actorsHK中设置

        # 场景操作
        self.scenario_operation = ScenarioOperation()
        self.ego_max_driven_distance = 150

    def create_behavior(self, scenario_init_action):
        """创建场景行为"""
        if self._should_use_genetic_algorithm(scenario_init_action):
            print(">> Using genetic algorithm for pedestrian crossing optimization")



            self.genetic_optimizer = SkoptGeneticOptimizer(self)
            self.optimized_params = self.genetic_optimizer.optimize_initial_state()

            self.actor_speed = self.optimized_params['actor_speed']
            self.trigger_distance_threshold = self.optimized_params['trigger_distance']
            self.position_offset = self.optimized_params['position_offset']

            opt_info = self.genetic_optimizer.get_optimization_info()
            print(f">> Optimization completed - Best fitness: {opt_info['best_fitness']:.4f}")
        else:
            print(">> Using default parameters for pedestrian crossing")

    def initialize_actorsHK(self):
        """初始化场景参与者"""
        if not self.optimized_params:
            self.genetic_optimizer = SkoptGeneticOptimizer(self)
            self.optimized_params = self.genetic_optimizer.optimize_initial_state()

            self.actor_speed = self.optimized_params['actor_speed']
            self.trigger_distance_threshold = self.optimized_params['trigger_distance']
            self.position_offset = self.optimized_params['position_offset']

        initial_transform = self.optimized_params.get("initial_transform", None)
        if initial_transform is None:
            raise RuntimeError("Optimized initial_transform not found!")

        # ✅ 用优化后的行人位置设置trigger_location
        self.trigger_location = initial_transform.location
        print(
            f">> trigger_location set to pedestrian position: ({self.trigger_location.x:.2f}, {self.trigger_location.y:.2f})")

        # 只生成行人，不生成障碍物！
        pedestrian_bp = self.world.get_blueprint_library().find('walker.pedestrian.0022')
        pedestrian_bp.set_attribute('role_name', 'pedestrian')

        self.actor_transform_list = [initial_transform]
        self.actor_type_list = ['walker.pedestrian.0022']
        # save_dynamic_object_crossing_json_incremental(self.optimized_params,config=self.config)

        try:
            # pedestrian = self.world.spawn_actor(pedestrian_bp, initial_transform)
            # self.other_actors = [pedestrian]
            # self.reference_actor = pedestrian

            self.other_actors = self.scenario_operation.initialize_vehicle_actors(
                self.actor_transform_list, self.actor_type_list
            )
            self.reference_actor = self.other_actors[0]

            # self.scenario_operation.other_actors = self.other_actors

            # 计算初始距离
            ego_loc = self.ego_vehicle.get_location()
            pedestrian_loc = self.reference_actor.get_location()
            self.initial_distance = math.hypot(
                pedestrian_loc.x - ego_loc.x,
                pedestrian_loc.y - ego_loc.y
            )
            self.last_pedestrian_location = pedestrian_loc

            # 计算相对位置
            ego_heading_rad = math.radians(self.ego_vehicle.get_transform().rotation.yaw)
            ego_forward = [math.cos(ego_heading_rad), math.sin(ego_heading_rad)]
            ego_right = [-math.sin(ego_heading_rad), math.cos(ego_heading_rad)]

            vec_to_ped = [pedestrian_loc.x - ego_loc.x, pedestrian_loc.y - ego_loc.y]
            forward_component = vec_to_ped[0] * ego_forward[0] + vec_to_ped[1] * ego_forward[1]
            lateral_component = vec_to_ped[0] * ego_right[0] + vec_to_ped[1] * ego_right[1]

            print(f"\n{'=' * 80}")
            print(f">> [Pedestrian Crossing] Scenario Initialized")
            print(f"{'=' * 80}")
            print(f"   Ego position: ({ego_loc.x:.2f}, {ego_loc.y:.2f})")
            print(f"   Pedestrian position: ({pedestrian_loc.x:.2f}, {pedestrian_loc.y:.2f})")
            print(f"   Pedestrian heading: {initial_transform.rotation.yaw:.2f}°")
            print(f"   Pedestrian speed (when triggered): {self.actor_speed:.2f} m/s")
            print(f"   Initial distance: {self.initial_distance:.2f}m")
            print(f"   Trigger distance: {self.trigger_distance_threshold:.2f}m")
            print(f"   Pedestrian relative position:")
            print(f"     - Ahead of ego: {forward_component:.2f}m")
            print(f"     - Right of ego (roadside): {abs(lateral_component):.2f}m")
            print(f"    Pedestrian will CROSS when distance < {self.trigger_distance_threshold:.2f}m")
            print(f"{'=' * 80}\n")

            # 初始状态：行人静止
            self._set_pedestrian_velocity(self.reference_actor, 0.0)

        except Exception as e:
            self.reference_actor = None
            self.other_actors = []
            print(f">> Failed to spawn pedestrian: {e}")
            import traceback
            traceback.print_exc()

    def update_behavior(self, scenario_action):
        """更新场景行为"""
        assert scenario_action is None, f'{self.name} should receive [None] action.'

        self.tick_count += 1

        if not self.other_actors:
            return

        pedestrian = self.other_actors[0]
        pedestrian_location = pedestrian.get_location()
        ego_location = self.ego_vehicle.get_location()

        # 计算当前距离
        if self.trigger_location is None:
            self.trigger_location = pedestrian_location

        current_distance = math.hypot(
            self.trigger_location.x - ego_location.x,
            self.trigger_location.y - ego_location.y
        )

        # 检查触发条件
        if not self.scenario_triggered:
            if self.tick_count % 3 == 0:
                progress = (self.initial_distance - current_distance) / (
                            self.initial_distance - self.trigger_distance_threshold) * 100.0
                print(
                    f">> [Approaching] distance(ego, pedestrian): {current_distance:.2f}m / {self.trigger_distance_threshold:.2f}m | Progress: {progress:.1f}%")

            if current_distance < self.trigger_distance_threshold:
                self.scenario_triggered = True

                ego_velocity = self.ego_vehicle.get_velocity()
                ego_speed = math.sqrt(ego_velocity.x ** 2 + ego_velocity.y ** 2)

                print(f"\n{'=' * 80}")
                print(f">> [TRIGGERED!] Pedestrian starts crossing!")
                print(f"   Distance: {current_distance:.2f}m < {self.trigger_distance_threshold:.2f}m")
                print(f"   Ego speed: {ego_speed:.2f} m/s")
                print(f"   Pedestrian crosses at: {self.actor_speed:.2f} m/s")
                print(f"{'=' * 80}\n")
            else:
                # 触发前保持静止
                self._set_pedestrian_velocity(pedestrian, 0.0)
                return

        # 触发后，行人横穿道路
        self.scenario_operation.walker_go_straight(self.actor_speed, 0)

        # 调试信息
        # if self.tick_count % 20 == 0:
        #     if self.last_pedestrian_location:
        #         moved_dist = math.hypot(
        #             pedestrian_location.x - self.last_pedestrian_location.x,
        #             pedestrian_location.y - self.last_pedestrian_location.y
        #         )
        #         print(f">> [Tick {self.tick_count}] Pedestrian crossing:")
        #         print(f"   Position: ({pedestrian_location.x:.2f}, {pedestrian_location.y:.2f})")
        #         print(f"   Moved: {moved_dist:.2f}m in 20 ticks")
        #         print(f"   Distance to ego: {current_distance:.2f}m")
        #
        #     self.last_pedestrian_location = pedestrian_location

    def _set_pedestrian_velocity(self, pedestrian, speed):
        """设置行人速度"""
        if speed == 0.0:
            # 完全静止
            pedestrian.set_target_velocity(carla.Vector3D(0, 0, 0))
        else:
            # 按照朝向方向移动
            transform = pedestrian.get_transform()
            forward_vec = transform.rotation.get_forward_vector()
            target_velocity = carla.Vector3D(
                forward_vec.x * speed,
                forward_vec.y * speed,
                0
            )
            pedestrian.set_target_velocity(target_velocity)

    def check_stop_condition(self):
        """检查停止条件"""
        if not self.other_actors:
            return True

        pedestrian = self.other_actors[0]
        ego_loc = self.ego_vehicle.get_location()
        pedestrian_loc = pedestrian.get_location()

        distance = math.hypot(
            pedestrian_loc.x - ego_loc.x,
            pedestrian_loc.y - ego_loc.y
        )

        # 行人距离过远或运行时间过长
        if distance > 50.0 or (self.scenario_triggered and self.tick_count > 500):
            print(f">> [Pedestrian Crossing] Scenario ended")
            print(f"   Final distance: {distance:.1f}m")
            return True

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