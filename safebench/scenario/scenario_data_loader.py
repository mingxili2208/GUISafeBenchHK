"""
1、计算插值轨迹
2、检查给定路径是否与当前路径集合中的任何路径重叠
3、场景数据加载器
"""
import numpy as np
from safebench.scenario.tools.route_manipulation import interpolate_trajectory


def calculate_interpolate_trajectory(config, world):
    # get route
    origin_waypoints_loc = []
    for loc in config.trajectory:
        origin_waypoints_loc.append(loc)
    route = interpolate_trajectory(world, origin_waypoints_loc, 2.0)

    # get [x, y] along the route
    waypoint_xy = []
    for transform_tuple in route:
        waypoint_xy.append([transform_tuple[0].location.x, transform_tuple[0].location.y])

    return waypoint_xy


def check_route_overlap(current_routes, route, distance_threshold=10):
    # 用于检查给定的路线是否与当前给定的路线集合产生重叠
    overlap = False
    for current_route in current_routes:
        for current_waypoint in current_route:
            for waypoint in route:
                distance = np.linalg.norm([current_waypoint[0] - waypoint[0], current_waypoint[1] - waypoint[1]])
                if distance < distance_threshold:
                    overlap = True
                    return overlap

    return overlap


class ScenarioDataLoader:
    """
    加载和管理场景数据,主要功能是初始化场景数据、重置场景索引计数器、选择不重叠的场景索引以及采样场景
    """
    def __init__(self, config_lists, num_scenario, town, world):
        self.num_scenario = num_scenario  # 代表一次性同时执行的场景数量
        self.config_lists = config_lists  # 标记有route, scenario.josn等配置文件
        self.town = town.lower()  # 将城镇名称转换为小写
        self.world = world
        self.routes = []

        # If using CARLA maps, manually check overlaps
        if 'safebench' not in self.town:
            for config in config_lists:
                self.routes.append(calculate_interpolate_trajectory(config, world))

        # 在当前城镇下,一共设计了多少个测试场景
        self.num_total_scenario = len(config_lists)
        self.reset_idx_counter()

    def reset_idx_counter(self):
        # 将场景索引重置为从0到总场景数量的列表
        self.scenario_idx = list(range(self.num_total_scenario))

    def _select_non_overlap_idx_safebench(self, remaining_ids, sample_num):
        # 根据区域选择不重叠的场景索引
        selected_idx = []
        current_regions = []
        for s_i in remaining_ids:
            if self.config_lists[s_i].route_region not in current_regions:
                selected_idx.append(s_i)
                if self.config_lists[s_i].route_region != "random":
                    current_regions.append(self.config_lists[s_i].route_region)
            if len(selected_idx) >= sample_num:
                break
        return selected_idx

    def _select_non_overlap_idx_carla(self, remaining_ids, sample_num):
        # 根据轨迹选择不重叠的场景索引
        selected_idx = []
        selected_routes = []
        for s_i in remaining_ids:
            if not check_route_overlap(selected_routes, self.routes[s_i]):
                selected_idx.append(s_i)
                selected_routes.append(self.routes[s_i])
            if len(selected_idx) >= sample_num:
                break
        return selected_idx

    def _select_non_overlap_idx(self, remaining_ids, sample_num):
        if 'safebench' in self.town:
            # If using SafeBench map, check overlap based on regions
            return self._select_non_overlap_idx_safebench(remaining_ids, sample_num)
        else:
            # If using CARLA maps, manually check overlaps
            return self._select_non_overlap_idx_carla(remaining_ids, sample_num)

    def __len__(self):
        return len(self.scenario_idx)

    def sampler(self):
        # self.num_scenario: 代表的是同时运行的场景数量,self.scenario_idx: 解析好的待测场景索引
        sample_num = np.min([self.num_scenario, len(self.scenario_idx)])
        # 选择第几个场景进行测试
        selected_idx = self._select_non_overlap_idx(self.scenario_idx, sample_num)
        selected_scenario = []
        for s_i in selected_idx:  # 遍历以适应可能得多个场景并行测试
            selected_scenario.append(self.config_lists[s_i])  # 根据返回的场景索引,确定测试场景的配置文件
            self.scenario_idx.remove(s_i)  # 选择完一个场景后,将其从待测场景索引中移除
        # 校验将要测试的场景数量不能超过设定的场景数量
        assert len(selected_scenario) <= self.num_scenario, f"number of scenarios is larger than {self.num_scenario}"
        return selected_scenario, len(selected_scenario)

