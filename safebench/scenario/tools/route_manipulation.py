"""
根据xml文件中定义的起始位置,从预先解析的map中的waypoints中获得从起点到终点的路径
"""

from safebench.carla_agents.navigation.local_planner import RoadOption
import matplotlib
matplotlib.use('TkAgg')
from safebench.carla_agents.navigation.global_route_planner import GlobalRoutePlanner


def interpolate_trajectory(world, waypoints_trajectory, hop_resolution=1.0):
    """
        Given some raw keypoints interpolate a full dense trajectory to be used by the user.
            :param world: an reference to the CARLA world so we can use the planner
            :param waypoints_trajectory: the current coarse trajectory
            :param hop_resolution: is the resolution, how dense is the provided trajectory going to be made
            :return: the full interpolated route both in GPS coordinates and also in its original form.
    """

    grp = GlobalRoutePlanner(world.get_map(), hop_resolution)
    route = []

    if len(waypoints_trajectory) == 1:
        route.append((waypoints_trajectory[0], RoadOption.VOID))

    for i in range(len(waypoints_trajectory) - 1):   # Goes until the one before the last.
        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        interpolated_trace = grp.trace_route(waypoint, waypoint_next)
        for wp_tuple in interpolated_trace:
            route.append((wp_tuple[0].transform, wp_tuple[1]))

    return route
