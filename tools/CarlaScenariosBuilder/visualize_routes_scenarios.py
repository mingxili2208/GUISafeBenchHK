"""
可视化选定的测试路线和交通物体的触发点信息
"""
import numpy as np
import os
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
from utilities import parse_route, parse_scenarios, select_waypoints, get_view_centers, get_map_centers

# 全局退出标志
exit_flag = False


def on_key_press(event):
    global exit_flag
    if event.key == 'escape':
        exit_flag = True  # 按下ESC时设置退出标志


def draw(ax, map_name, route_waypoints, scenario_config, centers, waypoints_sparse, zoom, mode):
    if zoom:
        draw_zoom(ax, route_waypoints, scenario_config, centers, waypoints_sparse, mode)
    else:
        draw_global(ax, map_name, route_waypoints, scenario_config, waypoints_sparse, mode)


def draw_global(ax, map_name, route_waypoints, scenario_config, waypoints_sparse, mode):
    """
    绘制全局视图，包括测试路线和交通物体触发点
    """
    ax.cla()

    # 绘制稀疏路径点
    ax.plot(waypoints_sparse[:, 0], -waypoints_sparse[:, 1], 'o', color='y')

    # 绘制测试路线
    if mode in ['both', 'route'] and route_waypoints is not None:
        for waypoints in route_waypoints:
            ax.plot(waypoints[:, 0], -waypoints[:, 1], '-o', color='b')
            ax.plot(waypoints[0, 0], -waypoints[0, 1], 'o', color='g')
            ax.plot(waypoints[-1, 0], -waypoints[-1, 1], 'o', color='r')
            ax.text(waypoints[0, 0] + 8, -waypoints[0, 1] + 8, "Start", bbox=dict(facecolor='green', alpha=0.7))
            ax.text(waypoints[-1, 0] + 8, -waypoints[-1, 1] + 8, "End", bbox=dict(facecolor='red', alpha=0.7))

    # 绘制触发点
    if mode in ['both', 'scenario'] and scenario_config is not None:
        waypoints = parse_scenarios(scenario_config)
        start_waypoint = waypoints[0]
        for end_waypoint in waypoints[1:]:
            ax.plot([start_waypoint[0], end_waypoint[0]], [-start_waypoint[1], -end_waypoint[1]], '--', color='b')
            ax.plot(end_waypoint[0], -end_waypoint[1], 'o', color='g')
            ax.text(end_waypoint[0] + 8, -end_waypoint[1] + 8, "Actor", bbox=dict(facecolor='green', alpha=0.7))
        ax.plot(start_waypoint[0], -start_waypoint[1], 'o', color='r')
        ax.text(start_waypoint[0] + 12, -start_waypoint[1] + 8, "Trigger", bbox=dict(facecolor='red', alpha=0.7))

    # 设置视图范围
    centers = get_map_centers(map_name)
    if centers is None:
        center = waypoints_sparse[:, :2].mean(0)
    else:
        center = centers.mean(0)

    dist = np.linalg.norm(waypoints_sparse[:, :2] - center, np.inf, axis=1).max() + 20
    x_min, x_max = center[0] - dist, center[0] + dist
    y_min, y_max = -center[1] - dist, -center[1] + dist
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


def draw_zoom(ax, route_waypoints, scenario_config, centers, waypoints_sparse, mode):
    """
    绘制局部放大视图，包括测试路线和交通物体触发点。
    """
    # 获取中心点
    if route_waypoints is not None:
        geo_center = route_waypoints[0][:, :2].mean(0)
    elif scenario_config is not None:
        waypoints = parse_scenarios(scenario_config)
        geo_center = waypoints[:, :2].mean(0)
    else:
        geo_center = waypoints_sparse[:, :2].mean(0)

    if centers is None:
        center = geo_center
    else:
        center_dists = np.linalg.norm(np.array(centers) - geo_center, axis=1)
        center = centers[center_dists.argmin()]

    dist = 200

    ax.cla()

    # 绘制稀疏路径点
    road_waypoints = select_waypoints(waypoints_sparse, center, dist)
    ax.plot(road_waypoints[:, 0], -road_waypoints[:, 1], 'o', color='y')

    # 绘制测试路线
    if mode in ['both', 'route'] and route_waypoints is not None:
        for waypoints in route_waypoints:
            ax.plot(waypoints[:, 0], -waypoints[:, 1], '-o', color='r')
            ax.plot(waypoints[0, 0], -waypoints[0, 1], 'o', color='g')
            ax.plot(waypoints[-1, 0], -waypoints[-1, 1], 'o', color='b')
            ax.text(waypoints[0, 0] + 8, -waypoints[0, 1] + 8, "Start", bbox=dict(facecolor='green', alpha=0.7))
            ax.text(waypoints[-1, 0] + 8, -waypoints[-1, 1] + 8, "End", bbox=dict(facecolor='red', alpha=0.7))

    # 绘制触发点
    if mode in ['both', 'scenario'] and scenario_config is not None:
        waypoints = parse_scenarios(scenario_config)
        start_waypoint = waypoints[0]
        for end_waypoint in waypoints[1:]:
            ax.plot([start_waypoint[0], end_waypoint[0]], [-start_waypoint[1], -end_waypoint[1]], '--', color='b')
            ax.plot(end_waypoint[0], -end_waypoint[1], 'o', color='g')
            ax.text(end_waypoint[0] + 8, -end_waypoint[1] + 8, "Actor", bbox=dict(facecolor='green', alpha=0.7))
        ax.plot(start_waypoint[0], -start_waypoint[1], 'o', color='r')
        ax.text(start_waypoint[0] + 12, -start_waypoint[1] + 8, "Trigger", bbox=dict(facecolor='red', alpha=0.7))

    x_min, x_max = center[0] - dist, center[0] + dist
    y_min, y_max = -center[1] - dist, -center[1] + dist
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


def set_title(ax, map_name, route_file, scenario_idx, mode, title=None):
    if title is None:
        title = "Left and right click to change route/scenario.\nMiddle click to zoom."
    title = f"Visualizing {mode} on map {map_name}.\nRoute: {route_file}, Route: {scenario_idx}.\n" + title
    ax.set_title(title, fontsize=12, loc='left')


def main(config):
    global exit_flag
    # 初始化交互模式
    plt.ion()

    # 加载路线
    scenario_id = config.scenario
    route_dir = os.path.join(config.save_dir, f"scenario_{scenario_id:02d}_routes")
    route_files = os.listdir(route_dir)
    route_files = list(filter(lambda x: x.lower().endswith(".xml"), route_files))
    route_files.sort()
    route_file_num = len(route_files)
    route_idx = 0
    route_file = os.path.join(route_dir, route_files[route_idx])
    route_waypoints, maps_names = parse_route(route_file)
    map_name = maps_names[0]

    # 加载触发点
    scenario_dir = os.path.join(config.save_dir, f"scenarios/scenario_{scenario_id:02d}.json")
    with open(scenario_dir, 'r') as f:
        scenario_json = json.load(f)
    scenario_configs = []
    for available_scenarios in scenario_json["available_scenarios"]:
        for map_name in available_scenarios:
            for available_event_configurations in available_scenarios[map_name]:
                for event_configuration in available_event_configurations["available_event_configurations"]:
                    if config.map == "None" or config.map == map_name:
                        event_configuration["map"] = map_name
                        scenario_configs.append(event_configuration)
    scenario_num = len(scenario_configs)
    current_rounte_filename = route_files[route_idx]
    # scenario_idx是形如scenario_01_route_00_weather_00中route_xx后接的数字
    scenario_idx = int(current_rounte_filename.split('_')[3])
    zoom = False
    mode = config.mode

    # 稀疏路径点
    waypoints_sparse = np.load(f"map_waypoints/{map_name}/sparse.npy")
    centers = get_view_centers(map_name)

    assert route_file_num > 0 and scenario_num > 0, "No route or scenario to visualize."

    def onclick(event):
        nonlocal route_idx, scenario_idx, zoom, map_name, route_file, route_waypoints, waypoints_sparse, centers

        new_route_id = route_idx

        if int(event.button) == 1:
            # 左键切换到下一条路线
            new_route_id = (route_idx + 1) % route_file_num
        elif int(event.button) == 3:
            # 右键切换到上一条路线
            new_route_id = (route_idx - 1) % route_file_num
        elif int(event.button) == 2:
            # 中键缩放
            zoom = not zoom

        if new_route_id != route_idx:
            route_idx = new_route_id
            route_file = os.path.join(route_dir, route_files[route_idx])
            route_waypoints, map_names = parse_route(route_file)
            if map_name != map_names[0]:
                map_name = map_names[0]
                waypoints_sparse = np.load(f"map_waypoints/{map_name}/sparse.npy")
                centers = get_view_centers(map_name)

        if 1 <= int(event.button) <= 3:
            current_rounte_filename = route_files[route_idx]
            scenario_idx = int(current_rounte_filename.split('_')[3])
            draw(ax, map_name, route_waypoints, scenario_configs[scenario_idx], centers, waypoints_sparse, zoom, mode)
            set_title(ax, map_name, route_file, scenario_idx, mode)
            plt.draw()

    # 可视化
    fig, ax = plt.subplots(figsize=(32, 32))
    draw(ax, map_name, route_waypoints, scenario_configs[scenario_idx], centers, waypoints_sparse, zoom, mode)
    set_title(ax, map_name, route_file, scenario_idx, mode)

    # 绑定事件
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    plt.show()

    # 主循环
    while not exit_flag:
        plt.pause(0.1)

    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default="center")
    parser.add_argument('--save_dir', type=str, default="scenario_data/center")
    parser.add_argument('--scenario', type=int, default=7)
    parser.add_argument('--mode', type=str, choices=['route', 'scenario', 'both'], default='both',
                        help="Visualization mode: 'route', 'scenario', or 'both'.")

    args = parser.parse_args()

    main(args)
