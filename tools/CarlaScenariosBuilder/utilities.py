import numpy as np
import os
import xml.etree.cElementTree as ET
import shutil
import math
import matplotlib.pyplot as plt


def _indent_xml(tree_or_root, space="  ", level=0):
    """
    Pretty-print XML for both Python 3.8 and 3.9+.
    """
    if hasattr(ET, "indent"):
        ET.indent(tree_or_root, space=space, level=level)
        return

    root = tree_or_root.getroot() if hasattr(tree_or_root, "getroot") else tree_or_root
    _indent_element(root, space=space, level=level)


def _indent_element(elem, space="  ", level=0):
    indent = "\n" + level * space
    child_indent = indent + space

    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = child_indent

        for child in elem:
            _indent_element(child, space=space, level=level + 1)

        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = indent
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def build_route(waypoints, route_id, town, save_file,
                weathers={'cloudiness': '0.0', 'dust_storm': '0.0', 'fog_density': '0.0', 'fog_distance': '0.0',
                          'fog_falloff': '0.0', 'mie_scattering_scale': '0.0', 'precipitation': '0.0',
                          'precipitation_deposits': '0.0', 'rayleigh_scattering_scale': '0.0331',
                          'scattering_intensity': '0.0', 'sun_altitude_angle': '0.0', 'sun_azimuth_angle': '0.0',
                          'wetness': '0.0', 'wind_intensity': '0.0'}):
    """
    :param waypoints: list of waypoints, each waypoint is a list of [x, y, z, pitch, yaw, roll]
    :param route_id: the id of the route
    :param town: 地图名称
    :param save_file: 保存的文件路径
    :param weathers: 根据Carla自带的天气信息进行设置，默认为Default天气
    """
    root = ET.Element("routes")
    route = ET.SubElement(root, "route", id=f'{route_id}', town=town)

    ET.SubElement(
        route,
        "weather",
        weathers
    )  # 可以在这里指出行驶路径的天气参数

    for waypoint in waypoints:
        x, y, z, pitch, yaw, roll = waypoint
        ET.SubElement(
            route,
            "waypoint",
            pitch=f"{pitch:.2f}",
            roll=f"{roll:.2f}",
            x=f"{x:.2f}",
            y=f"{y:.2f}",
            yaw=f"{yaw:.2f}",
            z=f"{z+2:.2f}"
        )

    tree = ET.ElementTree(root)
    # 将xml文件格式并且写入指定的文件save_file中
    _indent_xml(tree)
    tree.write(save_file, encoding='utf-8', xml_declaration=True)


def build_scenarios(waypoints, side_marks):
    """
    根据传入的路径点来生成场景配置
    """
    # 初始化空列表,用于存储格式化后的路径点
    scenario_waypoints = []
    for waypoint in waypoints:
        x, y, z, pitch, yaw, roll = waypoint
        point = {
            "pitch": f'{pitch:.2f}',
            "x": f'{x:.2f}',
            "y": f'{y:.2f}',
            "yaw": f'{yaw:.2f}',
            "z": f'{z+2:.2f}'
        }
        scenario_waypoints.append(point)

    # 使用第一个路径点作为场景的触发点，其他路径点作为其他actor的生成点
    config = {}
    # group other actors by side_marks ('left' or 'right')
    other_actors = {"left": [], "right": [], "center": []}
    for mark, actor in zip(side_marks, scenario_waypoints[1:]):
        if mark == "left":
            other_actors["left"].append(actor)
        elif mark == "right":
            other_actors["right"].append(actor)
        else:
            other_actors["center"].append(actor)
    # only keep non-empty sides
    config["other_actors"] = {k: v for k, v in other_actors.items() if v}
    config["transform"] = scenario_waypoints[0]

    return config


def parse_route(route_file):
    """
    解析路径文件,返回路径点和地图名称
    """
    tree = ET.parse(route_file)
    waypoints = []
    map_names = []
    for route in tree.iter("route"):
        map_name = route.attrib['town']
        map_names.append(map_name)
        waypoint_list = []  # the list of waypoints that can be found on this route
        for waypoint in route.iter('waypoint'):
            pitch = float(waypoint.attrib['pitch'])
            roll = float(waypoint.attrib['roll'])
            yaw = float(waypoint.attrib['yaw'])
            x = float(waypoint.attrib['x'])
            y = float(waypoint.attrib['y'])
            z = float(waypoint.attrib['z'])
            waypoint_list.append([x, y, z, pitch, yaw, roll])
        waypoints.append(waypoint_list)
    return np.asarray(waypoints), map_names


def parse_scenarios(scenario_config):
    """
    解析场景配置文件,返回路径点
    """
    waypoints = []
    trigger_point = scenario_config["transform"]
    x, y, z = float(trigger_point["x"]), float(trigger_point["y"]), float(trigger_point["z"])
    pitch, yaw = float(trigger_point["pitch"]), float(trigger_point["yaw"])
    roll = 0.0
    waypoints.append([x, y, z, pitch, yaw, roll])

    if "other_actors" in scenario_config:
        for actor_type in ["left", "right", "front"]:
            if actor_type in scenario_config["other_actors"]:
                for actor_config in scenario_config["other_actors"][actor_type]:
                    x, y, z = float(actor_config["x"]), float(actor_config["y"]), float(actor_config["z"])
                    pitch, yaw = float(actor_config["pitch"]), float(actor_config["yaw"])
                    roll = 0.0
                    waypoints.append([x, y, z, pitch, yaw, roll])
    return np.asarray(waypoints)


def select_waypoints(waypoints, center, distance):
    # 选择在给定中心点center附近距离小于distance的路径点
    mask = np.linalg.norm(waypoints[:, :2] - center, np.inf, axis=1) < distance
    waypoints = waypoints[mask]
    return waypoints


def rotate_waypoints(origin_waypoints, center, theta):
    """
    将路径点围绕着一个指定的中心点旋转一个给定的角度,并更新路径点的yaw角度
    """

    # waypoint format: x, y, z, pitch, yaw, roll
    m = np.asarray([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    center = np.array(center).reshape((1, 2))
    rotated_waypoints = origin_waypoints.copy()
    rotated_waypoints[:, :2] = (origin_waypoints[:, :2] - center) @ m.T + center
    # change the yaw angle
    rotated_waypoints[:, 4] = (rotated_waypoints[:, 4] + theta / np.pi * 180) % 360
    return rotated_waypoints


def get_nearist_waypoints(waypoint, waypoints):
    waypoint = waypoint.reshape((1, -1))
    waypoints_dist = np.linalg.norm(waypoints[:, :2] - waypoint[:, :2], axis=1)
    return waypoints_dist.argmin(), waypoints_dist.min()


def _get_batch_centers(x_num, y_num):
    map_size = 500
    centers = []
    for j in range(x_num):
        for i in range(y_num):
            centers.append([i, j])
    # 将中心点坐标转化为numpy数组,并且乘以地图尺寸
    centers = np.asarray(centers) * map_size
    # 将中心点坐标减去中心点坐标的均值,使中心对齐
    centers = centers - centers.mean(0)
    return centers


def _infer_map_center_from_waypoints(map_name):
    waypoint_file = os.path.join(
        os.path.dirname(__file__),
        "map_waypoints",
        map_name,
        "sparse.npy",
    )
    if not os.path.isfile(waypoint_file):
        return None

    waypoints = np.load(waypoint_file)
    if len(waypoints.shape) != 2 or waypoints.shape[1] < 2 or len(waypoints) == 0:
        raise ValueError(f"Waypoint file for map {map_name} is invalid: {waypoint_file}")

    min_xy = waypoints[:, :2].min(axis=0)
    max_xy = waypoints[:, :2].max(axis=0)
    center = (min_xy + max_xy) / 2
    return np.asarray([center])


def get_map_centers(map_name):
    # 处理香港地图的时候,人为指定地图中心
    if map_name == "center":
        centers = np.asarray([[750, 0]])
    elif map_name == "Town10HD_Opt":
        centers = np.asarray([[0, 0]])
    elif map_name == "WangJiao_2_w":
        centers = np.asarray([[-300, -400]])
    else:
        centers = _infer_map_center_from_waypoints(map_name)
        if centers is None:
            raise ValueError(
                f"Map {map_name} is not supported and no sparse waypoint file was found."
            )
    return centers


def get_view_centers(map_name):
    map_centers = get_map_centers(map_name)
    if map_centers is None:
        return None

    view_centers = []
    for map_center in map_centers:
        local_centers = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    # ignore the map center
                    continue
                local_centers.append([i * 100, j * 100])
        centers = np.asarray(local_centers) + map_center
        view_centers.append(centers)
    return np.vstack(view_centers)


def draw_waypoint_background(ax, road_waypoints, point_color='y', arrow_color='y', arrow_length=4.0, max_arrows=250):
    """
    使用 scatter + quiver 批量绘制地图点和方向。
    这样比为每个点单独创建一个 arrow patch 流畅很多。
    """
    if road_waypoints is None or len(road_waypoints) == 0:
        return

    ax.scatter(
        road_waypoints[:, 0],
        -road_waypoints[:, 1],
        s=10,
        c=point_color,
        linewidths=0,
        alpha=0.85,
        zorder=1,
    )

    arrow_stride = max(1, int(np.ceil(len(road_waypoints) / max_arrows)))
    arrow_waypoints = road_waypoints[::arrow_stride]
    yaw = np.deg2rad(arrow_waypoints[:, 4])
    dx = arrow_length * np.cos(yaw)
    dy = arrow_length * np.sin(yaw)

    ax.quiver(
        arrow_waypoints[:, 0],
        -arrow_waypoints[:, 1],
        dx,
        -dy,
        angles='xy',
        scale_units='xy',
        scale=1,
        color=arrow_color,
        width=0.002,
        headwidth=3.5,
        headlength=4.5,
        headaxislength=4.0,
        alpha=0.85,
        zorder=2,
    )


def copy_routes_and_scenarios(old_map_name, new_map_name):
    old_map_dir = f"scenario_origin/{old_map_name}"
    new_map_dir = f"scenario_origin/{new_map_name}"
    action = input(f"Careful! This action will removed all files in {new_map_dir}. You you want to pressed? [yes/no]\n")
    # 如果新地图目录已经存在,并且用户选择了yes,则删除新地图目录
    if os.path.isdir(new_map_dir) and action == 'yes':
        shutil.rmtree(new_map_dir)

    coord_shift = get_map_centers(new_map_name)[0: 1] - get_map_centers(old_map_name)[0: 1]
    for dir_path, dir_names, file_names in os.walk(old_map_dir):
        for file_name in file_names:
            # 只处理后缀为.npy的文件
            if file_name.endswith('.npy'):
                old_file_path = os.path.join(dir_path, file_name)
                new_dir = dir_path.replace(old_map_dir, new_map_dir)
                new_file_path = os.path.join(new_dir, file_name)
                os.makedirs(new_dir, exist_ok=True)
                waypoints = np.load(old_file_path)
                waypoints[:, :2] += coord_shift
                np.save(new_file_path, waypoints)
    print(f"copy scenarios from {old_map_name} to {new_map_name}")


def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return (norm_target, d_angle)


def compute_yaw_angle(yaw1, yaw2):
    """
    计算两个 yaw 之间的最小夹角（带符号）
    yaw 单位：度
    返回：绝对值（夹角大小），可选返回符号
    """
    diff = (yaw1 - yaw2 + 180) % 360 - 180  # 归一化到[-180, 180]
    return abs(diff), diff


if __name__ == '__main__':
    # copy_routes_and_scenarios("town_4intersection_2lane", "Town_Safebench_Light")

    # 从sparse.npy文件中加载路径点,并且绘制路径点
    a = np.load('map_waypoints/town_4intersection_2lane/sparse.npy')
    plt.scatter(a[:, 0], a[:, 1], color='b')
    # 获取给定地图的中心点坐标,并用红色散点图绘制这些中心点
    a = get_view_centers("town_4intersection_2lane")
    plt.scatter(a[:, 0], a[:, 1], color='r')
    plt.show()


