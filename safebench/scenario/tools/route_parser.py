import json
import math
import xml.etree.ElementTree as ET
import carla
from safebench.scenario.scenario_manager.scenario_config import ScenarioConfig
TRIGGER_THRESHOLD = 2.0  # Threshold to say if a trigger position is new or repeated, works for matching positions
TRIGGER_ANGLE_THRESHOLD = 10  # Threshold to say if two angles can be considering matching when matching transforms.


class RouteParser(object):
    """
    解析路线和场景配置参数
    """

    @staticmethod
    def parse_annotations_file(annotation_filename):
        """
        返回交通物体生成点和场景触发位置的注释
            :param annotation_filename: 场景配置的json文件
            :return:
        """
        with open(annotation_filename, 'r') as f:
            annotation_dict = json.loads(f.read())

        final_dict = {}
        for town_dict in annotation_dict['available_scenarios']:
            final_dict.update(town_dict)
        return final_dict

    @staticmethod
    def parse_routes_file(route_filename, scenario_file, single_route=None):
        """
        返回路线元素的列表
            :param route_filename: 测试路线的xml文件
            :param single_route: 如果不是None,则只返回这个路线
            :return: 包含路线的路径点, ID和城镇的列表
        """

        list_route_descriptions = []
        tree = ET.parse(route_filename)
        for route in tree.iter("route"):
            route_id = route.attrib['id']
            if single_route and route_id != single_route:
                continue

            new_config = ScenarioConfig()
            new_config.town = route.attrib['town']
            new_config.route_region = route.attrib['region'] if 'region' in route.attrib else None
            new_config.name = "RouteScenario_{}".format(route_id)
            new_config.weather = RouteParser.parse_weather(route)
            new_config.scenario_file = scenario_file

            waypoint_list = []  # the list of waypoints that can be found on this route
            for waypoint in route.iter('waypoint'):
                if len(waypoint_list) == 0:  # 只有将路径的xx.xml文件得到的第一个waypoint的位置和角度信息作为初始位置
                    pitch = float(waypoint.attrib['pitch'])
                    roll = float(waypoint.attrib['roll'])
                    yaw = float(waypoint.attrib['yaw'])
                    x = float(waypoint.attrib['x'])
                    y = float(waypoint.attrib['y'])
                    z = float(waypoint.attrib['z']) + 2.0  # avoid collision to the ground
                    initial_pose = carla.Transform(carla.Location(x, y, z), carla.Rotation(roll=roll, pitch=pitch, yaw=yaw))
                    new_config.initial_transform = initial_pose
                    new_config.initial_pose = initial_pose
                waypoint_list.append(carla.Location(x=float(waypoint.attrib['x']), y=float(waypoint.attrib['y']), z=float(waypoint.attrib['z'])))

            new_config.trajectory = waypoint_list
            list_route_descriptions.append(new_config)

        return list_route_descriptions

    @staticmethod
    def parse_weather(route):
        """
            Returns a carla.WeatherParameters with the corresponding weather for that route. 
            If the route has no weather attribute, the default one is triggered.
        """
        route_weather = route.find("weather")
        if route_weather is None:
            weather = carla.WeatherParameters(sun_altitude_angle=70)
        else:
            weather = carla.WeatherParameters()
            for weather_attrib in route.iter("weather"):
                if 'cloudiness' in weather_attrib.attrib:
                    weather.cloudiness = float(weather_attrib.attrib['cloudiness'])
                if 'dust_storm' in weather_attrib.attrib:
                    weather.dust_storm = float(weather_attrib.attrib['dust_storm'])
                if 'fog_density' in weather_attrib.attrib:
                    weather.fog_density = float(weather_attrib.attrib['fog_density'])
                if 'fog_distance' in weather_attrib.attrib:
                    weather.fog_distance = float(weather_attrib.attrib['fog_distance'])
                if 'fog_falloff' in weather_attrib.attrib:
                    weather.fog_falloff = float(weather_attrib.attrib['fog_falloff'])
                if 'mie_scattering_scale' in weather_attrib.attrib:
                    weather.mie_scattering_scale = float(weather_attrib.attrib['mie_scattering_scale'])
                if 'precipitation' in weather_attrib.attrib:
                    weather.precipitation = float(weather_attrib.attrib['precipitation'])
                if 'precipitation_deposits' in weather_attrib.attrib:
                    weather.precipitation_deposits = float(weather_attrib.attrib['precipitation_deposits'])
                if 'rayleigh_scattering_scale' in weather_attrib.attrib:
                    weather.rayleigh_scattering_scale = float(weather_attrib.attrib['rayleigh_scattering_scale'])
                if 'scattering_intensity' in weather_attrib.attrib:
                    weather.scattering_intensity = float(weather_attrib.attrib['scattering_intensity'])
                if 'sun_azimuth_angle' in weather_attrib.attrib:
                    weather.sun_azimuth_angle = float(weather_attrib.attrib['sun_azimuth_angle'])
                if 'sun_altitude_angle' in weather_attrib.attrib:
                    weather.sun_altitude_angle = float(weather_attrib.attrib['sun_altitude_angle'])
                if 'wetness' in weather_attrib.attrib:
                    weather.wetness = float(weather_attrib.attrib['wetness'])
                if 'wind_intensity' in weather_attrib.attrib:
                    weather.wind_intensity = float(weather_attrib.attrib['wind_intensity'])

        return weather

    @staticmethod
    def check_trigger_position(new_trigger, existing_triggers):
        """
        检查一个新的触发位置是否已经存在于现有的触发位置列表中,如果存在，则返回现有触发位置的ID
        该函数可以有效地避免重复添加相同的触发位置,确保每个触发位置只添加一次
            :param new_trigger: 一个字典,包含新的触发位置的坐标和角度信息
            :param existing_triggers:  一个字典,包含现有触发位置的ID及其=对应的坐标和角度信息
            :return:
        """
        for trigger_id in existing_triggers.keys():
            trigger = existing_triggers[trigger_id]
            dx = trigger['x'] - new_trigger['x']
            dy = trigger['y'] - new_trigger['y']
            distance = math.sqrt(dx * dx + dy * dy)

            dyaw = (trigger['yaw'] - new_trigger['yaw']) % 360
            if distance < TRIGGER_THRESHOLD and (dyaw < TRIGGER_ANGLE_THRESHOLD or dyaw > (360 - TRIGGER_ANGLE_THRESHOLD)):
                return trigger_id
        return None

    @staticmethod
    def convert_waypoint_float(waypoint):
        # 将航迹点的坐标和角度转换为浮点数
        waypoint['x'] = float(waypoint['x'])
        waypoint['y'] = float(waypoint['y'])
        waypoint['z'] = float(waypoint['z'])
        waypoint['yaw'] = float(waypoint['yaw'])

    def scan_route_for_scenariosHK(route_name, route_id, world_annotations):
        """
        返回可能在此路线中发生的场景的列表
        通过将场景中的位置与路线描述进行匹配
        """
        # the triggers dictionaries
        existent_triggers = {}

        # We have a table of IDs and trigger positions associated,包含ID和触发位置的字典
        possible_scenarios = {}

        # Keep track of the trigger ids being added
        latest_trigger_id = 0

        for town_name in world_annotations.keys():
            if town_name != route_name:
                continue

            # scenarios是一个列表(可以理解为按照功能场景划分的),每个元素是一个字典,包含场景的名称和可用的事件配置
            scenarios = world_annotations[town_name]
            for scenario in scenarios:  # For each existent scenario
                if "scenario_name" not in scenario:
                    raise ValueError('Scenario type not found in scenario description')

                scenario_name = scenario["scenario_name"]
                # 每一条测试路径对应一个交通物体的触发点
                event = scenario["available_event_configurations"][route_id]

                waypoint = {
                    "pitch": event['transform']['pitch'],
                    "x": event['transform']['x'],
                    "y": event['transform']['y'],
                    "yaw": event['transform']['yaw'],
                    "z": event['transform']['z']
                }

                RouteParser.convert_waypoint_float(waypoint)
                # We match a location for this scenario, create a scenario object so this scenario can be instantiated later
                other_vehicles = event['other_actors']
                # scenario_description是一个字典,包含场景的名称、其他参与者、场景触发点的位置以及trajectory中的匹配到的位置索引
                scenario_description = {
                    'name': scenario_name,
                    'other_actors': other_vehicles,
                    'trigger_position': waypoint,
                }
                # 检查一个新的触发位置是否已经存在于现有的触发位置列表中，如果不存在，则添加新的触发位置，并更新可能的场景列表
                trigger_id = RouteParser.check_trigger_position(waypoint, existent_triggers)
                if trigger_id is None:  # 如果触发点不存在,则说明历史existent_triggers中没有这个触发点,则需要添加
                    # This trigger does not exist create a new reference on existent triggers
                    existent_triggers.update({latest_trigger_id: waypoint})
                    # Update a reference for this trigger on the possible scenarios
                    possible_scenarios.update({latest_trigger_id: []})
                    trigger_id = latest_trigger_id
                    # Increment the latest trigger
                    latest_trigger_id += 1
                possible_scenarios[trigger_id].append(scenario_description)

        return possible_scenarios, existent_triggers
