"""
场景BEV渲染和测试可视化
"""
import math
import weakref
import carla
import pygame


# Colors
COLOR_BUTTER_0 = pygame.Color(252, 233, 79)
COLOR_BUTTER_1 = pygame.Color(237, 212, 0)
COLOR_BUTTER_2 = pygame.Color(196, 160, 0)

COLOR_ORANGE_0 = pygame.Color(252, 175, 62)
COLOR_ORANGE_1 = pygame.Color(245, 121, 0)
COLOR_ORANGE_2 = pygame.Color(209, 92, 0)

COLOR_CHOCOLATE_0 = pygame.Color(233, 185, 110)
COLOR_CHOCOLATE_1 = pygame.Color(193, 125, 17)
COLOR_CHOCOLATE_2 = pygame.Color(143, 89, 2)

COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)
COLOR_CHAMELEON_2 = pygame.Color(78, 154, 6)

COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
COLOR_SKY_BLUE_1 = pygame.Color(52, 101, 164)
COLOR_SKY_BLUE_2 = pygame.Color(32, 74, 135)

COLOR_PLUM_0 = pygame.Color(173, 127, 168)
COLOR_PLUM_1 = pygame.Color(117, 80, 123)
COLOR_PLUM_2 = pygame.Color(92, 53, 102)

COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

COLOR_ALUMINIUM_0 = pygame.Color(238, 238, 236)
COLOR_ALUMINIUM_1 = pygame.Color(211, 215, 207)
COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)
COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)
COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
COLOR_ALUMINIUM_4_5 = pygame.Color(66, 62, 64)
COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)


COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)


class Util(object):
    @staticmethod
    def blits(destination_surface, source_surfaces, rect=None, blend_mode=0):
        for surface in source_surfaces:
            destination_surface.blit(surface[0], surface[1], rect, blend_mode)

    @staticmethod
    def length(v):
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

    @staticmethod
    def get_bounding_box(actor):
        bb = actor.trigger_volume.extent
        corners = [
            carla.Location(x=-bb.x, y=-bb.y),
            carla.Location(x=bb.x, y=-bb.y),
            carla.Location(x=bb.x, y=bb.y),
            carla.Location(x=-bb.x, y=bb.y),
            carla.Location(x=-bb.x, y=-bb.y)
        ]
        corners = [x + actor.trigger_volume.location for x in corners]
        t = actor.get_transform()
        t.transform(corners)
        return corners


class MapImage(object):
    """
    生成整个城镇的地图
    """
    def __init__(self, carla_world, carla_map, pixels_per_meter, logger):
        # 记录日志，绘制整个城镇的地图
        logger.log('>> Drawing the map of the entire town. This may take a while...')
        self._pixels_per_meter = pixels_per_meter
        self.scale = 1.0

        # 生成路径点, 确定地图中路径点的边界，并在每个方向上增加一定的边距
        waypoints = carla_map.generate_waypoints(2.0)
        margin = 50
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        self.width = max(max_x - min_x, max_y - min_y)
        self._world_offset = (min_x, min_y)

        # 计算地图表面的宽度(以像素为单位)
        width_in_pixels = int(self._pixels_per_meter * self.width)
        # 创建一个pygame表面对象,用于绘制整个地图
        self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()

        # 渲染地图
        self.draw_road_map(self.big_map_surface, carla_world, carla_map, self.world_to_pixel, self.world_to_pixel_width)
        self.surface = self.big_map_surface

    def draw_road_map(self, map_surface, carla_world, carla_map, world_to_pixel, world_to_pixel_width):
        # 设置背景为黑色
        map_surface.fill(COLOR_BLACK)
        precision = 0.05

        def lane_marking_color_to_tango(lane_marking_color):
            # 将车道标记颜色转换为Tango颜色
            tango_color = COLOR_BLACK

            if lane_marking_color == carla.LaneMarkingColor.White:
                tango_color = COLOR_ALUMINIUM_2

            elif lane_marking_color == carla.LaneMarkingColor.Blue:
                tango_color = COLOR_SKY_BLUE_0

            elif lane_marking_color == carla.LaneMarkingColor.Green:
                tango_color = COLOR_CHAMELEON_0

            elif lane_marking_color == carla.LaneMarkingColor.Red:
                tango_color = COLOR_SCARLET_RED_0

            elif lane_marking_color == carla.LaneMarkingColor.Yellow:
                tango_color = COLOR_ORANGE_0

            return tango_color

        def draw_solid_line(surface, color, closed, points, width):
            # 绘制实线
            if len(points) >= 2:
                pygame.draw.lines(surface, color, closed, points, width)

        def draw_broken_line(surface, color, closed, points, width):
            # 绘制虚线
            broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]
            for line in broken_lines:
                pygame.draw.lines(surface, color, closed, line, width)

        def get_lane_markings(lane_marking_type, lane_marking_color, waypoints, sign):
            # 获取车道标记
            margin = 0.25
            marking_1 = [world_to_pixel(lateral_shift(w.transform, sign * w.lane_width * 0.5)) for w in waypoints]
            if lane_marking_type == carla.LaneMarkingType.Broken or (lane_marking_type == carla.LaneMarkingType.Solid):
                return [(lane_marking_type, lane_marking_color, marking_1)]
            else:
                marking_2 = [world_to_pixel(lateral_shift(w.transform, sign * (w.lane_width * 0.5 + margin * 2))) for w in waypoints]
                if lane_marking_type == carla.LaneMarkingType.SolidBroken:
                    return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1), (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]
                elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:
                    return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1), (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
                elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
                    return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1), (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]

                elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
                    return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1), (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]

            return [(carla.LaneMarkingType.NONE, carla.LaneMarkingColor.Other, [])]

        def draw_lane(surface, lane, color):
            # 绘制车道
            for side in lane:
                lane_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in side]
                lane_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in side]

                polygon = lane_left_side + [x for x in reversed(lane_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(surface, color, polygon, 5)
                    pygame.draw.polygon(surface, color, polygon)

        def draw_lane_marking(surface, waypoints):
            # 绘制车道标记
            # 左侧
            draw_lane_marking_single_side(surface, waypoints[0], -1)
            # 右侧
            draw_lane_marking_single_side(surface, waypoints[1], 1)

        def draw_lane_marking_single_side(surface, waypoints, sign):
            previous_marking_type = carla.LaneMarkingType.NONE
            previous_marking_color = carla.LaneMarkingColor.Other

            markings_list = []
            temp_waypoints = []
            current_lane_marking = carla.LaneMarkingType.NONE
            for sample in waypoints:
                lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking

                if lane_marking is None:
                    continue

                marking_type = lane_marking.type
                marking_color = lane_marking.color

                if current_lane_marking != marking_type:
                    markings = get_lane_markings(
                        previous_marking_type,
                        lane_marking_color_to_tango(previous_marking_color),
                        temp_waypoints,
                        sign
                    )
                    current_lane_marking = marking_type

                    for marking in markings:
                        markings_list.append(marking)

                    temp_waypoints = temp_waypoints[-1:]
                else:
                    temp_waypoints.append((sample))
                    previous_marking_type = marking_type
                    previous_marking_color = marking_color

            # 添加最后一个标记
            last_markings = get_lane_markings(
                previous_marking_type,
                lane_marking_color_to_tango(previous_marking_color),
                temp_waypoints,
                sign
            )
            for marking in last_markings:
                markings_list.append(marking)

            for markings in markings_list:
                if markings[0] == carla.LaneMarkingType.Solid:
                    draw_solid_line(surface, markings[1], False, markings[2], 2)
                elif markings[0] == carla.LaneMarkingType.Broken:
                    draw_broken_line(surface, markings[1], False, markings[2], 2)

        def lateral_shift(transform, shift):
            # 横向移动
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def draw_topology(carla_topology, index):
            # 绘制拓扑结构
            topology = [x[index] for x in carla_topology]
            topology = sorted(topology, key=lambda w: w.transform.location.z)
            set_waypoints = []
            for waypoint in topology:
                waypoints = [waypoint]

                nxt = waypoint.next(precision)
                if len(nxt) > 0:
                    nxt = nxt[0]
                    while nxt.road_id == waypoint.road_id:
                        waypoints.append(nxt)
                        nxt = nxt.next(precision)
                        if len(nxt) > 0:
                            nxt = nxt[0]
                        else:
                            break
                set_waypoints.append(waypoints)

                # 绘制路肩、停车位和人行道
                PARKING_COLOR = COLOR_ALUMINIUM_4_5
                SHOULDER_COLOR = COLOR_ALUMINIUM_5
                SIDEWALK_COLOR = COLOR_ALUMINIUM_3

                shoulder = [[], []]
                parking = [[], []]
                sidewalk = [[], []]

                for w in waypoints:
                    # 处理左侧车道
                    l = w.get_right_lane()
                    while l is not None and l.lane_type != carla.LaneType.Driving:
                        # 选择车道类型并且记录
                        if l.lane_type == carla.LaneType.Shoulder:
                            shoulder[0].append(l)

                        elif l.lane_type == carla.LaneType.Parking:
                            parking[0].append(l)

                        elif l.lane_type == carla.LaneType.Sidewalk:
                            sidewalk[0].append(l)

                        # 获取下一个左侧车道，若不存在则退出循环
                        try:
                            l = l.get_right_lane()
                        except:
                            l = None

                    # 处理右侧车道
                    r = w.get_left_lane()
                    while r is not None and r.lane_type != carla.LaneType.Driving:

                        # 检查车道类型并且记录
                        if r.lane_type == carla.LaneType.Shoulder:
                            shoulder[1].append(r)

                        elif r.lane_type == carla.LaneType.Parking:
                            parking[1].append(r)

                        elif r.lane_type == carla.LaneType.Sidewalk:
                            sidewalk[1].append(r)

                        # 获取下一个右侧车道，若不存在则退出循环
                        try:
                            r = r.get_left_lane()
                        except:
                            r = None

                draw_lane(map_surface, shoulder, SHOULDER_COLOR)
                draw_lane(map_surface, parking, PARKING_COLOR)
                draw_lane(map_surface, sidewalk, SIDEWALK_COLOR)

            # 绘制道路
            for waypoints in set_waypoints:
                waypoint = waypoints[0]
                road_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
                road_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

                polygon = road_left_side + [x for x in reversed(road_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon, 5)
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon)

                # 绘制车道标记
                if not waypoint.is_junction:
                    draw_lane_marking(map_surface, [waypoints, waypoints])

        topology = carla_map.get_topology()
        draw_topology(topology, 0)

    def world_to_pixel(self, location, offset=(0, 0)):
        # 将世界坐标转换为像素坐标
        x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def world_to_pixel_width(self, width):
        # 将世界宽度转换为像素宽度
        return int(self.scale * self._pixels_per_meter * width)


class BirdeyeRender(object):
    """
    在仿真环境中提供一个鸟瞰视图，以便更好地观察和分析车辆和行人的行为
    """
    def __init__(self, world, params, logger):
        self.params = params
        self.server_fps = 0.0
        self.simulation_time = 0
        self.server_clock = pygame.time.Clock()

        # world data
        self.world = world
        self.town_map = self.world.get_map()
        self.actors_with_transforms = []

        # hero actor
        self.hero_actor = None
        self.hero_id = None
        self.hero_transform = None
        self.heros_in_all_envs = []

        # the actors and map information
        self.vehicle_polygons = []
        self.walker_polygons = []
        self.waypoints = None
        self.red_light = False

        # create surface of the entire map, this take a long time
        self.map_image = MapImage(
            carla_world=self.world,
            carla_map=self.town_map,
            pixels_per_meter=self.params['pixels_per_meter'],
            logger=logger,
        )
        self.original_surface_size = min(self.params['screen_size'][0], self.params['screen_size'][1])
        self.surface_size = self.map_image.big_map_surface.get_width()

        # render actors
        self.actors_surface = pygame.Surface((self.map_image.surface.get_width(), self.map_image.surface.get_height()))
        self.actors_surface.set_colorkey(COLOR_BLACK)  # 任何绘制在self.actors_surface上的黑色都会被忽略

        # render waypoints
        self.waypoints_surface = pygame.Surface((self.map_image.surface.get_width(), self.map_image.surface.get_height()))
        self.waypoints_surface.set_colorkey(COLOR_BLACK)

        # render hero (scaled to have full observation in the birdeye image)
        scaled_original_size = self.original_surface_size * (1.0 / 0.62)
        self.hero_surface = pygame.Surface((scaled_original_size, scaled_original_size)).convert()

        self.result_surface = pygame.Surface((self.surface_size, self.surface_size)).convert()
        self.result_surface.set_colorkey(COLOR_BLACK)

        weak_self = weakref.ref(self)
        self.world.on_tick(lambda timestamp: BirdeyeRender.on_world_tick(weak_self, timestamp))

    def set_hero(self, hero_actor, hero_id):
        # 将传入的主角对象和ID保存到类的实例变量中,并记录主角的ID
        self.hero_actor = hero_actor
        self.hero_id = hero_id
        self.heros_in_all_envs.append(hero_id)

    def tick(self, clock):
        actors = self.world.get_actors()
        self.actors_with_transforms = [(actor, actor.get_transform()) for actor in actors]
        if self.hero_actor is not None:
            self.hero_transform = self.hero_actor.get_transform()

    @staticmethod
    def on_world_tick(weak_self, timestamp):
        # 防止对已经销毁的对象进行操作
        self = weak_self()
        if not self:
            return

        self.server_clock.tick()
        self.server_fps = self.server_clock.get_fps()
        self.simulation_time = timestamp.elapsed_seconds

    def _split_actors(self):
        # 将仿真环境中的演员（actors）分为车辆（vehicles）和行人（walkers）两类，并返回这两类演员的列表
        vehicles = []
        walkers = []

        for actor_with_transform in self.actors_with_transforms:
            actor = actor_with_transform[0]
            if 'vehicle' in actor.type_id:
                vehicles.append(actor_with_transform)
            elif 'walker.pedestrian' in actor.type_id:
                walkers.append(actor_with_transform)

        if self.hero_actor is not None and len(vehicles) > 1:
            location = self.hero_transform.location
            vehicle_list = [x[0] for x in vehicles if x[0].id != self.hero_actor.id]

            def distance(v): 
                return location.distance(v.get_location())
            
            for n, vehicle in enumerate(sorted(vehicle_list, key=distance)):
                if n > 15:
                    break

        return (vehicles, walkers)

    def _render_hist_actors(self, surface, actor_polygons, actor_type, world_to_pixel, num):
        # 在给定的 surface 上渲染历史演员（actors）的多边形
        lp = len(actor_polygons)
        color = COLOR_SKY_BLUE_0

        for i in range(max(0,lp-num), lp):
            for ID, poly in actor_polygons[i].items():
                corners = []
                for p in poly:
                    corners.append(carla.Location(x=p[0], y=p[1]))
                corners.append(carla.Location(x=poly[0][0], y=poly[0][1]))
                corners = [world_to_pixel(p) for p in corners]
                color_value = max(0.8 - 0.8/lp*(i+1), 0)
                if ID == self.hero_id or ID in self.heros_in_all_envs:
                    color = pygame.Color(255, math.floor(color_value*255), math.floor(color_value*255)) # red
                else:
                    if actor_type == 'vehicle':
                        # 绿色表示车辆
                        color = pygame.Color(math.floor(color_value*255), 255, math.floor(color_value*255))
                    elif actor_type == 'walker':
                        # 黄色表示行人
                        color = pygame.Color(255, 255, math.floor(color_value*255))
                pygame.draw.polygon(surface, color, corners)

    def render_waypoints(self, surface, waypoints, world_to_pixel):
        if self.red_light:
            color = pygame.Color(math.floor(0.5*255), 0, math.floor(0.5*255)) # purple
        else:
            color = pygame.Color(0,0,255) # blue
        corners = []
        for p in waypoints:
            corners.append(carla.Location(x=p[0], y=p[1]))
        corners = [world_to_pixel(p) for p in corners]
        route_width = 10
        pygame.draw.lines(surface, color, False, corners, route_width)

    def render_actors(self, surface, vehicles, walkers):
        self._render_hist_actors(surface, vehicles, 'vehicle', self.map_image.world_to_pixel, 10)
        self._render_hist_actors(surface, walkers, 'walker', self.map_image.world_to_pixel, 10)

    def clip_surfaces(self, clipping_rect):
        # 只在指定的矩形区域内进行绘制
        self.actors_surface.set_clip(clipping_rect)
        self.result_surface.set_clip(clipping_rect)

    def render(self, render_types=None):
        # clock tick
        self.tick(self.server_clock)

        if self.actors_with_transforms is None:
            return
        self.result_surface.fill(COLOR_BLACK)

        # Render Actors
        self.actors_surface.fill(COLOR_BLACK)
        self.render_actors(self.actors_surface, self.vehicle_polygons, self.walker_polygons)

        self.waypoints_surface.fill(COLOR_BLACK)
        self.render_waypoints(self.waypoints_surface, self.waypoints, self.map_image.world_to_pixel)

        # 合并表面
        if render_types is None:
            surfaces = [(self.map_image.surface, (0, 0)), (self.actors_surface, (0, 0)), (self.waypoints_surface, (0, 0)),]
        else:
            surfaces = []
            if 'roadmap' in render_types:
                surfaces.append((self.map_image.surface, (0, 0)))
            if 'waypoints' in render_types:
                surfaces.append((self.waypoints_surface, (0, 0)))
            if 'actors' in render_types:
                surfaces.append((self.actors_surface, (0, 0)))

        # 计算旋转角度,将视角调整为面向屏幕上方
        angle = 0.0 if self.hero_actor is None else self.hero_transform.rotation.yaw + 90.0

        # 渲染主角试图
        if self.hero_actor is not None:
            hero_location_screen = self.map_image.world_to_pixel(self.hero_transform.location)
            hero_front = self.hero_transform.get_forward_vector()
            translation_offset = (
                hero_location_screen[0] - self.hero_surface.get_width() / 2 + hero_front.x * self.params['pixels_ahead_vehicle'],
                hero_location_screen[1] - self.hero_surface.get_height() / 2 + hero_front.y * self.params['pixels_ahead_vehicle']
            )

            # Apply clipping rect
            clipping_rect = pygame.Rect(translation_offset[0], translation_offset[1], self.hero_surface.get_width(), self.hero_surface.get_height())
            self.clip_surfaces(clipping_rect)

            Util.blits(self.result_surface, surfaces)

            # Set background black
            self.hero_surface.fill(COLOR_BLACK)
            self.hero_surface.blit(self.result_surface, (-translation_offset[0], -translation_offset[1]))
            rotated_result_surface = pygame.transform.rotozoom(self.hero_surface, angle, 1.0).convert()
            return rotated_result_surface
        else:
            raise ValueError('hero_actor is None')
    