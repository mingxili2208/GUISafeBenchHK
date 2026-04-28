import math

import numpy as np
try:
    import pygame
except ImportError as exc:
    raise SystemExit(
        "pygame is required for the lightweight route/scenario viewer. "
        "Install it with `pip install pygame==2.3.0` or `pip install -r requirements.txt`."
    ) from exc


BACKGROUND_COLOR = (18, 22, 28)
POINT_COLOR = (238, 220, 90)
ARROW_COLOR = (245, 200, 66)
ROUTE_COLOR = (66, 154, 255)
ROUTE_ALT_COLOR = (255, 84, 84)
START_COLOR = (70, 210, 110)
END_COLOR = (255, 84, 84)
TRIGGER_COLOR = (255, 90, 90)
ACTOR_COLOR = (70, 210, 110)
GRID_COLOR = (36, 44, 54)
TEXT_COLOR = (238, 238, 238)
PANEL_COLOR = (18, 22, 28, 210)
BACKGROUND_MARGIN_PX = 48
POINT_MIN_RADIUS_PX = 1.6
POINT_MAX_RADIUS_PX = 2.6
POINT_EDGE_SOFTNESS_PX = 0.85
POINT_SUBPIXEL_BINS = 4
POINT_DENSITY_GAIN = 1.45
ARROW_MIN_SIZE_PX = 8
ARROW_MAX_SIZE_PX = 12
ARROW_WORLD_SPACING_PX = 28
ARROW_ANGLE_BINS = 64
ARROW_STAMP_SUPERSAMPLE = 4

class PygameMapViewer:
    def __init__(self, title, center, dist, window_size=(1280, 960)):
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Arial", 18)
        self.small_font = pygame.font.SysFont("Arial", 15)

        self.center = np.asarray(center, dtype=float)
        self.window_size = np.asarray(window_size, dtype=float)
        self.scale = max(0.1, min(self.window_size) / (2.0 * max(dist, 1.0)))

        self.dragging = False
        self.last_mouse_pos = None
        self.background_surface = None
        self.background_key = None
        self.point_stamp_cache = {}
        self.arrow_stamp_cache = {}

    def close(self):
        pygame.display.quit()
        pygame.quit()

    def tick(self, fps=60):
        self.clock.tick(fps)

    def update_window_size(self, size):
        self.window_size = np.asarray(size, dtype=float)
        self.invalidate_background()

    def invalidate_background(self):
        self.background_surface = None
        self.background_key = None

    def world_to_screen(self, point_xy):
        x = (point_xy[0] - self.center[0]) * self.scale + self.window_size[0] / 2.0
        y = -(point_xy[1] - self.center[1]) * self.scale + self.window_size[1] / 2.0
        return np.asarray([x, y], dtype=float)

    def world_to_screen_batch(self, points_xy):
        if len(points_xy) == 0:
            return np.empty((0, 2), dtype=float)

        screen = np.empty((len(points_xy), 2), dtype=float)
        screen[:, 0] = (points_xy[:, 0] - self.center[0]) * self.scale + self.window_size[0] / 2.0
        screen[:, 1] = -(points_xy[:, 1] - self.center[1]) * self.scale + self.window_size[1] / 2.0
        return screen

    def screen_to_world(self, screen_xy):
        x = (screen_xy[0] - self.window_size[0] / 2.0) / self.scale + self.center[0]
        y = -(screen_xy[1] - self.window_size[1] / 2.0) / self.scale + self.center[1]
        return np.asarray([x, y], dtype=float)

    def zoom_at(self, screen_xy, factor):
        screen_xy = np.asarray(screen_xy, dtype=float)
        world_before = self.screen_to_world(screen_xy)
        self.scale = float(np.clip(self.scale * factor, 0.2, 200.0))
        world_after = self.screen_to_world(screen_xy)
        self.center += world_before - world_after

    def pan_by_pixels(self, dx, dy):
        self.center[0] -= dx / self.scale
        self.center[1] += dy / self.scale

    def get_visible_mask(self, points_xy, margin_px=40):
        half_w = self.window_size[0] / (2.0 * self.scale) + margin_px / self.scale
        half_h = self.window_size[1] / (2.0 * self.scale) + margin_px / self.scale
        return (
            (points_xy[:, 0] >= self.center[0] - half_w)
            & (points_xy[:, 0] <= self.center[0] + half_w)
            & (points_xy[:, 1] >= self.center[1] - half_h)
            & (points_xy[:, 1] <= self.center[1] + half_h)
        )

    def pick_waypoint_index(self, waypoints_xy, mouse_pos, max_pixel_distance=12):
        click_world = self.screen_to_world(mouse_pos)
        world_threshold = max_pixel_distance / self.scale
        dist_sq = np.sum((waypoints_xy - click_world) ** 2, axis=1)
        idx = int(np.argmin(dist_sq))
        if dist_sq[idx] <= world_threshold ** 2:
            return idx
        return None

    def handle_view_event(self, event):
        changed = False

        if event.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            self.update_window_size(event.size)
            changed = True

        elif event.type == pygame.MOUSEWHEEL:
            mouse_pos = pygame.mouse.get_pos()
            factor = 1.15 if event.y > 0 else 1 / 1.15
            self.zoom_at(mouse_pos, factor)
            changed = True

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 2:
                self.dragging = True
                self.last_mouse_pos = event.pos
            elif event.button == 4:
                self.zoom_at(event.pos, 1.15)
                changed = True
            elif event.button == 5:
                self.zoom_at(event.pos, 1 / 1.15)
                changed = True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 2:
                self.dragging = False
                self.last_mouse_pos = None

        elif event.type == pygame.MOUSEMOTION and self.dragging and self.last_mouse_pos is not None:
            dx = event.pos[0] - self.last_mouse_pos[0]
            dy = event.pos[1] - self.last_mouse_pos[1]
            self.pan_by_pixels(dx, dy)
            self.last_mouse_pos = event.pos
            changed = True

        return changed

    def clear(self):
        self.screen.fill(BACKGROUND_COLOR)
        self._draw_grid()

    def _draw_grid(self):
        grid_world = self._grid_spacing_world()
        if grid_world is None:
            return

        half_w = self.window_size[0] / (2.0 * self.scale)
        half_h = self.window_size[1] / (2.0 * self.scale)
        x_min = self.center[0] - half_w
        x_max = self.center[0] + half_w
        y_min = self.center[1] - half_h
        y_max = self.center[1] + half_h

        x_start = math.floor(x_min / grid_world) * grid_world
        x_end = math.ceil(x_max / grid_world) * grid_world
        y_start = math.floor(y_min / grid_world) * grid_world
        y_end = math.ceil(y_max / grid_world) * grid_world

        x = x_start
        while x <= x_end:
            sx = self.world_to_screen((x, self.center[1]))[0]
            pygame.draw.line(self.screen, GRID_COLOR, (sx, 0), (sx, self.window_size[1]), width=1)
            x += grid_world

        y = y_start
        while y <= y_end:
            sy = self.world_to_screen((self.center[0], y))[1]
            pygame.draw.line(self.screen, GRID_COLOR, (0, sy), (self.window_size[0], sy), width=1)
            y += grid_world

    def _grid_spacing_world(self):
        target_px = 120
        raw = target_px / self.scale
        if raw <= 0:
            return None

        power = 10 ** math.floor(math.log10(raw))
        for step in (1, 2, 5, 10):
            spacing = power * step
            if spacing >= raw:
                return spacing
        return power * 10

    def draw_waypoints(self, road_waypoints, max_arrows=500):
        if road_waypoints is None or len(road_waypoints) == 0:
            return

        background_key = self._make_background_key(road_waypoints, max_arrows)
        if self.background_key != background_key or self.background_surface is None:
            self.background_surface = self._build_background_surface(road_waypoints, max_arrows)
            self.background_key = background_key

        if self.background_surface is not None:
            self.screen.blit(self.background_surface, (0, 0))

    def _make_background_key(self, road_waypoints, max_arrows):
        if road_waypoints is None:
            return None

        try:
            data_ptr = int(road_waypoints.__array_interface__['data'][0])
        except Exception:
            data_ptr = id(road_waypoints)

        return (
            int(self.window_size[0]),
            int(self.window_size[1]),
            round(float(self.center[0]), 5),
            round(float(self.center[1]), 5),
            round(float(self.scale), 6),
            data_ptr,
            int(len(road_waypoints)),
            int(max_arrows),
        )

    def _build_background_surface(self, road_waypoints, max_arrows):
        width = int(self.window_size[0])
        height = int(self.window_size[1])
        if width <= 0 or height <= 0:
            return None

        background_surface = pygame.Surface((width, height), pygame.SRCALPHA, 32)
        mask = self.get_visible_mask(road_waypoints[:, :2], margin_px=BACKGROUND_MARGIN_PX)
        visible = road_waypoints[mask]
        if len(visible) == 0:
            return background_surface.convert_alpha()

        screen_points = self.world_to_screen_batch(visible[:, :2])
        point_alpha = self._render_point_alpha(screen_points, width, height)
        if point_alpha is not None:
            background_surface.blit(self._alpha_to_surface(point_alpha, POINT_COLOR), (0, 0))

        arrow_alpha = self._render_arrow_alpha(visible, screen_points, width, height, max_arrows=max_arrows)
        if arrow_alpha is not None:
            background_surface.blit(self._alpha_to_surface(arrow_alpha, ARROW_COLOR), (0, 0))

        return background_surface.convert_alpha()

    def _alpha_to_surface(self, alpha, color):
        height, width = alpha.shape
        surface = pygame.Surface((width, height), pygame.SRCALPHA, 32)
        rgb_view = pygame.surfarray.pixels3d(surface)
        alpha_view = pygame.surfarray.pixels_alpha(surface)

        rgb_view[:, :, 0].fill(color[0])
        rgb_view[:, :, 1].fill(color[1])
        rgb_view[:, :, 2].fill(color[2])
        alpha_view[:, :] = alpha.T

        del rgb_view
        del alpha_view
        return surface

    def _render_point_alpha(self, screen_points, width, height):
        if len(screen_points) == 0:
            return None

        radius_px = float(
            np.clip(1.35 + 0.22 * math.log1p(max(self.scale, 0.1)), POINT_MIN_RADIUS_PX, POINT_MAX_RADIUS_PX)
        )
        radius_key = int(round(radius_px * 4.0))
        radius = int(math.ceil(radius_px + POINT_EDGE_SOFTNESS_PX + 1.0))

        alpha_accum = np.zeros((height, width), dtype=np.float32)
        xs = np.floor(screen_points[:, 0]).astype(np.int32)
        ys = np.floor(screen_points[:, 1]).astype(np.int32)
        fx = np.clip(screen_points[:, 0] - xs, 0.0, 0.999)
        fy = np.clip(screen_points[:, 1] - ys, 0.0, 0.999)
        phase_x = np.minimum((fx * POINT_SUBPIXEL_BINS).astype(np.int32), POINT_SUBPIXEL_BINS - 1)
        phase_y = np.minimum((fy * POINT_SUBPIXEL_BINS).astype(np.int32), POINT_SUBPIXEL_BINS - 1)
        valid = (
            (xs >= -radius)
            & (xs < width + radius)
            & (ys >= -radius)
            & (ys < height + radius)
        )
        for sx, sy, px, py in zip(xs[valid], ys[valid], phase_x[valid], phase_y[valid]):
            stamp = self._get_point_stamp(radius_key, int(px), int(py))
            self._stamp_add(alpha_accum, stamp, int(sx), int(sy))

        alpha = 255.0 * (1.0 - np.exp(-POINT_DENSITY_GAIN * alpha_accum))
        return np.clip(alpha, 0.0, 255.0).astype(np.uint8)

    def _get_point_stamp(self, radius_key, phase_x, phase_y):
        key = (radius_key, phase_x, phase_y)
        cached = self.point_stamp_cache.get(key)
        if cached is not None:
            return cached

        radius_px = radius_key / 4.0
        radius = int(math.ceil(radius_px + POINT_EDGE_SOFTNESS_PX))
        size = radius * 2 + 1

        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        offset_x = phase_x / POINT_SUBPIXEL_BINS
        offset_y = phase_y / POINT_SUBPIXEL_BINS
        dist = np.sqrt((xx.astype(np.float32) - offset_x) ** 2 + (yy.astype(np.float32) - offset_y) ** 2)
        edge0 = max(radius_px - POINT_EDGE_SOFTNESS_PX, 0.0)
        edge1 = radius_px + POINT_EDGE_SOFTNESS_PX
        t = np.clip((dist - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
        stamp = (1.0 - t * t * (3.0 - 2.0 * t)).astype(np.float32)
        self.point_stamp_cache[key] = stamp
        return stamp

    def _render_arrow_alpha(self, visible, screen_points, width, height, max_arrows=500):
        if len(visible) == 0:
            return None

        arrow_size_px = int(np.clip(round(6.5 + 0.38 * self.scale), ARROW_MIN_SIZE_PX, ARROW_MAX_SIZE_PX))
        arrow_spacing_world = ARROW_WORLD_SPACING_PX / max(self.scale, 1e-6)

        grid = np.floor(visible[:, :2] / arrow_spacing_world).astype(int)
        selected_indices = []
        occupied_cells = set()
        for idx, cell in enumerate(grid):
            key = (int(cell[0]), int(cell[1]))
            if key in occupied_cells:
                continue
            occupied_cells.add(key)
            selected_indices.append(idx)
            if len(selected_indices) >= max_arrows:
                break

        if not selected_indices:
            return None

        arrow_points = visible[selected_indices]
        arrow_pos = np.rint(screen_points[selected_indices]).astype(np.int32)
        alpha = np.zeros((height, width), dtype=np.uint8)

        yaw = np.deg2rad(arrow_points[:, 4])
        angle_bins = np.mod(np.rint(yaw / (2.0 * math.pi) * ARROW_ANGLE_BINS), ARROW_ANGLE_BINS).astype(np.int32)

        for (sx, sy), angle_bin in zip(arrow_pos, angle_bins):
            stamp = self._get_arrow_stamp(arrow_size_px, int(angle_bin))
            radius = stamp.shape[0] // 2
            if sx < -radius or sx >= width + radius or sy < -radius or sy >= height + radius:
                continue
            self._stamp_max(alpha, stamp, int(sx), int(sy))

        return alpha

    def _get_arrow_stamp(self, size_px, angle_bin):
        key = (int(size_px), int(angle_bin))
        cached = self.arrow_stamp_cache.get(key)
        if cached is not None:
            return cached

        theta = (angle_bin / ARROW_ANGLE_BINS) * (2.0 * math.pi)
        ss = ARROW_STAMP_SUPERSAMPLE
        pad = max(3, int(round(size_px * 0.6)))
        size = size_px * 2 + pad * 2 + 1
        ss_size = size * ss
        center = np.asarray([ss_size / 2.0, ss_size / 2.0], dtype=np.float32)

        direction = np.asarray([math.cos(theta), -math.sin(theta)], dtype=np.float32)
        perp = np.asarray([-direction[1], direction[0]], dtype=np.float32)

        surface = pygame.Surface((ss_size, ss_size), pygame.SRCALPHA, 32)
        white = (255, 255, 255, 255)

        shaft_len_back = size_px * ss * 0.62
        shaft_len_front = size_px * ss * 0.18
        shaft_half_width = max(1.0, size_px * ss * 0.12)
        shaft_start = center - direction * shaft_len_back
        shaft_end = center + direction * shaft_len_front
        shaft_poly = [
            shaft_start + perp * shaft_half_width,
            shaft_end + perp * shaft_half_width,
            shaft_end - perp * shaft_half_width,
            shaft_start - perp * shaft_half_width,
        ]
        pygame.draw.polygon(surface, white, [(float(x), float(y)) for x, y in shaft_poly])

        head_tip = center + direction * (size_px * ss * 0.92)
        head_base = center - direction * (size_px * ss * 0.06)
        head_half_width = size_px * ss * 0.34
        head_poly = [
            head_tip,
            head_base + perp * head_half_width,
            head_base - perp * head_half_width,
        ]
        pygame.draw.polygon(surface, white, [(float(x), float(y)) for x, y in head_poly])

        scaled = pygame.transform.smoothscale(surface, (size, size))
        alpha_view = pygame.surfarray.pixels_alpha(scaled)
        stamp = alpha_view.T.copy()
        del alpha_view
        self.arrow_stamp_cache[key] = stamp
        return stamp

    def _stamp_add(self, target, stamp, center_x, center_y):
        radius_y = stamp.shape[0] // 2
        radius_x = stamp.shape[1] // 2
        height, width = target.shape

        dst_x0 = max(0, center_x - radius_x)
        dst_x1 = min(width, center_x + radius_x + 1)
        dst_y0 = max(0, center_y - radius_y)
        dst_y1 = min(height, center_y + radius_y + 1)
        if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
            return

        src_x0 = dst_x0 - (center_x - radius_x)
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y0 = dst_y0 - (center_y - radius_y)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        target[dst_y0:dst_y1, dst_x0:dst_x1] += stamp[src_y0:src_y1, src_x0:src_x1]

    def _stamp_max(self, target, stamp, center_x, center_y):
        radius_y = stamp.shape[0] // 2
        radius_x = stamp.shape[1] // 2
        height, width = target.shape

        dst_x0 = max(0, center_x - radius_x)
        dst_x1 = min(width, center_x + radius_x + 1)
        dst_y0 = max(0, center_y - radius_y)
        dst_y1 = min(height, center_y + radius_y + 1)
        if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
            return

        src_x0 = dst_x0 - (center_x - radius_x)
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y0 = dst_y0 - (center_y - radius_y)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        np.maximum(
            target[dst_y0:dst_y1, dst_x0:dst_x1],
            stamp[src_y0:src_y1, src_x0:src_x1],
            out=target[dst_y0:dst_y1, dst_x0:dst_x1],
        )

    def draw_polyline(self, points_xy, color, width=2, point_radius=4):
        if points_xy is None or len(points_xy) == 0:
            return

        screen_points = self.world_to_screen_batch(points_xy)
        int_points = [(int(x), int(y)) for x, y in screen_points]
        if len(int_points) >= 2:
            pygame.draw.lines(self.screen, color, False, int_points, width)
        for point in int_points:
            pygame.draw.circle(self.screen, color, point, point_radius)

    def draw_marker(self, point_xy, color, label=None, radius=5, offset=(10, -14)):
        screen_point = self.world_to_screen(point_xy)
        pos = (int(screen_point[0]), int(screen_point[1]))
        pygame.draw.circle(self.screen, color, pos, radius)
        if label:
            self.draw_label(label, (pos[0] + offset[0], pos[1] + offset[1]), color)

    def draw_label(self, text, position, accent_color=None):
        text_surface = self.small_font.render(text, True, TEXT_COLOR)
        padding = 4
        panel = pygame.Surface(
            (text_surface.get_width() + padding * 2, text_surface.get_height() + padding * 2),
            pygame.SRCALPHA,
        )
        panel.fill(PANEL_COLOR)
        if accent_color is not None:
            pygame.draw.rect(panel, accent_color, panel.get_rect(), width=1)
        panel.blit(text_surface, (padding, padding))
        self.screen.blit(panel, position)

    def draw_overlay(self, title, lines):
        rendered_title = self.font.render(title, True, TEXT_COLOR)
        rendered_lines = [self.small_font.render(line, True, TEXT_COLOR) for line in lines]

        width = rendered_title.get_width()
        height = rendered_title.get_height() + 10
        for line in rendered_lines:
            width = max(width, line.get_width())
            height += line.get_height() + 4
        width += 20
        height += 14

        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill(PANEL_COLOR)
        self.screen.blit(panel, (12, 12))
        self.screen.blit(rendered_title, (22, 20))

        y = 20 + rendered_title.get_height() + 8
        for line in rendered_lines:
            self.screen.blit(line, (22, y))
            y += line.get_height() + 4

    def present(self):
        pygame.display.flip()
