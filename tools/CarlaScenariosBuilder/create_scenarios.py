"""
Create scenario trigger and actor points with a lightweight pygame viewer.
"""

import argparse
import os

import numpy as np
try:
    import pygame
except ImportError as exc:
    raise SystemExit(
        "pygame is required for create_scenarios.py. "
        "Install it with `pip install pygame==2.3.0` or `pip install -r requirements.txt`."
    ) from exc

from create_routes import load_route
from pygame_viewer import (
    ACTOR_COLOR,
    END_COLOR,
    PygameMapViewer,
    ROUTE_ALT_COLOR,
    START_COLOR,
    TRIGGER_COLOR,
)
from utilities import get_map_centers, get_nearist_waypoints


DEFAULT_DIST = 200


def load_scenario(config, waypoints_sparse, save_dir):
    center = None
    selected_waypoints_idx = []
    if config.route_idx >= 0:
        scenario_file = os.path.join(save_dir, f"scenario_{config.route_idx:02d}.npy")
        if os.path.isfile(scenario_file):
            selected_waypoints = np.load(scenario_file)
            center = selected_waypoints[:, :2].mean(axis=0)
            for waypoint in selected_waypoints:
                idx, dist = get_nearist_waypoints(waypoint, waypoints_sparse)
                if dist > 5:
                    print(
                        f"waypoint {waypoint} can not be found on the map, "
                        f"assigned to the nearest waypoint {waypoints_sparse[idx]}"
                    )
                selected_waypoints_idx.append(idx)
    return center, selected_waypoints_idx


def save_scenario(config, save_dir, selected_waypoints, side_marks):
    route_idx = config.route_idx
    if route_idx < 0:
        route_idx = 0
        while os.path.isfile(os.path.join(save_dir, f"scenario_{route_idx:02d}.npy")):
            route_idx += 1

    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"scenario_{route_idx:02d}.npy")
    np.save(save_file, selected_waypoints)
    save_file_side = os.path.join(save_dir, f"scenario_{route_idx:02d}_sides.npy")
    np.save(save_file_side, side_marks)
    return route_idx, save_file


def _build_overlay_lines(config, route_id, selected_route_waypoints_idx, selected_trigger_waypoints_idx, viewer, status_message):
    zoom_ratio = viewer.scale / (min(viewer.window_size) / (2.0 * DEFAULT_DIST))
    return [
        f"Map: {config.map}    Scenario: {config.scenario:02d}    Route: {route_id:02d}",
        (
            "Left click: select/remove trigger or actor    Right click: save scenario    "
            "Wheel: zoom    Middle drag: pan    ESC: exit"
        ),
        (
            f"Route points: {len(selected_route_waypoints_idx)}    "
            f"Selected trigger/actors: {len(selected_trigger_waypoints_idx)}    "
            f"View center: ({viewer.center[0]:.1f}, {viewer.center[1]:.1f})    "
            f"Zoom: {zoom_ratio:.2f}x"
        ),
        status_message,
    ]


def _draw_scene(
    viewer,
    config,
    route_id,
    waypoints_sparse,
    selected_route_waypoints_idx,
    selected_trigger_waypoints_idx,
    status_message,
):
    viewer.clear()
    viewer.draw_waypoints(waypoints_sparse)

    if selected_route_waypoints_idx:
        route_waypoints = np.take(waypoints_sparse, np.array(selected_route_waypoints_idx), axis=0)
        viewer.draw_polyline(route_waypoints[:, :2], ROUTE_ALT_COLOR, width=3, point_radius=4)
        viewer.draw_marker(route_waypoints[0, :2], START_COLOR, "Start")
        if len(selected_route_waypoints_idx) > 1:
            viewer.draw_marker(route_waypoints[-1, :2], END_COLOR, "End")

    if selected_trigger_waypoints_idx:
        trigger_waypoints = np.take(waypoints_sparse, np.array(selected_trigger_waypoints_idx), axis=0)
        trigger_screen = viewer.world_to_screen(trigger_waypoints[0, :2])
        viewer.draw_marker(trigger_waypoints[0, :2], TRIGGER_COLOR, "Trigger")
        for actor_waypoint in trigger_waypoints[1:]:
            actor_screen = viewer.world_to_screen(actor_waypoint[:2])
            pygame.draw.line(
                viewer.screen,
                TRIGGER_COLOR,
                (int(trigger_screen[0]), int(trigger_screen[1])),
                (int(actor_screen[0]), int(actor_screen[1])),
                width=1,
            )
            viewer.draw_marker(actor_waypoint[:2], ACTOR_COLOR, "Actor")

    viewer.draw_overlay(
        "Create Scenario Viewer",
        _build_overlay_lines(
            config,
            route_id,
            selected_route_waypoints_idx,
            selected_trigger_waypoints_idx,
            viewer,
            status_message,
        ),
    )
    viewer.present()


def main(config):
    waypoints_sparse = np.load(f"map_waypoints/{config.map}/sparse.npy")
    scenario_id = config.scenario

    scenario_save_dir = os.path.join("scenario_origin", config.map, f"scenario_{scenario_id:02d}_scenarios")
    routes_dir = os.path.join("scenario_origin", config.map, f"scenario_{scenario_id:02d}_routes")

    if config.route_idx == -1:
        route_files = sorted(f for f in os.listdir(routes_dir) if f.startswith("route_") and f.endswith(".npy"))
        route_ids = [int(f.split("_")[1].split(".")[0]) for f in route_files]
    else:
        route_ids = [config.route_idx]

    if not route_ids:
        raise ValueError(f"No routes found in {routes_dir}")

    current_route_index = 0
    selected_route_waypoints_idx = []
    selected_trigger_waypoints_idx = []
    initial_center = np.asarray(get_map_centers(config.map)[0], dtype=float)

    def load_current_route():
        nonlocal selected_route_waypoints_idx, selected_trigger_waypoints_idx, initial_center
        route_id = route_ids[current_route_index]
        route_cfg = argparse.Namespace(**vars(config))
        route_cfg.route = route_id

        loaded_center, selected_route_waypoints_idx = load_route(route_cfg, waypoints_sparse, routes_dir)
        if not selected_route_waypoints_idx:
            raise ValueError(f"Route {route_id:02d} not found or empty in {routes_dir}")

        scenario_cfg = argparse.Namespace(**vars(config))
        scenario_cfg.route_idx = route_id
        scenario_center, selected_trigger_waypoints_idx = load_scenario(
            scenario_cfg,
            waypoints_sparse,
            scenario_save_dir,
        )

        if scenario_center is not None:
            initial_center = np.asarray(scenario_center, dtype=float)
        elif loaded_center is not None:
            initial_center = np.asarray(loaded_center, dtype=float)
        else:
            route_waypoints = np.take(waypoints_sparse, np.array(selected_route_waypoints_idx), axis=0)
            initial_center = route_waypoints[:, :2].mean(axis=0)

    load_current_route()

    viewer = PygameMapViewer(
        title=f"Create Scenarios - {config.map} - scenario {scenario_id:02d}",
        center=initial_center,
        dist=DEFAULT_DIST,
    )

    status_message = "Ready. Pick trigger first, then actor locations."
    dirty = True
    running = True

    while running:
        changed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if viewer.handle_view_event(event):
                changed = True
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                if event.key == pygame.K_r:
                    viewer.center = np.asarray(initial_center, dtype=float)
                    viewer.scale = max(0.1, min(viewer.window_size) / (2.0 * DEFAULT_DIST))
                    status_message = "View reset."
                    changed = True

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    idx = viewer.pick_waypoint_index(waypoints_sparse[:, :2], event.pos)
                    if idx is not None:
                        if idx in selected_trigger_waypoints_idx:
                            selected_trigger_waypoints_idx.remove(idx)
                            status_message = f"Removed point {idx}."
                        else:
                            selected_trigger_waypoints_idx.append(idx)
                            if len(selected_trigger_waypoints_idx) == 1:
                                status_message = f"Trigger set at waypoint {idx}."
                            else:
                                status_message = f"Added actor point {idx}."
                        changed = True

                elif event.button == 3:
                    if len(selected_trigger_waypoints_idx) < 1:
                        status_message = "Need at least 1 waypoint to create a scenario."
                        changed = True
                        continue

                    selected_waypoints = np.take(
                        waypoints_sparse,
                        np.array(selected_trigger_waypoints_idx),
                        axis=0,
                    )
                    start_wp = np.take(waypoints_sparse, selected_route_waypoints_idx[0], axis=0)
                    end_wp = np.take(waypoints_sparse, selected_route_waypoints_idx[-1], axis=0)
                    start_pos, end_pos = start_wp[:2], end_wp[:2]

                    side_marks = []
                    for i, wp in enumerate(selected_waypoints):
                        actor_pos = wp[:2]
                        cross = (
                            (actor_pos[0] - start_pos[0]) * (end_pos[1] - start_pos[1])
                            - (actor_pos[1] - start_pos[1]) * (end_pos[0] - start_pos[0])
                        )
                        if config.scenario in [3, 4, 5, 6, 7]:
                            side_marks.append("center")
                        elif cross > 0:
                            side_marks.append("left")
                            wp[4] = (wp[4] + 90) % 360
                        elif cross < 0:
                            side_marks.append("right")
                            wp[4] = (wp[4] - 90) % 360
                        else:
                            side_marks.append("center")
                        selected_waypoints[i] = wp

                    scenario_cfg = argparse.Namespace(**vars(config))
                    scenario_cfg.route_idx = route_ids[current_route_index]
                    route_id, save_file = save_scenario(
                        scenario_cfg,
                        scenario_save_dir,
                        selected_waypoints,
                        np.array(side_marks),
                    )

                    if config.route_idx == -1:
                        current_route_index += 1
                        if current_route_index >= len(route_ids):
                            status_message = f"All routes processed. Last save: {save_file}"
                        else:
                            load_current_route()
                            viewer.center = np.asarray(initial_center, dtype=float)
                            viewer.scale = max(0.1, min(viewer.window_size) / (2.0 * DEFAULT_DIST))
                            status_message = (
                                f"Scenario {route_id:02d} saved to {save_file}. "
                                f"Now editing route {route_ids[current_route_index]:02d}."
                            )
                    else:
                        status_message = f"Scenario {route_id:02d} saved to {save_file}."
                    changed = True

        route_id = route_ids[min(current_route_index, len(route_ids) - 1)]
        if dirty or changed:
            _draw_scene(
                viewer,
                config,
                route_id,
                waypoints_sparse,
                selected_route_waypoints_idx,
                selected_trigger_waypoints_idx,
                status_message,
            )
            dirty = False

        viewer.tick(60)

    viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="TsimShaTsui-1029-3")
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--route_idx", type=int, default=-1)
    parser.add_argument(
        "--road",
        type=str,
        default="auto",
        choices=["auto", "intersection", "straight"],
        help="Retained for compatibility; the lightweight viewer does not use this value.",
    )

    args = parser.parse_args()
    main(args)
