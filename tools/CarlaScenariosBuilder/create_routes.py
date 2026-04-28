"""
Create and edit scenario routes with a lightweight pygame viewer.
"""

import argparse
import os

import numpy as np
try:
    import pygame
except ImportError as exc:
    raise SystemExit(
        "pygame is required for create_routes.py. "
        "Install it with `pip install pygame==2.3.0` or `pip install -r requirements.txt`."
    ) from exc

from pygame_viewer import END_COLOR, PygameMapViewer, ROUTE_COLOR, START_COLOR
from utilities import get_map_centers, get_nearist_waypoints


DEFAULT_DIST = 200


def save_waypoints(config, save_dir, selected_waypoints):
    route_id = config.route
    if config.route < 0:
        route_id = 0
        while os.path.isfile(os.path.join(save_dir, f"route_{route_id:02d}.npy")):
            route_id += 1

    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"route_{route_id:02d}.npy")
    np.save(save_file, selected_waypoints)
    return route_id, save_file


def _match_waypoints_to_indices(waypoints_sparse, selected_waypoints):
    selected_waypoints_idx = []
    for waypoint in selected_waypoints:
        idx, dist = get_nearist_waypoints(waypoint, waypoints_sparse)
        if dist > 5:
            print(
                f"waypoint {waypoint} can not be found on the map, "
                f"assigned to the nearest waypoint {waypoints_sparse[idx]}"
            )
        selected_waypoints_idx.append(idx)
    return selected_waypoints_idx


def load_route(config, waypoints_sparse, save_dir):
    center = None
    selected_waypoints_idx = []
    if config.route >= 0:
        route_file = os.path.join(save_dir, f"route_{config.route:02d}.npy")
        if os.path.isfile(route_file):
            selected_waypoints = np.load(route_file)
            center = selected_waypoints[:, :2].mean(axis=0)
            selected_waypoints_idx = _match_waypoints_to_indices(waypoints_sparse, selected_waypoints)

    return center, selected_waypoints_idx


def _build_overlay_lines(config, selected_waypoints_idx, viewer, status_message):
    zoom_ratio = viewer.scale / (min(viewer.window_size) / (2.0 * DEFAULT_DIST))
    return [
        f"Map: {config.map}    Scenario: {config.scenario:02d}",
        (
            "Left click: select/remove waypoint    Right click: save route    "
            "Wheel: zoom    Middle drag: pan    ESC: exit"
        ),
        (
            f"Selected points: {len(selected_waypoints_idx)}    "
            f"View center: ({viewer.center[0]:.1f}, {viewer.center[1]:.1f})    "
            f"Zoom: {zoom_ratio:.2f}x"
        ),
        status_message,
    ]


def _draw_scene(viewer, config, waypoints_sparse, selected_waypoints_idx, status_message):
    viewer.clear()
    viewer.draw_waypoints(waypoints_sparse)

    if selected_waypoints_idx:
        selected_waypoints = np.take(waypoints_sparse, np.array(selected_waypoints_idx), axis=0)
        viewer.draw_polyline(selected_waypoints[:, :2], ROUTE_COLOR, width=3, point_radius=4)
        viewer.draw_marker(selected_waypoints[0, :2], START_COLOR, "Start")
        if len(selected_waypoints_idx) > 1:
            viewer.draw_marker(selected_waypoints[-1, :2], END_COLOR, "End")

    viewer.draw_overlay(
        "Create Route Viewer",
        _build_overlay_lines(config, selected_waypoints_idx, viewer, status_message),
    )
    viewer.present()


def main(config):
    waypoints_sparse = np.load(f"map_waypoints/{config.map}/sparse.npy")

    scenario_id = config.scenario
    save_dir = os.path.join("scenario_origin", config.map, f"scenario_{scenario_id:02d}_routes")

    initial_center = np.asarray(get_map_centers(config.map)[0], dtype=float)
    selected_waypoints_idx = []

    loaded_center, loaded_selected_idx = load_route(config, waypoints_sparse, save_dir)
    if len(loaded_selected_idx) > 0:
        initial_center = np.asarray(loaded_center, dtype=float)
        selected_waypoints_idx = loaded_selected_idx

    viewer = PygameMapViewer(
        title=f"Create Routes - {config.map} - scenario {scenario_id:02d}",
        center=initial_center,
        dist=DEFAULT_DIST,
    )

    status_message = "Ready. Build a route with at least 2 waypoints."
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
                    viewer.center = np.asarray(get_map_centers(config.map)[0], dtype=float)
                    viewer.scale = max(0.1, min(viewer.window_size) / (2.0 * DEFAULT_DIST))
                    status_message = "View reset."
                    changed = True

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    idx = viewer.pick_waypoint_index(waypoints_sparse[:, :2], event.pos)
                    if idx is not None:
                        if idx in selected_waypoints_idx:
                            selected_waypoints_idx.remove(idx)
                            status_message = f"Removed waypoint {idx}."
                        else:
                            selected_waypoints_idx.append(idx)
                            status_message = f"Added waypoint {idx}."
                        changed = True

                elif event.button == 3:
                    if len(selected_waypoints_idx) < 2:
                        status_message = "Need at least 2 waypoints to create a route."
                    else:
                        selected_waypoints = np.take(
                            waypoints_sparse,
                            np.array(selected_waypoints_idx),
                            axis=0,
                        )
                        route_id, save_file = save_waypoints(config, save_dir, selected_waypoints)
                        if config.route < 0:
                            selected_waypoints_idx.clear()
                            status_message = f"Route {route_id:02d} saved to {save_file}. Keep creating routes."
                        else:
                            status_message = f"Route {route_id:02d} updated at {save_file}."
                    changed = True

        if dirty or changed:
            _draw_scene(viewer, config, waypoints_sparse, selected_waypoints_idx, status_message)
            dirty = False

        viewer.tick(60)

    viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="TsimShaTsui-1029-3")
    parser.add_argument("--save_dir", type=str, default="scenario_data/center")
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--route", type=int, default=-1)
    parser.add_argument(
        "--road",
        type=str,
        default="auto",
        choices=["auto", "intersection", "straight"],
        help="Retained for compatibility; the lightweight viewer does not use this value.",
    )

    args = parser.parse_args()
    main(args)
