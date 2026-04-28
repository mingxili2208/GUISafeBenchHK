import argparse
import copy
import json
import os
import re
import shutil

import carla
import numpy as np

from utilities import build_route, get_map_centers


STANDARD_EXPORT = "standard"
ADV_EXPORT = "adv"
ADV_PARAMETERS = ["genetic_algorithm", True]
SCENARIO_DIR_PATTERN = re.compile(r"scenario_(\d{2})_routes$")


def create_route_hongkong(config, selected_waypoints):
    scenario_id = config.scenario
    save_dir = os.path.join(config.save_dir, f"scenario_{scenario_id:02d}_routes")
    os.makedirs(save_dir, exist_ok=True)

    real_route_waypoints = [np.array(route_waypoint) for route_waypoint in selected_waypoints]
    return [np.array(real_route_waypoints)]


def get_static_public_attributes(cls):
    return {
        key: getattr(cls, key)
        for key in dir(cls)
        if not key.startswith('_')
        and not callable(getattr(cls, key))
        and key in cls.__dict__
        and not isinstance(cls.__dict__.get(key), property)
    }


def object_to_numeric_dict(obj):
    return {
        key: str(getattr(obj, key))
        for key in dir(obj)
        if not key.startswith('_')
        and not callable(getattr(obj, key))
        and isinstance(getattr(obj, key), (int, float))
    }


def _sort_key(filename):
    numbers = re.findall(r"\d+", filename)
    return int(numbers[0]) if numbers else -1


def _discover_scenarios(origin_dir):
    scenario_ids = set()
    if not os.path.isdir(origin_dir):
        raise FileNotFoundError(f"origin_dir does not exist: {origin_dir}")

    for name in os.listdir(origin_dir):
        match = SCENARIO_DIR_PATTERN.fullmatch(name)
        if match:
            scenario_ids.add(int(match.group(1)))
    return sorted(scenario_ids)


def _resolve_origin_dir(config):
    return getattr(config, "origin_dir", os.path.join("scenario_origin", config.map))


def _resolve_export_format(config):
    return getattr(config, "export_format", ADV_EXPORT)


def save_routes(config, save_dir, route, weather_id, weather):
    route_id = 0
    save_file = os.path.join(save_dir, f"scenario_{config.scenario:02d}_route_{route_id:02d}_weather_{weather_id:02d}.xml")
    while os.path.isfile(save_file):
        route_id += 1
        save_file = os.path.join(save_dir, f"scenario_{config.scenario:02d}_route_{route_id:02d}_weather_{weather_id:02d}.xml")

    build_route(route, route_id, config.map, save_file, weathers=weather)
    return route_id


def _build_index_entry(export_kind, base_entry):
    entry = dict(base_entry)
    if export_kind == STANDARD_EXPORT:
        entry.update({
            "scenario_folder": "standard",
            "risk_level": None,
            "parameters": None,
        })
    elif export_kind == ADV_EXPORT:
        entry.update({
            "scenario_folder": "adv_init_state",
            "risk_level": None,
            "parameters": copy.deepcopy(ADV_PARAMETERS),
        })
    else:
        raise ValueError(f"unsupported export kind: {export_kind}")
    return entry


def _write_index_files(save_dir, scenario_id, base_entries, export_format):
    export_kinds = []
    if export_format in {STANDARD_EXPORT, "both"}:
        export_kinds.append(STANDARD_EXPORT)
    if export_format in {ADV_EXPORT, "both"}:
        export_kinds.append(ADV_EXPORT)
    if not export_kinds:
        raise ValueError(f"unsupported export format: {export_format}")

    os.makedirs(save_dir, exist_ok=True)
    for export_kind in export_kinds:
        entries = [_build_index_entry(export_kind, base_entry) for base_entry in base_entries]
        filename = f"{export_kind}_scenario_{scenario_id:02d}.json"
        with open(os.path.join(save_dir, filename), "w") as handle:
            json.dump(entries, handle, indent=2)


def main(config):
    np.set_printoptions(suppress=True)

    origin_dir = _resolve_origin_dir(config)
    export_format = _resolve_export_format(config)
    scenario_ids = _discover_scenarios(origin_dir) if config.scenario < 0 else [config.scenario]

    centers = get_map_centers(config.map)
    weather_parameters = list(get_static_public_attributes(carla.WeatherParameters).values())

    for scenario_id in scenario_ids:
        config.scenario = scenario_id
        source_dir = os.path.join(origin_dir, f"scenario_{scenario_id:02d}_routes")
        route_file_names = sorted(
            [name for name in os.listdir(source_dir) if name.startswith("route_") and name.endswith(".npy")],
            key=_sort_key,
        )

        save_dir = os.path.join(config.save_dir, f"scenario_{scenario_id:02d}_routes")
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        base_entries = []
        data_id = 0
        export_route_num = 0

        for route_file_name in route_file_names:
            route_file = os.path.join(source_dir, route_file_name)
            selected_waypoints = np.load(route_file)

            for _ in centers:
                for scenario_route in create_route_hongkong(config, selected_waypoints):
                    for weather_id, weather_item in enumerate(weather_parameters):
                        weather = object_to_numeric_dict(weather_item)
                        route_id = save_routes(config, save_dir, scenario_route, weather_id=weather_id, weather=weather)
                        base_entries.append({
                            "data_id": data_id,
                            "scenario_id": scenario_id,
                            "route_id": route_id,
                            "weather_id": weather_id,
                        })
                        data_id += 1
                        export_route_num += 1

        _write_index_files(config.save_dir, scenario_id, base_entries, export_format)
        print(f"{export_route_num} routes of scenario {scenario_id} exported from {source_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='center')
    parser.add_argument('--origin_dir', type=str, default=None, help="defaults to scenario_origin/<map>")
    parser.add_argument('--save_dir', type=str, default="scenario_data/center")
    parser.add_argument('--scenario', type=int, default=1)
    parser.add_argument(
        '--format',
        dest='export_format',
        type=str,
        default=ADV_EXPORT,
        choices=[STANDARD_EXPORT, ADV_EXPORT, 'both'],
    )

    args = parser.parse_args()
    if args.origin_dir is None:
        args.origin_dir = os.path.join("scenario_origin", args.map)

    main(args)
