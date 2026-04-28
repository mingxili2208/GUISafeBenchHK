import argparse
import json
import os
import re

import numpy as np

from utilities import build_scenarios


SCENARIO_DIR_PATTERN = re.compile(r"scenario_(\d{2})_scenarios$")
SCENARIO_NAME_LIST = [
    "",
    "DynamicObjectCrossing",
    "VehicleTurningRoute",
    "OtherLeadingVehicle",
    "ManeuverOppositeDirection",
    "OppositeVehicleRunningRedLight",
    "SignalizedJunctionLeftTurn",
    "SignalizedJunctionRightTurn",
    "NoSignalJunctionCrossingRoute",
]


def create_scenario_hongkong(selected_waypoints, side_marks):
    scenario_config = build_scenarios(selected_waypoints, side_marks)
    return [scenario_config]


def _sort_key(filename):
    numbers = re.findall(r"\d+", filename)
    return int(numbers[0]) if numbers else -1


def _resolve_origin_dir(config):
    return getattr(config, "origin_dir", os.path.join("scenario_origin", config.map))


def _discover_scenarios(origin_dir):
    scenario_ids = set()
    if not os.path.isdir(origin_dir):
        raise FileNotFoundError(f"origin_dir does not exist: {origin_dir}")

    for name in os.listdir(origin_dir):
        match = SCENARIO_DIR_PATTERN.fullmatch(name)
        if match:
            scenario_ids.add(int(match.group(1)))
    return sorted(scenario_ids)


def _collect_route_ids(origin_dir, scenario_id):
    routes_dir = os.path.join(origin_dir, f"scenario_{scenario_id:02d}_routes")
    if not os.path.isdir(routes_dir):
        return set()

    return {
        int(name.replace("route_", "").replace(".npy", ""))
        for name in os.listdir(routes_dir)
        if name.startswith("route_") and name.endswith(".npy")
    }


def save_scenarios(config, scenario_id, scenarios_configs):
    save_dir = os.path.join(config.save_dir, "scenarios")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"scenario_{scenario_id:02d}.json")

    scenario_json = {
        "available_scenarios": [
            {
                config.map: [
                    {
                        "available_event_configurations": scenarios_configs,
                        "scenario_name": SCENARIO_NAME_LIST[scenario_id],
                    }
                ]
            }
        ]
    }
    with open(save_file, 'w') as handle:
        json.dump(scenario_json, handle, indent=2)


def main(config):
    np.set_printoptions(suppress=True)

    origin_dir = _resolve_origin_dir(config)
    scenario_ids = _discover_scenarios(origin_dir) if config.scenario < 0 else [config.scenario]

    for scenario_id in scenario_ids:
        source_dir = os.path.join(origin_dir, f"scenario_{scenario_id:02d}_scenarios")
        route_ids = _collect_route_ids(origin_dir, scenario_id)
        scenario_file_names = sorted(
            [
                name for name in os.listdir(source_dir)
                if name.endswith(".npy")
                and "sides" not in name
                and int(name.replace("scenario_", "").replace(".npy", "")) in route_ids
            ],
            key=_sort_key,
        )

        all_scenario_configs = []
        for scenario_file_name in scenario_file_names:
            scenario_file = os.path.join(source_dir, scenario_file_name)
            selected_waypoints = np.load(scenario_file)
            scenario_actor_sides_file = scenario_file.replace(".npy", "_sides.npy")
            side_marks = np.load(scenario_actor_sides_file)
            all_scenario_configs += create_scenario_hongkong(selected_waypoints, side_marks)

        save_scenarios(config, scenario_id, all_scenario_configs)
        print(f"{len(all_scenario_configs)} scenarios of scenario {scenario_id} exported from {source_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='center')
    parser.add_argument('--origin_dir', type=str, default=None, help="defaults to scenario_origin/<map>")
    parser.add_argument('--save_dir', type=str, default="scenario_data/center")
    parser.add_argument('--scenario', type=int, default=-1)

    args = parser.parse_args()
    if args.origin_dir is None:
        args.origin_dir = os.path.join("scenario_origin", args.map)

    main(args)
