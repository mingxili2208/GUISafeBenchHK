"""
Export routes, scenario annotations, and scenario index JSON files for runtime use.
"""
from copy import deepcopy
import argparse
import os
import re
import yaml

from export_routes import main as export_routes
from export_scenarios import main as export_scenarios


SCENARIO_DIR_PATTERN = re.compile(r"scenario_(\d{2})_(routes|scenarios)$")
KNOWN_CONFIG_KEYS = {
    "map",
    "origin_dir",
    "save_dir",
    "scenario",
    "export_format",
    "format",
    "skip_cleanup",
}
DEFAULT_CONFIG_FILENAMES = [
    "export_config.yaml",
    "export_config.example.yaml",
]


def try_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        print(f"[warning] failed to remove {path}: {exc}")


def _sort_key(filename):
    numbers = re.findall(r"\d+", filename)
    return int(numbers[0]) if numbers else -1


def _discover_scenario_ids(origin_dir):
    scenario_ids = set()
    if not os.path.isdir(origin_dir):
        raise FileNotFoundError(f"origin_dir does not exist: {origin_dir}")

    for name in os.listdir(origin_dir):
        match = SCENARIO_DIR_PATTERN.fullmatch(name)
        if match:
            scenario_ids.add(int(match.group(1)))
    return sorted(scenario_ids)


def cleanup_unused_routes(scenario_id, origin_dir):
    routes_dir = os.path.join(origin_dir, f"scenario_{scenario_id:02d}_routes")
    scenarios_dir = os.path.join(origin_dir, f"scenario_{scenario_id:02d}_scenarios")
    if not os.path.isdir(routes_dir) or not os.path.isdir(scenarios_dir):
        print(f"[warning] skip cleanup for scenario {scenario_id:02d}: missing {routes_dir} or {scenarios_dir}")
        return

    tmp_prefix = "._tmp_rename_"
    for directory in [routes_dir, scenarios_dir]:
        for filename in os.listdir(directory):
            if filename.startswith(tmp_prefix):
                try_remove(os.path.join(directory, filename))

    route_files = sorted(
        [name for name in os.listdir(routes_dir) if name.startswith("route_") and name.endswith(".npy")],
        key=_sort_key,
    )
    removed_count = 0
    for route_file in route_files:
        route_idx = route_file.replace("route_", "").replace(".npy", "")
        scenario_file = os.path.join(scenarios_dir, f"scenario_{route_idx}.npy")
        if not os.path.exists(scenario_file):
            try_remove(os.path.join(routes_dir, route_file))
            removed_count += 1
            print(f"[cleanup] removed route without matching scenario: {route_file}")

    if removed_count:
        print(f"[cleanup] removed {removed_count} orphan route files for scenario {scenario_id:02d}")

    remaining_routes = sorted(
        [name for name in os.listdir(routes_dir) if name.startswith("route_") and name.endswith(".npy")],
        key=_sort_key,
    )
    for new_idx, route_file in enumerate(remaining_routes):
        old_idx = route_file.replace("route_", "").replace(".npy", "")

        old_route_path = os.path.join(routes_dir, route_file)
        tmp_route_path = os.path.join(routes_dir, f"{tmp_prefix}{new_idx:02d}.npy")
        os.replace(old_route_path, tmp_route_path)

        old_scenario_path = os.path.join(scenarios_dir, f"scenario_{old_idx}.npy")
        if os.path.exists(old_scenario_path):
            tmp_scenario_path = os.path.join(scenarios_dir, f"{tmp_prefix}{new_idx:02d}.npy")
            os.replace(old_scenario_path, tmp_scenario_path)

        old_sides_path = os.path.join(scenarios_dir, f"scenario_{old_idx}_sides.npy")
        if os.path.exists(old_sides_path):
            tmp_sides_path = os.path.join(scenarios_dir, f"{tmp_prefix}_sides_{new_idx:02d}.npy")
            os.replace(old_sides_path, tmp_sides_path)

    for new_idx in range(len(remaining_routes)):
        tmp_route_path = os.path.join(routes_dir, f"{tmp_prefix}{new_idx:02d}.npy")
        final_route_path = os.path.join(routes_dir, f"route_{new_idx:02d}.npy")
        if os.path.exists(tmp_route_path):
            os.replace(tmp_route_path, final_route_path)

        tmp_scenario_path = os.path.join(scenarios_dir, f"{tmp_prefix}{new_idx:02d}.npy")
        final_scenario_path = os.path.join(scenarios_dir, f"scenario_{new_idx:02d}.npy")
        if os.path.exists(tmp_scenario_path):
            os.replace(tmp_scenario_path, final_scenario_path)

        tmp_sides_path = os.path.join(scenarios_dir, f"{tmp_prefix}_sides_{new_idx:02d}.npy")
        final_sides_path = os.path.join(scenarios_dir, f"scenario_{new_idx:02d}_sides.npy")
        if os.path.exists(tmp_sides_path):
            os.replace(tmp_sides_path, final_sides_path)

    print(f"[cleanup] scenario {scenario_id:02d} normalized to {len(remaining_routes)} route/scenario pairs")


def _normalize_config(config):
    if not getattr(config, "origin_dir", None):
        config.origin_dir = os.path.join("scenario_origin", config.map)
    if not getattr(config, "save_dir", None):
        config.save_dir = os.path.join("scenario_data", config.map)
    return config


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"cannot parse boolean value: {value}")


def _load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}

    if not isinstance(config_data, dict):
        raise ValueError(f"config file must contain a mapping: {config_path}")

    unknown_keys = sorted(set(config_data.keys()) - KNOWN_CONFIG_KEYS)
    if unknown_keys:
        print(f"[warning] unused config keys in {config_path}: {', '.join(unknown_keys)}")

    if "format" in config_data and "export_format" not in config_data:
        config_data["export_format"] = config_data["format"]
    config_data.pop("format", None)

    if "skip_cleanup" in config_data:
        config_data["skip_cleanup"] = _parse_bool(config_data["skip_cleanup"])
    if "scenario" in config_data and config_data["scenario"] is not None:
        config_data["scenario"] = int(config_data["scenario"])

    base_dir = os.path.dirname(os.path.abspath(config_path))
    for key in ["origin_dir", "save_dir"]:
        value = config_data.get(key)
        if value and not os.path.isabs(value):
            config_data[key] = os.path.normpath(os.path.join(base_dir, value))

    return config_data


def _build_config(args):
    cli_values = vars(args).copy()
    config_path = cli_values.pop("config_path", None)

    merged = {}
    if not config_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for filename in DEFAULT_CONFIG_FILENAMES:
            candidate = os.path.join(script_dir, filename)
            if os.path.isfile(candidate):
                config_path = candidate
                print(f"[config] auto-loading {candidate}")
                break

    if config_path:
        resolved_config_path = os.path.abspath(config_path)
        merged.update(_load_yaml_config(resolved_config_path))
        print(f"[config] loaded {resolved_config_path}")

    # If CLI --map overrides the yaml's map, clear the yaml-derived origin_dir /
    # save_dir so _normalize_config re-derives them from the new map name.
    cli_map = cli_values.get("map")
    if cli_map is not None and cli_map != merged.get("map"):
        merged.pop("origin_dir", None)
        merged.pop("save_dir", None)

    for key, value in cli_values.items():
        if value is not None:
            merged[key] = value

    if "map" not in merged:
        merged["map"] = "TsimShaTsui-1029-3"
    if "scenario" not in merged:
        merged["scenario"] = 1
    if "export_format" not in merged:
        merged["export_format"] = "standard"
    if "skip_cleanup" not in merged:
        merged["skip_cleanup"] = False

    return argparse.Namespace(**merged)


def main(config):
    config = _normalize_config(config)
    scenario_ids = _discover_scenario_ids(config.origin_dir) if config.scenario < 0 else [config.scenario]

    if getattr(config, "skip_cleanup", False):
        print("[cleanup] skipped by --skip-cleanup")
    else:
        for scenario_id in scenario_ids:
            cleanup_unused_routes(scenario_id, config.origin_dir)

    print(f"[export] exporting routes using format={config.export_format}")
    export_routes(deepcopy(config))
    print("[export] exporting scenario annotations")
    export_scenarios(deepcopy(config))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', type=str, default=None, help="optional yaml config file")
    parser.add_argument('--map', type=str, default=None)
    parser.add_argument('--origin_dir', type=str, default=None, help="defaults to scenario_origin/<map>")
    parser.add_argument('--save_dir', type=str, default=None, help="defaults to scenario_data/<map>")
    parser.add_argument('--scenario', type=int, default=None, help="scenario id to export; use -1 for all")
    parser.add_argument(
        '--format',
        dest='export_format',
        type=str,
        default=None,
        choices=['standard', 'adv', 'both'],
        help="which scenario index JSON format(s) to export",
    )
    parser.add_argument(
        '--skip-cleanup',
        action='store_true',
        default=None,
        help="skip route/scenario cleanup and renumbering before export",
    )

    args = parser.parse_args()
    main(_build_config(args))
