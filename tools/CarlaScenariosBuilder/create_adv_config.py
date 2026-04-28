"""
生成SafeBench场景配置文件的脚本
读取scenario_data/center目录下的XML文件，生成对应的JSON配置文件
"""

import os
import json
import re
import argparse
from pathlib import Path


def parse_xml_filename(filename):
    """
    解析XML文件名，提取scenario_id, route_id, weather_id
    例如: scenario_01_route_00_weather_15.xml -> (1, 0, 15)
    """
    pattern = r'scenario_(\d+)_route_(\d+)_weather_(\d+)\.xml'
    match = re.match(pattern, filename)
    if match:
        scenario_id = int(match.group(1))
        route_id = int(match.group(2))
        weather_id = int(match.group(3))
        return scenario_id, route_id, weather_id
    return None


def generate_scenario_config(scenario_data_dir, save_dir, scenario_folder="adv_init_state", skip_combinations=None):
    """
    生成场景配置文件

    Args:
        scenario_data_dir: scenario_data目录路径
        save_dir: 保存配置文件的目录
        scenario_folder: 场景文件夹名称 ("standard" 或 "adv_init_state")
        skip_combinations: 要跳过的组合列表，格式[(scenario_id, route_id), ...]
    """
    if skip_combinations is None:
        skip_combinations = []

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 收集所有XML文件
    xml_files = []
    center_dir = Path(scenario_data_dir) / "center"

    print(f"Looking for XML files in: {center_dir}")

    # 遍历所有scenario_XX_routes目录
    for scenario_dir in center_dir.glob("scenario_*_routes"):
        print(f"Checking directory: {scenario_dir}")
        if scenario_dir.is_dir():
            # 收集该目录下的所有XML文件
            for xml_file in scenario_dir.glob("*.xml"):
                xml_files.append((xml_file.name, xml_file))
                print(f"Found XML file: {xml_file}")

    print(f"Found {len(xml_files)} XML files")

    # 按scenario_id分组处理
    scenarios_data = {}

    # 先按scenario_id分组收集所有文件
    for xml_filename, xml_filepath in sorted(xml_files):
        parsed = parse_xml_filename(xml_filename)
        if parsed is None:
            print(f"Warning: Cannot parse filename {xml_filename}")
            continue

        scenario_id, route_id, weather_id = parsed

        # 检查是否需要跳过此组合
        if (scenario_id, route_id) in skip_combinations:
            print(f"Skipping scenario {scenario_id}, route {route_id}")
            continue

        # 为每个scenario_id创建数据列表
        if scenario_id not in scenarios_data:
            scenarios_data[scenario_id] = []

        # 存储文件信息
        scenarios_data[scenario_id].append({
            "filename": xml_filename,
            "filepath": xml_filepath,
            "route_id": route_id,
            "weather_id": weather_id
        })

    # 为每个scenario生成配置文件，data_id从0开始独立计数
    total_routes = 0
    for scenario_id, scenario_files in scenarios_data.items():
        scenario_routes = []

        # 对每个scenario内的文件按route_id排序，确保顺序一致
        scenario_files_sorted = sorted(scenario_files, key=lambda x: x["route_id"])

        # 为当前scenario生成data_id，从0开始
        for data_id, file_info in enumerate(scenario_files_sorted):
            data = {
                "data_id": data_id,  # 每个scenario独立从0开始计数
                "scenario_folder": scenario_folder,
                "scenario_id": scenario_id,
                "route_id": file_info["route_id"],
                "weather_id": file_info["weather_id"],
                "risk_level": None,
                "parameters": [
                    "genetic_algorithm" if scenario_folder == "adv_init_state" else "standard",
                    True
                ]
            }
            scenario_routes.append(data)

        # 保存当前scenario的配置文件
        filename = f'{scenario_folder}_scenario_{scenario_id:02d}.json'
        filepath = os.path.join(save_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(scenario_routes, f, indent=2)

        route_count = len(scenario_routes)
        total_routes += route_count
        print(f"{route_count} routes of scenario {scenario_id} exported to {filepath}")
        print(f"Data IDs for scenario {scenario_id}: {[route['data_id'] for route in scenario_routes]}")

    print(f"\nTotal: {total_routes} routes exported across {len(scenarios_data)} scenarios")
    return scenarios_data


def main():
    parser = argparse.ArgumentParser(description='Generate SafeBench scenario configuration files')
    parser.add_argument('--scenario_data_dir', type=str,
                        default='scenario_data',
                        help='Path to scenario_data directory')
    parser.add_argument('--save_dir', type=str,
                        default='./scenarios',
                        help='Directory to save generated config files')
    parser.add_argument('--scenario_folder', type=str,
                        choices=['standard', 'adv_init_state'],
                        default='adv_init_state',
                        help='Scenario folder type')
    parser.add_argument('--skip_combinations', type=str, nargs='*', default=[],
                        help='Skip specific scenario-route combinations, format: "5-0" "3-2"')

    args = parser.parse_args()

    # 解析跳过的组合
    skip_combinations = []
    for combo in args.skip_combinations:
        try:
            scenario_id, route_id = map(int, combo.split('-'))
            skip_combinations.append((scenario_id, route_id))
        except ValueError:
            print(f"Warning: Invalid skip combination format: {combo}")

    if skip_combinations:
        print(f"Will skip combinations: {skip_combinations}")

    # 生成配置文件
    generate_scenario_config(
        scenario_data_dir=args.scenario_data_dir,
        save_dir=args.save_dir,
        scenario_folder=args.scenario_folder,
        skip_combinations=skip_combinations
    )


if __name__ == '__main__':
    main()