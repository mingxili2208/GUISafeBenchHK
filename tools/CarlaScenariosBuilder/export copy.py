"""
在预先创建好路线和场景之后,将其导出到指定的目录中,同时进行一些文件操作以确保数据的一致性和完整性
"""
from copy import deepcopy
from export_routes import main as export_routes
from export_scenarios import main as export_scenarios
import argparse
import os
import re

def try_remove(path):
    """尝试删除文件，如果不存在则忽略"""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"[警告] 无法删除文件 {path}: {e}")

def cleanup_unused_routes(scenario_id, origin_dir):
    """
    检查 route 和 scenario 文件夹：
      1. 删除没有对应场景文件的 route
      2. 重排序号（保持两个文件夹一致）
    """
    routes_dir = os.path.join(origin_dir, f"scenario_{scenario_id:02d}_routes")
    scenarios_dir = os.path.join(origin_dir, f"scenario_{scenario_id:02d}_scenarios")

    # 确保目录存在
    if not os.path.exists(routes_dir) or not os.path.exists(scenarios_dir):
        print(f"[错误] 目录不存在: {routes_dir} 或 {scenarios_dir}")
        return

    # 1. 预先清理残留的临时文件（防止上次崩溃导致残留）
    tmp_prefix = "._tmp_rename_"
    for d in [routes_dir, scenarios_dir]:
        for f in os.listdir(d):
            if f.startswith(tmp_prefix):
                try_remove(os.path.join(d, f))

    # 2. 清理无效路线
    # 按数字顺序排序，防止 route_10 排在 route_2 前面
    def sort_key(f):
        nums = re.findall(r'\d+', f)
        return int(nums[0]) if nums else 0

    route_files = sorted([f for f in os.listdir(routes_dir) if f.startswith("route_") and f.endswith(".npy")], key=sort_key)
    removed_count = 0

    for route_file in route_files:
        route_idx = route_file.replace("route_", "").replace(".npy", "")
        scenario_file = f"scenario_{route_idx}.npy"

        # 删除无对应场景文件的route文件
        if not os.path.exists(os.path.join(scenarios_dir, scenario_file)):
            try_remove(os.path.join(routes_dir, route_file))
            removed_count += 1
            print(f"[清理] 删除无效路线: {route_file}")

    print(f"[清理] 完成，删除了 {removed_count} 条无效路线")

    # 3. 重新获取剩余文件并按数字排序
    remaining_routes = sorted([f for f in os.listdir(routes_dir) if f.startswith("route_") and f.endswith(".npy")], key=sort_key)

    print(f"[重编号] 开始处理 {len(remaining_routes)} 个文件...")

    # 4. 重命名逻辑
    # 使用 os.replace 而不是 os.rename，因为 os.replace 在 Windows 上可以覆盖已存在的文件，避免 FileExistsError
    for i, route_file in enumerate(remaining_routes):
        old_idx = route_file.replace("route_", "").replace(".npy", "")

        # --- Route 文件重命名 ---
        old_route_path = os.path.join(routes_dir, route_file)
        tmp_route_path = os.path.join(routes_dir, f"{tmp_prefix}{i:02d}.npy")
        os.replace(old_route_path, tmp_route_path)

        # --- Scenario 文件重命名 ---
        old_scn_path = os.path.join(scenarios_dir, f"scenario_{old_idx}.npy")
        if os.path.exists(old_scn_path):
            tmp_scn_path = os.path.join(scenarios_dir, f"{tmp_prefix}{i:02d}.npy")
            os.replace(old_scn_path, tmp_scn_path)

        # --- Sides 文件重命名 ---
        old_sides_path = os.path.join(scenarios_dir, f"scenario_{old_idx}_sides.npy")
        if os.path.exists(old_sides_path):
            tmp_sides_path = os.path.join(scenarios_dir, f"{tmp_prefix}_sides_{i:02d}.npy")
            os.replace(old_sides_path, tmp_sides_path)

    # 5. 将临时文件重命名为最终文件名 (00, 01, 02...)
    for i in range(len(remaining_routes)):
        # Route
        tmp_route_path = os.path.join(routes_dir, f"{tmp_prefix}{i:02d}.npy")
        final_route_path = os.path.join(routes_dir, f"route_{i:02d}.npy")
        if os.path.exists(tmp_route_path):
            os.replace(tmp_route_path, final_route_path)

        # Scenario
        tmp_scn_path = os.path.join(scenarios_dir, f"{tmp_prefix}{i:02d}.npy")
        final_scn_path = os.path.join(scenarios_dir, f"scenario_{i:02d}.npy")
        if os.path.exists(tmp_scn_path):
            os.replace(tmp_scn_path, final_scn_path)

        # Sides
        tmp_sides_path = os.path.join(scenarios_dir, f"{tmp_prefix}_sides_{i:02d}.npy")
        final_sides_path = os.path.join(scenarios_dir, f"scenario_{i:02d}_sides.npy")
        if os.path.exists(tmp_sides_path):
            os.replace(tmp_sides_path, final_sides_path)

    print("[重编号] 完成，序号已重新排列从 00 开始")


def main(config):
    cleanup_unused_routes(config.scenario, config.origin_dir)
    # 导出函数调用
    try:
        print("[导出] 开始导出路线...")
        export_routes(deepcopy(config))
        print("[导出] 开始导出场景...")
        export_scenarios(deepcopy(config))
    except Exception as e:
        print(f"[错误] 导出过程中发生错误: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='sham_shui_po_scene')
    # 指定手动选取路线和场景触发位置的npy文件的存储位置
    parser.add_argument('--origin_dir', type=str, default="scenario_origin/sham_shui_po_scene")
    # 指定行驶route和场景触发点位置的存储位置
    parser.add_argument('--save_dir', type=str, default="scenario_data/sham_shui_po_scene")
    # 指定需要导出的场景ID,默认导出所有场景
    parser.add_argument('--scenario', type=int, default=1)
    args = parser.parse_args()

    main(args)