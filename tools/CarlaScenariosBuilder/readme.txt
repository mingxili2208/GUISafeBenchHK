map_waypoints: 以npy形式存储着地图的可行驶路径点
scenario_origin: 存储着在可视化界面上点选测试路线和交通物体npy信息
scenario_data: 存储着测试路线（eg,scenario_01_routes）,对抗交通物体触发位置（eg,scenarios）以及描述测试路线和交通物体的json文件

第一步：get_map_data.py: 获取地图的可行驶路径点和路网信息，存储为npy文件
第二步：create_routes.py： 手工在加载的地图上选择测试路线的脚本；create_scenarios.py: 手工在加载的地图上选择对抗交通物体触发位置的脚本
第三步：export.py: 设计好的测试路线、交通物体信息和触发位置连同预设天气一起导出为xml和json场景文件。它会调用export_routes.py和export_scenarios.py
第四步：visualize_routes.py: 可视化测试路线的脚本
visualize_scenarios.py: 可视化对抗交通物体触发位置的脚本
visualize_routes_scenarios.py: 可视化测试路线和对抗交通物体触发位置的脚本

