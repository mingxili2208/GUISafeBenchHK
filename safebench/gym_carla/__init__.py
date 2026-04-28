"""
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 17:18:58
Description:
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
"""

from gym.envs.registration import register
# 调用register函数来注册一个新的Gym环境,id指定环境的唯一标识符,entry_point指定环境的入口点为safebench.gym_carla.envs模块中的CarlaEnv类
register(
    id='carla-v0',
    entry_point='safebench.gym_carla.envs:CarlaEnv',
)
