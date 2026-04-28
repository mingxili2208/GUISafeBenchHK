"""
为TCP自动驾驶算法定制的环境包装器
基于原始env_wrapper.py，支持TCP所需的观测类型

主要变更:
1. 添加obs_type=4支持TCP的原始尺寸相机图像
2. 使用CarlaEnvTCP替代CarlaEnv
"""

import copy
import gym
import numpy as np
import pygame


class VectorWrapperTCP:
    """ 
        The interface to control a list of environments, customized for TCP.
    """

    def __init__(self, env_params, scenario_config, world, birdeye_render, display, logger):
        self.logger = logger
        self.world = world
        self.num_scenario = scenario_config['num_scenario']
        self.ROOT_DIR = scenario_config['ROOT_DIR']
        self.frame_skip = scenario_config['frame_skip']  
        self.render = scenario_config['render']

        self.env_list = []
        self.action_space_list = []
        for _ in range(self.num_scenario):
            env = carla_env_tcp(env_params, birdeye_render=birdeye_render, display=display, world=world, logger=logger)
            self.env_list.append(env)
            self.action_space_list.append(env.action_space)

        self.finished_env = [False] * self.num_scenario
        # self.running_results = {}
    
    def obs_postprocess(self, obs_list):
        # 对于TCP，返回obs列表而不是numpy数组，因为包含字典
        return obs_list

    def get_ego_vehicles(self):
        ego_vehicles = []
        for env in self.env_list:
            if env.ego_vehicle is not None:
                ego_vehicles.append(env.ego_vehicle)
        return ego_vehicles

    def get_static_obs(self, scenario_configs):
        static_obs_list = []
        for s_i in range(len(scenario_configs)):
            static_obs = self.env_list[s_i].get_static_obs(scenario_configs[s_i])
            static_obs_list.append(static_obs)
        return static_obs_list

    def reset(self, scenario_configs, scenario_init_action):
        obs_list = []
        info_list = []
        for s_i in range(len(scenario_configs)):
            config = scenario_configs[s_i]
            obs, info = self.env_list[s_i].reset(config=config, env_id=s_i, scenario_init_action=scenario_init_action[s_i])
            obs_list.append(obs)
            info_list.append(info)
            info_list[s_i].update({'data_id': config.data_id})

        self.finished_env = [False] * self.num_scenario
        for s_i in range(len(scenario_configs), self.num_scenario):
            self.finished_env[s_i] = True

        for s_i in range(len(scenario_configs)):
            info_list[s_i].update({'s_id': s_i})

        return self.obs_postprocess(obs_list), info_list

    def step(self, ego_actions, scenario_actions, scenario_configs):
        # apply action
        action_idx = 0  # action idx should match the env that is not finished
        for e_i in range(self.num_scenario):
            if not self.finished_env[e_i]:
                processed_action = self.env_list[e_i]._postprocess_action(ego_actions[action_idx])
                self.env_list[e_i].step_before_tick(processed_action, scenario_actions[action_idx])
                action_idx += 1

        # tick all scenarios
        for _ in range(self.frame_skip):
            self.world.tick()


        obs_list = []
        reward_list = []
        done_list = []
        info_list = []
        for e_i in range(self.num_scenario):
            if not self.finished_env[e_i]:
                current_env = self.env_list[e_i]
                obs, reward, done, info = current_env.step_after_tick()

                # info['data_id'] = current_env.config.data_id
                # info['s_id'] = e_i
                info['data_id'] = scenario_configs[e_i].data_id
                if done:
                    self.finished_env[e_i] = True
                    # if current_env.config.data_id in self.running_results.keys():
                    if current_env.config.data_id in self.env_list[e_i].running_results.keys():
                        self.logger.log('Scenario with data_id {} is duplicated'.format(current_env.config.data_id))
                    self.env_list[e_i].running_results[current_env.config.data_id] = copy.deepcopy(
                        current_env.scenario_manager.running_record
                    )

                obs_list.append(obs)
                reward_list.append(reward)
                done_list.append(done)
                info_list.append(info)
        
        rewards = np.array(reward_list)
        dones = np.array(done_list)
        infos = np.array(info_list)

        if self.render:
            pygame.display.flip()
        return self.obs_postprocess(obs_list), rewards, dones, infos

    def all_scenario_done(self):
        if np.sum(self.finished_env) == self.num_scenario:
            return True
        else:
            return False

    def clean_up(self):
        import gc
        from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider

        self.running_results = {}
        for e_i in range(self.num_scenario):
            try:
                self.env_list[e_i].clean_up()
            except Exception as cleanup_error:
                self.logger.log(f'>> Env cleanup failed for index {e_i}: {cleanup_error}', 'yellow')

        try:
            self.world.tick()
        except Exception as world_error:
            self.logger.log(f'>> World tick after cleanup failed: {world_error}', 'yellow')

        CarlaDataProvider.clear_actor_tracking_maps()
        gc.collect()


class ObservationWrapperTCP(gym.Wrapper):
    """为TCP定制的观测包装器"""
    
    def __init__(self, env, obs_type):
        super().__init__(env)
        self._env = env

        self.is_running = False
        self.obs_type = obs_type
        self._build_obs_space()

        act_dim = 2
        act_lim = np.ones((act_dim), dtype=np.float32)
        self.action_space = gym.spaces.Box(-act_lim, act_lim, dtype=np.float32)

    def get_static_obs(self, config):
        return self._env.get_static_obs(config)

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        return self._preprocess_obs(obs), info

    def step_before_tick(self, ego_action, scenario_action):
        self._env.step_before_tick(ego_action=ego_action, scenario_action=scenario_action)

    def step_after_tick(self):
        obs, reward, done, info = self._env.step_after_tick()
        self.is_running = self._env.is_running
        reward, info = self._preprocess_reward(reward, info)
        obs = self._preprocess_obs(obs)
        return obs, reward, done, info

    def _build_obs_space(self):
        if self.obs_type == 0:
            obs_dim = 4
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
        elif self.obs_type == 1:
            obs_dim = 11
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
        elif self.obs_type == 2 or self.obs_type == 3:
            obs_dim = 128
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
        elif self.obs_type == 4:
            # TCP专用：原始尺寸相机图像
            obs_dim = 256
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
        else:
            raise NotImplementedError

    def _preprocess_obs(self, obs):
        """
        根据不同的obs_type,对环境数据进行预处理
        obs_type=4: 返回TCP所需的原始尺寸相机图像
        """
        if self.obs_type == 0:
            return obs['state'][:4].astype(np.float64)
        elif self.obs_type == 1:
            new_obs = np.array([
                obs['state'][0], obs['state'][1], obs['state'][2], obs['state'][3],
                obs.get('command', 4), 
                obs.get('forward_vector', [0, 0])[0], obs.get('forward_vector', [0, 0])[1],
                obs.get('node_forward', [0, 0])[0], obs.get('node_forward', [0, 0])[1],
                obs.get('target_forward', [0, 0])[0], obs.get('target_forward', [0, 0])[1]
            ])
            return new_obs
        elif self.obs_type == 2:
            return {"img": obs['birdeye'], "states": obs['state'][:4].astype(np.float64)}
        elif self.obs_type == 3:
            return {"img": obs['camera'], "states": obs['state'][:4].astype(np.float64)}
        elif self.obs_type == 4:
            # TCP专用：使用原始尺寸相机图像
            return {"img": obs['camera_raw'], "states": obs['state'][:4].astype(np.float64)}
        else:
            raise NotImplementedError

    def _preprocess_reward(self, reward, info):
        return reward, info

    def _postprocess_action(self, action):
        return action

    def clear_up(self):
        self._env.clear_up()


def carla_env_tcp(env_params, birdeye_render=None, display=None, world=None, logger=None):
    """创建TCP专用的CARLA环境"""
    from safebench.gym_carla.envs.carla_env_tcp import CarlaEnvTCP
    
    # 注册gym环境（如果尚未注册）
    env_id = 'carla-tcp-v0'
    if env_id not in gym.envs.registry.env_specs:
        gym.register(
            id=env_id,
            entry_point='safebench.gym_carla.envs.carla_env_tcp:CarlaEnvTCP',
        )
    
    return ObservationWrapperTCP(
        CarlaEnvTCP(
            env_params=env_params, 
            birdeye_render=birdeye_render,
            display=display, 
            world=world, 
            logger=logger,
        ), 
        obs_type=env_params['obs_type']
    )
