"""
1、根据不同的地图配置初始化仿真环境和渲染器
2、数据加载,使用ScenarioDataLoader加载场景数据
3、动作获取与环境交互
4、经验存储,将动作、观测值、奖励等信息存储到经验回放缓冲区中
5、根据不同的策略on-policy/off-policy,从经验回放缓冲区中采样数据
6、策略训练,在评估过程中记录和打印结果,并保存视频和结果
7、环境清理,关闭环境并释放资源
"""

import numpy as np
import torch


class RouteReplayBuffer:
    """
        This buffer supports parallel storing transitions from multiple trajectories.
    """
    
    def __init__(self, num_total_scenarios, mode, buffer_capacity=1000):
        self.mode = mode
        self.buffer_capacity = buffer_capacity
        # 代表一个episode中预先设计好的场景数量
        self.num_total_scenarios = num_total_scenarios
        self.buffer_len = 0

        # 管理动态的、时间步粒度的交互数据
        self.reset_buffer()

        # 管理静态的、回合粒度的初始化数据
        self.reset_init_buffer()

    def reset_buffer(self):
        self.buffer_ego_actions = [[] for _ in range(self.num_total_scenarios)]
        self.buffer_scenario_actions = [[] for _ in range(self.num_total_scenarios)]
        self.buffer_obs = [[] for _ in range(self.num_total_scenarios)]
        self.buffer_next_obs = [[] for _ in range(self.num_total_scenarios)]
        self.buffer_rewards = [[] for _ in range(self.num_total_scenarios)]
        self.buffer_dones = [[] for _ in range(self.num_total_scenarios)]
        self.buffer_additional_dict = [{} for _ in range(self.num_total_scenarios)]

    def reset_init_buffer(self):
        self.buffer_static_obs = []
        self.buffer_init_action = []
        self.buffer_episode_reward = []
        self.buffer_init_additional_dict = {}
        self.init_buffer_len = 0

    def finish_one_episode(self):
        # get total reward for episode
        for s_i in range(self.num_total_scenarios):
            dones = np.where(self.buffer_dones[s_i])[0]
            start_ = dones[-2] if len(dones) > 1 else -1
            end_ = dones[-1]
            self.buffer_episode_reward.append(np.sum(self.buffer_rewards[s_i][start_+1:end_+1]))

    def store(self, data_list, additional_dict):
        ego_actions = data_list[0]
        scenario_actions = data_list[1]
        obs = data_list[2]
        next_obs = data_list[3]
        rewards = data_list[4]
        dones = data_list[5]
        self.buffer_len += len(rewards)

        # separate trajectories according to infos
        for s_i in range(len(additional_dict)):
            sid = additional_dict[s_i]['data_id']  # data_id是具体的第几个scenario的序列索引
            self.buffer_ego_actions[sid].append(ego_actions[s_i])  # 添加自车动作
            self.buffer_scenario_actions[sid].append(scenario_actions[s_i])  # 添加场景动作
            self.buffer_obs[sid].append(obs[s_i])  # 添加观测
            self.buffer_next_obs[sid].append(next_obs[s_i])  # 添加下一个观测
            self.buffer_rewards[sid].append(rewards[s_i])  # 添加奖励
            self.buffer_dones[sid].append(dones[s_i])  # 添加完成标志

            # store additional information in given dict (e.g., cost and actor_info)
            for key in additional_dict[s_i].keys():
                if key == 'data_id':  # 原先是scenario_id
                    continue
                if key not in self.buffer_additional_dict[sid].keys():
                    self.buffer_additional_dict[sid][key] = []
                self.buffer_additional_dict[sid][key].append(additional_dict[s_i][key])

    def store_init(self, data_list, additional_dict=None):
        static_obs = data_list[0]
        scenario_init_action = data_list[1]
        self.buffer_static_obs.append(static_obs)
        self.buffer_init_action.append(scenario_init_action)
        self.init_buffer_len += len(scenario_init_action)

        # store additional information in given dict
        if additional_dict:
            for key in additional_dict.keys():
                if key not in self.buffer_init_additional_dict.keys():
                    self.buffer_init_additional_dict[key] = []
                self.buffer_init_additional_dict[key].append(additional_dict[key])

    def sample_init(self, batch_size):
        # 从存储的初始化数据中随机选择一个批次,并返回一个包含这些数据的字典
        num_trajectory = len(self.buffer_init_action)  # 获取初始化动作的数量
        start_idx = np.max([0, num_trajectory - self.buffer_capacity])  # 计算开始索引，确保不超过缓冲区容量


        # 选择最新的静态观测数据、初始化动作和奖励
        prepared_static_obs = self.buffer_static_obs[start_idx:]
        prepared_init_action = self.buffer_init_action[start_idx:]
        prepared_episode_reward = self.buffer_episode_reward[start_idx:]

        # 随即选择一个批次的索引
        sample_index = np.random.randint(0, len(prepared_init_action), size=batch_size)
        static_obs = np.concatenate(prepared_static_obs, axis=0)[sample_index]  # 根据索引获取静态观测数据
        init_action = np.concatenate(prepared_init_action, axis=0)[sample_index]  # 根据索引获取初始化动作
        episode_reward = np.array(prepared_episode_reward)[sample_index]  # 根据索引获取回合奖励
        # 创建一个包含这些采样数据的字典
        batch = {
            'static_obs': static_obs,
            'init_action': init_action,
            'episode_reward': episode_reward,
        }

        # 遍历初始化缓冲区中的额外信息
        for key in self.buffer_init_additional_dict.keys():
            batch[key] = torch.cat(self.buffer_init_additional_dict[key][start_idx:])[sample_index]
        return batch

    def sample(self, batch_size):
        # prepare concatenated list
        prepared_ego_actions = []
        prepared_scenario_actions = []
        prepared_obs = []
        prepared_next_obs = []
        prepared_rewards = []
        prepared_dones = []
        prepared_infos = {}

        # get the length of each sub-buffer
        samples_per_trajectory = self.buffer_capacity // self.num_total_scenarios # assume average over all sub-buffer
        for sid in range(self.num_total_scenarios):
            # select the latest samples starting from the end of buffer
            num_trajectory = len(self.buffer_rewards[sid])
            start_idx = np.max([0, num_trajectory - samples_per_trajectory])

            # concat
            prepared_ego_actions += self.buffer_ego_actions[sid][start_idx:]
            prepared_scenario_actions += self.buffer_scenario_actions[sid][start_idx:]
            prepared_obs += self.buffer_obs[sid][start_idx:]
            prepared_next_obs += self.buffer_next_obs[sid][start_idx:]
            prepared_rewards += self.buffer_rewards[sid][start_idx:]
            prepared_dones += self.buffer_dones[sid][start_idx:]

            # add additional information 
            for k_i in self.buffer_additional_dict[sid].keys():
                if k_i not in prepared_infos.keys():
                    prepared_infos[k_i] = []
                prepared_infos[k_i] += self.buffer_additional_dict[sid][k_i][start_idx:]

        # sample from concatenated list
        # the first sample does not have previous state ()
        sample_index = np.random.randint(1, len(prepared_rewards), size=batch_size)

        # prepare batch 
        action = prepared_ego_actions if self.mode == 'train_agent' else prepared_scenario_actions
        batch = {
            'action': np.stack(action)[sample_index],                 # action
            'state': np.stack(prepared_obs)[sample_index, :],         # state
            'n_state': np.stack(prepared_next_obs)[sample_index, :],  # next state
            'reward': np.stack(prepared_rewards)[sample_index],       # reward
            'done': np.stack(prepared_dones)[sample_index],           # done
        }

        # add additional information to the batch
        batch_info = {} 
        for k_i in prepared_infos.keys():
            if k_i == 'route_waypoints':
                continue
            batch_info[k_i] = np.stack(prepared_infos[k_i])[sample_index-1]
            batch_info['n_' + k_i] = np.stack(prepared_infos[k_i])[sample_index]

        # combine two dicts
        batch.update(batch_info)
        return batch


class PerceptionReplayBuffer:
    """
        This buffer supports parallel storing image states and labels for object detection
    """
    
    def __init__(self, num_scenario, mode, buffer_capacity=1000):
        self.mode = mode
        self.buffer_capacity = buffer_capacity
        self.num_scenario = num_scenario
        self.buffer_len = 0

        # buffers for different data type
        self.buffer_bbox_label = [[] for _ in range(num_scenario)]          # perception labels
        self.buffer_predictions = [[] for _ in range(num_scenario)]         # perception outputs
        self.buffer_scenario_actions = [[] for _ in range(num_scenario)]    # synthetic textures (attack)
        self.buffer_obs = [[] for _ in range(num_scenario)]                 # image observations (FPV observation)
        self.buffer_loss = [[] for _ in range(num_scenario)]                # object detection loss (IoU, class, etc.)
    
    def finish_one_episode(self):
        pass

    def reset_init_buffer(self):
        self.buffer_static_obs = []
        self.buffer_init_action = []
        self.buffer_episode_reward = []
        self.buffer_init_additional_dict = {}
        self.init_buffer_len = 0
    
    def store_init(self, data_list, additional_dict=None):
        pass
    
    def store(self, data_list, additional_dict=None):
        ego_actions = data_list[0]
        scenario_actions = data_list[1]
        obs = data_list[2]
        self.buffer_len += len(ego_actions)

        # separate trajectories according to infos
        for s_i in range(len(additional_dict)):
            sid = additional_dict[s_i]['data_id']
            self.buffer_predictions[sid].append(ego_actions[s_i]['od_result'])
            self.buffer_scenario_actions[sid].append(scenario_actions[s_i]['attack'])
            self.buffer_obs[sid].append(obs[s_i]['img'])
            self.buffer_bbox_label[sid].append(additional_dict[s_i]['bbox_label'])
            self.buffer_loss[sid].append(additional_dict[s_i]['iou_loss'])

    def sample(self, batch_size):
        # prepare concatenated list
        prepared_bbox_label = []
        prepared_predictions = []
        prepared_obs = []
        prepared_scenario_actions = []
        prepared_loss = []
        # get the length of each sub-buffer
        samples_per_trajectory = self.buffer_capacity // self.num_scenario # assume average over all sub-buffer
        for s_i in range(self.num_scenario):
            # select the latest samples starting from the end of buffer
            num_trajectory = len(self.buffer_loss[s_i])
            start_idx = np.max([0, num_trajectory - samples_per_trajectory])

            # concat
            prepared_bbox_label += self.buffer_bbox_label[s_i][start_idx:]
            prepared_predictions += self.buffer_predictions[s_i][start_idx:]
            prepared_scenario_actions += self.buffer_scenario_actions[s_i][start_idx:]
            prepared_obs += self.buffer_obs[s_i][start_idx:]
            prepared_loss += self.buffer_loss[s_i][start_idx:]
        # sample from concatenated list
        sample_index = np.random.randint(0, len(prepared_loss), size=batch_size)

        batch = {
            'label': np.stack(prepared_bbox_label)[sample_index, :],        
            # 'prediction': np.stack(prepared_predictions)[sample_index, :],     # TODO: Multiple/empty predictions should be stacked together
            # 'attack': np.stack(prepared_scenario_actions)[sample_index, :],
            # 'attack': torch.stack(prepared_scenario_actions)[sample_index, :],
            'image': np.stack(prepared_obs)[sample_index, :],
            'loss': np.stack(prepared_loss)[sample_index],                       # scalar with 1D 
        }
        
        return batch
