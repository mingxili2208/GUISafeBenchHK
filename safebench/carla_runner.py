"""
场景运行器的类
场景可视化、场景数据加载和采样、场景测试环境
"""
import copy
import json
import numpy as np
import carla
import os
import pygame
from tqdm import tqdm
import time
from safebench.gym_carla.env_wrapper import VectorWrapper
from safebench.gym_carla.env_wrapper_tcp import VectorWrapperTCP
from safebench.gym_carla.envs.render import BirdeyeRender
from safebench.gym_carla.replay_buffer import RouteReplayBuffer, PerceptionReplayBuffer
from safebench.agent import AGENT_POLICY_LIST
from safebench.scenario import SCENARIO_POLICY_LIST
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_data_loader import ScenarioDataLoader
from safebench.scenario.tools.scenario_utils import scenario_parse
from safebench.util.logger import Logger, setup_logger_kwargs
from safebench.util.metric_util import get_route_scores, get_perception_scores
try:
    import safebench.scenario.scenario_definition.adv_init_state as adv_init_state_module
except ImportError:
    adv_init_state_module = None

class CarlaRunner:
    def __init__(self, agent_config, scenario_config):
        self.scenario_config = scenario_config
        self.agent_config = agent_config
        self.GLOBAL_CARLA_ENV = None

        self.seed = scenario_config['seed']
        self.exp_name = scenario_config['exp_name']
        self.output_dir = scenario_config['output_dir']
        self.mode = scenario_config['mode']
        self.save_video = scenario_config['save_video']
        self.control_path = os.environ.get('SAFEBENCH_JOB_CONTROL_PATH')

        self.render = scenario_config['render']
        self.num_scenario = scenario_config['num_scenario']
        self.fixed_delta_seconds = scenario_config['fixed_delta_seconds']
        self.scenario_category = scenario_config['scenario_category']

        # continue training flag
        self.continue_agent_training = scenario_config['continue_agent_training']
        self.continue_scenario_training = scenario_config['continue_scenario_training']

        # apply settings to carla
        self.client = carla.Client('localhost', scenario_config['port'])
        # 设置较大的超时值，防止清理阶段（actor批量销毁/world.tick）因服务端偶发慢响应
        # 触发 C++ 析构器内 TimeoutException → std::terminate() 崩溃
        self.client.set_timeout(120.0)
        self.world = None
        self.env = None

        # env_params 主要用于渲染的BirdEyeView展示
        self.env_params = {
            'auto_ego': scenario_config['auto_ego'],
            'obs_type': agent_config['obs_type'],
            'scenario_category': self.scenario_category,               # planning or perception
            'ROOT_DIR': scenario_config['ROOT_DIR'],
            'warm_up_steps': 9,                                        # number of ticks after spawning the vehicles
            'disable_lidar': True,                                     # show bird-eye view lidar or not
            'display_size': 256,                                       # screen size of one bird-eye view window
            'obs_range': 64,                                           # observation range (meter)
            'd_behind': 12,                                            # distance behind the ego vehicle (meter)
            'max_past_step': 1,                                        # the number of past steps to draw
            'discrete': False,                                         # whether to use discrete control space
            'discrete_acc': [-3.0, 0.0, 3.0],                          # discrete value of accelerations
            'discrete_steer': [-0.2, 0.0, 0.2],                        # discrete value of steering angles
            'continuous_accel_range': [-3.0, 3.0],                     # continuous acceleration range
            'continuous_steer_range': [-0.3, 0.3],                     # continuous steering angle range
            'max_episode_step': scenario_config['max_episode_step'],   # maximum timesteps per episode
            'max_waypt': 20,                                           # maximum number of waypoints
            'lidar_bin': 0.125,                                        # bin size of lidar sensor (meter)
            'out_lane_thres': 4,                                       # threshold for out of lane (meter)
            'desired_speed': 25,                                       # desired speed (km/h)
            'image_sz': 1024,                                          # TODO: move to config of od scenario
        }

        # pass config from scenario to agent
        agent_config['mode'] = scenario_config['mode']
        agent_config['ego_action_dim'] = scenario_config['ego_action_dim']
        agent_config['ego_state_dim'] = scenario_config['ego_state_dim']
        agent_config['ego_action_limit'] = scenario_config['ego_action_limit']

        # 定义日志记录器
        logger_kwargs = setup_logger_kwargs(
            self.exp_name, 
            self.output_dir, 
            self.seed,
            agent=agent_config['policy_type'],
            scenario=scenario_config['policy_type'],
            scenario_category=self.scenario_category
        )
        self.logger = Logger(**logger_kwargs)  # 对Kwargs进行解包成字典

        # prepare parameters
        if self.mode == 'train_agent':
            self.buffer_capacity = agent_config['buffer_capacity']
            self.eval_in_train_freq = agent_config['eval_in_train_freq']
            self.save_freq = agent_config['save_freq']
            self.train_episode = agent_config['train_episode']
            self.logger.save_config(agent_config)
            self.logger.create_training_dir()
        elif self.mode == 'train_scenario':
            self.buffer_capacity = scenario_config['buffer_capacity']
            self.eval_in_train_freq = scenario_config['eval_in_train_freq']
            self.save_freq = scenario_config['save_freq']
            self.train_episode = scenario_config['train_episode']
            self.logger.save_config(scenario_config)
            self.logger.create_training_dir()
        elif self.mode == 'eval':
            self.save_freq = scenario_config['save_freq']
            self.logger.log('>> Evaluation Mode, skip config saving', 'yellow')
            self.logger.create_eval_dir(load_existing_results=True)
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}.")

        self.logger.log('>> Agent Policy: ' + agent_config['policy_type'])
        self.logger.log('>> Scenario Policy: ' + scenario_config['policy_type'])

        if self.scenario_config['auto_ego']:
            self.logger.log('>> Using auto-polit for ego vehicle, action of policy will be ignored', 'yellow')
        if scenario_config['policy_type'] == 'ordinary' and self.mode != 'train_agent':
            self.logger.log('>> Ordinary scenario can only be used in agent training', 'red')
            raise Exception()
        self.logger.log('>> ' + '-' * 40)

        self.agent_policy = AGENT_POLICY_LIST[agent_config['policy_type']](agent_config, logger=self.logger)
        self.scenario_policy = SCENARIO_POLICY_LIST[scenario_config['policy_type']](scenario_config, logger=self.logger)
        if self.save_video:
            assert self.mode == 'eval', 'only allow video saving in eval mode'
            self.logger.init_video_recorder()

    def _init_world(self, town):
        self.logger.log(f'>> Initializing carla world: {town}')
        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(self.scenario_config['tm_port'])
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def _init_renderer(self):
        self.logger.log('>> Initializing pygame birdeye renderer')
        pygame.init()
        flag = pygame.HWSURFACE | pygame.DOUBLEBUF
        if not self.render:
            flag = flag | pygame.HIDDEN

        if self.scenario_category == 'planning':
            if self.env_params['disable_lidar']:
                window_size = (self.env_params['display_size'] * 2, self.env_params['display_size'] * self.num_scenario)
            else:
                window_size = (self.env_params['display_size'] * 3, self.env_params['display_size'] * self.num_scenario)
        else:
            window_size = (self.env_params['display_size'], self.env_params['display_size'] * self.num_scenario)
        self.display = pygame.display.set_mode(window_size, flag)

        pixels_per_meter = self.env_params['display_size'] / self.env_params['obs_range']
        pixels_ahead_vehicle = (self.env_params['obs_range'] / 2 - self.env_params['d_behind']) * pixels_per_meter
        self.birdeye_params = {
            'screen_size': [self.env_params['display_size'], self.env_params['display_size']],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle,
        }
        self.birdeye_render = BirdeyeRender(self.world, self.birdeye_params, logger=self.logger)

    def train(self, data_loader, start_episode=0):
        buffer_cls = RouteReplayBuffer if self.scenario_category == 'planning' else PerceptionReplayBuffer
        replay_buffer = buffer_cls(len(data_loader), self.mode, self.buffer_capacity)

        for e_i in tqdm(range(start_episode, self.train_episode)):
            data_loader.reset_idx_counter()
            replay_buffer.reset_init_buffer()
            replay_buffer.reset_buffer()
            while len(data_loader) > 0:
                sampled_scenario_configs, _ = data_loader.sampler()
                static_obs = self.env.get_static_obs(sampled_scenario_configs)
                scenario_init_action, additional_dict = self.scenario_policy.get_init_action(static_obs)
                obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)
                replay_buffer.store_init([static_obs, scenario_init_action], additional_dict=additional_dict)

                self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos, static_obs=static_obs)

                while not self.env.all_scenario_done():
                    ego_actions = self.agent_policy.get_action(obs, infos, deterministic=False)
                    scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=False)

                    next_obs, rewards, dones, infos = self.env.step(
                        ego_actions=ego_actions,
                        scenario_actions=scenario_actions,
                        scenario_configs=sampled_scenario_configs,
                    )
                    replay_buffer.store([ego_actions, scenario_actions, obs, next_obs, rewards, dones], additional_dict=infos)
                    obs = copy.deepcopy(next_obs)

                    if self.mode == 'train_agent' and self.agent_policy.type == 'offpolicy':
                        self.agent_policy.train(replay_buffer)
                    elif self.mode == 'train_scenario' and self.scenario_policy.type == 'offpolicy':
                        self.scenario_policy.train(replay_buffer)

                self.env.clean_up()

            # Allow CARLA 5 seconds to fully tear down streaming sockets and
            # release resources from the previous scenario before starting the
            # next one.  This reduces the per-episode double-close race window.
            if self.logger is not None:
                self.logger.log('>> Waiting 5s for CARLA cleanup between episodes...', color='yellow')
            time.sleep(5)

            replay_buffer.finish_one_episode()
            self.logger.add_training_results('episode', e_i)
            print(f'Episode {e_i} Reward: {np.sum(replay_buffer.buffer_episode_reward)}')
            self.logger.add_training_results('episode_reward', np.sum(replay_buffer.buffer_episode_reward))
            self.logger.save_training_results()

            if self.mode == 'train_agent' and self.agent_policy.type == 'onpolicy':
                self.agent_policy.train(replay_buffer)
            elif self.mode == 'train_scenario' and self.scenario_policy.type in ['init_state', 'onpolicy']:
                self.scenario_policy.train(replay_buffer)

            if (e_i + 1) % self.eval_in_train_freq == 0:
                pass

            if (e_i + 1) % self.save_freq == 0:
                if self.mode == 'train_agent':
                    self.agent_policy.save_model(e_i)
                if self.mode == 'train_scenario':
                    self.scenario_policy.save_model(e_i)

    def _collect_batch_scenario_names(self, sampled_scenario_configs, num_sampled_scenario):
        scenario_names = []
        for scenario_i in range(num_sampled_scenario):
            fallback_name = (
                f'scenario_{sampled_scenario_configs[scenario_i].scenario_id:02d}'
                f'_route_{sampled_scenario_configs[scenario_i].route_id:02d}'
            )
            try:
                env_scenario_list = self.env.env_list[scenario_i].scenario_manager.scenario_list
                if env_scenario_list and len(env_scenario_list) > 0 and getattr(env_scenario_list[0], 'name', None):
                    scenario_names.append(env_scenario_list[0].name)
                else:
                    scenario_names.append(fallback_name)
            except Exception:
                scenario_names.append(fallback_name)
        return scenario_names

    def _build_batch_metadata(self, sampled_scenario_configs, scenario_names):
        return {
            'data_ids': [config.data_id for config in sampled_scenario_configs],
            'scenario_ids': [config.scenario_id for config in sampled_scenario_configs],
            'route_ids': [config.route_id for config in sampled_scenario_configs],
            'scenario_names': scenario_names,
            'town': getattr(sampled_scenario_configs[0], 'town', None) if sampled_scenario_configs else None,
            'content_tag': self.scenario_category,
        }

    def persist_eval_progress(self, reason='manual'):
        if self.mode != 'eval':
            return
        self.logger.log(f'>> Persisting evaluation progress due to {reason}', 'yellow')
        self.logger.save_eval_results()

    def _read_control_request(self):
        if not self.control_path:
            return None
        try:
            with open(self.control_path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except Exception:
            return None
        action = payload.get('action')
        if action in {'pause', 'stop'}:
            return action
        return None

    def _mark_control_completed(self, action, finished_scenarios, total_scenarios):
        if not self.control_path:
            return
        payload = {}
        try:
            if os.path.exists(self.control_path):
                with open(self.control_path, 'r', encoding='utf-8') as handle:
                    payload = json.load(handle)
        except Exception:
            payload = {}
        payload.update(
            {
                'completed_action': action,
                'completed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'finished_scenarios': finished_scenarios,
                'total_scenarios': total_scenarios,
            }
        )
        with open(self.control_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def eval(self, data_loader):
        num_finished_scenario = 0
        data_loader.reset_idx_counter()

        if adv_init_state_module is not None:
            adv_init_state_module.GLOBAL_CARLA_ENV = self.env
        while len(data_loader) > 0:
            sampled_scenario_configs, num_sampled_scenario = data_loader.sampler()
            num_finished_scenario += num_sampled_scenario

            score_list = {config.data_id: [] for config in sampled_scenario_configs}
            batch_completed = False

            try:
                static_obs = self.env.get_static_obs(sampled_scenario_configs)
                self.scenario_policy.load_model(sampled_scenario_configs)
                scenario_init_action, _ = self.scenario_policy.get_init_action(static_obs, deterministic=True)

                self.env.obs = None
                self.env.infos = None
                self.env.static_obs = static_obs

                obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)
                self.env.obs = obs
                self.env.infos = infos

                self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos, static_obs=static_obs)

                while not self.env.all_scenario_done():
                    ego_actions = self.agent_policy.get_action(obs, infos, deterministic=True)
                    scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=True)

                    obs, rewards, _, infos = self.env.step(
                        ego_actions=ego_actions,
                        scenario_actions=scenario_actions,
                        scenario_configs=sampled_scenario_configs,
                    )

                    if self.save_video:
                        if self.scenario_category == 'planning':
                            self.logger.add_frame(pygame.surfarray.array3d(self.display).transpose(1, 0, 2))
                        else:
                            self.logger.add_frame({s_i['scenario_id']: ego_actions[n_i]['annotated_image'] for n_i, s_i in enumerate(infos)})

                    reward_idx = 0
                    for s_i in infos:
                        score = rewards[reward_idx] if self.scenario_category == 'planning' else 1 - infos[reward_idx]['iou_loss']
                        score_list[s_i['data_id']].append(score)
                        reward_idx += 1

                batch_completed = True
            finally:
                try:
                    if batch_completed:
                        self.logger.log('>> All scenarios are completed. Clearning up all actors')
                    else:
                        self.logger.log('>> Batch interrupted. Cleaning up spawned actors', 'yellow')
                    if self.env is not None:
                        self.env.clean_up()
                except Exception as cleanup_error:
                    self.logger.log(f'>> Clean up failed: {cleanup_error}', 'red')

            # Allow CARLA 5 seconds to fully tear down streaming sockets and
            # release resources from the previous scenario before starting the
            # next one.  This gives CARLA breathing room between batches.
            if self.logger is not None:
                self.logger.log('>> Waiting 5s for CARLA cleanup between scenario batches...', color='yellow')
            time.sleep(5)

            scenario_names = self._collect_batch_scenario_names(sampled_scenario_configs, num_sampled_scenario)
            batch_metadata = self._build_batch_metadata(sampled_scenario_configs, scenario_names)

            self.logger.log(
                f'[{num_finished_scenario}/{data_loader.num_total_scenario}] Ranking scores for batch scenario:',
                'yellow',
            )
            batch_ranking_scores = {}
            for data_id, scores in score_list.items():
                mean_score = float(np.mean(scores)) if scores else None
                batch_ranking_scores[str(data_id)] = mean_score
                self.logger.log('\t Env id ' + str(data_id) + ': ' + str(mean_score), 'yellow')

            batch_running_results = {}
            for scenario_i in range(num_sampled_scenario):
                batch_running_results.update(self.env.env_list[scenario_i].running_results)

            all_running_results = self.logger.add_eval_results(records=batch_running_results)
            score_function = get_route_scores if self.scenario_category == 'planning' else get_perception_scores
            all_scores = score_function(all_running_results)
            self.logger.add_eval_results(scores=all_scores)
            self.logger.print_eval_results()
            self.logger.save_eval_results()

            batch_summary = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'finished_scenarios': num_finished_scenario,
                'total_scenarios': data_loader.num_total_scenario,
                'data_ids': batch_metadata['data_ids'],
                'scenario_ids': batch_metadata['scenario_ids'],
                'route_ids': batch_metadata['route_ids'],
                'scenario_names': batch_metadata['scenario_names'],
                'batch_ranking_scores': batch_ranking_scores,
                'aggregate_scores': all_scores,
                'video_output': None,
                'video_error': None,
            }

            if self.save_video:
                try:
                    batch_summary['video_output'] = self.logger.save_video_with_metadata(batch_metadata)
                except Exception as video_error:
                    batch_summary['video_error'] = str(video_error)
                    self.logger.log(f'>> Failed to save video: {video_error}', 'red')

            self.logger.save_eval_batch_summary(batch_summary)

            control_action = self._read_control_request()
            if control_action in {'pause', 'stop'}:
                self.logger.log(
                    f'>> Received {control_action} request. Stop after current scenario batch.',
                    'yellow',
                )
                self._mark_control_completed(
                    control_action,
                    num_finished_scenario,
                    data_loader.num_total_scenario,
                )
                self.persist_eval_progress(reason=control_action)
                return control_action

        self.logger.save_eval_results()
        return None

    def run(self):
        config_by_map = scenario_parse(self.scenario_config, self.logger)
        for town in config_by_map.keys():
            self._init_world(town)

            start_time = time.time()
            self._init_renderer()
            end_time = time.time()
            print(f'Renderer initialization time: {end_time - start_time} seconds')

            if self.agent_config['policy_type'] == 'tcp':
                self.logger.log('>> Using TCP Vector Wrapper for environment')
                self.env = VectorWrapperTCP(
                    self.env_params,
                    self.scenario_config,
                    self.world,
                    self.birdeye_render,
                    self.display,
                    self.logger,
                )
            else:
                self.logger.log('>> Using Standard Vector Wrapper for environment')
                self.env = VectorWrapper(
                    self.env_params,
                    self.scenario_config,
                    self.world,
                    self.birdeye_render,
                    self.display,
                    self.logger,
                )

            data_loader = ScenarioDataLoader(config_by_map[town], self.num_scenario, town, self.world)
            if self.mode == 'eval':
                self.agent_policy.load_model()
                self.agent_policy.set_mode('eval')
                self.scenario_policy.set_mode('eval')
                control_action = self.eval(data_loader)
                if control_action in {'pause', 'stop'}:
                    break
            elif self.mode == 'train_agent':
                start_episode = self.check_continue_training(self.agent_policy)
                self.scenario_policy.load_model()
                self.agent_policy.set_mode('train')
                self.scenario_policy.set_mode('eval')
                self.train(data_loader, start_episode)
            elif self.mode == 'train_scenario':
                start_episode = self.check_continue_training(self.scenario_policy)
                self.agent_policy.load_model()
                self.agent_policy.set_mode('eval')
                self.scenario_policy.set_mode('train')
                self.train(data_loader, start_episode)
            else:
                raise NotImplementedError(f'Unsupported mode: {self.mode}.')

    def check_continue_training(self, policy):
        policy.load_model()
        if policy.continue_episode == 0:
            start_episode = 0
            self.logger.log('>> Previous checkpoint not found. Training from scratch.')
        else:
            start_episode = policy.continue_episode
            self.logger.log('>> Continue training from previous checkpoint.')
        return start_episode

    def close(self):
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        pygame.quit()
        if self.env:
            try:
                self.env.clean_up()
            except Exception as cleanup_error:
                self.logger.log(f'>> Final environment cleanup failed: {cleanup_error}', 'red')
