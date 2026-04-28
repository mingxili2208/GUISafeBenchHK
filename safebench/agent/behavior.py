import numpy as np
from safebench.agent.base_policy import BasePolicy
from safebench.carla_agents.navigation.behavior_agent import BehaviorAgent


class CarlaBehaviorAgent(BasePolicy):
    name = 'behavior'
    type = 'unlearnable'

    """ This is just an example for testing, which always goes straight. """
    def __init__(self, agent_config, logger):
        self.logger = logger
        self.num_scenario = agent_config['num_scenario']
        self.ego_action_dim = agent_config['ego_action_dim']
        self.model_path = agent_config['model_path']
        self.mode = 'train'
        self.continue_episode = 0
        self.route = None
        self.controller_list = []
        behavior_list = ["cautious", "normal", "aggressive"]
        self.behavior = behavior_list[1]

    def set_ego_and_route(self, ego_vehicles, info, static_obs=None):
        self.ego_vehicles = ego_vehicles
        self.controller_list = []
        for e_i in range(len(ego_vehicles)):
            controller = BehaviorAgent(self.ego_vehicles[e_i], behavior=self.behavior, target_route_speed=static_obs[e_i]['target_speed'])
            dest_waypoint = info[e_i]['route_waypoints'][-1]
            location = dest_waypoint.transform.location
            controller.set_destination(location) # 为每个控制器设置路线
            self.controller_list.append(controller)

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, obs, infos, deterministic=False):
        actions = []
        for e_idx, e_i in enumerate(infos):
            # select the controller that matches the scenario_id
            try:
                control = self.controller_list[e_idx].run_step()
                # print(f"success to get control for ego vehicle {e_i['data_id']}")  # debug
            except Exception as e:
                print(e)
                control = self.controller_list[0].run_step()
            throttle = control.throttle
            steer = control.steer
            actions.append([throttle, steer]) 
        actions = np.array(actions, dtype=np.float32)
        return actions

    def load_model(self):
        pass

    def save_model(self):
        pass
