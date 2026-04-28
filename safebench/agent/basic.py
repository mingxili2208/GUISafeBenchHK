import numpy as np
from safebench.agent.base_policy import BasePolicy
from safebench.carla_agents.navigation.basic_agent import BasicAgent


class CarlaBasicAgent(BasePolicy):
    name = 'basic'
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

        # parameters for PID controller
        self.target_speed = agent_config['target_speed']
        dt = agent_config['dt']
        lateral_KP = agent_config['lateral_KP']
        lateral_KI = agent_config['lateral_KI']
        lateral_KD = agent_config['lateral_KD']
        longitudinal_KP = agent_config['longitudinal_KP']
        longitudinal_KI = agent_config['longitudinal_KI']
        longitudinal_KD = agent_config['longitudinal_KD']
        max_steering = agent_config['max_steering']
        max_throttle = agent_config['max_throttle']

        self.opt_dict = {
            'lateral_control_dict': {'K_P': lateral_KP, 'K_I': lateral_KI, 'K_D': lateral_KD, 'dt': dt},
            'longitudinal_control_dict': {'K_P': longitudinal_KP, 'K_I': longitudinal_KI, 'K_D': longitudinal_KD, 'dt': dt},
            'max_steering': max_steering,
            'max_throttle': max_throttle,
        }

    def set_ego_and_route(self, ego_vehicles, info):
        self.ego_vehicles = ego_vehicles
        self.controller_list = []
        for e_i in range(len(ego_vehicles)):
            controller = BasicAgent(self.ego_vehicles[e_i], target_speed=self.target_speed, opt_dict=self.opt_dict)
            dest_waypoint = info[e_i]['route_waypoints'][-1]
            location = dest_waypoint.transform.location
            controller.set_destination(location) # set route for each controller
            self.controller_list.append(controller)

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, obs, infos, deterministic=False):
        actions = []
        for e_i in infos:
            # select the controller that matches the scenario_id
            control = self.controller_list[e_i['scenario_id']].run_step()
            throttle = control.throttle
            steer = control.steer
            actions.append([throttle, steer]) 
        actions = np.array(actions, dtype=np.float32)
        return actions

    def load_model(self):
        pass

    def save_model(self):
        pass
