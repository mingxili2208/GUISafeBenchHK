"""
将被测算法的类名和类对象一一对应起来，方便后续的调用
"""

# for planning scenario
from safebench.agent.dummy import DummyAgent
from safebench.agent.rl.sac import SAC
from safebench.agent.rl.ddpg import DDPG
from safebench.agent.rl.ppo import PPO
from safebench.agent.rl.td3 import TD3
from safebench.agent.basic import CarlaBasicAgent
from safebench.agent.behavior import CarlaBehaviorAgent
try:
    from safebench.agent.tcp import TCPAgent
    _tcp_import_error = None
except ImportError as exc:
    _tcp_import_error = exc

    class TCPAgent:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TCPAgent is unavailable because its dependencies could not be imported. "
                "Use a non-tcp agent config such as behavior/basic, or fix the TCP package setup."
            ) from _tcp_import_error


# 注意这里列出了所有的被测算法类型,包括carla自带AD,感知（YOLO,faster_rcnn）和规划（SAC,DDPG,PPO,TD3）等
AGENT_POLICY_LIST = {
    'dummy': DummyAgent,
    'basic': CarlaBasicAgent,
    'behavior': CarlaBehaviorAgent,
    'sac': SAC,
    'ddpg': DDPG,
    'ppo': PPO,
    'td3': TD3,
    'tcp': TCPAgent,
}
