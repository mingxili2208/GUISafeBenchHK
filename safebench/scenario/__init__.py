"""
该脚本主要用于定义场景生成策略,将场景生成策略和类型进行映射
"""

# collect policy models from scenarios
from safebench.scenario.scenario_policy.dummy_policy import DummyPolicy
from safebench.scenario.scenario_policy.reinforce_continuous import REINFORCE
from safebench.scenario.scenario_policy.normalizing_flow_policy import NormalizingFlow
from safebench.scenario.scenario_policy.hardcode_policy import HardCodePolicy
from safebench.scenario.scenario_policy.rl.sac import SAC

# 这里列出了可用的场景生成策略,例如知识的、、硬编码、随即、强化学习、生成式模型等
SCENARIO_POLICY_LIST = {
    'standard': DummyPolicy,
    'ordinary': DummyPolicy,
    'advsim': HardCodePolicy,
    'advtraj': HardCodePolicy,
    'human': HardCodePolicy,
    'random': HardCodePolicy,
    'lc': REINFORCE,
    'nf': NormalizingFlow,
    'sac': SAC,
}
