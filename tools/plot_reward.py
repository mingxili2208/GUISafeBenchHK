"""
绘制强化学习训练过程中的episode和reward的关系图
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

# 打开存储评估结果的文件
with open('../log/exp/exp_ppo_standard_seed_0/training_results/results.pkl', 'rb') as f:
    data = pkl.load(f)

# 提取episode和reward
episode = data['episode']
reward = data['episode_reward']

# 绘制episode和reward的关系图
plt.plot(episode, reward)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.grid()
plt.xlim([0, 100])
plt.tight_layout()
plt.savefig('reward.png', dpi=300)

