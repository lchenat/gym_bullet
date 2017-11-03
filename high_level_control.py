#!/usr/bin/env python3

import numpy as np
import torch as th

import gym
import hierarchical_envs

from drl.policies import DiscretePolicy
from drl.models import Baseline
from drl.algos import Reinforce
from drl.env_converter import EnvWrapper

env = gym.make('DiscreteOrientation-v0')
env = EnvWrapper(env)

model, critic =  Baseline(env.state_size, env.action_size, layer_sizes=(4, 4), discrete=True)
policy = DiscretePolicy(model)
agent = Reinforce(policy=policy, critic=critic, update_frequency=1000000000,
                  critic_weight=1.0,
                  entropy_weight=0.001,
                  grad_clip=0.5)
agent.load_state_dict(th.load('./high_level.pth'))

def get_direction(state):
    action, _ = agent(state)
    return action

if __name__ == '__main__':
    state = env.reset()
    for _ in range(10):
        print(get_direction(state))
        state, _, _, _ = env.step(get_direction(state))
