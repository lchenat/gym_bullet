#!/usr/bin/env python3

from .pb_envs import *
from gym.envs.registration import registry, register, make, spec

# ------------bullet-------------

register(
    id='DiscreteOrientation-v0',
    entry_point='hierarchical_envs.orientation_env:StochasticDiscreteOrientation',
    timestep_limit=5000,
    reward_threshold=9500.0, )

