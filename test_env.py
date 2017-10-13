import numpy as np 
import gym
import time
from hierarchical_envs.pb_envs.gym_locomotion_envs import InsectBulletEnv

# env = Walker2DBulletEnv(render=True)
env = gym.make('InsectBulletEnv-v0', render=True)
# env = InsectBulletEnv(render=True)
o = env.reset()
while True:
    env.render(mode='human')
    next_o, r, d, info = env.step(env.action_space.sample())
    time.sleep(0.05)
