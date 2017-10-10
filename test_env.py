import numpy as np 
import gym
from pybullet_my_envs.gym_locomotion_envs import SwimmerBulletEnv, Walker2DBulletEnv, HumanoidBulletEnv
import time

# env = Walker2DBulletEnv(render=True)
env = HumanoidBulletEnv(render=True)
o = env.reset()
while True:
        next_o, r, d, info = env.step(env.action_space.sample())
        time.sleep(0.05)
