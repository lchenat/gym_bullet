import numpy as np 
import gym
from pybullet_my_envs.gym_locomotion_envs import SwimmerBulletEnv

env = SwimmerBulletEnv(render=True)
o = env.reset()
while True:
	next_o, r, d, info = env.step(env.action_space.sample())
	if d:
		break
