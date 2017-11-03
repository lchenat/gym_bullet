from hierarchical_envs.pb_envs.gym_locomotion_envs import InsectBulletEnv
import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
# from rllab.sampler.utils import rollout
#from pybullet_my_envs.gym_locomotion_envs import Ant6BulletEnv, AntBulletEnv, SwimmerBulletEnv
from hierarchical_envs.pb_envs.gym_locomotion_envs import InsectBulletEnv, AntBulletEnv, SwimmerBulletEnv
from rllab.envs.gym_wrapper import GymEnv

import numpy as np
from rllab.misc import tensor_utils
import time

north_x = 0
north_y = 1e3

def simple_high(states):
	x, y, tx, ty = states
	if x < tx - 50:
		return 0
	if x > tx + 50:
		return np.pi
	if y < ty:
		return np.pi / 2
	return -np.pi / 2


def rollout(env, pi_low, pi_high, tx=700, ty=0, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    x, y, z = env.robot.body_xyz
    r, p, yaw = env.robot.body_rpy
    target_theta = np.arctan2(
        ty - y,
        tx - x)
    angle_to_target = target_theta - yaw
 
    print('direction: ', o[0], o[1])
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a_high = pi_high([x, y, tx, ty]) 
        feed_o[0] = np.cos(a_high) # get direction
        feed_o[1] = np.sin(a_high)
        a, agent_info = agent.get_action(feed_o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('low_level', type=str,
                        help='path to lower_level policy')
    parser.add_argument('--max_path_length', type=int, default=500,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    args = parser.parse_args()

    data = joblib.load(args.file)
    pi_low = data['policy']
    pi_high = simple_high
	env = GymEnv(InsectBulletEnv(render=True, d=0.75, r_init=None, d_angle=True))
    while True:
        path = rollout(env, pi_low, pi_high, max_path_length=args.max_path_length,
                       animated=True, speedup=args.speedup)
        if not query_yes_no('Continue simulation?'):
            break
