#!/usr/bin/env python3

import gym
import gym.spaces
import numpy as np

from PIL import Image
from numpy.linalg import norm


def manhattan(x):
    return sum(x)


norm = manhattan


class StochasticDiscreteOrientation(gym.Env):

    def __init__(self, size_noise=0.05, map_size=50.0, goal_radius=0.05):
        super(StochasticDiscreteOrientation, self).__init__()
        self.metadata = {
            'render.modes':  ['human', 'rgb_array']
        }
        self.size_noise = size_noise
        self.map_size = map_size
        self.actions = [
            np.array([1, 0]),  # right
            np.array([-1, 0]),  # left
            np.array([0, 1]),  # up
            np.array([0, -1]),  # down
        ]
        self.goal_radius = map_size * goal_radius
        self._reset()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=-map_size,
                                                high=map_size,
                                                shape=(4, ))
        self.prev_action = None

    @property
    def _state(self):
        return np.concatenate([self.position, self.goal_position], axis=0)

    def _step(self, action):
        move = self.actions[action] + self.size_noise * np.random.randn(1)
        self.position += move
        np.clip(self.position, -self.map_size, self.map_size, self.position)
        dist_to_goal = norm(self.goal_position - self.position)
        reward = (norm(self.goal_position) - dist_to_goal)
        reward /= norm(self.goal_position) + 1e-6
        reward *= abs(reward)**2
        if not self.prev_action == action:
            reward *= 0.1
        self.prev_action = action
        done = dist_to_goal <= self.goal_radius
        if done:
            reward *= 5.0
        info = {'true_reward': None}
        return self._state, reward, done, info

    def _reset(self):
        self.position = np.zeros((2))
        self.goal_position = np.random.randn(2) * self.map_size * 0.9
        np.clip(self.goal_position, -self.map_size, self.map_size, self.goal_position)
        return self._state

    def _render(self, mode='human', close=False):
        if close:
            return
        map_view = np.zeros((int(self.map_size) * 2,
                             int(self.map_size) * 2,
                             3), dtype=np.uint8)
        map_view += 255
        r = int(self.goal_radius)
        gp_x, gp_y = self.goal_position.astype('int') + int(self.map_size)
        map_view[gp_x - r:gp_x + r, gp_y - r:gp_y + r, 1:] = 0
        p_x, p_y = self.position.astype('int') + int(self.map_size)
        map_view[p_x - r:p_x + r, p_y - r:p_y + r, :] = 0
        if mode == 'rgb_array':
            return map_view
        if mode == 'human':
            img = Image.fromarray(map_view, 'RGB')
            img.show()

    def _close(self):
        pass

    def _seed(self, seed=None):
        np.random.seed(seed)


if __name__ == '__main__':
    asdf = StochasticDiscreteOrientation()
    asdf.render()
    for _ in range(1000):
        asdf.step(0)
    asdf.render()
