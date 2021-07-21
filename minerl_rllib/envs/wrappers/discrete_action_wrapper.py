import os

import gym
import numpy as np
from sklearn.neighbors import NearestNeighbors


class MineRLDiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, num_actions=32, data_dir=None):
        super().__init__(env)
        self.num_actions = num_actions
        if data_dir is None:
            data_dir = os.environ.get('MINERL_DATA_ROOT', 'data')
        kmeans_file = os.path.join(data_dir, f'{num_actions}-means', f'{env.env_spec.name}.npy')
        self.kmeans = np.load(kmeans_file)
        self.action_space = gym.spaces.Discrete(num_actions)
        self.nearest_neighbors = NearestNeighbors(n_neighbors=1).fit(self.kmeans)

    def action(self, action: int):
        return self.kmeans[action]

    def reverse_action(self, action: np.ndarray):
        action = self.env.reverse_action(action)
        action = np.reshape(action, (1, 64))
        distances, indices = self.nearest_neighbors.kneighbors(action)
        return int(indices[0].item())
