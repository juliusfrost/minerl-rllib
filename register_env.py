import copy
import os

import numpy as np
from sklearn.neighbors import NearestNeighbors
import gym
import gym.wrappers
import minerl
from ray.tune.registry import register_env


class MineRLObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Tuple((self.observation_space['pov'], self.observation_space['vector']))

    def observation(self, observation):
        return observation['pov'], observation['vector']


class MineRLActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = self.action_space['vector']

    def action(self, action):
        return dict(vector=action)

    def reverse_action(self, action):
        return action['vector']


class MineRLDiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, num_actions=32, kmeans_file=None):
        super().__init__(env)
        self.num_actions = num_actions
        if kmeans_file is None:
            data_path = os.environ.get('MINERL_DATA_ROOT', 'data')
            kmeans_file = os.path.join(data_path, f'{num_actions}-means', f'{env.env_spec.name}.npy')
        self.kmeans = np.load(kmeans_file)
        self.action_space = gym.spaces.Discrete(num_actions)
        self.nearest_neighbors = NearestNeighbors(n_neighbors=1).fit(self.kmeans)

    def action(self, action: int):
        return self.kmeans[action]

    def reverse_action(self, action: np.ndarray):
        action = np.reshape(action, (1, 64))
        distances, indices = self.nearest_neighbors.kneighbors(action)
        return int(indices[0].item())


class MineRLRewardPenaltyWrapper(gym.wrappers.TransformReward):
    def __init__(self, env, reward_penalty=0.001):
        super().__init__(env, lambda r: r - reward_penalty)


class MineRLTimeLimitWrapper(gym.wrappers.TimeLimit):
    def __init__(self, env):
        super().__init__(env, env.env_spec.max_episode_steps)


def wrap(env: minerl.env.MineRLEnv, discrete=False, num_actions=32):
    env = MineRLTimeLimitWrapper(env)
    env = MineRLObservationWrapper(env)
    env = MineRLActionWrapper(env)
    if discrete:
        env = MineRLDiscreteActionWrapper(env, num_actions)
    return env


for env_spec in minerl.herobraine.envs.obfuscated_envs:
    kwargs = dict(
        observation_space=env_spec.observation_space,
        action_space=env_spec.action_space,
        docstr=env_spec.get_docstring(),
        xml=os.path.join(minerl.herobraine.env_spec.MISSIONS_DIR, env_spec.xml),
        env_spec=env_spec,
    )
    env_kwargs = copy.deepcopy(kwargs)

    register_env(env_spec.name, lambda env_config: wrap(minerl.env.MineRLEnv(**env_kwargs)))
