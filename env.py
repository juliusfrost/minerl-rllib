import copy
import os
from collections import deque

import gym
import gym.wrappers
import minerl
import numpy as np
from ray.tune.registry import register_env
from ray.tune.utils.util import merge_dicts
from sklearn.neighbors import NearestNeighbors


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
        action = np.reshape(action, (1, 64))
        distances, indices = self.nearest_neighbors.kneighbors(action)
        return int(indices[0].item())


class MineRLRewardPenaltyWrapper(gym.wrappers.TransformReward):
    def __init__(self, env, reward_penalty=0.001):
        super().__init__(env, lambda r: r - reward_penalty)


class MineRLTimeLimitWrapper(gym.wrappers.TimeLimit):
    def __init__(self, env):
        super().__init__(env, env.env_spec.max_episode_steps)


class MineRLActionRepeat(gym.Wrapper):
    def __init__(self, env, action_repeat=1):
        super().__init__(env)
        self.action_repeat = action_repeat

    def step(self, action):
        obs, reward, done, info = None, None, None, None
        for _ in range(self.action_repeat):
            obs, reward, done, info = self.env.step(action)
            if done:
                break
        return obs, reward, done, info


class MineRLObservationStack(gym.Wrapper):
    def __init__(self, env, num_stack, lz4_compress=False):
        super().__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)
        self.tuple = False
        self.tuple_len = 0
        if isinstance(self.observation_space, gym.spaces.Tuple):
            self.tuple = True
            new_spaces = []
            for space in self.observation_space:
                if isinstance(space, gym.spaces.Box):
                    low = np.repeat(space.low[np.newaxis, ...], num_stack, axis=0)
                    high = np.repeat(space.high[np.newaxis, ...], num_stack, axis=0)
                    new_space = gym.spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)
                    new_spaces.append(new_space)
                else:
                    raise NotImplementedError
                self.tuple_len += 1
            self.observation_space = gym.spaces.Tuple(new_spaces)
        elif isinstance(self.observation_space, gym.spaces.Box):
            low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
            high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)
        else:
            raise NotImplementedError

    def _get_observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        if not self.tuple:
            return gym.wrappers.frame_stack.LazyFrames(list(self.frames), self.lz4_compress)
        obs = []
        for i in range(self.tuple_len):
            frames = [f[i] for f in self.frames]
            obs.append(gym.wrappers.frame_stack.LazyFrames(frames, self.lz4_compress))
        return obs

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self._get_observation()


def wrap(env: minerl.env.MineRLEnv, discrete=False, num_actions=32, data_dir=None, num_stack=1, action_repeat=1):
    env = MineRLTimeLimitWrapper(env)
    env = MineRLObservationWrapper(env)
    env = MineRLActionWrapper(env)
    if discrete:
        env = MineRLDiscreteActionWrapper(env, num_actions, data_dir=data_dir)
    if num_stack > 1:
        env = MineRLObservationStack(env, num_stack)
    if action_repeat > 1:
        env = MineRLActionRepeat(env, action_repeat)
    return env


def register(discrete=False, num_actions=32, data_dir=None, **kwargs):
    """
    Must be called to register the MineRL environments for RLlib
    """
    for env_spec in minerl.herobraine.envs.obfuscated_envs:
        env_kwargs = copy.deepcopy(dict(
            observation_space=env_spec.observation_space,
            action_space=env_spec.action_space,
            docstr=env_spec.get_docstring(),
            xml=os.path.join(minerl.herobraine.env_spec.MISSIONS_DIR, env_spec.xml),
            env_spec=env_spec,
        ))
        wrap_kwargs = dict(discrete=discrete, num_actions=num_actions, data_dir=data_dir)

        def env_creator(env_config):
            return wrap(minerl.env.MineRLEnv(**env_kwargs), **merge_dicts(wrap_kwargs, env_config))

        register_env(env_spec.name, env_creator)
