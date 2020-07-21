import copy
import os
from collections import deque

import gym
import gym.wrappers
import minerl
import numpy as np
from ray.tune.registry import register_env
from sklearn.neighbors import NearestNeighbors


class MineRLRandomDebugEnv(gym.Env):
    def __init__(self):
        super(MineRLRandomDebugEnv, self).__init__()
        pov_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3))
        vector_space = gym.spaces.Box(low=-1.2000000476837158, high=1.2000000476837158, shape=(64,))
        self.observation_space = gym.spaces.Dict(dict(pov=pov_space, vector=vector_space))
        action_space = gym.spaces.Box(low=-1.0499999523162842, high=1.0499999523162842, shape=(64,))
        self.action_space = gym.spaces.Dict(dict(vector=action_space))
        self.done = False
        self.t = 0
        self.name = 'MineRLDebug-v0'

    def _obs(self):
        return self.observation_space.sample()

    def step(self, action):
        obs = self._obs()
        reward = 0
        if self.t < 100:
            self.done = False
        else:
            self.done = True
        info = {}
        self.t += 1
        return obs, reward, self.done, info

    def reset(self):
        self.done = False
        self.t = 0
        return self._obs()

    def render(self, mode='human'):
        pass


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
        total_reward = 0
        for _ in range(self.action_repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


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
                    new_space = gym.spaces.Box(low=low, high=high, dtype=space.dtype)
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


class MineRLGrayScale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        pov_space, vector_space = self.observation_space
        assert isinstance(pov_space, gym.spaces.Box)
        low = np.min(pov_space.low, axis=2, keepdims=True)
        high = np.max(pov_space.high, axis=2, keepdims=True)
        pov_space = gym.spaces.Box(low, high, dtype=pov_space.dtype)
        self.observation_space = gym.spaces.Tuple((pov_space, vector_space))

    def observation(self, observation):
        pov, vector = observation
        gray_scaled_pov = np.mean(pov, axis=2, keepdims=True)
        return gray_scaled_pov, vector


class MineRLDeterministic(gym.Wrapper):
    def __init__(self, env, seed: int):
        super().__init__(env)
        self._set_seed = seed

    def reset(self, **kwargs):
        self.seed(self._set_seed)
        return self.env.reset()


def wrap(env, discrete=False, num_actions=32, data_dir=None, num_stack=1, action_repeat=1,
         gray_scale=False, seed=None):
    if isinstance(env, minerl.env.MineRLEnv):
        env = MineRLTimeLimitWrapper(env)
    env = MineRLObservationWrapper(env)
    env = MineRLActionWrapper(env)
    if discrete:
        env = MineRLDiscreteActionWrapper(env, num_actions, data_dir=data_dir)
    if gray_scale:
        env = MineRLGrayScale(env)
    if num_stack > 1:
        env = MineRLObservationStack(env, num_stack)
    if action_repeat > 1:
        env = MineRLActionRepeat(env, action_repeat)
    if seed is not None:
        env = MineRLDeterministic(env, seed)
    return env


def register(discrete=False, num_actions=32, data_dir=None, **kwargs):
    """
    Must be called to register the MineRL environments for RLlib
    """
    wrap_kwargs = dict(discrete=discrete, num_actions=num_actions, data_dir=data_dir)
    wrap_kwargs.update(kwargs)

    for env_spec in minerl.herobraine.envs.obfuscated_envs:
        env_kwargs = copy.deepcopy(dict(
            observation_space=env_spec.observation_space,
            action_space=env_spec.action_space,
            docstr=env_spec.get_docstring(),
            xml=os.path.join(minerl.herobraine.env_spec.MISSIONS_DIR, env_spec.xml),
            env_spec=env_spec,
        ))

        def env_creator(env_config):
            wrap_kwargs.update(env_config)
            return wrap(minerl.env.MineRLEnv(**env_kwargs), **wrap_kwargs)

        register_env(env_spec.name, env_creator)

    def env_creator(env_config):
        wrap_kwargs.update(env_config)
        return wrap(MineRLRandomDebugEnv(), **wrap_kwargs)

    register_env('MineRLRandomDebug-v0', env_creator)
