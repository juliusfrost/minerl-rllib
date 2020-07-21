from collections import deque

import gym
import gym.wrappers
import numpy as np


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
