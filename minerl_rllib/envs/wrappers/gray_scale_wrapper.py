import gym
import numpy as np


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
