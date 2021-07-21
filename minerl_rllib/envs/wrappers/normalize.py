import gym
import numpy as np
from gym.wrappers import TransformReward, TransformObservation


def normalize(a, prev_low, prev_high, new_low, new_high):
    return (a - prev_low) / (prev_high - prev_low) * (new_high - new_low) + new_low


class MineRLNormalizeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, pov_low=0.0, pov_high=1.0, vec_low=-1.0, vec_high=1.0):
        super().__init__(env)
        self._old_pov_space: gym.spaces.Box = self.env.observation_space.spaces[0]
        self._old_vec_space: gym.spaces.Box = self.env.observation_space.spaces[1]
        self._pov_space = gym.spaces.Box(
            pov_low, pov_high, self._old_pov_space.low.shape, np.float32
        )
        self._vec_space = gym.spaces.Box(
            vec_low, vec_high, self._old_vec_space.low.shape, np.float32
        )
        self.observation_space = gym.spaces.Tuple((self._pov_space, self._vec_space))

    def observation(self, observation):
        pov, vec = observation
        pov = normalize(
            pov,
            self._old_pov_space.low,
            self._old_pov_space.high,
            self._pov_space.low,
            self._pov_space.high,
        )
        vec = normalize(
            vec,
            self._old_vec_space.low,
            self._old_vec_space.high,
            self._vec_space.low,
            self._vec_space.high,
        )
        return pov, vec


class MineRLNormalizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self._old_vec_space: gym.spaces.Box = self.env.action_space
        self._vec_space = gym.spaces.Box(
            low, high, self._old_vec_space.low.shape, np.float32
        )
        self.action_space = self._vec_space

    def action(self, action):
        return normalize(
            action,
            self._old_vec_space.low,
            self._old_vec_space.high,
            self._vec_space.low,
            self._vec_space.high,
        )

    def reverse_action(self, action):
        action = self.env.reverse_action(action)
        return normalize(
            action,
            self._vec_space.low,
            self._vec_space.high,
            self._old_vec_space.low,
            self._old_vec_space.high,
        )


class MineRLRewardScaleWrapper(TransformReward):
    def __init__(self, env, reward_scale=1.0):
        def f(reward):
            return reward * reward_scale

        super().__init__(env, f)
