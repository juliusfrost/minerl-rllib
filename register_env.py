import copy
import os

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


class MineRLRewardPenaltyWrapper(gym.wrappers.TransformReward):
    def __init__(self, env, reward_penalty=0.001):
        super().__init__(env, lambda r: r - reward_penalty)


class MineRLTimeLimitWrapper(gym.wrappers.TimeLimit):
    def __init__(self, env):
        super().__init__(env, env.env_spec.max_episode_steps)


def wrap(env):
    env = MineRLTimeLimitWrapper(env)
    env = MineRLObservationWrapper(env)
    env = MineRLActionWrapper(env)
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
