import copy
import os

import filelock
import gym
import minerl
from ray.tune.registry import register_env

from envs.env import MineRLRandomDebugEnv
from envs.wrappers import wrap


class LazyMineRLEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._env = None
        self.observation_space = kwargs.get('observation_space')
        self.action_space = kwargs.get('action_space')
        self.env_spec = kwargs.get('env_spec')
        super().__init__()

    def reset(self, **kwargs):
        if self._env is None:
            with filelock.FileLock('minerl_env.lock'):
                self._env = minerl.env.MineRLEnv(*self._args, **self._kwargs)
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def render(self, mode='human'):
        return self._env.render(mode)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self._env, name)


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
            return wrap(LazyMineRLEnv(**env_kwargs), **wrap_kwargs)

        register_env(env_spec.name, env_creator)

    def env_creator(env_config):
        wrap_kwargs.update(env_config)
        return wrap(MineRLRandomDebugEnv(), **wrap_kwargs)

    register_env('MineRLRandomDebug-v0', env_creator)
