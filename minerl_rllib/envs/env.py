import os

import filelock
import gym
import gym.wrappers
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.envs import (
    BASIC_ENV_SPECS,
    COMPETITION_ENV_SPECS,
    BASALT_COMPETITION_ENV_SPECS,
)
from minerl.herobraine.envs import MINERL_OBTAIN_DIAMOND_OBF_V0 as DEBUG_ENV_SPEC
from minerl_wrappers import wrap
from ray.tune.registry import register_env

from minerl_rllib.generate_kmeans import main as generate_kmeans


class LazyMineRLEnv(gym.Env):
    def __init__(self, env_spec, **kwargs):
        self._kwargs = kwargs
        self.env_spec: EnvSpec = env_spec
        self._env = None
        super().__init__()

    def init_env(self):
        with filelock.FileLock("minerl_env.lock"):
            self._env = self.env_spec.make(**self._kwargs)

    def reset(self, **kwargs):
        if self._env is None:
            self.init_env()
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def render(self, mode="human"):
        return self._env.render(mode)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        if self._env is None:
            self.init_env()
        return getattr(self._env, name)


class MineRLRandomDebugEnv(gym.Env):
    def __init__(self):
        super(MineRLRandomDebugEnv, self).__init__()
        pov_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3))
        vector_space = gym.spaces.Box(
            low=-1.2000000476837158, high=1.2000000476837158, shape=(64,)
        )
        self.observation_space = gym.spaces.Dict(
            dict(pov=pov_space, vector=vector_space)
        )
        action_space = gym.spaces.Box(
            low=-1.0499999523162842, high=1.0499999523162842, shape=(64,)
        )
        self.action_space = gym.spaces.Dict(dict(vector=action_space))
        self.done = False
        self.t = 0
        self.env_spec = DEBUG_ENV_SPEC

    def _obs(self):
        return self.observation_space.sample()

    def step(self, action):
        obs = self._obs()
        reward = 0.0
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

    def render(self, mode="human"):
        pass


def wrap_env(env, env_config, env_name):
    if env_config.get("kmeans", False):
        # generate kmeans actions or load from data path if exists
        # sets the minerl-wrappers config to use these action means
        kmeans_config = env_config.get("kmeans_config", {})
        args = [
            "--env",
            env_name,
            "--num-actions",
            str(kmeans_config.get("num_actions", 32)),
            "--data-dir",
            kmeans_config.get("data_dir", os.getenv("MINERL_DATA_ROOT", "data"))
        ]
        with filelock.FileLock("minerl_env.lock"):
            means = generate_kmeans(args)
        if env_config.get("diamond", False):
            if "diamond_config" in env_config:
                env_config["diamond_config"]["action_choices"] = means
            else:
                env_config["diamond_config"] = {"action_choices": means}
        if env_config.get("pfrl_2020", False):
            if "pfrl_2020_config" in env_config:
                env_config["pfrl_2020_config"]["action_choices"] = means
            else:
                env_config["pfrl_2020_config"] = {"action_choices": means}
    env = wrap(env, **env_config)
    return env


def register_minerl_envs():
    for env_spec in (
        BASIC_ENV_SPECS + COMPETITION_ENV_SPECS + BASALT_COMPETITION_ENV_SPECS
    ):

        def env_creator(env_config):
            env = LazyMineRLEnv(env_spec)
            env = wrap_env(env, env_config, env_spec.name)
            return env

        register_env(env_spec.name, env_creator)

    def env_creator(env_config):
        return wrap(MineRLRandomDebugEnv(), **env_config)

    register_env("MineRLRandomDebug-v0", env_creator)
