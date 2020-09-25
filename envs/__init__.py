import copy
import os

import minerl
from ray.tune.registry import register_env

from envs.env import LazyMineRLEnv, MineRLRandomDebugEnv
from envs.wrappers import wrap


def register():
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

        def env_creator(env_config):
            return wrap(LazyMineRLEnv(**env_kwargs), **env_config)

        register_env(env_spec.name, env_creator)

    def env_creator(env_config):
        return wrap(MineRLRandomDebugEnv(), **env_config)

    register_env('MineRLRandomDebug-v0', env_creator)
