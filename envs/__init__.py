import copy
import os

import minerl
from ray.tune.registry import register_env

from envs.env import MineRLRandomDebugEnv
from envs.wrappers import wrap


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