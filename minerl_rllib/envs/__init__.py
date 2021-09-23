from minerl_rllib.envs.env import register_minerl_envs
from minerl_rllib.envs.input import register_minerl_input


def register():
    """
    Must be called to register the MineRL environments for RLlib
    """
    register_minerl_envs()
    register_minerl_input()
