from envs.env import register_minerl_envs
from envs.input import register_minerl_input


def register():
    """
    Must be called to register the MineRL environments for RLlib
    """
    register_minerl_envs()
    register_minerl_input()
