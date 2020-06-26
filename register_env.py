import os
import gym
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
    
    
for env_spec in minerl.herobraine.envs.ENVS:
    def env_creator(env_config):
        kwargs=dict(
            observation_space=env_spec.observation_space,
            action_space=env_spec.observation_space,
            docstr=env_spec.get_docstring(),
            xml=os.path.join(minerl.herobraine.env_spec.MISSIONS_DIR, env_spec.xml),
            env_spec=env_spec,
        )
        env = minerl.env.MineRLEnv(**kwargs)
        env = MineRLActionWrapper(MineRLObservationWrapper(env))
        return env
    register_env(env_spec.name, env_creator)