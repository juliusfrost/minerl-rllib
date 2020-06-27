import os
import gym
import minerl
from ray.tune.registry import register_env
from ray.rllib.env.external_env import ExternalEnv


class MineRL(ExternalEnv):
    def __init__(self, env_config: dict):
        import minerl
        env = gym.make(env_config['name'])
        env = MineRLActionWrapper(MineRLObservationWrapper(env))
        self.env = env
        super().__init__(self.env.action_space, self.env.observation_space, max_concurrent=1)

    def run(self):
        while True:
            episode_id = self.start_episode()
            obs = self.env.reset()
            done = False
            while not done:
                action = self.get_action(episode_id, obs)
                obs, reward, done, info = self.env.step(action)
                self.log_returns(episode_id, reward, info)
            self.end_episode(episode_id, obs)


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
        env_config.update(dict(name=env_spec.name))
        env = MineRL(env_config)
        return env


    register_env(env_spec.name, env_creator)
