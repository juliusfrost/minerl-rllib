import os
import gym
import minerl
from ray.tune.registry import register_env
from ray.rllib.env.external_env import ExternalEnv


class MineRL(minerl.env.MineRLEnv, ExternalEnv):
    def __init__(self, **kwargs):
        minerl.env.MineRLEnv.__init__(self, **kwargs)
        ExternalEnv.__init__(self, self.action_space, self.observation_space, max_concurrent=1)

    def run(self):
        while True:
            episode_id = self.start_episode()
            obs = self.env.reset()
            done = False
            while not done:
                action = self.get_action(episode_id, obs)
                obs, reward, done, info = self.step(action)
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
        kwargs = dict(
            observation_space=env_spec.observation_space,
            action_space=env_spec.action_space,
            docstr=env_spec.get_docstring(),
            xml=os.path.join(minerl.herobraine.env_spec.MISSIONS_DIR, env_spec.xml),
            env_spec=env_spec,
        )
        env = MineRL(**kwargs)
        env = MineRLActionWrapper(MineRLObservationWrapper(env))
        return env


    register_env(env_spec.name, env_creator)
