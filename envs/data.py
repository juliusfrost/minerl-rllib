import gym
import minerl
from ray.rllib.offline import InputReader

from envs.wrappers import wrap


class MinerRLDataEnv(gym.Env):
    def __init__(self, data_pipeline: minerl.data.DataPipeline):
        super().__init__()
        self.data_pipeline = data_pipeline
        self.env_spec = data_pipeline.spec
        self.observation_space = self.env_spec.observation_space
        self.action_space = self.env_spec.action_space
        self.trajectory_names = data_pipeline.get_trajectory_names()
        self.index = 0
        self.iterator = None

    def step(self, action):
        obs, action, reward, next_obs, done = self.trajectory[self.step_index]
        self.step_index += 1
        pass

    def reset(self):
        self.trajectory = list(self.data_pipeline.load_data(self.trajectory_names[0]))
        self.index += 1
        self.index %= len(self.trajectory_names)
        self.step_index = 0
        obs, _, _, _, _ = self.trajectory[self.step_index]
        return obs

    def render(self, mode='human'):
        pass


class MineRLReader(InputReader):
    def __init__(self, env_config, environment, data_dir, num_workers, worker_batch_size=4, minimum_size_to_dequeue=32):
        super().__init__()
        self.data_pipeline = minerl.data.make(environment, data_dir, num_workers, worker_batch_size,
                                              minimum_size_to_dequeue, force_download=False)
        self.env_config = env_config
        self.env_config.update({'data_dir': data_dir})

    def get_env(self, env):
        return wrap(env, **self.env_config)

    def next(self):
        self.data_pipeline.batch_iter()
