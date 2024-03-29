import os
import shutil

import gym
import minerl
import numpy as np
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline import InputReader
from ray.rllib.offline.json_writer import JsonWriter

from minerl_rllib.envs.wrappers import wrap


class MinerRLDataEnv(gym.Env):
    def __init__(self, data_pipeline: minerl.data.DataPipeline):
        super().__init__()
        self.data_pipeline = data_pipeline
        self.env_spec = data_pipeline.spec
        self.observation_space = self.env_spec.observation_space
        self.action_space = self.env_spec.action_space
        self.trajectory_names = data_pipeline.get_trajectory_names()
        self.index = 0
        # used by step() and reset()
        self.trajectory = None
        self.step_index = 0

    def step(self, action):
        prev_obs, action, reward, obs, done = self.trajectory[self.step_index]
        info = {
            "prev_obs": prev_obs,
            "action": action,
        }
        self.step_index += 1
        if self.step_index >= len(self.trajectory):
            if not done:
                print("Encountered end of trajectory when done returned False!")
            done = True
        return obs, reward, done, info

    def reset(self):
        self.trajectory = list(
            self.data_pipeline.load_data(self.trajectory_names[self.index])
        )
        self.index += 1
        self.index %= len(self.trajectory_names)
        self.step_index = 0
        obs, _, _, _, _ = self.trajectory[self.step_index]
        return obs

    def render(self, mode="human"):
        pass


def wrap_env(env: MinerRLDataEnv, env_config):
    return wrap(env, **env_config)


def write_jsons(
    environment,
    data_dir,
    env_config,
    save_path,
    overwrite=False,
    fail_safe=True,
    **kwargs,
):
    data_pipeline = minerl.data.make(environment, data_dir, **kwargs)
    env = MinerRLDataEnv(data_pipeline)
    env = wrap_env(env, env_config)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if len(os.listdir(save_path)) != 0:
        abs_save_path = os.path.abspath(save_path)
        if overwrite:
            print(f"Overwriting! {abs_save_path}")
            shutil.rmtree(abs_save_path)
        else:
            if fail_safe:
                print(f"Json data already exists at {abs_save_path}")
                return
            else:
                raise ValueError(
                    f"Directory {abs_save_path} not empty!"
                    f"Cannot overwrite existing data automatically, please delete old data if unused."
                )

    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(save_path)
    prep = get_preprocessor(env.observation_space)(env.observation_space)

    for eps_id, trajectory_name in enumerate(env.trajectory_names):
        t = 0
        prev_action = None
        prev_reward = 0
        done = False
        try:
            obs = env.reset()
        except TypeError:
            continue
        while not done:
            new_obs, reward, done, info = env.step(env.action_space.sample())
            action = info["action"]
            action = env.reverse_action(action)
            if prev_action is None:
                prev_action = np.zeros_like(action)

            batch_builder.add_values(
                t=t,
                eps_id=eps_id,
                agent_index=0,
                obs=prep.transform(obs),
                actions=action,
                action_prob=1.0,  # put the true action probability here
                rewards=reward,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                dones=done,
                infos={"trajectory_name": trajectory_name},
                new_obs=prep.transform(new_obs),
            )
            obs = new_obs
            prev_action = action
            prev_reward = reward
            t += 1
        writer.write(batch_builder.build_and_reset())


class MineRLReader(InputReader):
    def __init__(
        self,
        env_config,
        environment,
        data_dir,
        num_workers,
        worker_batch_size=4,
        minimum_size_to_dequeue=32,
    ):
        super().__init__()
        self.data_pipeline = minerl.data.make(
            environment,
            data_dir,
            num_workers,
            worker_batch_size,
            minimum_size_to_dequeue,
            force_download=False,
        )
        self.env_config = env_config
        self.env_config.update({"data_dir": data_dir})

    def get_env(self, env):
        return wrap(env, **self.env_config)

    def next(self):
        self.data_pipeline.batch_iter()
