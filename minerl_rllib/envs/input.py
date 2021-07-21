import os

import gym
import numpy as np
from ray.rllib.models.preprocessors import get_preprocessor

from ray.rllib.offline import IOContext
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import SampleBatchType
from ray.tune.registry import register_input

from minerl_rllib.envs.data import MinerRLDataEnv, wrap_env


class MineRLInputReader(InputReader):
    def __init__(self, ioctx: IOContext = None):
        super().__init__()
        print("Input reader initialization success!")
        import minerl

        input_config = ioctx.input_config
        env_name = ioctx.config.get("env")
        env_config = ioctx.config.get("env_config", {})
        self.data = minerl.data.make(
            env_name,
            data_dir=os.getenv(
                "MINERL_DATA_ROOT", input_config.get("data_dir", "data")
            ),
            num_workers=input_config.get("num_workers", 4),
            worker_batch_size=input_config.get("worker_batch_size", 32),
            minimum_size_to_dequeue=input_config.get("minimum_size_to_dequeue", 32),
            force_download=input_config.get("force_download", False),
        )
        batch_size = input_config.get("batch_size", 1)
        seq_len = input_config.get("seq_len", 32)
        num_epochs = input_config.get("num_epochs", -1)
        preload_buffer_size = input_config.get("preload_buffer_size", 2)
        seed = input_config.get("seed", None)
        include_metadata = input_config.get("include_metadata", False)
        self.generator = self.data.batch_iter(
            batch_size,
            seq_len,
            num_epochs=num_epochs,
            preload_buffer_size=preload_buffer_size,
            seed=seed,
            include_metadata=include_metadata,
        )
        env = MinerRLDataEnv(self.data)
        env = wrap_env(env, env_config)
        print(env.observation_space)
        self.prep = get_preprocessor(env.observation_space)(env.observation_space)

        env_ptr = env
        self.obs_fns = []
        self.action_fns = []
        self.reverse_action_fns = []
        self.reward_fns = []
        while hasattr(env_ptr, "env"):
            if isinstance(env_ptr, gym.ObservationWrapper):
                self.obs_fns.append(env_ptr.observation)
            if isinstance(env_ptr, gym.ActionWrapper):
                self.action_fns.append(env_ptr.action)
                self.reverse_action_fns.append(env_ptr.reverse_action)
            if isinstance(env_ptr, gym.RewardWrapper):
                self.reward_fns.append(env_ptr.reward)
            env_ptr = env_ptr.env

    def process_obs(self, obs):
        for i in range(len(self.obs_fns) - 1, -1, -1):
            f = self.obs_fns[i]
            obs = f(obs)
        return self.prep.transform(obs)

    def process_action(self, action):
        for i in range(len(self.action_fns) - 1, -1, -1):
            f = self.action_fns[i]
            action = f(action)
        return action

    def process_reverse_action(self, action):
        for i in range(len(self.reverse_action_fns)):
            f = self.reverse_action_fns[i]
            action = f(action)
        return action

    def process_reward(self, reward):
        for i in range(len(self.reward_fns) - 1, -1, -1):
            f = self.reward_fns[i]
            reward = f(reward)
        return reward

    def next(self) -> SampleBatchType:
        obs, action, reward, next_obs, done = self.process_batch(next(self.generator))
        d = {
            SampleBatch.OBS: obs,
            SampleBatch.ACTIONS: action,
            SampleBatch.NEXT_OBS: next_obs,
            SampleBatch.REWARDS: reward,
            SampleBatch.DONES: done,
        }
        # recursive_print(d)
        return SampleBatch(d)

    def process_batch(self, batch):
        obs, action, reward, next_obs, done = batch
        batch_n, batch_t = obs["vector"].shape[0], obs["vector"].shape[1]
        action = [
            [
                self.process_reverse_action({"vector": action["vector"][i][t]})
                for i in range(batch_n)
            ]
            for t in range(batch_t)
        ]
        reward = [
            [self.process_reward(reward[i][t]) for i in range(batch_n)]
            for t in range(batch_t)
        ]
        obs = [
            [
                self.process_obs(
                    {"pov": obs["pov"][i][t], "vector": obs["vector"][i][t]}
                )
                for i in range(batch_n)
            ]
            for t in range(batch_t)
        ]
        next_obs = [
            [
                self.process_obs(
                    {"pov": next_obs["pov"][i][t], "vector": next_obs["vector"][i][t]}
                )
                for i in range(batch_n)
            ]
            for t in range(batch_t)
        ]
        # print(len(obs), len(obs[0]), obs[0][0].shape)
        return (
            np.array(obs).squeeze(1),
            np.array(action).squeeze(1),
            np.array(reward).squeeze(1),
            np.array(next_obs).squeeze(1),
            np.transpose(done).squeeze(1),
        )


def minerl_input_creator(ioctx: IOContext = None):
    return MineRLInputReader(ioctx=ioctx)


def recursive_print(d):
    if isinstance(d, dict):
        print("{")
        for k, v in d.items():
            print(f"key={k}")
            recursive_print(v)
        print("}")
    elif isinstance(d, list) or isinstance(d, tuple):
        print("[")
        for v in d:
            recursive_print(v)
        print("]")
    elif isinstance(d, np.ndarray):
        print(d.shape)
    else:
        print(d)


def register_minerl_input():
    register_input("minerl", minerl_input_creator)
