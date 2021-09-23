import collections
import copy
import os

import gym
from minerl.data.data_pipeline import DataPipeline, tree_slice


def patch_data_pipeline():
    # copied from the DataPipeline class and removed tqdm
    def load_data(
        self,
        stream_name: str,
        skip_interval=0,
        include_metadata=False,
        include_monitor_data=False,
    ):
        """Iterates over an individual trajectory named stream_name.

        Args:
            stream_name (str): The stream name desired to be iterated through.
            skip_interval (int, optional): How many sices should be skipped.. Defaults to 0.
            include_metadata (bool, optional): Whether or not meta data about the loaded trajectory should be included.. Defaults to False.
            include_monitor_data (bool, optional): Whether to include all of the monitor data from the environment. Defaults to False.
        Yields:
            A tuple of (state, player_action, reward_from_action, next_state, is_next_state_terminal).
            These are tuples are yielded in order of the episode.
        """
        if "/" in stream_name:
            file_dir = stream_name
        else:
            file_dir = os.path.join(self.data_dir, stream_name)

        if DataPipeline._is_blacklisted(stream_name):
            raise RuntimeError(
                "This stream is corrupted (and will be removed in the next version of the data!)"
            )

        seq = DataPipeline._load_data_pyfunc(
            file_dir,
            -1,
            None,
            self.environment,
            skip_interval=skip_interval,
            include_metadata=include_metadata,
        )

        observation_seq, action_seq, reward_seq, next_observation_seq, done_seq = seq[
            :5
        ]
        remainder = iter(seq[5:])

        monitor_seq = next(remainder) if include_monitor_data else None
        meta = next(remainder) if include_monitor_data else None

        # make a copty
        gym_spec = gym.envs.registration.spec(self.environment)
        target_space = copy.deepcopy(self.observation_space)

        x = list(target_space.spaces.items())
        target_space.spaces = collections.OrderedDict(
            sorted(x, key=lambda x: x[0] if x[0] != "pov" else "z")
        )

        # Now we just need to slice the dict.
        for idx in range(len(reward_seq)):
            # Wrap in dict
            action_dict = tree_slice(action_seq, idx)
            observation_dict = tree_slice(observation_seq, idx)
            next_observation_dict = tree_slice(next_observation_seq, idx)

            yield_list = [
                observation_dict,
                action_dict,
                reward_seq[idx],
                next_observation_dict,
                done_seq[idx],
            ]
            yield yield_list + (
                ([tree_slice(monitor_seq, idx)] if include_monitor_data else [])
                + ([meta] if include_metadata else [])
            )

    DataPipeline.load_data = load_data
