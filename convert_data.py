import argparse
import os

import gym
import minerl
import numpy as np
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline.json_writer import JsonWriter

from envs import register

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', default=os.getenv('MINERL_DATA_ROOT', 'data'))
parser.add_argument('--save-path', default=None)
parser.add_argument('--env', default=None)


def main():
    args = parser.parse_args()

    if args.save_path is None:
        save_path = os.path.join(args.data_path, 'rllib')
    else:
        save_path = args.save_path

    if args.env is None:
        env_list = []
        for env_spec in minerl.herobraine.envs.obfuscated_envs:
            env_list.append(env_spec.name)
    else:
        env_list = [args.env]

    register()

    for env_name in env_list:
        env = gym.make(env_name)
        env = env.MineRLObservationWrapper(env.MineRLActionWrapper(env))

        batch_builder = SampleBatchBuilder()
        writer = JsonWriter(os.path.join(save_path, env_name))
        prep = get_preprocessor(env.observation_space)(env.observation_space)

        env.close()

        data = minerl.data.make(env_name, data_dir=args.data_path)

        for trajectory_name in data.get_trajectory_names():
            t = 0
            prev_action = None
            prev_reward = 0
            done = False
            obs = None
            info = None
            for obs, action, reward, next_obs, done in data.load_data(trajectory_name):
                obs = (obs['pov'], obs['vector'])
                next_obs = (next_obs['pov'], next_obs['vector'])
                action = action['vector']
                if prev_action is None:
                    prev_action = np.zeros_like(action)

                batch_builder.add_values(
                    t=t,
                    eps_id=trajectory_name,
                    agent_index=0,
                    obs=prep.transform(obs),
                    actions=action,
                    action_prob=1.0,  # put the true action probability here
                    rewards=reward,
                    prev_actions=prev_action,
                    prev_rewards=prev_reward,
                    dones=done,
                    infos=info,
                    new_obs=prep.transform(next_obs))
                prev_action = action
                prev_reward = reward
                t += 1
            writer.write(batch_builder.build_and_reset())


if __name__ == '__main__':
    main()
