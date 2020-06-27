import os
import argparse
import yaml
import pprint
import ray
from ray import tune

import register_env

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='MineRLNavigateDenseVectorObf-v0', help='environment name')
parser.add_argument('--config', type=str, default=None,
                    help='Config file to load algorithm from. Defaults to algorithm argument choice.')
parser.add_argument('--algorithm', type=str, default='sac',
                    help='Choose algorithm from those implemented. Used if config argument not set.')
parser.add_argument('--mode', choices=['online', 'offline', 'mixed'], default='online')
parser.add_argument('--data-path', default=None,
                    help='Must be set for offline and mixed training. Must be rllib json format.')
parser.add_argument('--mixing-ratio', default=0.5,
                    help='How much to sample from data over the environment')
parser.add_argument('--framework', choices=['torch', 'tf', 'tfe'], default='torch')


def main():
    args = parser.parse_args()
    ray.init()

    if args.config is not None:
        algo_config = yaml.safe_load(open(args.config))
    elif os.path.exists(os.path.join('config', args.algorithm + '.yaml')):
        config_file = os.path.join('config', args.algorithm + '.yaml')
        algo_config = yaml.safe_load(open(config_file))
    else:
        raise FileNotFoundError(f'Config {args.config} or algorithm {args.algorithm} not found!')

    print('\nArguments:')
    pprint.pprint(args)
    print('\nConfig:')
    pprint.pprint(algo_config)
    print()

    config = algo_config['config']
    config.update(dict(
        framework=args.framework
    ))

    if args.mode == 'offline':
        config.update(dict(
            explore=False,
            input=args.data_path,
            input_evaluation=['simulation'],
        ))
    elif args.mode == 'mixed':
        config.update(dict(
            input={args.data_path: args.mixing_ratio, 'sample': (1 - args.mixing_ratio)},
            input_evaluation=['simulation'],
        ))

    tune.run(
        algo_config['run'],
        name='MineRL-' + algo_config['run'],
        local_dir='./results',
        checkpoint_freq=1000,
        checkpoint_at_end=True,
        checkpoint_score_attr='episode_reward_mean',
        keep_checkpoints_num=50,
        global_checkpoint_period=int(60 * 60 * 8),
        stop={'time_total_s': int(60 * 60 * 47.5), 'timesteps_total': 8000000},
        config=config
    )


if __name__ == '__main__':
    main()
