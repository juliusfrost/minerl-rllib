import argparse
import json
import os

import yaml
from minerl.herobraine.envs import obfuscated_envs
from ray.tune.utils import merge_dicts

from envs.data import write_jsons

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default=os.getenv('MINERL_DATA_ROOT', 'data'),
                    help='path to the data directory of the MineRL data')
parser.add_argument('--save-path', default=None, type=str,
                    help='directory to write jsons. defaults to the rllib subdirectory of the MineRL data path')
parser.add_argument('--env', type=str, default=None, help='Environment name to write jsons')
parser.add_argument('-f', '--config-file', default=None, type=str, help='config file to load environment config')
parser.add_argument('--env-config', default='{}', type=json.loads,
                    help='specifies environment configuration options. overrides config file specifications')
parser.add_argument('--overwrite', action='store_true', help='overwrite existing data if directory not empty')


def get_save_path(data_dir, env_config, env_name=None):
    save_dir = 'env-config'
    for key, value in env_config.items():
        save_dir += f'--{key}-{value}'
    save_path = os.path.join(data_dir, 'rllib', save_dir)
    if env_name is not None:
        save_path = os.path.join(save_path, env_name)
    return save_path


def main():
    args = parser.parse_args()

    env_list = []

    env_config = {}
    if args.config_file is not None:
        config = yaml.safe_load(open(args.config_file))
        settings = list(config.values())[0]
        if 'config' in settings:
            if 'env_config' in settings['config']:
                env_config = settings['config']['env_config']
            if 'env' in settings['config']:
                env_list.append(settings['config']['env'])
        if 'env' in settings:
            env_list.append(settings['env'])
    else:
        if args.env is None:
            for env_spec in obfuscated_envs:
                env_list.append(env_spec.name)
        else:
            env_list.append(args.env)
    env_config = merge_dicts(env_config, args.env_config)

    if args.save_path is None:
        save_path = get_save_path(args.data_dir, env_config)
    else:
        save_path = args.save_path
    print(f'saving jsons to {save_path}')

    for env_name in env_list:
        print(f'Writing data to json files for environment {env_name}')
        env_save_path = os.path.join(save_path, env_name)
        write_jsons(env_name, args.data_dir, env_config, env_save_path, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
