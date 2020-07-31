import argparse
import json
import os

import yaml
from minerl.herobraine.envs import obfuscated_envs

from envs.data import write_jsons

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default=os.getenv('MINERL_DATA_ROOT', 'data'),
                    help='path to the data directory of the MineRL data')
parser.add_argument('--save-dir', default=None, type=str,
                    help='directory to write jsons. defaults to the rllib subdirectory of the MineRL data path')
parser.add_argument('--env', type=str, default=None, help='Environment name to write jsons')
parser.add_argument('--preprocess', action='store_true', help='whether to preprocess observations')
parser.add_argument('--config-file', default=None, type=str, help='config file to load environment config')
parser.add_argument('--env-config', default='{}', type=json.loads,
                    help='specifies environment configuration options. overrides config file specifications')


def main():
    args = parser.parse_args()

    if args.save_path is None:
        save_path = os.path.join(args.data_path, 'rllib')
    else:
        save_path = args.save_path

    env_list = []

    env_config = {}
    if args.config_file is not None:
        config = yaml.safe_load(open(args.config_file))
        settings = config[list(config.keys())[0]]
        if 'config' in settings:
            if 'env_config' in settings['config']:
                env_config = settings['config']['env_config']
            if 'env' in settings['config']:
                env_list.append(settings['config']['env'])
        if 'env' in settings:
            env_list.append(settings['env'])

    if args.env is None and len(env_list) == 0:
        for env_spec in obfuscated_envs:
            env_list.append(env_spec.name)
    else:
        env_list.append(args.env)

    for env_name in env_list:
        write_jsons(env_name, args.data_dir, env_config, save_path)


if __name__ == '__main__':
    main()
