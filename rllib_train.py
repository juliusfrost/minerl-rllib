"""
Runs the training process through the RLLib train script
See help with:
$ python rllib_train.py --help
"""
import subprocess

import yaml
from ray.rllib.train import create_parser, run

import envs
import models

envs.register()
models.register()


def generate_kmeans(env):
    command = f'python generate_kmeans.py --env {env}'
    print('running:', command)
    subprocess.run(command.split())


def convert_data(args):
    command = f'python convert_data.py -f {args.config_file}'
    print('running:', command)
    subprocess.run(command.split())


def check_data(args):
    if args.config_file is not None:
        config = yaml.safe_load(open(args.config_file))
        settings = list(config.values())[0]
        if 'env' in settings:
            env = settings['env']
        if 'config' in settings:
            if 'env' in settings['config']:
                env = settings['config']['env']
            if 'env_config' in settings['config']:
                env_config = settings['config']['env_config']
                if env_config.get('discrete', False):
                    generate_kmeans(env)
            if 'input' in settings['config']:
                convert_data(args)


def main():
    parser = create_parser()
    args = parser.parse_args()
    check_data(args)
    run(args, parser)


if __name__ == '__main__':
    main()
