"""
Runs the training process through the RLLib train script
See help with:
$ python rllib_train.py --help
"""

from ray.rllib.train import create_parser, run

from minerl_rllib import models, envs

envs.register()
models.register()


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)


if __name__ == '__main__':
    main()
