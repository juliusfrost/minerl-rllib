"""
Runs the training process through the RLLib train script
Example command:
$ python rllib_train.py -f config/minerl-impala-debug.yaml
See help with:
$ python rllib_train.py --help
"""

from ray.rllib.train import create_parser, run

import envs
import models

envs.register()
models.register()


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)


if __name__ == '__main__':
    main()
