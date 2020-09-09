from ray.rllib.rollout import create_parser, deprecation_warning, run

import envs
import models

envs.register()
models.register()


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Old option: monitor, use video-dir instead.
    if args.monitor:
        deprecation_warning("--monitor", "--video-dir=[some dir]")
    # User tries to record videos, but no-render is set: Error.
    if (args.monitor or args.video_dir) and args.no_render:
        raise ValueError(
            "You have --no-render set, but are trying to record rollout videos"
            " (via options --video-dir/--monitor)! "
            "Either unset --no-render or do not use --video-dir/--monitor.")
    # --use_shelve w/o --out option.
    if args.use_shelve and not args.out:
        raise ValueError(
            "If you set --use-shelve, you must provide an output file via "
            "--out as well!")
    # --track-progress w/o --out option.
    if args.track_progress and not args.out:
        raise ValueError(
            "If you set --track-progress, you must provide an output file via "
            "--out as well!")

    run(args, parser)


if __name__ == '__main__':
    main()
