import logging

import gym
import minerl


def main():
    logging.basicConfig(level=logging.DEBUG)
    env = gym.make("MineRLNavigateDense-v0")
    env.reset()
    env.close()


if __name__ == "__main__":
    main()
