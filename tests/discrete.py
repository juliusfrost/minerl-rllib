import os
import numpy as np
import gym
from tqdm import tqdm
from minerl_rllib.envs import wrap


def test_discrete():
    os.chdir('..')
    env = gym.make('MineRLTreechopVectorObf-v0')
    env = wrap(env, discrete=True, num_actions=32)

    for _ in tqdm(range(100)):
        print(env.reverse_action(np.random.randn(1, 64)))

    done = False
    obs = env.reset()
    for _ in tqdm(range(100)):
        if done:
            break
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)


if __name__ == '__main__':
    test_discrete()
