import gym


class MineRLActionRepeat(gym.Wrapper):
    def __init__(self, env, action_repeat=1):
        super().__init__(env)
        self.action_repeat = action_repeat

    def step(self, action):
        obs, reward, done, info = None, None, None, None
        total_reward = 0
        for _ in range(self.action_repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
