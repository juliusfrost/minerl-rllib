import gym.wrappers


class MineRLRewardPenaltyWrapper(gym.wrappers.TransformReward):
    def __init__(self, env, reward_penalty=0.001):
        super().__init__(env, lambda r: r - reward_penalty)