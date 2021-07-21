import gym


class MineRLDeterministic(gym.Wrapper):
    def __init__(self, env, seed: int):
        super().__init__(env)
        self._set_seed = seed

    def reset(self, **kwargs):
        self.seed(self._set_seed)
        return self.env.reset()
