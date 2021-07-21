import gym


class MineRLObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Tuple(
            (self.observation_space["pov"], self.observation_space["vector"])
        )

    def observation(self, observation):
        return observation["pov"], observation["vector"]
