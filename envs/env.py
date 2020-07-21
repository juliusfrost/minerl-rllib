import gym
import gym.wrappers
import minerl


class MineRLRandomDebugEnv(gym.Env):
    def __init__(self):
        super(MineRLRandomDebugEnv, self).__init__()
        pov_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3))
        vector_space = gym.spaces.Box(low=-1.2000000476837158, high=1.2000000476837158, shape=(64,))
        self.observation_space = gym.spaces.Dict(dict(pov=pov_space, vector=vector_space))
        action_space = gym.spaces.Box(low=-1.0499999523162842, high=1.0499999523162842, shape=(64,))
        self.action_space = gym.spaces.Dict(dict(vector=action_space))
        self.done = False
        self.t = 0
        self.env_spec = minerl.herobraine.envs.MINERL_OBTAIN_DIAMOND_OBF_V0

    def _obs(self):
        return self.observation_space.sample()

    def step(self, action):
        obs = self._obs()
        reward = 0
        if self.t < 100:
            self.done = False
        else:
            self.done = True
        info = {}
        self.t += 1
        return obs, reward, self.done, info

    def reset(self):
        self.done = False
        self.t = 0
        return self._obs()

    def render(self, mode='human'):
        pass
