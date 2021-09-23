import gym


class MineRLActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = self.action_space["vector"]

    def action(self, action):
        return dict(vector=action)

    def reverse_action(self, action):
        return action["vector"]
