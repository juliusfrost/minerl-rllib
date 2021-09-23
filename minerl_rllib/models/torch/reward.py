import torch.nn as nn


def baseline(reward_embed_size=128):
    reward_network = nn.Sequential(
        nn.Linear(1, reward_embed_size),
        nn.Tanh(),
    )
    return reward_network, reward_embed_size
