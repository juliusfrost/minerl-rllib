import torch.nn as nn


def baseline(action_embed_size=128, discrete=False, n: int = None):
    if discrete:
        action_network = nn.Embedding(n, action_embed_size)
    else:
        action_network = nn.Sequential(
            nn.Linear(64, action_embed_size),
            nn.ELU(),
        )
    return action_network, action_embed_size
