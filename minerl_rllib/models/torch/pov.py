import torch.nn as nn


def baseline(pov_embed_size=256):
    pov_network = nn.Sequential(
        nn.Conv2d(3, 64, 4, 4),
        nn.ReLU(),
        nn.Conv2d(64, 128, 4, 4),
        nn.ReLU(),
        nn.Conv2d(128, pov_embed_size, 4, 4),
        nn.ReLU(),
    )
    return pov_network, pov_embed_size
