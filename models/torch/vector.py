import torch.nn as nn


def baseline(vector_embed_size=512):
    vector_network = nn.Sequential(
        nn.Linear(64, vector_embed_size),
        nn.ELU(),
    )
    return vector_network, vector_embed_size
