import minerl_rllib.models.torch.baseline


def register():
    """
    Registers all models as available for RLlib
    """
    minerl_rllib.models.torch.baseline.register()
