import models.torch


def register():
    """
    Registers all models as available for RLlib
    """
    models.torch.baseline.register()
