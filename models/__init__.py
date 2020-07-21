import models.torch.action
import models.torch.baseline
import models.torch.pov
import models.torch.reward
import models.torch.vector


def register():
    """
    Registers all models as available for RLlib
    """
    models.torch.baseline.register()
