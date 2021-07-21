from ray.rllib.agents.sac.sac_torch_model import SACTorchModel
from ray.rllib.utils.framework import TensorType


class CQLModel(SACTorchModel):
    def value_function(self) -> TensorType:
        pass

    def import_from_h5(self, h5_file: str) -> None:
        pass
