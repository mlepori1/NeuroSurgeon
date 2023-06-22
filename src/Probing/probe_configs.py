from transformers import PretrainedConfig
from ..Models.model_configs import *


class CircuitProbeConfig(PretrainedConfig):
    def __init__(
        self,
        probe_updates: str,
        circuit_config: CircuitConfig,
        resid_config: ResidualUpdateModelConfig,
    ):
        super().__init__()
        self.probe_updates = probe_updates
        self.circuit_config = circuit_config
        self.resid_config = resid_config
