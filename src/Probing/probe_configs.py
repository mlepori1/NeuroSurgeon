from transformers import PretrainedConfig
from typing import List
from ..Models.model_configs import CircuitConfig


class ResidualUpdateModelConfig(PretrainedConfig):
    def __init__(
        self,
        model_type: str,
        target_layers: List[int],
        mlp: bool,
        attn: bool,
        circuit: bool = False,
        base: bool = True,
    ):
        super().__init__()
        self.model_type = model_type
        self.target_layers = target_layers
        self.mlp = mlp
        self.attn = attn
        self.circuit = circuit
        self.base = base


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
