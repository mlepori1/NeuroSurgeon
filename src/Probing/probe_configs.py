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
        updates: bool = True,
        stream: bool = False,
    ):
        super().__init__()
        self.model_type = model_type
        self.target_layers = target_layers
        self.mlp = mlp
        self.attn = attn
        self.circuit = circuit
        self.base = base
        self.updates = updates
        self.stream = stream


class CircuitProbeConfig(PretrainedConfig):
    def __init__(
        self,
        probe_activations: str,
        circuit_config: CircuitConfig,
        resid_config: ResidualUpdateModelConfig,
    ):
        super().__init__()
        self.probe_activations = probe_activations
        self.circuit_config = circuit_config
        self.resid_config = resid_config


class SubnetworkProbeConfig(PretrainedConfig):
    def __init__(
        self,
        probe_activations: str,
        intermediate_size: int,
        n_classes: int,
        circuit_config: CircuitConfig,
        resid_config: ResidualUpdateModelConfig,
    ):
        super().__init__()
        self.probe_activations = probe_activations
        self.circuit_config = circuit_config
        self.resid_config = resid_config
        self.intermediate_size = intermediate_size
        self.n_classes = n_classes
