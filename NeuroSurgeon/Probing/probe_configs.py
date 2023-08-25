from typing import List

from transformers import PretrainedConfig

from ..Models.model_configs import CircuitConfig


class ResidualUpdateModelConfig(PretrainedConfig):
    def __init__(
        self,
        model_type: str,
        target_layers: List[int],
        mlp: bool,
        attn: bool,
        updates: bool = True,
        stream: bool = False,
    ):
        super().__init__()
        self.model_type = model_type
        self.target_layers = target_layers
        self.mlp = mlp
        self.attn = attn
        self.updates = updates
        self.stream = stream


class SubnetworkProbeConfig(PretrainedConfig):
    def __init__(
        self,
        probe_vectors: str,
        n_classes: int,
        circuit_config: CircuitConfig,
        resid_config: ResidualUpdateModelConfig,
        intermediate_size: int = -1,
        labeling: str = "sequence",
    ):
        super().__init__()
        self.probe_vectors = probe_vectors
        self.circuit_config = circuit_config
        self.resid_config = resid_config
        self.intermediate_size = intermediate_size
        self.n_classes = n_classes
        self.labeling = labeling
