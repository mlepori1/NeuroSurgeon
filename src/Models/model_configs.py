from transformers import PretrainedConfig
from typing import List


class CircuitConfig(PretrainedConfig):
    def __init__(
        self,
        mask_method: str = "continuous_sparsification",
        mask_hparams: dict = {},
        target_layers: List[str] = [],
        freeze_base: bool = True,
        add_l0: bool = True,
        l0_lambda: float = 1e-8,
    ):
        super().__init__()
        self.mask_method = mask_method
        self.mask_hparams = mask_hparams
        self.target_layers = target_layers
        self.freeze_base = freeze_base
        self.add_l0 = add_l0
        self.l0_lambda = l0_lambda


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
