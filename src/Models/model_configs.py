from transformers import PretrainedConfig
from typing import List

class CircuitConfig(PretrainedConfig):

    def __init__(
            self,
            mask_method: str = "continuous_sparsification",
            mask_hparams: dict = {},
            target_layers: List[str] = [],
            freeze_base: bool = True,
            add_l0: bool = True, # Will throw error if no loss in output
            l0_lambda: float = 1E-8 
    ):
        
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
            base: bool,
            target_layers: List[int],
            mlp: bool,
            attn: bool,
    ):
        super().__init__()
        self.model_type = model_type
        self.base = base
        self.target_layers = target_layers
        self.mlp = mlp
        self.attn = attn