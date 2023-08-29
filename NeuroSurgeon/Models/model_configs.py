from typing import List

from transformers import PretrainedConfig


class CircuitConfig(PretrainedConfig):
    """A config object to define the behavior of a CircuitModel

    :param mask_method: Which masking technique to use. Options include:
        ["continuous_sparsification", "hard_concrete", "magnitude_pruning"], defaults to "continuous_sparsification"
    :type mask_method: str, optional
    :param mask_hparams: A dictionary defining hyperparameters specific to the specified mask method. See the documentation for each layer for details about these hyperparameters, defaults to {}

        - For "continuous_sparsification": Requires "ablation", "mask_unit", "mask_bias", "mask_init_value"
        - For "hard_concrete": Requires "ablation", "mask_unit", "mask_bias", "mask_init_percentage"
        - For "magnitude_pruning": Requires "ablation", "mask_bias", "prune_percentage"

    :type mask_hparams: dict, optional
    :param target_layers: A list of layers to turn into mask layers.
        These layer names can be obtained from the wrapped model's state dict. They must correspond to nn.Linear, GPT-style Conv1D, nn.Conv2d, or nn.Conv1d layers. defaults to []
    :type target_layers: List[str], optional
    :param freeze_base: Whether to freeze the weights and biases of the wrapped model, defaults to True
    :type freeze_base: bool, optional
    :param add_l0: Whether to add L0 regularization to the loss computed during a transformer model's forward pass, defaults to True
    :type add_l0: bool, optional
    :param l0_lambda: The weighting of the L0 regularization, should usually be scaled with parameter count, defaults to 1e-8
    :type l0_lambda: float, optional
    """

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
