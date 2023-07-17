from abc import abstractmethod
import torch.nn as nn
import torch


class MaskLayer(nn.Module):
    """Base Class for masking layers. Inherits from nn.Module"""

    def __init__(self, ablation: str, mask_unit: str, mask_bias: bool):
        super().__init__()
        self.ablation = ablation
        self.mask_unit = mask_unit
        self.mask_bias = mask_bias
        self.use_masks = True  # Default behavior is to use the masks, but this can change for specific evaluations

    @property
    def ablation(self):
        return self._ablation

    @ablation.setter
    def ablation(self, value):
        if value not in ["none", "randomly_sampled", "zero_ablate", "random_ablate, complement_sampled"]:
            raise ValueError(
                "Only none, randomly_sampled, zero_ablate, random_ablate, and complement_sampled are supported"
            )
        self._ablation = value

    @property
    def mask_bias(self):
        return self._mask_bias

    @mask_bias.setter
    def mask_bias(self, value):
        self._mask_bias = value

    @property
    def mask_unit(self):
        return self._mask_unit

    @mask_unit.setter
    def mask_unit(self, value):
        self._mask_unit = value

    @property
    def use_masks(self):
        return self._use_masks

    @use_masks.setter
    def use_masks(self, value):
        self._use_masks = value

    def train(self, train_bool):
        self.training = train_bool

    def calculate_l0(self):
        l0 = torch.sum(self._compute_mask("weight_mask_params"))
        if self.mask_bias:
            l0 += torch.sum(self._compute_mask("bias_mask_params"))
        return l0

    def calculate_max_l0(self):
        max_l0 = len(self._compute_mask("weight_mask_params").reshape(-1))
        if self.mask_bias:
            max_l0 += len(self._compute_mask("bias_mask_params").reshape(-1))
        return max_l0

    @abstractmethod
    def _compute_mask(self, param_type):
        pass

    @abstractmethod
    def _init_mask(self):
        pass

    @abstractmethod
    def reset_parameters(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def from_layer(self):
        pass
