from abc import abstractmethod
import torch.nn as nn
import torch


class MaskLayer(nn.Module):
    """Base Class for masking layers. Inherits from nn.Module"""

    def __init__(self, bias: bool, ablation: str, mask_bias: bool):
        self.bias = bias
        self.ablation = ablation
        self.mask_bias = mask_bias

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, value):
        self._bias = value

    @property
    def ablation(self):
        return self._ablation

    @ablation.setter
    def ablation(self, value):
        if value not in ["none, randomly_sampled", "zero_ablate", "random_ablate"]:
            raise ValueError(
                "Only none, randomly_sampled, zero_ablate, random_ablate are supported"
            )
        self._ablation = value

    @property
    def mask_bias(self):
        return self._mask_bias

    @mask_bias.setter
    def mask_bias(self, value):
        if value == True and self.bias == False:
            raise ValueError("Cannot mask bias if bias is set to false")
        self._mask_bias = value

    def train(self, train_bool):
        self.training = train_bool

    def calculate_l0(self):
        l0 = torch.sum(self._compute_mask("weight_mask_params"))
        if self.mask_bias():
            l0 += torch.sum(self._compute_mask("bias_mask_params"))
        return l0

    def calculate_max_l0(self):
        max_l0 = len(self._compute_mask("weight_mask_params").reshape(-1))
        if self.mask_bias():
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
