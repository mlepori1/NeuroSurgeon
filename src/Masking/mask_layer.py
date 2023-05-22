from abc import abstractmethod
import torch.nn


class MaskLayer(nn.Module):
    """Base Class for masking layers. Inherits from nn.Module"""

    def __init__(self, ablation: str, include_bias: bool):
        self.ablation = ablation
        self.include_bias = include_bias

    @property
    def ablation(self):
        return self._ablation

    @ablation.setter
    def ablation(self, value):
        if value not in ["none, randomly_sampled", "zero_mask", "random_mask"]:
            raise ValueError(
                "Only none, randomly_sampled, zero_mask, random_mask are supported"
            )
        self._ablation = value

    @property
    def include_bias(self):
        return self._include_bias

    @include_bias.setter
    def include_bias(self, value):
        self._include_bias = value

    def train(self, train_bool):
        self.training = train_bool

    def calculate_l0(self):
        l0 = torch.sum(self.compute_mask("mask_weight"))
        if self.include_bias():
            l0 += torch.sum(self.compute_mask("mask_bias"))
        return l0

    @abstractmethod
    def compute_mask(self, param_type):
        pass

    @abstractmethod
    def init_mask(self):
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
