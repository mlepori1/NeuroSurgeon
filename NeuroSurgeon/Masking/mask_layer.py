from abc import abstractmethod

import torch
import torch.nn as nn


class MaskLayer(nn.Module):
    """This is an abstract class that defines the minimum functionality of a mask layer.
    All mask layers inherit from this class.

    :param ablation: A string that determines how masks are produced from the mask layer parameters. Valid options include:

        - none: Producing a standard binary mask
        - zero_ablate: Inverting the standard binary mask. Used for pruning discovered subnetworks.
        - random_ablate: Inverting the standard binary mask and reinitializing zero'd elements. Used for pruning discovered subnetworks.
        - randomly_sampled: Sampling a random binary mask of the same size as the standard mask.
        - complement_sampled: Sampling a random binary mask of the same size as the standard mask from the complement set of entries as the standard mask.

    :type ablation: str
    :param mask_unit: A string that determines whether masks are produced at the weight or neuron level. Valid options include ["neuron", "weight"]
    :type mask_unit: str
    :param mask_bias: Determines whether to mask bias terms in addition to weight terms.
    :type mask_bias: bool
    """

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
        if value not in [
            "none",
            "randomly_sampled",
            "zero_ablate",
            "random_ablate",
            "complement_sampled",
        ]:
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
        """Returns the L0 norm of the mask. This is used for L0 regularization and for reporting on mask size

        :return The total L0 norm of the mask
        :rtype float
        """
        l0 = torch.sum(self._compute_mask("weight_mask_params"))
        if self.mask_bias:
            l0 += torch.sum(self._compute_mask("bias_mask_params"))
        return l0

    def calculate_max_l0(self):
        """Returns the maximum L0 norm of the mask (i.e. the number of prunable weights/neurons in the layer).

        :return The maximum L0 norm of the mask
        :rtype float
        """
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
