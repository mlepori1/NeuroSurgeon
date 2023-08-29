import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D as GPTConv1D

from ..Masking.contsparse_layer import (
    ContSparseConv1d,
    ContSparseConv2d,
    ContSparseGPTConv1D,
    ContSparseLayer,
    ContSparseLinear,
)
from ..Masking.hardconcrete_layer import (
    HardConcreteConv1d,
    HardConcreteConv2d,
    HardConcreteGPTConv1D,
    HardConcreteLinear,
)
from ..Masking.magprune_layer import (
    MagPruneConv1d,
    MagPruneConv2d,
    MagPruneGPTConv1D,
    MagPruneLinear,
)
from ..Masking.mask_layer import MaskLayer
from .model_configs import CircuitConfig


class CircuitModel(nn.Module):
    """CircuitModel is a wrapper around a Transformers model (or custom nn.Module that also returns an object from a forward pass), which replaces particular layers
    with MaskLayers. These MaskLayers modify the weight matrices of particular layers according to some strategy, which
    is defined in the CircuitConfig.

    :param config: A configuration object defining the how the CircuitModel wrapper modifes a model
    :type config: CircuitConfig
    :param model: The model to modify
    :type model: nn.Module
    """

    def __init__(self, config: CircuitConfig, model: nn.Module):
        super().__init__()
        self.config = config
        self.wrapped_model = model
        self.temperature = 1.0

        self._replace_target_layers()
        self._handle_model_freezing()

    def _handle_model_freezing(self):
        # Put all mask parameters in train mode, put all others in eval mode
        self.train(True)

        # Remove gradients from all parameters except mask params
        if self.config.freeze_base:
            for layer in self.wrapped_model.modules():
                if not issubclass(type(layer), MaskLayer):
                    layer.train(False)
                if hasattr(layer, "weight") and layer.weight != None:
                    layer.weight.requires_grad = False
                if hasattr(layer, "bias") and layer.bias != None:
                    layer.bias.requires_grad = False
            for layer in self.wrapped_model.modules():
                if issubclass(type(layer), MaskLayer):
                    layer.train(True)

    def _replace_target_layers(self):
        # Iterate through target layers and replace with mask layers

        torch2masked = self._create_torch2masked(self.config)
        masked_args = self._create_masked_args(self.config)

        for target_layer in self.config.target_layers:
            if "." in target_layer:
                layer_path = target_layer.split(".")
            else:
                layer_path = [target_layer]

            earlier_component = None
            current_component = self.wrapped_model
            try:
                for component_path in layer_path:
                    earlier_component = current_component
                    current_component = getattr(earlier_component, component_path)
            except:
                raise ValueError(f"{target_layer} not found in network")

            try:
                layer_type = type(current_component)
                masked_layer = torch2masked[layer_type].from_layer(
                    current_component, *masked_args
                )

                setattr(earlier_component, component_path, masked_layer)
            except:
                raise ValueError(
                    f"{target_layer} is a {layer_type}, which is not a supported layer type"
                )

    def _create_torch2masked(self, config):
        if config.mask_method == "continuous_sparsification":
            return {
                nn.Linear: ContSparseLinear,
                nn.Conv2d: ContSparseConv2d,
                nn.Conv1d: ContSparseConv1d,
                GPTConv1D: ContSparseGPTConv1D,
            }
        elif config.mask_method == "magnitude_pruning":
            return {
                nn.Linear: MagPruneLinear,
                nn.Conv2d: MagPruneConv2d,
                nn.Conv1d: MagPruneConv1d,
                GPTConv1D: MagPruneGPTConv1D,
            }
        elif config.mask_method == "hard_concrete":
            return {
                nn.Linear: HardConcreteLinear,
                nn.Conv2d: HardConcreteConv2d,
                nn.Conv1d: HardConcreteConv1d,
                GPTConv1D: HardConcreteGPTConv1D,
            }
        else:
            raise ValueError(
                "Only Continuous_Sparsification, Hard_Concrete, and Magnitude_Pruning is supported at this time"
            )

    def _create_masked_args(self, config):
        if config.mask_method == "continuous_sparsification":
            return [
                config.mask_hparams["ablation"],
                config.mask_hparams["mask_unit"],
                config.mask_hparams["mask_bias"],
                config.mask_hparams["mask_init_value"],
            ]
        if config.mask_method == "hard_concrete":
            return [
                config.mask_hparams["ablation"],
                config.mask_hparams["mask_unit"],
                config.mask_hparams["mask_bias"],
                config.mask_hparams["mask_init_percentage"],
            ]
        if config.mask_method == "magnitude_pruning":
            return [
                config.mask_hparams["ablation"],
                config.mask_hparams["mask_bias"],
                config.mask_hparams["prune_percentage"],
            ]
        else:
            raise ValueError(
                "Only Continuous_Sparsification, Hard_Concrete, and Magnitude_Pruning is supported at this time"
            )

    def train(self, train_bool=True):
        """Similar to the normal nn.Module train function, except keeps the underlying model weights
        frozen if that is specified by the config argument

        :param train_bool: Whether to put the model in train mode or eval mode, defaults to True
        :type train_bool: bool
        """
        self.training = train_bool

        if self.config.freeze_base:
            for layer in self.wrapped_model.modules():
                if not issubclass(type(layer), MaskLayer):
                    layer.train(
                        False
                    )  # Keep all non-masklayers train=False, to handle dropout, batchnorm, etc
            for layer in self.wrapped_model.modules():
                if issubclass(type(layer), MaskLayer):
                    layer.train(
                        train_bool
                    )  # Set masklayers to train or eval, all weights and biases are frozen anyway
        else:
            for layer in self.wrapped_model.modules():
                layer.train(train_bool)

    def compute_l0_statistics(self):
        """Compute overall l0, max masking parameters, per-layer-l0 statistics

        :return: A dictionary containing the computed statistics. Keys are "total_l0", "max_l0", "layer2l0", "layer2maxl0"
        :rtype: dict
        """
        train_bool = self.training
        self.eval()
        total_l0 = 0
        max_l0 = 0
        layer2l0 = {}
        layer2maxl0 = {}
        for name, layer in self.wrapped_model.named_modules():
            if issubclass(type(layer), MaskLayer):
                layer_l0 = layer.calculate_l0()
                layer_max_l0 = layer.calculate_max_l0()
                total_l0 += layer_l0
                max_l0 += layer_max_l0
                layer2l0[name] = layer_l0
                layer2maxl0[name] = layer_max_l0
        self.train(train_bool)
        return {
            "total_l0": total_l0,
            "max_l0": max_l0,
            "layer2l0": layer2l0,
            "layer2maxl0": layer2maxl0,
        }

    def _compute_l0_loss(self):
        # add L0 loss when training
        if self.training:
            total_l0 = 0.0
            for _, layer in self.wrapped_model.named_modules():
                if issubclass(type(layer), MaskLayer):
                    layer_l0 = layer.calculate_l0()
                    total_l0 += layer_l0
            return total_l0
        else:
            return 0.0

    @property
    def temperature(self):
        """The temperature parameter for ContSparseLayers ONLY

        :return: Temperature parameter
        :rtype: float
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if self.config.mask_method != "continuous_sparsification":
            warnings.warn(
                "Temperature is ignored when not using continuous sparsification"
            )
        if value < 0:
            raise ValueError("Temperature must be > 0")

        self._temperature = value
        for module in self.modules():
            if issubclass(type(module), ContSparseLayer):
                module.temperature = value

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        """Forward call of the wrapped model, adds L0 regurization if specified by the config"""
        output = self.wrapped_model(**kwargs, return_dict=True)
        if self.config.add_l0:
            if hasattr(output, "loss") and output.loss is not None:
                output.loss = output.loss + (
                    self.config.l0_lambda * self._compute_l0_loss()
                )
        return output

    def set_ablate_mode(self, ablation, force_resample=False):
        """Changes the ablate mode of the MaskLayers from what is specified in the CircuitConfig

        :param ablation: A string that determines how masks are produced from the mask layer parameters. Valid options include:

            - none: Producing a standard binary mask
            - zero_ablate: Inverting the standard binary mask. Used for pruning discovered subnetworks.
            - random_ablate: Inverting the standard binary mask and reinitializing zero'd elements. Used for pruning discovered subnetworks.
            - randomly_sampled: Sampling a random binary mask of the same size as the standard mask.
            - complement_sampled: Sampling a random binary mask of the same size as the standard mask from the complement set of entries as the standard mask.

        :type ablation: str
        :param force_resample: If setting ablation=["randomly_sampled", "complement_sampled"], whether to randomly resample the generated mask, defaults to False
        :type force_resample: bool, optional
        """
        for module in self.modules():
            if issubclass(type(module), MaskLayer):
                module.ablation = ablation
                module.force_resample = force_resample

    def use_masks(self, value, name_list=None):
        """This function can be used to turn off masking behavior for either the entire model, or particular layers (if name_list!=None)

        :param value: Whether to use masks or not
        :type value: bool
        :param name_list: If set, it determines which subset of MaskLayers to turn on or off, defaults to None
        :type name_list: List[str], optional
        """
        for name, module in self.named_modules():
            if name_list is not None:
                if issubclass(type(module), MaskLayer) and name in name_list:
                    module.use_masks = value
            else:
                if issubclass(type(module), MaskLayer):
                    module.use_masks = value
