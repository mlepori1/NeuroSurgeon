import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D as GPTConv1D
from transformers import PreTrainedModel
from .model_configs import CircuitConfig
from ..Masking.contsparse_layer import (
    ContSparseLayer,
    ContSparseLinear,
    ContSparseConv1d,
    ContSparseConv2d,
    ContSparseGPTConv1D,
)
from ..Masking.magprune_layer import (
    MagPruneLinear,
    MagPruneConv1d,
    MagPruneConv2d,
    MagPruneGPTConv1D,
)
from ..Masking.hardconcrete_layer import (
    HardConcreteLinear,
    HardConcreteConv1d,
    HardConcreteConv2d,
    HardConcreteGPTConv1D,
)
from ..Masking.mask_layer import MaskLayer
import warnings


class CircuitModel(nn.Module):
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
        # Compute overall l0, max masking parameters, per-layer-l0 statistics
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
        # Call forward of model, add l0 regularization if appropriate
        output = self.wrapped_model(**kwargs, return_dict=True)
        if self.config.add_l0:
            if hasattr(output, "loss") and output.loss is not None:
                output.loss = output.loss + (
                    self.config.l0_lambda * self._compute_l0_loss()
                )
        return output

    def set_ablate_mode(self, ablation, force_resample=False):
        # change ablate mode for model
        for module in self.modules():
            if issubclass(type(module), MaskLayer):
                module.ablation = ablation
                module.force_resample = force_resample

    def use_masks(self, value, name_list=None):
        for name, module in self.named_modules():
            if name_list is not None:
                if issubclass(type(module), MaskLayer) and name in name_list:
                    module.use_masks = value
            else:
                if issubclass(type(module), MaskLayer):
                    module.use_masks = value
