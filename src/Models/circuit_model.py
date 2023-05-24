import torch
import torch.nn as nn
from transformers import PretrainedModel
from Models.model_configs import CircuitConfig
from contsparse_layer import ContSparseLayer, ContSparseLinear, ContSparseConv1d, ContSparseConv2d
from mask_layer import MaskLayer
import warnings

class CircuitModel(PretrainedModel):

    def __init__(self, config: CircuitConfig, model: PretrainedModel):
        super().__init__()
        self.config = config
        self.model = model
        
        self._replace_target_layers()
        self._handle_model_freezing()

    def _handle_model_freezing(self):
        # Remove gradients from all parameters except mask params
        if self.config.freeze_base:
            for layer in self.model.modules():
                if hasattr(layer, "weight") and layer.weight != None:
                    layer.weight.requires_grad = False
                if hasattr(layer, "bias") and layer.bias != None: 
                    layer.bias.requires_grad = False

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
            current_component = self.model
            for component_path in layer_path:
                earlier_component = current_component
                current_component = getattr(earlier_component, component_path)
            
            layer_type = type(current_component)
            masked_layer = torch2masked[layer_type](current_component, *masked_args)
            setattr(earlier_component, component_path, masked_layer)

    def _create_torch2masked(self, config):
        if config.mask_method == "Continuous_Sparsification":
            try:
                return {
                    nn.Linear : ContSparseLinear,
                    nn.Conv2d: ContSparseConv2d,
                    nn.Conv1d: ContSparseConv1d,
                }
            except:
                raise ValueError("Only support masking Linear, Conv1d, and Conv2d layers at this time")
        else:
            raise ValueError("Only Continuous_Sparsification is supported at this time")

    def _create_masked_args(self, config):
        if config.mask_method == "Continuous_Sparsification":
            return [config.mask_hparams["ablation"], config.mask_hparams["mask_bias"], config.mask_hparams["mask_init_value"]]
        else:
            raise ValueError("Only Continuous_Sparsification is supported at this time")
        
    def train(self, train_bool):
        if self.config.freeze_base:
            for layer in self.model.modules():
                if not issubclass(MaskLayer):
                    layer.train(False) # Keep all non-masklayers train=False, to handle dropout, batchnorm, etc
            for layer in self.model.modules():
                if issubclass(MaskLayer):
                    layer.train(train_bool) # Set masklayers to train or eval, all weights and biases are frozen anyway        

    def compute_l0_statistics(self):
        # Compute overall l0, max masking parameters, per-layer-l0 statistics
        self.eval()
        total_l0 = 0
        max_l0 = 0
        layer2l0 = {}
        layer2maxl0 = {}
        for name, layer in self.model.named_modules():
            if issubclass(layer, MaskLayer):
                layer_l0 = layer.calculate_l0()
                layer_max_l0 = layer.calculate_max_l0()
                total_l0 += layer_l0
                max_l0 += layer_max_l0
                layer2l0[name] = layer_l0
                layer2maxl0[name] = layer_max_l0
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
            for _, layer in self.model.named_modules():
                if issubclass(layer, MaskLayer):
                    layer_l0 = layer.calculate_l0()
                    total_l0 += layer_l0
            return total_l0
        else: return 0.0

    @property
    def temperature(self):
        return self._temperature
    
    @temperature.setter
    def temperature(self, value):
        if self.config.mask_method != "Continuous_Sparsification":
            warnings.warn("Temperature is ignored when not using continuous sparsification")
        if value < 0:
            raise ValueError("Temperature must be > 0")

        self._temperature = value
        for module in self.modules():
            if issubclass(module, ContSparseLayer):
                module.temperature = value

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        # Call forward of model, add l0 regularization if appropriate
        output = self.model(**input, return_dict=True)
        if self.config.add_l0:
            if hasattr(output, "loss"):
                output.loss = output.loss + (self.config.l0_lambda * self._compute_l0_loss())
            else:
                raise ValueError("Cannot add L0 Regularization when underlying model doesn't return loss")
        return output

    def __getattr__(self, name):
        # Override this to call functions from the underlying model if not found in this class
        return getattr(self.model, name)

    def set_ablate_mode(self, ablation):
        # change ablate mode for model
        for module in self.modules():
            if issubclass(module, MaskLayer):
                module.ablation = ablation