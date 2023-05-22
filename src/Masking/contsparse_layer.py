from mask_layer import MaskLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


class ContSparseLayer(MaskLayer):
    def __init__(self, ablation: str, include_bias: bool, mask_init_value: float):
        super().__init__(ablation, include_bias)
        self.mask_init_value = mask_init_value
        self.temperature = 1.0

    @property
    def mask_init_value(self):
        return self._mask_init_value

    @mask_init_value.setter
    def mask_init_value(self, value):
        self._mask_init_value = value

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    def sample_random_mask(self, param_type, force_resample=False):
        """Used to create a binary mask that contains the same number of ones and zeros as a normal ablated mask,
        but drawn from the complement of the trained binary mask. This is done to assess whether ablating a trained
        subnetwork yields greater performance degredation than ablating a random subnetwork.

        Sample a random mask once and then use it to evaluate a whole dataset. Create more models like this to
        get a distribution over random mask samples
        """
        if hasattr(self, "sampled_" + param_type) and not force_resample:
            return getattr(self, "sampled_" + param_type)
        else:
            if param_type == "weight_mask_params":
                mask_param = self.weight_mask_params
            elif param_type == "bias_mask_params":
                mask_param = self.bias_mask_params
            else:
                raise ValueError(
                    "Only weight_mask_params and bias_mask_params are supported param_types"
                )

            # Get hard mask
            sampled_size = torch.sum((mask_param > 0).int())
            if sampled_size > torch.sum((mask_param < 0).int()):
                raise ValueError(
                    "Trying to sample random masks, but original mask contains > 50 percent of weights"
                )

            # Sample sample_size number of weights from the complement of the mask given by mask_weight
            complement_mask_weights = mask_param < 0
            sample_complement_indices = complement_mask_weights.nonzero(
                as_tuple=False
            )  # get indices of complement weights
            # shuffle the indices of possible sampled weights, take the first sample_size indices as your sampled mask
            sample_complement_indices = sample_complement_indices[
                torch.randperm(sample_complement_indices.size(0))
            ][:sampled_size]
            # Reformat indices into tuple form for indexing into tensor
            sample_complement_indices = (
                sample_complement_indices[:, 0],
                sample_complement_indices[:, 1],
            )
            # Create a mask with just the sampled indices removed to run random ablation experiments
            sampled_mask = torch.ones(mask_param.shape)
            sampled_mask[sample_complement_indices] = 0.0
            sampled_mask = nn.Parameter(sampled_mask, requires_grad=False).cuda()

            setattr(self, "sampled_" + param_type, sampled_mask)

            return sampled_mask

    def compute_mask(self, param_type):
        if param_type == "weight_mask_params":
            mask_param = self.weight_mask_params
        elif param_type == "bias_mask_params":
            mask_param = self.bias_mask_params
        else:
            raise ValueError(
                "Only weight_mask_params and bias_mask_params are supported param_types"
            )

        hard_mask = not self.training or mask_param.requires_grad == False
        if (self.ablation == "none") and hard_mask:
            mask = (mask_param > 0).float()  # Hard Mask when not training
        elif self.ablation == "randomly_sampled":
            mask = (
                self.get_sampled_mask()
            )  # Generates a randomly sampled mask of equal size to trained mask
        elif (self.ablation != "none") and hard_mask:
            mask = (mask_param <= 0).float()  # Used for subnetwork ablation
        else:
            mask = F.sigmoid(
                self.temperature * mask_param
            )  # Generate a soft mask for training

        return mask


class ContSparseLinear(ContSparseLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        ablation: str = "none",
        include_bias: bool = False,
        mask_init_value: float = 0.0,
    ):
        super().__init__(ablation, include_bias, mask_init_value)

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.zeros(out_features, in_features))  # type: ignore

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))  # type: ignore
        else:
            self.register_parameter("bias", None)  # type: ignore

        self.reset_parameters()

        # Create a random tensor to reinit ablated parameters
        if self.ablation == "random_mask":
            self.random_weight = nn.Parameter(torch.zeros(self.weight.shape))
            init.kaiming_uniform_(self.random_weight, a=math.sqrt(5))
            self.random_weight.requires_grad = False

            if self.include_bias and self.bias is not None:
                self.random_bias = nn.Parameter(torch.zeros(self.bias.shape))
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.random_weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.random_bias, -bound, bound)
                self.random_bias.requires_grad = False

    @classmethod
    def from_layer(layer: nn.Linear, ablation, include_bias, mask_init_value):
        if layer.bias is not None:
            bias = True
        else:
            bias = False

        cont_sparse = ContSparseLinear(
            layer.in_feature,
            layer.out_features,
            bias,
            ablation,
            include_bias,
            mask_init_value,
        )
        cont_sparse.weight = layer.weight
        if bias:
            cont_sparse.bias = layer.bias

    def init_mask(self):
        self.weight_mask_params = nn.Parameter(torch.zeros(self.weight.shape))
        nn.init.constant_(self.weight_mask_params, self.mask_init_value)

        if self.include_bias and self.bias is not None:
            self.bias_mask_params = nn.Parameter(torch.zeros(self.bias.shape))
            nn.init.constant_(self.bias_mask_params, self.mask_init_value)

    def reset_parameters(self):
        """Reset network parameters."""
        self.init_mask()

        init.kaiming_uniform_(
            self.weight, a=math.sqrt(5)
        )  # Update Linear reset to match torch 1.12 https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:  # type: ignore
        """Perform the forward pass.
        Parameters
        ----------
        data : torch.Tensor
            N-dimensional tensor, with last dimension `in_features`
        Returns
        -------
        torch.Tensor
            N-dimensional tensor, with last dimension `out_features`
        """
        self.weight_mask = self.compute_mask("weight_mask_params")
        if self.ablation == "random":
            masked_weight = (
                self.weight * self.weight_mask
            )  # This will give you the inverse weights, 0's for ablated weights
            masked_weight += (
                ~self.weight_mask.bool()
            ).float() * self.random_weight  # Invert the mask to target the 0'd weights, make them random
        else:
            masked_weight = self.weight * self.weight_mask

        if self.include_bias and self.bias is not None:
            self.bias_mask = self.compute_mask("bias_mask_params")
            if self.ablation == "random":
                masked_bias = (
                    self.bias * self.bias_mask
                )  # This will give you the inverse weights, 0's for ablated weights
                masked_bias += (
                    ~self.bias_mask.bool()
                ).float() * self.random_bias  # Invert the mask to target the 0'd weights, make them random
            else:
                masked_bias = self.bias * self.bias_mask
        else:
            masked_bias = self.bias

        out = F.linear(data, masked_weight, masked_bias)
        return out


class ContSparseConv2D(ContSparseLayer):
    pass

class ContSparseConv1D(ContSparseLayer):
    pass