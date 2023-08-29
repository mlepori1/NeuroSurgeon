import math
import warnings
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers.pytorch_utils import Conv1D

from .mask_layer import MaskLayer


class MagPruneLayer(MaskLayer):
    """An abstract class defining the basic functionality of Magnitude Pruning layers.
    Magnitude pruning is not a differentiable masking strategy - it deterministically prunes the N% lowest magnitude weights.
    Masking always occurs at the weight-level. This pruning strategy might serve as a baseline to compare against gradient-based masking strategies (i.e. Continuous Sparsification or Hard Concrete Masking).
    This is a common strategy in model pruning, notably in work analyzing the Lottery Ticket Hypothesis (https://arxiv.org/abs/1803.03635)

    :param ablation: A string that determines how masks are produced from the mask layer parameters. Valid options include:

        - none: Producing a standard binary mask
        - zero_ablate: Inverting the standard binary mask. Used for pruning discovered subnetworks.
        - random_ablate: Inverting the standard binary mask and reinitializing zero'd elements. Used for pruning discovered subnetworks.
        - randomly_sampled: Sampling a random binary mask of the same size as the standard mask.
        - complement_sampled: Sampling a random binary mask of the same size as the standard mask from the complement set of entries as the standard mask.

    :type ablation: str
    :param mask_bias: Determines whether to mask bias terms in addition to weight terms.
    :type mask_bias: bool
    :param prune_percentage: Determines the percentage of weights to prune
    :type prune_percentage: float
    """

    def __init__(self, ablation: str, mask_bias: bool, prune_percentage: float):
        super().__init__(ablation, "weight", mask_bias)
        self.prune_percentage = prune_percentage
        self.force_resample = False

    @property
    def prune_percentage(self):
        return self._prune_percentage

    @prune_percentage.setter
    def prune_percentage(self, value):
        self._prune_percentage = value

    @property
    def force_resample(self):
        return self._force_resample

    @force_resample.setter
    def force_resample(self, value):
        self._force_resample = value

    def _sample_mask_from_complement(self, param_type):
        """Used to create a binary mask that contains the same number of ones and zeros as a normal ablated mask,
        but where the ablated parameters are drawn from the complement of the trained binary mask.
        This is done to assess whether ablating a trained subnetwork yields greater performance degredation than
        ablating a random subnetwork.

        Sample a random mask once and then use it to evaluate a whole dataset. Setting layer.force_resample = True
        forces the layer to resample a mask.
        """
        if hasattr(self, "sampled_" + param_type) and not self.force_resample:
            return getattr(self, "sampled_" + param_type)
        # Setting force_resample=True makes mask get resampled
        self.force_resample = False

        if param_type == "weight_mask_params":
            base_param = self.weight
        elif param_type == "bias_mask_params":
            base_param = self.bias
        else:
            raise ValueError(
                "Only weight_mask_params and bias_mask_params are supported param_types"
            )

        # Make sure that one can sample from the complement
        if self.prune_percentage <= 0.5:
            raise ValueError(
                "Trying to sample random masks, but original mask contains > 50 percent of weights"
            )

        # Get hard mask
        threshold = torch.quantile(base_param.reshape(-1), self.prune_percentage)
        mask = (base_param > threshold).float()
        sampled_size = torch.sum(mask).int()
        # Sample sample_size number of weights from the complement of the mask given by mask_weight
        complement_mask_weights = base_param < threshold
        sample_complement_indices = complement_mask_weights.nonzero(
            as_tuple=False
        )  # get indices of complement weights
        # shuffle the indices of possible sampled weights, take the first sample_size indices as your sampled mask
        sample_complement_indices = sample_complement_indices[
            torch.randperm(sample_complement_indices.size(0))
        ][:sampled_size]

        # Reformat indices into tuple form for indexing into tensor
        idxs = sample_complement_indices.shape[1]

        sample_complement_indices = [
            sample_complement_indices[:, idx] for idx in range(idxs)
        ]
        sample_complement_indices = tuple(sample_complement_indices)

        # Create a mask with just the sampled indices removed to compare ablating random subnetworks to ablating magnitude pruned subnetworks
        sampled_mask = torch.ones(base_param.shape)
        sampled_mask[sample_complement_indices] = 0.0
        sampled_mask = nn.Parameter(sampled_mask, requires_grad=False)

        # Get device of mask param and send parameter to it
        sampled_mask = sampled_mask.to(base_param.device)

        setattr(self, "sampled_" + param_type, sampled_mask)

        return sampled_mask

    def _sample_mask_randomly(self, param_type):
        """Used to create a binary mask that contains the same number of ones and zeros as a normal ablated mask,
        but where the ablated parameters are randomly drawn.
        This is done to assess whether ablating a trained subnetwork yields greater performance degredation than
        ablating a random subnetwork.

        Sample a random mask once and then use it to evaluate a whole dataset. Setting layer.force_resample = True
        forces the layer to resample a mask.
        """
        if hasattr(self, "sampled_" + param_type) and not self.force_resample:
            return getattr(self, "sampled_" + param_type)
        # Setting force_resample=True makes mask get resampled once
        self.force_resample = False

        if param_type == "weight_mask_params":
            base_param = self.weight
        elif param_type == "bias_mask_params":
            base_param = self.bias
        else:
            raise ValueError(
                "Only weight_mask_params and bias_mask_params are supported param_types"
            )

        # Get number of parameters to ablate
        threshold = torch.quantile(base_param.reshape(-1), self.prune_percentage)
        mask = (base_param > threshold).float()
        sampled_size = torch.sum(mask).int()

        # Get all indices
        all_indices_mask = torch.ones(base_param.shape)
        all_indices = all_indices_mask.nonzero(as_tuple=False)
        # shuffle the all indices, take the first sample_size indices as your sampled mask
        sampled_indices = all_indices[torch.randperm(all_indices.size(0))][
            :sampled_size
        ]

        # Reformat indices into tuple form for indexing into tensor
        idxs = sampled_indices.shape[1]

        sampled_indices = [sampled_indices[:, idx] for idx in range(idxs)]
        sampled_indices = tuple(sampled_indices)

        # Create a mask with just the sampled indices removed to compare ablating random subnetworks to ablating learned subnetworks
        sampled_mask = torch.ones(base_param.shape)
        sampled_mask[sampled_indices] = 0.0
        sampled_mask = nn.Parameter(sampled_mask, requires_grad=False)

        # Get device of mask param and send parameter to it
        sampled_mask = sampled_mask.to(base_param.device)

        setattr(self, "sampled_" + param_type, sampled_mask)

        return sampled_mask

    def _compute_mask(self, param_type):
        """This function maps mask_parameters to masks. The behavior of this function is determined
        by the ablation parameter.
        """
        if param_type == "weight_mask_params":
            base_param = self.weight
        elif param_type == "bias_mask_params":
            base_param = self.bias
        else:
            raise ValueError(
                "Only weight_mask_params and bias_mask_params are supported param_types"
            )

        if self.ablation == "none":
            threshold = torch.quantile(base_param.reshape(-1), self.prune_percentage)
            # Give the expected behavior of masking all weights at prune_percentage = 1.0, none at 0.0
            if self.prune_percentage == 1.0:
                mask = (base_param > threshold).float()
            else:
                mask = (base_param >= threshold).float()
        elif self.ablation == "complement_sampled":
            mask = self._sample_mask_from_complement(
                param_type
            )  # Generates a randomly sampled mask of equal size to trained mask from complement of subnetwork
        elif self.ablation == "randomly_sampled":
            mask = self._sample_mask_randomly(param_type)
        elif self.ablation != "none":
            threshold = torch.quantile(base_param.reshape(-1), self.prune_percentage)
            mask = (
                base_param < threshold
            ).float()  # Inverse hard mask for subnetwork ablation

        return mask

    @abstractmethod
    def _generate_random_values(self, param_type):
        # Used to create a mask of random values for random_ablate condition
        pass

    def _compute_random_ablation(self, param_type):
        """Computes the inverse of the standard binary mask and reinitializes zero'd elements. Used for pruning discovered subnetworks."""
        if param_type == "weight":
            params = self.weight
            params_mask = self.weight_mask
            random_params = self._generate_random_values("weight")
        elif param_type == "bias":
            params = self.bias
            params_mask = self.bias_mask
            random_params = self._generate_random_values("bias")
        else:
            raise ValueError("Only accepts weight and bias")

        masked_params = (
            params * params_mask
        )  # This will give you a mask with 0's for subnetwork weights
        masked_params += (
            ~params_mask.bool()
        ).float() * random_params  # Invert the mask to target the 0'd weights, make them random
        return masked_params


class MagPruneLinear(MagPruneLayer):
    """A Linear Layer that implements Magnitude Pruning.

    :param in_features: Size of each input sample
    :type in_features: int
    :param out_features: Size of each output sample
    :type out_features: int
    :param bias: If set to False, the layer will not learn an additive bias. Default: True
    :type bias: bool
    :param ablation: A string that determines how masks are produced from the mask layer parameters. Valid options include:

        - none: Producing a standard binary mask
        - zero_ablate: Inverting the standard binary mask. Used for pruning discovered subnetworks.
        - random_ablate: Inverting the standard binary mask and reinitializing zero'd elements. Used for pruning discovered subnetworks.
        - randomly_sampled: Sampling a random binary mask of the same size as the standard mask.
        - complement_sampled: Sampling a random binary mask of the same size as the standard mask from the complement set of entries as the standard mask.

    :type ablation: str
    :param mask_bias: Determines whether to mask bias terms in addition to weight terms. Default: False
    :type mask_bias: bool
    :param prune_percentage: The percentage of weights to prune. Default: 0.2
    :type prune_percentage: float
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        ablation: str = "none",
        mask_bias: bool = False,
        prune_percentage: float = 0.2,
    ):
        super().__init__(ablation, mask_bias, prune_percentage)

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))  # type: ignore

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))  # type: ignore
        else:
            if self.mask_bias:
                raise ValueError("Cannot mask bias if there is no bias")
            self.register_parameter("bias", None)  # type: ignore

        self.reset_parameters()

    @classmethod
    def from_layer(
        self,
        layer: nn.Linear,
        ablation: str = "none",
        mask_bias: bool = False,
        prune_percentage: float = 0.2,
    ):
        """Creates a MagPruneLinear layer from a nn.Linear layer.

        :param layer: An instance of a nn.Linear layer.
        :type layer: nn.Linear
        :param ablation: A string that determines how masks are produced from the mask layer parameters. Valid options include:

            - none: Producing a standard binary mask
            - zero_ablate: Inverting the standard binary mask. Used for pruning discovered subnetworks.
            - random_ablate: Inverting the standard binary mask and reinitializing zero'd elements. Used for pruning discovered subnetworks.
            - randomly_sampled: Sampling a random binary mask of the same size as the standard mask.
            - complement_sampled: Sampling a random binary mask of the same size as the standard mask from the complement set of entries as the standard mask.

        :type ablation: str
        :param mask_bias: Determines whether to mask bias terms in addition to weight terms. Default: False
        :type mask_bias: bool
        :param prune_percentage: The percentage of weights to prune. Default: 0.2
        :type prune_percentage: float

        :return: Magnitude Pruning Linear layer with the same weights as the layer argument
        :rtype: MagPruneLinear
        """
        if layer.bias is not None:
            bias = True
        else:
            bias = False

        if not bias and mask_bias:
            mask_bias = False
            warnings.warn(
                f"Cannot mask bias for layer {layer} because {layer} has no bias term"
            )

        mag_prune = MagPruneLinear(
            layer.in_features,
            layer.out_features,
            bias,
            ablation=ablation,
            mask_bias=mask_bias,
            prune_percentage=prune_percentage,
        )
        mag_prune.weight = layer.weight
        if bias:
            mag_prune.bias = layer.bias

        return mag_prune

    def _init_mask(self):
        # Included to conform to mask_layer abstract class
        pass

    def reset_parameters(self):
        """Reset network parameters."""
        init.kaiming_uniform_(
            self.weight, a=math.sqrt(5)
        )  # Update Linear reset to match torch 1.12 https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def _generate_random_values(self, param_type):
        if hasattr(self, "random_" + param_type):
            return getattr(self, "random_" + param_type)

        if param_type == "weight":
            self.random_weight = nn.Parameter(torch.zeros(self.weight.shape))
            init.kaiming_uniform_(self.random_weight, a=math.sqrt(5))
            self.random_weight.requires_grad = False
            return self.random_weight

        elif param_type == "bias":
            self.random_bias = nn.Parameter(torch.zeros(self.bias.shape))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.random_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.random_bias, -bound, bound)
            self.random_bias.requires_grad = False
            return self.random_bias

        else:
            raise ValueError("generate_random_values only supports weights and biases")

    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:  # type: ignore
        """Performs a forward pass

        :param data: Input tensors
        :type data: torch.Tensor
        :return: Output tensor
        :rtype: torch.Tensor
        """
        self.weight_mask = self._compute_mask("weight_mask_params")

        if not self.use_masks:
            masked_weight = self.weight
        elif self.ablation == "random_ablate":
            masked_weight = self._compute_random_ablation("weight")
        else:
            masked_weight = self.weight * self.weight_mask

        if self.mask_bias:
            self.bias_mask = self._compute_mask("bias_mask_params")
            if not self.use_masks:
                masked_bias = self.bias
            elif self.ablation == "random_ablate":
                masked_bias = self._compute_random_ablation("bias")
            else:
                masked_bias = self.bias * self.bias_mask
        else:
            masked_bias = self.bias

        out = F.linear(data, masked_weight, masked_bias)
        return out


class MagPruneGPTConv1D(MagPruneLayer):
    """A GPT-style Conv1D Layer that implements Magnitude Pruning.

    :param nf: Number of output features
    :type nf: int
    :param nx: Number of input features
    :type nx: int
    :param ablation: A string that determines how masks are produced from the mask layer parameters. Valid options include:

        - none: Producing a standard binary mask
        - zero_ablate: Inverting the standard binary mask. Used for pruning discovered subnetworks.
        - random_ablate: Inverting the standard binary mask and reinitializing zero'd elements. Used for pruning discovered subnetworks.
        - randomly_sampled: Sampling a random binary mask of the same size as the standard mask.
        - complement_sampled: Sampling a random binary mask of the same size as the standard mask from the complement set of entries as the standard mask.

    :type ablation: str
    :param mask_bias: Determines whether to mask bias terms in addition to weight terms. This has no effect on this layer, as there are no bias terms. Default: False
    :type mask_bias: bool
    :param prune_percentage: The percentage of weights to prune. Default: 0.2
    :type prune_percentage: float
    """

    def __init__(
        self,
        nf,
        nx,
        ablation: str = "none",
        mask_bias: bool = False,
        prune_percentage: float = 0.2,
    ):
        super().__init__(ablation, mask_bias, prune_percentage)

        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

        self._init_mask()

    @classmethod
    def from_layer(
        self,
        layer: Conv1D,
        ablation: str = "none",
        mask_bias: bool = False,
        prune_percentage: float = 0.2,
    ):
        """Creates a MagPruneGPTConv1D layer from a Conv1D layer.

        :param nf: Number of output features
        :type nf: int
        :param nx: Number of input features
        :type nx: int
        :param ablation: A string that determines how masks are produced from the mask layer parameters. Valid options include:

            - none: Producing a standard binary mask
            - zero_ablate: Inverting the standard binary mask. Used for pruning discovered subnetworks.
            - random_ablate: Inverting the standard binary mask and reinitializing zero'd elements. Used for pruning discovered subnetworks.
            - randomly_sampled: Sampling a random binary mask of the same size as the standard mask.
            - complement_sampled: Sampling a random binary mask of the same size as the standard mask from the complement set of entries as the standard mask.

        :type ablation: str
        :param mask_bias: Determines whether to mask bias terms in addition to weight terms. This has no effect on this layer, as there are no bias terms. Default: False
        :type mask_bias: bool
        :param prune_percentage: The percentage of weights to prune. Default: 0.2
        :type prune_percentage: float

        :return: Magnitude Pruning GPTConv1D layer with the same weights as the layer argument
        :rtype: MagPruneGPTConv1D
        """
        mag_prune = MagPruneGPTConv1D(
            layer.nf,
            layer.weight.shape[
                0
            ],  # For some reason, this layer doesn't store in_features as an attribute
            ablation=ablation,
            mask_bias=mask_bias,  # This class always has a bias term
            prune_percentage=prune_percentage,
        )
        mag_prune.weight = layer.weight
        mag_prune.bias = layer.bias

        return mag_prune

    def _init_mask(self):
        pass

    def _generate_random_values(self, param_type):
        if hasattr(self, "random_" + param_type):
            return getattr(self, "random_" + param_type)

        if param_type == "weight":
            self.random_weight = nn.Parameter(torch.zeros(self.weight.shape))
            nn.init.normal_(self.random_weight, std=0.02)
            self.random_weight.requires_grad = False
            return self.random_weight

        elif param_type == "bias":
            self.random_bias = nn.Parameter(torch.zeros(self.bias.shape))
            self.random_bias.requires_grad = False
            return self.random_bias

        else:
            raise ValueError("generate_random_values only supports weights and biases")

    def forward(self, x):
        """Performs a forward pass

        :param data: Input tensors
        :type data: torch.Tensor
        :return: Output tensor
        :rtype: torch.Tensor
        """
        self.weight_mask = self._compute_mask("weight_mask_params")

        if not self.use_masks:
            masked_weight = self.weight
        elif self.ablation == "random_ablate":
            masked_weight = self._compute_random_ablation("weight")
        else:
            masked_weight = self.weight * self.weight_mask

        if self.mask_bias:
            self.bias_mask = self._compute_mask("bias_mask_params")
            if not self.use_masks:
                masked_bias = self.bias
            elif self.ablation == "random_ablate":
                masked_bias = self._compute_random_ablation("bias")
            else:
                masked_bias = self.bias * self.bias_mask
        else:
            masked_bias = self.bias

        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(masked_bias, x.view(-1, x.size(-1)), masked_weight)
        x = x.view(size_out)
        return x


class _MagPruneConv(MagPruneLayer):
    """Abstract class used for both Conv1d and Conv2d Magnitude Pruning layers"""

    def __init__(
        self,
        layer_fn,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        ablation: str = "none",
        mask_bias: bool = False,
        prune_percentage: float = 0.2,
    ):
        super().__init__(ablation, mask_bias, prune_percentage)

        self._base_layer = layer_fn(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.weight = self._base_layer.weight
        if bias:
            self.bias = self._base_layer.bias
        else:
            if self.mask_bias:
                raise ValueError("Cannot mask bias if there is no bias")
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self._base_layer.reset_parameters()
        self.weight = self._base_layer.weight
        self.bias = self._base_layer.bias

    def _init_mask(self):
        pass

    def _generate_random_values(self, param_type):
        # Create a random tensor to reinit ablated parameters
        if hasattr(self, "random_" + param_type):
            return getattr(self, "random_" + param_type)

        if param_type == "weight":
            self.random_weight = nn.Parameter(
                torch.empty(self._base_layer.weight.size())
            )
            init.kaiming_uniform_(self.random_weight, a=math.sqrt(5))
            self.random_weight.requires_grad = False
            return self.random_weight

        elif param_type == "bias":
            self.random_bias = nn.Parameter(torch.empty(self._base_layer.bias.size()))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.random_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.random_bias, -bound, bound)
            self.random_bias.requires_grad = False
            return self.random_bias

        else:
            raise ValueError("generate_random_values only supports weights and biases")

    def forward(self, x):
        self.weight_mask = self._compute_mask("weight_mask_params")

        if not self.use_masks:
            masked_weight = self.weight
        elif self.ablation == "random_ablate":
            masked_weight = self._compute_random_ablation("weight")
        else:
            masked_weight = self.weight * self.weight_mask

        if self.mask_bias:
            self.bias_mask = self._compute_mask("bias_mask_params")
            if not self.use_masks:
                masked_bias = self.bias
            elif self.ablation == "random_ablate":
                masked_bias = self._compute_random_ablation("bias")
            else:
                masked_bias = self.bias * self.bias_mask
        else:
            masked_bias = self.bias

        out = self._base_layer._conv_forward(x, masked_weight, masked_bias)
        return out


class MagPruneConv2d(_MagPruneConv):
    """A Conv2d layer that implements Magnitude Pruning.

    :param in_channels: Number of channels in the input image
    :type in_channels: int
    :param out_channels: Number of channels produced by the convolution
    :type out_channels: int
    :param kernel_size: Size of the convolving kernel
    :type kernel_size: int or tuple
    :param padding: Padding added to all four sides of the input. Default: 0
    :type padding: int
    :param stride:  Stride of the convolution. Default: 1
    :type stride: int
    :param dilation: Spacing between kernel elements. Default: 1
    :type dilation: int or tuple
    :param groups:  Number of blocked connections from input channels to output channels. Default: 1
    :type groups: int
    :param bias: If True, adds a learnable bias to the output. Default: True
    :type bias: bool
    :param padding_mode:  'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    :type padding_mode:  str
    :param ablation: A string that determines how masks are produced from the mask layer parameters. Valid options include:

        - none: Producing a standard binary mask
        - zero_ablate: Inverting the standard binary mask. Used for pruning discovered subnetworks.
        - random_ablate: Inverting the standard binary mask and reinitializing zero'd elements. Used for pruning discovered subnetworks.
        - randomly_sampled: Sampling a random binary mask of the same size as the standard mask.
        - complement_sampled: Sampling a random binary mask of the same size as the standard mask from the complement set of entries as the standard mask.

    :type ablation: str
    :param mask_bias: Determines whether to mask bias terms in addition to weight terms. Default: False
    :type mask_bias: bool
    :param prune_percentage: The percentage of weights to prune. Default: 0.2
    :type prune_percentage: float
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        ablation: str = "none",
        mask_bias: bool = False,
        prune_percentage: float = 0.2,
    ):
        layer_fn = nn.Conv2d
        super().__init__(
            layer_fn,
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            ablation=ablation,
            mask_bias=mask_bias,
            prune_percentage=prune_percentage,
        )

    @classmethod
    def from_layer(
        self,
        layer: nn.Conv2d,
        ablation: str = "none",
        mask_bias: bool = False,
        prune_percentage: float = 0.2,
    ):
        """Create a MagPruneConv2d layer from a nn.Conv2d layer

        :param layer: A nn.Conv2d layer
        :type layer: nn.Conv2d
        :param ablation: A string that determines how masks are produced from the mask layer parameters. Valid options include:

            - none: Producing a standard binary mask
            - zero_ablate: Inverting the standard binary mask. Used for pruning discovered subnetworks.
            - random_ablate: Inverting the standard binary mask and reinitializing zero'd elements. Used for pruning discovered subnetworks.
            - randomly_sampled: Sampling a random binary mask of the same size as the standard mask.
            - complement_sampled: Sampling a random binary mask of the same size as the standard mask from the complement set of entries as the standard mask.

        :type ablation: str
        :param mask_bias: Determines whether to mask bias terms in addition to weight terms. Default: False
        :type mask_bias: bool
        :param prune_percentage: The percentage of weights to prune. Default: 0.2
        :type prune_percentage: float

        :return: Magnitude Pruning Conv2d layer with the same weights as the layer argument
        :rtype: MagPruneConv2d
        """
        if layer.bias is not None:
            bias = True
        else:
            bias = False

        if not bias and mask_bias:
            mask_bias = False
            warnings.warn(
                f"Cannot mask bias for layer {layer} because {layer} has no bias term"
            )

        mag_prune = MagPruneConv2d(
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=bias,
            padding_mode=layer.padding_mode,
            ablation=ablation,
            mask_bias=mask_bias,
            prune_percentage=prune_percentage,
        )
        mag_prune.weight = layer.weight
        if bias:
            mag_prune.bias = layer.bias

        return mag_prune


class MagPruneConv1d(_MagPruneConv):
    """A Conv1d layer that implements Magnitude Pruning.

    :param in_channels: Number of channels in the input image
    :type in_channels: int
    :param out_channels: Number of channels produced by the convolution
    :type out_channels: int
    :param kernel_size: Size of the convolving kernel
    :type kernel_size: int or tuple
    :param padding: Padding added to all four sides of the input. Default: 0
    :type padding: int
    :param stride:  Stride of the convolution. Default: 1
    :type stride: int
    :param dilation: Spacing between kernel elements. Default: 1
    :type dilation: int or tuple
    :param groups:  Number of blocked connections from input channels to output channels. Default: 1
    :type groups: int
    :param bias: If True, adds a learnable bias to the output. Default: True
    :type bias: bool
    :param padding_mode:  'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    :type padding_mode:  str
    :param ablation: A string that determines how masks are produced from the mask layer parameters. Valid options include:

        - none: Producing a standard binary mask
        - zero_ablate: Inverting the standard binary mask. Used for pruning discovered subnetworks.
        - random_ablate: Inverting the standard binary mask and reinitializing zero'd elements. Used for pruning discovered subnetworks.
        - randomly_sampled: Sampling a random binary mask of the same size as the standard mask.
        - complement_sampled: Sampling a random binary mask of the same size as the standard mask from the complement set of entries as the standard mask.

    :type ablation: str
    :param mask_bias: Determines whether to mask bias terms in addition to weight terms. Default: False
    :type mask_bias: bool
    :param prune_percentage: The percentage of weights to prune. Default: 0.2
    :type prune_percentage: float
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        ablation: str = "none",
        mask_bias: bool = False,
        prune_percentage: float = 0.2,
    ):
        layer_fn = nn.Conv1d
        super().__init__(
            layer_fn,
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            ablation=ablation,
            mask_bias=mask_bias,
            prune_percentage=prune_percentage,
        )

    @classmethod
    def from_layer(self, layer: nn.Conv1d, ablation, mask_bias, prune_percentage):
        """Create a MagPruneConv1d layer from a nn.Conv1d layer

        :param layer: A nn.Conv1d layer
        :type layer: nn.Conv1d
        :param ablation: A string that determines how masks are produced from the mask layer parameters. Valid options include:

            - none: Producing a standard binary mask
            - zero_ablate: Inverting the standard binary mask. Used for pruning discovered subnetworks.
            - random_ablate: Inverting the standard binary mask and reinitializing zero'd elements. Used for pruning discovered subnetworks.
            - randomly_sampled: Sampling a random binary mask of the same size as the standard mask.
            - complement_sampled: Sampling a random binary mask of the same size as the standard mask from the complement set of entries as the standard mask.

        :type ablation: str
        :param mask_bias: Determines whether to mask bias terms in addition to weight terms. Default: False
        :type mask_bias: bool
        :param prune_percentage: The percentage of weights to prune. Default: 0.2
        :type prune_percentage: float

        :return: Magnitude Pruning Conv1d layer with the same weights as the layer argument
        :rtype: MagPruneConv1d
        """
        if layer.bias is not None:
            bias = True
        else:
            bias = False

        if not bias and mask_bias:
            mask_bias = False
            warnings.warn(
                f"Cannot mask bias for layer {layer} because {layer} has no bias term"
            )

        mag_prune = MagPruneConv1d(
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=bias,
            padding_mode=layer.padding_mode,
            ablation=ablation,
            mask_bias=mask_bias,
            prune_percentage=prune_percentage,
        )
        mag_prune.weight = layer.weight
        if bias:
            mag_prune.bias = layer.bias

        return mag_prune
