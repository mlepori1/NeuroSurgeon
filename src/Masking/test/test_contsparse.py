from ..contsparse_layer import (
    ContSparseLinear,
    ContSparseConv1d,
    ContSparseConv2d,
    ContSparseGPTConv1D,
)
import torch
import pytest
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

### Test ContSparseLinear Layer


def test_linear_init():
    mask_fill_val = 19.0
    layer = ContSparseLinear(
        10,
        10,
        bias=True,
        ablation="none",
        mask_bias=True,
        mask_init_value=mask_fill_val,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "weight_mask_params")
    assert hasattr(layer, "bias")
    assert hasattr(layer, "bias_mask_params")
    assert torch.all(
        layer.weight_mask_params
        == torch.full(layer.weight_mask_params.shape, mask_fill_val)
    )
    assert torch.all(
        layer.bias_mask_params
        == torch.full(layer.bias_mask_params.shape, mask_fill_val)
    )

    layer = ContSparseLinear(
        10,
        10,
        bias=True,
        ablation="none",
        mask_bias=False,
        mask_init_value=mask_fill_val,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "weight_mask_params")
    assert hasattr(layer, "bias")
    assert not hasattr(layer, "bias_mask_params")
    assert torch.all(
        layer.weight_mask_params
        == torch.full(layer.weight_mask_params.shape, mask_fill_val)
    )

    layer = ContSparseLinear(
        10,
        10,
        bias=False,
        ablation="none",
        mask_bias=False,
        mask_init_value=mask_fill_val,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "weight_mask_params")
    assert layer.bias is None
    assert not hasattr(layer, "bias_mask_params")
    assert torch.all(
        layer.weight_mask_params
        == torch.full(layer.weight_mask_params.shape, mask_fill_val)
    )

    # Assert that you cannot mask a bias that isn't there
    with pytest.raises(Exception):
        layer = ContSparseLinear(
            10,
            10,
            bias=False,
            ablation="none",
            mask_bias=True,
            mask_init_value=mask_fill_val,
        )


def test_linear_from_layer():
    base_layer = nn.Linear(10, 20)
    masked_layer = ContSparseLinear.from_layer(
        base_layer,
        ablation="none",
        mask_unit="weight",
        mask_bias=True,
        mask_init_value=1.0,
    )

    # Ensure that masked layer is initialized correctly
    assert hasattr(masked_layer, "weight_mask_params")
    assert hasattr(masked_layer, "bias_mask_params")

    ipt_tensor = torch.Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    base_out = base_layer(ipt_tensor)

    # Assert that test mode masked_layer produces same output as base layer (because init_value is > 0)
    masked_layer.train(False)
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out == base_out)

    # Assert that train mode masked_layer produces diff output as base layer
    masked_layer.train(True)
    masked_out = masked_layer(ipt_tensor)
    assert not torch.all(masked_out == base_out)

    masked_layer = ContSparseLinear.from_layer(
        base_layer,
        ablation="none",
        mask_unit="weight",
        mask_bias=True,
        mask_init_value=-1.0,
    )

    # Assert that test mode masked_layer produces diff output as base layer (because init_value is < 0)
    masked_layer.train(False)
    masked_out = masked_layer(ipt_tensor)
    assert not torch.all(masked_out == base_out)

    # Assert that train mode masked_layer produces diff output as base layer
    masked_layer.train(True)
    masked_out = masked_layer(ipt_tensor)
    assert not torch.all(masked_out == base_out)


def test_linear_use_mask():
    base_layer = nn.Linear(10, 20)
    masked_layer = ContSparseLinear.from_layer(
        base_layer,
        ablation="none",
        mask_unit="weight",
        mask_bias=True,
        mask_init_value=-1.0,
    )

    ipt_tensor = torch.Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    base_out = base_layer(ipt_tensor)

    # Assert that test mode masked_layer produces different output as base layer (because init_value is < 0)
    masked_layer.train(False)
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out != base_out)

    # Assert that turning off masks gives same output as base layer
    masked_layer.use_masks = False
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out == base_out)


def test_linear_zero_ablate():
    layer = ContSparseLinear(
        10, 10, bias=True, ablation="zero_ablate", mask_bias=False, mask_init_value=0.0
    )
    ipt = torch.ones(10)
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.5
    )
    layer.train(False)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))
    # Assert that positive params are masked and negative ones are not
    to_ablate = layer.weight_mask_params > 0
    assert torch.all(layer.weight_mask == ~to_ablate)


def test_linear_random_ablate():
    layer = ContSparseLinear(
        10,
        10,
        bias=True,
        ablation="random_ablate",
        mask_bias=False,
        mask_init_value=0.0,
    )
    ipt = torch.ones(10)
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.5
    )
    layer.train(False)
    rand_out = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))
    # Assert that positive params are masked and negative ones are not
    to_ablate = layer.weight_mask_params > 0
    assert torch.all(layer.weight_mask == ~to_ablate)

    # Assert that random ablation is different than zero ablation
    layer = ContSparseLinear(
        10, 10, bias=True, ablation="zero_ablate", mask_bias=False, mask_init_value=0.0
    )
    layer.train(False)
    zero_out = layer(ipt)  # Weight mask computed during forward pass
    assert torch.all(rand_out != zero_out)


def test_linear_randomly_sampled_ablate():
    layer = ContSparseLinear(
        10,
        10,
        bias=True,
        ablation="randomly_sampled",
        mask_bias=False,
        mask_init_value=0.0,
    )
    ipt = torch.ones(10)
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.6
    )
    layer.train(False)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))

    # Assert that the layer is masking out the same number of parameters as are in the subnetwork
    subnetwork_params = layer.weight_mask_params > 0
    assert torch.sum(~layer.weight_mask.bool()) == torch.sum(subnetwork_params)

    # Ensure no error is thrown if the subnetwork params contain >50% of parameters
    layer = ContSparseLinear(
        10,
        10,
        bias=True,
        ablation="randomly_sampled",
        mask_bias=False,
        mask_init_value=0.0,
    )
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.3
    )
    layer.train(False)
    _ = layer(ipt)


def test_linear_complement_sampled_ablate():
    layer = ContSparseLinear(
        10,
        10,
        bias=True,
        ablation="complement_sampled",
        mask_bias=False,
        mask_init_value=0.0,
    )
    ipt = torch.ones(10)
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.6
    )
    layer.train(False)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))

    # Assert that all subnetwork parameters are unmasked
    subnetwork_params = layer.weight_mask_params > 0
    assert torch.all(layer.weight_mask.bool()[subnetwork_params])

    # Assert that the layer is masking out the same number of parameters as are in the subnetwork
    assert torch.sum(~layer.weight_mask.bool()) == torch.sum(subnetwork_params)

    # Assert that an error is thrown if the subnetwork params contain >50% of parameters
    layer = ContSparseLinear(
        10,
        10,
        bias=True,
        ablation="complement_sampled",
        mask_bias=False,
        mask_init_value=0.0,
    )
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.3
    )
    layer.train(False)
    with pytest.raises(Exception):
        _ = layer(ipt)  # Weight mask computed during forward pass


def test_linear_masking():
    # Test that masking every param gives the zero vector
    layer = ContSparseLinear(
        10, 10, bias=True, ablation="none", mask_bias=True, mask_init_value=-0.1
    )
    ipt = torch.ones(10)
    layer.train(False)
    out = layer(ipt)
    assert torch.all(out == torch.zeros(out.shape))

    # Test that masking every weight param gives the bias vector
    layer = ContSparseLinear(
        10, 10, bias=True, ablation="none", mask_bias=False, mask_init_value=-0.1
    )
    ipt = torch.ones(10)
    layer.train(False)
    out = layer(ipt)
    assert torch.all(out == layer.bias)

    # Test that masking one full neuron gives a particular zero entry
    layer = ContSparseLinear(
        10, 15, bias=True, ablation="none", mask_bias=True, mask_init_value=1.0
    )

    w_params = layer.weight_mask_params
    w_params.requires_grad = False
    w_params[5, :] = torch.zeros(10)
    w_params.requires_grad = True
    layer.weight_mask_params = w_params

    b_params = layer.bias_mask_params
    b_params.requires_grad = False
    b_params[5] = 0
    b_params.requires_grad = True
    layer.bias_mask_params = b_params

    ipt = torch.ones(10)
    layer.train(False)
    out = layer(ipt)
    assert torch.all(out[5] == 0.0)


def test_linear_l0_calc():
    layer = ContSparseLinear(
        10, 15, bias=True, ablation="none", mask_bias=True, mask_init_value=1.0
    )
    l_train_l0 = layer.calculate_l0()
    l_train_l0_max = layer.calculate_max_l0()

    # Assert that training L0 calculation is working
    assert l_train_l0.item().is_integer() == False  # Mask entries should not be binary
    assert l_train_l0 != l_train_l0_max

    # Assert that test L0 calculation is working
    layer.train(False)
    l_test_l0 = layer.calculate_l0()
    l_test_l0_max = layer.calculate_max_l0()
    assert l_test_l0.item().is_integer()  # Mask entries should be binary
    assert l_test_l0 == l_test_l0_max
    assert l_test_l0 != l_train_l0


def test_linear_temperature():
    layer = ContSparseLinear(
        10, 10, bias=True, ablation="none", mask_bias=True, mask_init_value=1.0
    )

    # Assert that changing the temperature changes JUST the train mask
    train_l0_1 = layer.calculate_l0()
    layer.train(False)
    test_l0_1 = layer.calculate_l0()

    layer.train(True)
    layer.temperature = 2.0
    train_l0_2 = layer.calculate_l0()
    layer.train(False)
    test_l0_2 = layer.calculate_l0()

    assert train_l0_1 != train_l0_2
    assert test_l0_1 == test_l0_2

    # Assert that driving up the temperature past floating point precision makes train and test masks the same
    layer.temperature = 1000
    train_l0 = layer.calculate_l0()
    layer.train(False)
    test_l0 = layer.calculate_l0()

    assert train_l0 == test_l0


def test_linear_neuron_pruning():
    # Assert that masking out a single element in the mask matrix knocks out a full neuron
    layer = ContSparseLinear(
        10,
        20,
        bias=True,
        ablation="none",
        mask_unit="neuron",
        mask_bias=True,
        mask_init_value=1.0,
    )

    w_params = layer.weight_mask_params
    w_params.requires_grad = False
    w_params[5] = 0.0
    w_params.requires_grad = True
    layer.weight_mask_params = w_params

    b_params = layer.bias_mask_params
    b_params.requires_grad = False
    b_params[5] = 0
    b_params.requires_grad = True
    layer.bias_mask_params = b_params

    ipt = torch.ones(10)
    layer.train(False)
    out = layer(ipt)
    assert torch.all(out[5] == 0.0)


def test_linear_neuron_l0():
    # Assert that L0 norm is always a multiple of the # of input parameters
    layer = ContSparseLinear(
        10,
        20,
        bias=True,
        ablation="none",
        mask_unit="neuron",
        mask_bias=False,
        mask_init_value=1.0,
    )

    w_params = layer.weight_mask_params
    w_params.requires_grad = False
    w_params[0] = 0.0
    w_params.requires_grad = True
    layer.weight_mask_params = w_params
    layer.train(False)

    assert layer.calculate_l0() == 190

    w_params = layer.weight_mask_params
    w_params.requires_grad = False
    w_params[1] = 0.0
    w_params.requires_grad = True
    layer.weight_mask_params = w_params
    layer.train(False)

    assert layer.calculate_l0() == 180

    w_params = layer.weight_mask_params
    w_params.requires_grad = False
    w_params[2] = 0.0
    w_params.requires_grad = True
    layer.weight_mask_params = w_params
    layer.train(False)

    assert layer.calculate_l0() == 170


### Test ContSparseGPTConv1d Layer


def test_gptconv1d_init():
    mask_fill_val = 19.0
    layer = ContSparseGPTConv1D(
        10, 10, ablation="none", mask_bias=True, mask_init_value=mask_fill_val
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "weight_mask_params")
    assert hasattr(layer, "bias")
    assert hasattr(layer, "bias_mask_params")
    assert torch.all(
        layer.weight_mask_params
        == torch.full(layer.weight_mask_params.shape, mask_fill_val)
    )
    assert torch.all(
        layer.bias_mask_params
        == torch.full(layer.bias_mask_params.shape, mask_fill_val)
    )

    layer = ContSparseGPTConv1D(
        10, 10, ablation="none", mask_bias=False, mask_init_value=mask_fill_val
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "weight_mask_params")
    assert hasattr(layer, "bias")
    assert not hasattr(layer, "bias_mask_params")
    assert torch.all(
        layer.weight_mask_params
        == torch.full(layer.weight_mask_params.shape, mask_fill_val)
    )


def test_gptconv1d_from_layer():
    base_layer = Conv1D(
        20, 10
    )  # For some reason, out_features comes first in this classes constructor...
    masked_layer = ContSparseGPTConv1D.from_layer(
        base_layer,
        ablation="none",
        mask_unit="weight",
        mask_bias=True,
        mask_init_value=1.0,
    )

    # Ensure that masked layer is initialized correctly
    assert hasattr(masked_layer, "weight_mask_params")
    assert hasattr(masked_layer, "bias_mask_params")

    ipt_tensor = torch.Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    base_out = base_layer(ipt_tensor)

    # Assert that test mode masked_layer produces same output as base layer (because init_value is > 0)
    masked_layer.train(False)
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out == base_out)

    # Assert that train mode masked_layer produces diff output as base layer
    masked_layer.train(True)
    masked_out = masked_layer(ipt_tensor)
    assert not torch.all(masked_out == base_out)

    masked_layer = ContSparseGPTConv1D.from_layer(
        base_layer,
        ablation="none",
        mask_unit="weight",
        mask_bias=True,
        mask_init_value=-1.0,
    )

    # Assert that test mode masked_layer produces diff output as base layer (because init_value is < 0)
    masked_layer.train(False)
    masked_out = masked_layer(ipt_tensor)
    assert not torch.all(masked_out == base_out)

    # Assert that train mode masked_layer produces diff output as base layer
    masked_layer.train(True)
    masked_out = masked_layer(ipt_tensor)
    assert not torch.all(masked_out == base_out)


def test_gptconv1d_use_mask():
    base_layer = Conv1D(
        20, 10
    )  # For some reason, out_features comes first in this classes constructor...
    masked_layer = ContSparseGPTConv1D.from_layer(
        base_layer,
        ablation="none",
        mask_unit="weight",
        mask_bias=True,
        mask_init_value=-1.0,
    )

    ipt_tensor = torch.Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    base_out = base_layer(ipt_tensor)

    # Assert that test mode masked_layer produces different output as base layer (because init_value is < 0)
    masked_layer.train(False)
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out != base_out)

    # Assert that turning off masks gives same output as base layer
    masked_layer.use_masks = False
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out == base_out)


def test_gptconv1d_zero_ablate():
    layer = ContSparseGPTConv1D(
        10, 10, ablation="zero_ablate", mask_bias=False, mask_init_value=0.0
    )
    ipt = torch.ones(10)
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.5
    )
    layer.train(False)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))
    # Assert that positive params are masked and negative ones are not
    to_ablate = layer.weight_mask_params > 0
    assert torch.all(layer.weight_mask == ~to_ablate)


def test_gptconv1d_random_ablate():
    layer = ContSparseGPTConv1D(
        10, 10, ablation="random_ablate", mask_bias=False, mask_init_value=0.0
    )
    ipt = torch.ones(10)
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.5
    )
    layer.train(False)
    rand_out = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))
    # Assert that positive params are masked and negative ones are not
    to_ablate = layer.weight_mask_params > 0
    assert torch.all(layer.weight_mask == ~to_ablate)

    # Assert that random ablation is different than zero ablation
    layer = ContSparseGPTConv1D(
        10, 10, ablation="zero_ablate", mask_bias=False, mask_init_value=0.0
    )
    layer.train(False)
    zero_out = layer(ipt)  # Weight mask computed during forward pass
    assert torch.all(rand_out != zero_out)


def test_gptconv1d_randomly_sampled_ablate():
    layer = ContSparseGPTConv1D(
        10, 10, ablation="randomly_sampled", mask_bias=False, mask_init_value=0.0
    )
    ipt = torch.ones(10)
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.6
    )
    layer.train(False)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))

    # Assert that the layer is masking out the same number of parameters as are in the subnetwork
    subnetwork_params = layer.weight_mask_params > 0
    assert torch.sum(~layer.weight_mask.bool()) == torch.sum(subnetwork_params)

    # Assert that no error is thrown if the subnetwork params contain >50% of parameters
    layer = ContSparseGPTConv1D(
        10, 10, ablation="randomly_sampled", mask_bias=False, mask_init_value=0.0
    )
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.3
    )
    layer.train(False)
    _ = layer(ipt)  # Weight mask computed during forward pass


def test_gptconv1d_complement_sampled_ablate():
    layer = ContSparseGPTConv1D(
        10, 10, ablation="complement_sampled", mask_bias=False, mask_init_value=0.0
    )
    ipt = torch.ones(10)
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.6
    )
    layer.train(False)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))

    # Assert that all subnetwork parameters are unmasked
    subnetwork_params = layer.weight_mask_params > 0
    assert torch.all(layer.weight_mask.bool()[subnetwork_params])

    # Assert that the layer is masking out the same number of parameters as are in the subnetwork
    assert torch.sum(~layer.weight_mask.bool()) == torch.sum(subnetwork_params)

    # Assert that an error is thrown if the subnetwork params contain >50% of parameters
    layer = ContSparseGPTConv1D(
        10, 10, ablation="complement_sampled", mask_bias=False, mask_init_value=0.0
    )
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.3
    )
    layer.train(False)
    with pytest.raises(Exception):
        _ = layer(ipt)  # Weight mask computed during forward pass


def test_gptconv1d_masking():
    # Test that masking every param gives the zero vector
    layer = ContSparseGPTConv1D(
        10, 10, ablation="none", mask_bias=True, mask_init_value=-0.1
    )
    ipt = torch.ones(10)
    layer.train(False)
    out = layer(ipt)
    assert torch.all(out == torch.zeros(out.shape))

    # Test that masking every weight param gives the bias vector
    layer = ContSparseGPTConv1D(
        10, 10, ablation="none", mask_bias=False, mask_init_value=-0.1
    )
    ipt = torch.ones(10)
    layer.train(False)
    out = layer(ipt)
    assert torch.all(
        out == layer.bias
    )  # Bias is initialized at 0's anyway for GPTConv1D, so kind of redundant...

    # Test that masking one neuron gives a particular zero entry
    layer = ContSparseGPTConv1D(
        15, 10, ablation="none", mask_bias=True, mask_init_value=1.0
    )

    w_params = layer.weight_mask_params
    w_params.requires_grad = False
    w_params[:, 5] = torch.zeros(10)
    w_params.requires_grad = True
    layer.weight_mask_params = w_params

    b_params = layer.bias_mask_params
    b_params.requires_grad = False
    b_params[5] = 0
    b_params.requires_grad = True
    layer.bias_mask_params = b_params

    ipt = torch.ones(10)
    layer.train(False)
    out = layer(ipt)
    assert torch.all(out[5] == 0.0)


def test_gptconv1d_l0_calc():
    layer = ContSparseGPTConv1D(
        10, 15, ablation="none", mask_bias=True, mask_init_value=1.0
    )
    l_train_l0 = layer.calculate_l0()
    l_train_l0_max = layer.calculate_max_l0()

    # Assert that training L0 calculation is working
    assert l_train_l0.item().is_integer() == False  # Mask entries should not be binary
    assert l_train_l0 != l_train_l0_max

    # Assert that test L0 calculation is working
    layer.train(False)
    l_test_l0 = layer.calculate_l0()
    l_test_l0_max = layer.calculate_max_l0()
    assert l_test_l0.item().is_integer()  # Mask entries should be binary
    assert l_test_l0 == l_test_l0_max
    assert l_test_l0 != l_train_l0


def test_gptconv1d_temperature():
    layer = ContSparseGPTConv1D(
        10, 10, ablation="none", mask_bias=True, mask_init_value=1.0
    )

    # Assert that changing the temperature changes JUST the train mask
    train_l0_1 = layer.calculate_l0()
    layer.train(False)
    test_l0_1 = layer.calculate_l0()

    layer.train(True)
    layer.temperature = 2.0
    train_l0_2 = layer.calculate_l0()
    layer.train(False)
    test_l0_2 = layer.calculate_l0()

    assert train_l0_1 != train_l0_2
    assert test_l0_1 == test_l0_2

    # Assert that driving up the temperature past floating point precision makes train and test masks the same
    layer.temperature = 1000
    train_l0 = layer.calculate_l0()
    layer.train(False)
    test_l0 = layer.calculate_l0()

    assert train_l0 == test_l0


def test_gptconv1d_neuron_pruning():
    # Assert that masking out a single element in the mask matrix knocks out a full neuron
    layer = ContSparseGPTConv1D(
        20, 10, ablation="none", mask_unit="neuron", mask_bias=True, mask_init_value=1.0
    )

    w_params = layer.weight_mask_params
    w_params.requires_grad = False
    w_params[0, 5] = 0.0
    w_params.requires_grad = True
    layer.weight_mask_params = w_params

    b_params = layer.bias_mask_params
    b_params.requires_grad = False
    b_params[5] = 0
    b_params.requires_grad = True
    layer.bias_mask_params = b_params

    ipt = torch.ones(10)
    layer.train(False)
    out = layer(ipt)
    assert torch.all(out[5] == 0.0)


def test_gptconv1d_neuron_l0():
    # Assert that L0 norm is always a multiple of the # of input parameters
    layer = ContSparseGPTConv1D(
        10,
        20,
        ablation="none",
        mask_unit="neuron",
        mask_bias=False,
        mask_init_value=1.0,
    )

    w_params = layer.weight_mask_params
    w_params.requires_grad = False
    w_params[0, 0] = 0.0
    w_params.requires_grad = True
    layer.weight_mask_params = w_params
    layer.train(False)

    assert layer.calculate_l0() == 180

    w_params = layer.weight_mask_params
    w_params.requires_grad = False
    w_params[0, 1] = 0.0
    w_params.requires_grad = True
    layer.weight_mask_params = w_params
    layer.train(False)

    assert layer.calculate_l0() == 160

    w_params = layer.weight_mask_params
    w_params.requires_grad = False
    w_params[0, 2] = 0.0
    w_params.requires_grad = True
    layer.weight_mask_params = w_params
    layer.train(False)

    assert layer.calculate_l0() == 140


### Test ContSparseConv1d Layer


def test_conv1d_init():
    mask_fill_val = 19.0
    layer = ContSparseConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="none",
        mask_bias=True,
        mask_init_value=mask_fill_val,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "weight_mask_params")
    assert hasattr(layer, "bias")
    assert hasattr(layer, "bias_mask_params")
    assert torch.all(
        layer.weight_mask_params
        == torch.full(layer.weight_mask_params.shape, mask_fill_val)
    )
    assert torch.all(
        layer.bias_mask_params
        == torch.full(layer.bias_mask_params.shape, mask_fill_val)
    )

    layer = ContSparseConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="none",
        mask_bias=False,
        mask_init_value=mask_fill_val,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "weight_mask_params")
    assert hasattr(layer, "bias")
    assert not hasattr(layer, "bias_mask_params")
    assert torch.all(
        layer.weight_mask_params
        == torch.full(layer.weight_mask_params.shape, mask_fill_val)
    )

    layer = ContSparseConv1d(
        1,
        10,
        3,
        bias=False,
        ablation="none",
        mask_bias=False,
        mask_init_value=mask_fill_val,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "weight_mask_params")
    assert layer.bias is None
    assert not hasattr(layer, "bias_mask_params")
    assert torch.all(
        layer.weight_mask_params
        == torch.full(layer.weight_mask_params.shape, mask_fill_val)
    )

    # Assert that you cannot mask a bias that isn't there
    with pytest.raises(Exception):
        layer = ContSparseConv1d(
            1,
            10,
            3,
            bias=False,
            ablation="none",
            mask_bias=True,
            mask_init_value=mask_fill_val,
        )


def test_conv1d_from_layer():
    base_layer = nn.Conv1d(1, 10, 3)
    masked_layer = ContSparseConv1d.from_layer(
        base_layer,
        ablation="none",
        mask_unit="weight",
        mask_bias=True,
        mask_init_value=1.0,
    )

    # Ensure that masked layer is initialized correctly
    assert hasattr(masked_layer, "weight_mask_params")
    assert hasattr(masked_layer, "bias_mask_params")

    ipt_tensor = torch.Tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0]])
    base_out = base_layer(ipt_tensor)

    # Assert that test mode masked_layer produces same output as base layer (because init_value is > 0)
    masked_layer.train(False)
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out == base_out)

    # Assert that train mode masked_layer produces diff output as base layer
    masked_layer.train(True)
    masked_out = masked_layer(ipt_tensor)
    assert not torch.all(masked_out == base_out)

    masked_layer = ContSparseConv1d.from_layer(
        base_layer,
        ablation="none",
        mask_unit="weight",
        mask_bias=True,
        mask_init_value=-1.0,
    )

    # Assert that test mode masked_layer produces diff output as base layer (because init_value is < 0)
    masked_layer.train(False)
    masked_out = masked_layer(ipt_tensor)
    assert not torch.all(masked_out == base_out)

    # Assert that train mode masked_layer produces diff output as base layer
    masked_layer.train(True)
    masked_out = masked_layer(ipt_tensor)
    assert not torch.all(masked_out == base_out)


def test_conv1d_use_mask():
    base_layer = nn.Conv1d(1, 10, 3)
    masked_layer = ContSparseConv1d.from_layer(
        base_layer,
        ablation="none",
        mask_unit="weight",
        mask_bias=True,
        mask_init_value=-1.0,
    )

    ipt_tensor = torch.Tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0]])
    base_out = base_layer(ipt_tensor)

    # Assert that test mode masked_layer produces different output as base layer (because init_value is < 0)
    masked_layer.train(False)
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out != base_out)

    # Assert that turning off masks gives same output as base layer
    masked_layer.use_masks = False
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out == base_out)


def test_conv1d_zero_ablate():
    layer = ContSparseConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="zero_ablate",
        mask_bias=False,
        mask_init_value=0.0,
    )
    ipt = torch.ones(1, 10)
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.5
    )
    layer.train(False)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))
    # Assert that positive params are masked and negative ones are not
    to_ablate = layer.weight_mask_params > 0
    assert torch.all(layer.weight_mask == ~to_ablate)


def test_conv1d_random_ablate():
    layer = ContSparseConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="random_ablate",
        mask_bias=False,
        mask_init_value=0.0,
    )
    ipt = torch.ones(1, 10)
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.5
    )
    layer.train(False)
    rand_out = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))
    # Assert that positive params are masked and negative ones are not
    to_ablate = layer.weight_mask_params > 0
    assert torch.all(layer.weight_mask == ~to_ablate)

    # Assert that random ablation is different than zero ablation
    layer = ContSparseConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="zero_ablate",
        mask_bias=False,
        mask_init_value=0.0,
    )
    layer.train(False)
    zero_out = layer(ipt)  # Weight mask computed during forward pass
    assert torch.all(rand_out != zero_out)


def test_conv1d_randomly_sampled_ablate():
    layer = ContSparseConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="randomly_sampled",
        mask_bias=False,
        mask_init_value=0.0,
    )
    ipt = torch.ones(1, 10)
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.7
    )
    layer.train(False)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))

    # Assert that the layer is masking out the same number of parameters as are in the subnetwork
    subnetwork_params = layer.weight_mask_params > 0
    assert torch.sum(~layer.weight_mask.bool()) == torch.sum(subnetwork_params)

    # Assert that no error is thrown if the subnetwork params contain >50% of parameters
    layer = ContSparseConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="randomly_sampled",
        mask_bias=False,
        mask_init_value=0.0,
    )
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.1
    )
    layer.train(False)
    _ = layer(ipt)  # Weight mask computed during forward pass


def test_conv1d_complement_sampled_ablate():
    layer = ContSparseConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="complement_sampled",
        mask_bias=False,
        mask_init_value=0.0,
    )
    ipt = torch.ones(1, 10)
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.7
    )
    layer.train(False)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))

    # Assert that all subnetwork parameters are unmasked
    subnetwork_params = layer.weight_mask_params > 0
    assert torch.all(layer.weight_mask.bool()[subnetwork_params])

    # Assert that the layer is masking out the same number of parameters as are in the subnetwork
    assert torch.sum(~layer.weight_mask.bool()) == torch.sum(subnetwork_params)

    # Assert that an error is thrown if the subnetwork params contain >50% of parameters
    layer = ContSparseConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="complement_sampled",
        mask_bias=False,
        mask_init_value=0.0,
    )
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.1
    )
    layer.train(False)
    with pytest.raises(Exception):
        _ = layer(ipt)  # Weight mask computed during forward pass


def test_conv1d_masking():
    # Test that masking every param gives the zero vector
    layer = ContSparseConv1d(
        1, 10, 3, bias=True, ablation="none", mask_bias=True, mask_init_value=-0.1
    )
    ipt = torch.ones(1, 10)
    layer.train(False)
    out = layer(ipt)
    assert torch.all(out == torch.zeros(out.shape))

    # Test that masking every weight param gives the bias vector
    layer = ContSparseConv1d(
        1, 10, 3, bias=True, ablation="none", mask_bias=False, mask_init_value=-0.1
    )
    ipt = torch.ones(1, 10)
    layer.train(False)
    out = layer(ipt)
    assert torch.all(out[:, 0] == layer.bias)

    # Test that masking one filter gives a particular zero entry
    layer = ContSparseConv1d(
        1, 10, 3, bias=True, ablation="none", mask_bias=True, mask_init_value=1.0
    )

    w_params = layer.weight_mask_params
    w_params.requires_grad = False

    w_params[5, :, :] = torch.zeros(3)
    w_params.requires_grad = True
    layer.weight_mask_params = w_params

    b_params = layer.bias_mask_params
    b_params.requires_grad = False
    b_params[5] = 0
    b_params.requires_grad = True
    layer.bias_mask_params = b_params

    ipt = torch.ones(1, 10)
    layer.train(False)
    out = layer(ipt)
    assert torch.all(out[5, :] == 0.0)


def test_conv1d_l0_calc():
    layer = ContSparseConv1d(
        1, 10, 3, bias=True, ablation="none", mask_bias=True, mask_init_value=1.0
    )
    l_train_l0 = layer.calculate_l0()
    l_train_l0_max = layer.calculate_max_l0()

    # Assert that training L0 calculation is working
    assert l_train_l0.item().is_integer() == False  # Mask entries should not be binary
    assert l_train_l0 != l_train_l0_max

    # Assert that test L0 calculation is working
    layer.train(False)
    l_test_l0 = layer.calculate_l0()
    l_test_l0_max = layer.calculate_max_l0()
    assert l_test_l0.item().is_integer()  # Mask entries should be binary
    assert l_test_l0 == l_test_l0_max
    assert l_test_l0 != l_train_l0


def test_conv1d_temperature():
    layer = ContSparseConv1d(
        1, 10, 3, bias=True, ablation="none", mask_bias=True, mask_init_value=1.0
    )

    # Assert that changing the temperature changes JUST the train mask
    train_l0_1 = layer.calculate_l0()
    layer.train(False)
    test_l0_1 = layer.calculate_l0()

    layer.train(True)
    layer.temperature = 2.0
    train_l0_2 = layer.calculate_l0()
    layer.train(False)
    test_l0_2 = layer.calculate_l0()

    assert train_l0_1 != train_l0_2
    assert test_l0_1 == test_l0_2

    # Assert that driving up the temperature past floating point precision makes train and test masks the same
    layer.temperature = 1000
    train_l0 = layer.calculate_l0()
    layer.train(False)
    test_l0 = layer.calculate_l0()

    assert train_l0 == test_l0


def test_conv1d_neuron_pruning():
    # Assert that masking out a single element in the mask matrix knocks out a full neuron
    layer = ContSparseConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="none",
        mask_unit="neuron",
        mask_bias=True,
        mask_init_value=1.0,
    )

    w_params = layer.weight_mask_params
    w_params.requires_grad = False

    w_params[5] = 0
    w_params.requires_grad = True
    layer.weight_mask_params = w_params

    b_params = layer.bias_mask_params
    b_params.requires_grad = False
    b_params[5] = 0
    b_params.requires_grad = True
    layer.bias_mask_params = b_params

    ipt = torch.ones(1, 10)
    layer.train(False)
    out = layer(ipt)
    assert torch.all(out[5, :] == 0.0)


def test_conv1d_neuron_l0():
    # Assert that L0 norm is always a multiple of the # of channel parameters
    layer = ContSparseConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="none",
        mask_unit="neuron",
        mask_bias=False,
        mask_init_value=1.0,
    )

    w_params = layer.weight_mask_params
    w_params.requires_grad = False
    w_params[0] = 0.0
    w_params.requires_grad = True
    layer.weight_mask_params = w_params
    layer.train(False)

    assert layer.calculate_l0() == 27

    w_params = layer.weight_mask_params
    w_params.requires_grad = False
    w_params[1] = 0.0
    w_params.requires_grad = True
    layer.weight_mask_params = w_params
    layer.train(False)

    assert layer.calculate_l0() == 24

    w_params = layer.weight_mask_params
    w_params.requires_grad = False
    w_params[2] = 0.0
    w_params.requires_grad = True
    layer.weight_mask_params = w_params
    layer.train(False)

    assert layer.calculate_l0() == 21


### Test ContSparseConv2d Layer


def test_conv2d_init():
    mask_fill_val = 19.0
    layer = ContSparseConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="none",
        mask_bias=True,
        mask_init_value=mask_fill_val,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "weight_mask_params")
    assert hasattr(layer, "bias")
    assert hasattr(layer, "bias_mask_params")
    assert torch.all(
        layer.weight_mask_params
        == torch.full(layer.weight_mask_params.shape, mask_fill_val)
    )
    assert torch.all(
        layer.bias_mask_params
        == torch.full(layer.bias_mask_params.shape, mask_fill_val)
    )

    layer = ContSparseConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="none",
        mask_bias=False,
        mask_init_value=mask_fill_val,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "weight_mask_params")
    assert hasattr(layer, "bias")
    assert not hasattr(layer, "bias_mask_params")
    assert torch.all(
        layer.weight_mask_params
        == torch.full(layer.weight_mask_params.shape, mask_fill_val)
    )

    layer = ContSparseConv2d(
        1,
        10,
        3,
        bias=False,
        ablation="none",
        mask_bias=False,
        mask_init_value=mask_fill_val,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "weight_mask_params")
    assert layer.bias is None
    assert not hasattr(layer, "bias_mask_params")
    assert torch.all(
        layer.weight_mask_params
        == torch.full(layer.weight_mask_params.shape, mask_fill_val)
    )

    # Assert that you cannot mask a bias that isn't there
    with pytest.raises(Exception):
        layer = ContSparseConv2d(
            1,
            10,
            3,
            bias=False,
            ablation="none",
            mask_bias=True,
            mask_init_value=mask_fill_val,
        )


def test_conv2d_from_layer():
    base_layer = nn.Conv2d(1, 10, 3)
    masked_layer = ContSparseConv2d.from_layer(
        base_layer,
        ablation="none",
        mask_unit="weight",
        mask_bias=True,
        mask_init_value=1.0,
    )

    # Ensure that masked layer is initialized correctly
    assert hasattr(masked_layer, "weight_mask_params")
    assert hasattr(masked_layer, "bias_mask_params")

    ipt_tensor = torch.Tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        ]
    )
    base_out = base_layer(ipt_tensor)

    # Assert that test mode masked_layer produces same output as base layer (because init_value is > 0)
    masked_layer.train(False)
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out == base_out)

    # Assert that train mode masked_layer produces diff output as base layer
    masked_layer.train(True)
    masked_out = masked_layer(ipt_tensor)
    assert not torch.all(masked_out == base_out)

    masked_layer = ContSparseConv2d.from_layer(
        base_layer,
        ablation="none",
        mask_unit="weight",
        mask_bias=True,
        mask_init_value=-1.0,
    )

    # Assert that test mode masked_layer produces diff output as base layer (because init_value is < 0)
    masked_layer.train(False)
    masked_out = masked_layer(ipt_tensor)
    assert not torch.all(masked_out == base_out)

    # Assert that train mode masked_layer produces diff output as base layer
    masked_layer.train(True)
    masked_out = masked_layer(ipt_tensor)
    assert not torch.all(masked_out == base_out)


def test_conv2d_use_mask():
    base_layer = nn.Conv2d(1, 10, 3)
    masked_layer = ContSparseConv2d.from_layer(
        base_layer,
        ablation="none",
        mask_unit="weight",
        mask_bias=True,
        mask_init_value=-1.0,
    )

    ipt_tensor = torch.Tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        ]
    )
    base_out = base_layer(ipt_tensor)

    # Assert that test mode masked_layer produces different output as base layer (because init_value is < 0)
    masked_layer.train(False)
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out != base_out)

    # Assert that turning off masks gives same output as base layer
    masked_layer.use_masks = False
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out == base_out)


def test_conv2d_zero_ablate():
    layer = ContSparseConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="zero_ablate",
        mask_bias=False,
        mask_init_value=0.0,
    )
    ipt = torch.ones(1, 10, 10)
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.5
    )
    layer.train(False)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))
    # Assert that positive params are masked and negative ones are not
    to_ablate = layer.weight_mask_params > 0
    assert torch.all(layer.weight_mask == ~to_ablate)


def test_conv2d_random_ablate():
    layer = ContSparseConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="random_ablate",
        mask_bias=False,
        mask_init_value=0.0,
    )
    ipt = torch.ones(1, 10, 10)
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.5
    )
    layer.train(False)
    rand_out = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))
    # Assert that positive params are masked and negative ones are not
    to_ablate = layer.weight_mask_params > 0
    assert torch.all(layer.weight_mask == ~to_ablate)

    # Assert that random ablation is different than zero ablation
    layer = ContSparseConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="zero_ablate",
        mask_bias=False,
        mask_init_value=0.0,
    )
    layer.train(False)
    zero_out = layer(ipt)  # Weight mask computed during forward pass
    assert torch.all(rand_out != zero_out)


def test_conv2d_randomly_sampled_ablate():
    layer = ContSparseConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="randomly_sampled",
        mask_bias=False,
        mask_init_value=0.0,
    )
    ipt = torch.ones(1, 10, 10)
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.7
    )
    layer.train(False)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))

    # Assert that the layer is masking out the same number of parameters as are in the subnetwork
    subnetwork_params = layer.weight_mask_params > 0
    assert torch.sum(~layer.weight_mask.bool()) == torch.sum(subnetwork_params)

    # Assert that no error is thrown if the subnetwork params contain >50% of parameters
    layer = ContSparseConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="randomly_sampled",
        mask_bias=False,
        mask_init_value=0.0,
    )
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.1
    )
    layer.train(False)
    _ = layer(ipt)  # Weight mask computed during forward pass


def test_conv2d_complement_sampled_ablate():
    layer = ContSparseConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="complement_sampled",
        mask_bias=False,
        mask_init_value=0.0,
    )
    ipt = torch.ones(1, 10, 10)
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.7
    )
    layer.train(False)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))

    # Assert that all subnetwork parameters are unmasked
    subnetwork_params = layer.weight_mask_params > 0
    assert torch.all(layer.weight_mask.bool()[subnetwork_params])

    # Assert that the layer is masking out the same number of parameters as are in the subnetwork
    assert torch.sum(~layer.weight_mask.bool()) == torch.sum(subnetwork_params)

    # Assert that an error is thrown if the subnetwork params contain >50% of parameters
    layer = ContSparseConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="complement_sampled",
        mask_bias=False,
        mask_init_value=0.0,
    )
    layer.weight_mask_params = nn.Parameter(
        torch.rand(layer.weight_mask_params.size()) - 0.1
    )
    layer.train(False)
    with pytest.raises(Exception):
        _ = layer(ipt)  # Weight mask computed during forward pass


def test_conv2d_masking():
    # Test that masking every param gives the zero vector
    layer = ContSparseConv2d(
        1, 10, 3, bias=True, ablation="none", mask_bias=True, mask_init_value=-0.1
    )
    ipt = torch.ones(1, 10, 10)
    layer.train(False)
    out = layer(ipt)
    assert torch.all(out == torch.zeros(out.shape))

    # Test that masking every weight param gives the bias vector
    layer = ContSparseConv2d(
        1, 10, 3, bias=True, ablation="none", mask_bias=False, mask_init_value=-0.1
    )
    layer.train(False)
    out = layer(ipt)
    assert torch.all(out[:, 0, 0] == layer.bias)

    # Test that masking one filter gives a particular zero entry
    layer = ContSparseConv2d(
        1, 10, 3, bias=True, ablation="none", mask_bias=True, mask_init_value=1.0
    )

    w_params = layer.weight_mask_params
    w_params.requires_grad = False

    w_params[5, :, :] = torch.zeros(3)
    w_params.requires_grad = True
    layer.weight_mask_params = w_params

    b_params = layer.bias_mask_params
    b_params.requires_grad = False
    b_params[5] = 0
    b_params.requires_grad = True
    layer.bias_mask_params = b_params

    layer.train(False)
    out = layer(ipt)
    assert torch.all(out[5, :] == 0.0)


def test_conv2d_l0_calc():
    layer = ContSparseConv2d(
        1, 10, 3, bias=True, ablation="none", mask_bias=True, mask_init_value=1.0
    )
    l_train_l0 = layer.calculate_l0()
    l_train_l0_max = layer.calculate_max_l0()

    # Assert that training L0 calculation is working
    assert l_train_l0.item().is_integer() == False  # Mask entries should not be binary
    assert l_train_l0 != l_train_l0_max

    # Assert that test L0 calculation is working
    layer.train(False)
    l_test_l0 = layer.calculate_l0()
    l_test_l0_max = layer.calculate_max_l0()
    assert l_test_l0.item().is_integer()  # Mask entries should be binary
    assert l_test_l0 == l_test_l0_max
    assert l_test_l0 != l_train_l0


def test_conv2d_temperature():
    layer = ContSparseConv2d(
        1, 10, 3, bias=True, ablation="none", mask_bias=True, mask_init_value=1.0
    )

    # Assert that changing the temperature changes JUST the train mask
    train_l0_1 = layer.calculate_l0()
    layer.train(False)
    test_l0_1 = layer.calculate_l0()

    layer.train(True)
    layer.temperature = 2.0
    train_l0_2 = layer.calculate_l0()
    layer.train(False)
    test_l0_2 = layer.calculate_l0()

    assert train_l0_1 != train_l0_2
    assert test_l0_1 == test_l0_2

    # Assert that driving up the temperature past floating point precision makes train and test masks the same
    layer.temperature = 1000
    train_l0 = layer.calculate_l0()
    layer.train(False)
    test_l0 = layer.calculate_l0()

    assert train_l0 == test_l0


def test_conv2d_neuron_pruning():
    # Assert that masking out a single element in the mask matrix knocks out a full neuron
    layer = ContSparseConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="none",
        mask_unit="neuron",
        mask_bias=True,
        mask_init_value=1.0,
    )

    w_params = layer.weight_mask_params
    w_params.requires_grad = False

    w_params[5] = 0
    w_params.requires_grad = True
    layer.weight_mask_params = w_params

    b_params = layer.bias_mask_params
    b_params.requires_grad = False
    b_params[5] = 0
    b_params.requires_grad = True
    layer.bias_mask_params = b_params

    ipt = torch.ones(1, 10, 10)
    layer.train(False)
    out = layer(ipt)
    assert torch.all(out[5, :] == 0.0)


def test_conv2d_neuron_l0():
    # Assert that L0 norm is always a multiple of the # channel parameters
    layer = ContSparseConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="none",
        mask_unit="neuron",
        mask_bias=False,
        mask_init_value=1.0,
    )

    w_params = layer.weight_mask_params
    w_params.requires_grad = False
    w_params[0] = 0.0
    w_params.requires_grad = True
    layer.weight_mask_params = w_params
    layer.train(False)

    assert layer.calculate_l0() == 81

    w_params = layer.weight_mask_params
    w_params.requires_grad = False
    w_params[1] = 0.0
    w_params.requires_grad = True
    layer.weight_mask_params = w_params
    layer.train(False)

    assert layer.calculate_l0() == 72

    w_params = layer.weight_mask_params
    w_params.requires_grad = False
    w_params[2] = 0.0
    w_params.requires_grad = True
    layer.weight_mask_params = w_params
    layer.train(False)

    assert layer.calculate_l0() == 63
