import pytest
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from ..magprune_layer import (
    MagPruneConv1d,
    MagPruneConv2d,
    MagPruneGPTConv1D,
    MagPruneLinear,
)

### Test MagPruneLinear Layer


def test_linear_init():
    prune_percentage = 1.0
    layer = MagPruneLinear(
        10,
        10,
        bias=True,
        ablation="none",
        mask_bias=True,
        prune_percentage=prune_percentage,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "bias")

    layer = MagPruneLinear(
        10,
        10,
        bias=True,
        ablation="none",
        mask_bias=False,
        prune_percentage=prune_percentage,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "bias")

    layer = MagPruneLinear(
        10,
        10,
        bias=False,
        ablation="none",
        mask_bias=False,
        prune_percentage=prune_percentage,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert layer.bias is None

    # Assert that you cannot mask a bias that isn't there
    with pytest.raises(Exception):
        layer = MagPruneLinear(
            10,
            10,
            bias=False,
            ablation="none",
            mask_bias=True,
            prune_percentage=prune_percentage,
        )


def test_linear_init_percentage():
    # Assert that the number of parameters that are masked is correct
    prune_percentage = 0.9
    layer = MagPruneLinear(
        1000,
        100,
        bias=True,
        ablation="none",
        mask_bias=True,
        prune_percentage=prune_percentage,
    )
    mask = layer._compute_mask("weight_mask_params")
    assert torch.sum(mask) == 10000

    prune_percentage = 0.1
    layer = MagPruneLinear(
        1000,
        100,
        bias=True,
        ablation="none",
        mask_bias=True,
        prune_percentage=prune_percentage,
    )
    mask = layer._compute_mask("weight_mask_params")
    assert torch.sum(mask) == 90000

    prune_percentage = 0.5
    layer = MagPruneLinear(
        1000,
        100,
        bias=True,
        ablation="none",
        mask_bias=True,
        prune_percentage=prune_percentage,
    )
    mask = layer._compute_mask("weight_mask_params")
    assert torch.sum(mask) == 50000


def test_linear_from_layer():
    base_layer = nn.Linear(10, 20)
    masked_layer = MagPruneLinear.from_layer(
        base_layer,
        ablation="none",
        mask_bias=True,
        prune_percentage=0.8,
    )

    ipt_tensor = torch.Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    base_out = base_layer(ipt_tensor)

    # Assert that test and train mode outputs are the same as each other, and different from the base output
    masked_layer.train(False)
    masked_out_test = masked_layer(ipt_tensor)
    assert not torch.all(masked_out_test == base_out)

    # Assert that train mode masked_layer produces diff output as base layer
    masked_layer.train(True)
    masked_out_train = masked_layer(ipt_tensor)
    assert not torch.all(masked_out_train == base_out)

    assert torch.all(masked_out_train == masked_out_test)

    masked_layer = MagPruneLinear.from_layer(
        base_layer,
        ablation="none",
        mask_bias=True,
        prune_percentage=0.0,
    )

    # Assert that test and train mode outputs are the same as each other, and the same as the base output
    masked_layer.train(False)
    masked_out_test = masked_layer(ipt_tensor)
    assert torch.all(masked_out_test == base_out)

    masked_layer.train(True)
    masked_out_train = masked_layer(ipt_tensor)
    assert torch.all(masked_out_train == base_out)
    assert torch.all(masked_out_train == masked_out_test)


def test_linear_use_mask():
    base_layer = nn.Linear(10, 20)
    masked_layer = MagPruneLinear.from_layer(
        base_layer,
        ablation="none",
        mask_bias=True,
        prune_percentage=0.8,
    )

    ipt_tensor = torch.Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    base_out = base_layer(ipt_tensor)

    # Assert that test mode masked_layer produces different output as base layer
    masked_layer.train(False)
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out != base_out)

    # Assert that turning off masks gives same output as base layer
    masked_layer.use_masks = False
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out == base_out)


def test_linear_zero_ablate():
    layer = MagPruneLinear(
        10,
        10,
        bias=True,
        ablation="zero_ablate",
        mask_bias=False,
        prune_percentage=0.2,
    )
    ipt = torch.ones(10)
    # Get weight mask params that have negatives and positives
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))
    # Assert that max weight param is masked out
    to_ablate = torch.max(layer.weight)
    assert layer.weight_mask[layer.weight == to_ablate] == 0


def test_linear_random_ablate():
    layer = MagPruneLinear(
        10,
        10,
        bias=True,
        ablation="random_ablate",
        mask_bias=False,
        prune_percentage=0.2,
    )
    ipt = torch.ones(10)
    rand_out = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))
    # Assert that max weight param is masked out
    to_ablate = torch.max(layer.weight)
    assert layer.weight_mask[layer.weight == to_ablate] == 0

    # Assert that random ablation is different than zero ablation
    layer = MagPruneLinear(
        10,
        10,
        bias=True,
        ablation="zero_ablate",
        mask_bias=False,
        prune_percentage=0.2,
    )
    zero_out = layer(ipt)  # Weight mask computed during forward pass
    assert torch.all(rand_out != zero_out)


def test_linear_randomly_sampled_ablate():
    layer = MagPruneLinear(
        10,
        10,
        bias=True,
        ablation="randomly_sampled",
        mask_bias=False,
        prune_percentage=0.8,
    )
    ipt = torch.ones(10)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))

    # Assert that the layer is masking out the same number of parameters as are in the subnetwork
    assert (
        torch.sum(~layer.weight_mask.bool())
        / torch.sum(torch.ones(layer.weight_mask.shape))
        == 1 - layer.prune_percentage
    )

    # Ensure no error is thrown if the subnetwork params contain >50% of parameters
    layer = MagPruneLinear(
        10,
        10,
        bias=True,
        ablation="randomly_sampled",
        mask_bias=False,
        prune_percentage=0.2,
    )
    _ = layer(ipt)


def test_linear_complement_sampled_ablate():
    layer = MagPruneLinear(
        10,
        10,
        bias=True,
        ablation="complement_sampled",
        mask_bias=False,
        prune_percentage=0.8,
    )
    ipt = torch.ones(10)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))

    # Assert that the layer is masking out the same number of parameters as are in the subnetwork
    assert (
        torch.sum(~layer.weight_mask.bool())
        / torch.sum(torch.ones(layer.weight_mask.shape))
        == 1 - layer.prune_percentage
    )

    # Assert that an error is thrown if the subnetwork params contain >50% of parameters
    layer = MagPruneLinear(
        10,
        10,
        bias=True,
        ablation="complement_sampled",
        mask_bias=False,
        prune_percentage=0.2,
    )
    with pytest.raises(Exception):
        _ = layer(ipt)  # Weight mask computed during forward pass


def test_linear_masking():
    # Test that masking every param gives the zero vector
    layer = MagPruneLinear(
        10, 10, bias=True, ablation="none", mask_bias=True, prune_percentage=1.0
    )
    ipt = torch.ones(10)
    out = layer(ipt)
    assert torch.all(out == torch.zeros(out.shape))

    # Test that masking every weight param gives the bias vector
    layer = MagPruneLinear(
        10, 10, bias=True, ablation="none", mask_bias=False, prune_percentage=1.0
    )
    ipt = torch.ones(10)
    out = layer(ipt)
    assert torch.all(out == layer.bias)


def test_linear_l0_calc():
    layer = MagPruneLinear(
        10, 15, bias=True, ablation="none", mask_bias=True, prune_percentage=0.5
    )
    l_train_l0 = layer.calculate_l0()
    l_train_l0_max = layer.calculate_max_l0()

    # Assert that training L0 calculation is working
    assert l_train_l0.item().is_integer() == True  # Mask entries should be binary
    assert l_train_l0 != l_train_l0_max

    # Assert that test L0 calculation is working
    layer.train(False)
    l_test_l0 = layer.calculate_l0()
    l_test_l0_max = layer.calculate_max_l0()
    assert l_test_l0.item().is_integer()  # Mask entries should be binary
    assert l_test_l0 != l_test_l0_max
    assert l_test_l0 == l_train_l0


### Test MagPruneGPTConv1d Layer


def test_gptconv1d_init():
    prune_percentage = 1.0
    layer = MagPruneGPTConv1D(
        10, 10, ablation="none", mask_bias=True, prune_percentage=prune_percentage
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "bias")

    layer = MagPruneGPTConv1D(
        10, 10, ablation="none", mask_bias=False, prune_percentage=prune_percentage
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "bias")


def test_gptconv1d_init_percentage():
    # Assert that the number of parameters that would be masked for a given forward pass sample
    # is close to the expected number
    prune_percentage = 0.9
    layer = MagPruneGPTConv1D(
        1000, 100, ablation="none", mask_bias=False, prune_percentage=prune_percentage
    )
    mask = layer._compute_mask("weight_mask_params")
    assert torch.sum(mask) == 10000

    prune_percentage = 0.1
    layer = MagPruneGPTConv1D(
        1000, 100, ablation="none", mask_bias=False, prune_percentage=prune_percentage
    )
    mask = layer._compute_mask("weight_mask_params")
    assert torch.sum(mask) == 90000

    prune_percentage = 0.5
    layer = MagPruneGPTConv1D(
        1000, 100, ablation="none", mask_bias=False, prune_percentage=prune_percentage
    )
    mask = layer._compute_mask("weight_mask_params")
    assert torch.sum(mask) == 50000


def test_gptconv1d_from_layer():
    base_layer = Conv1D(
        20, 10
    )  # For some reason, out_features comes first in this classes constructor...
    masked_layer = MagPruneGPTConv1D.from_layer(
        base_layer,
        ablation="none",
        mask_bias=True,
        prune_percentage=0.8,
    )

    ipt_tensor = torch.Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    base_out = base_layer(ipt_tensor)

    # Assert that test mode masked_layer produces same output as train masked_layer, different from base layer
    masked_layer.train(False)
    masked_out_test = masked_layer(ipt_tensor)
    assert not torch.all(masked_out_test == base_out)

    masked_layer.train(True)
    masked_out_train = masked_layer(ipt_tensor)
    assert not torch.all(masked_out_train == base_out)
    assert torch.all(masked_out_test == masked_out_train)

    masked_layer = MagPruneGPTConv1D.from_layer(
        base_layer,
        ablation="none",
        mask_bias=True,
        prune_percentage=0.0,
    )

    # Assert that test mode masked_layer produces same out as train, and base
    masked_layer.train(False)
    masked_out_test = masked_layer(ipt_tensor)
    assert torch.all(masked_out_test == base_out)

    # Assert that train mode masked_layer produces diff output as base layer
    masked_layer.train(True)
    masked_out_train = masked_layer(ipt_tensor)
    assert torch.all(masked_out_train == base_out)
    assert torch.all(masked_out_train == masked_out_test)


def test_gptconv1d_use_mask():
    base_layer = Conv1D(
        20, 10
    )  # For some reason, out_features comes first in this classes constructor...
    masked_layer = MagPruneGPTConv1D.from_layer(
        base_layer,
        ablation="none",
        mask_bias=True,
        prune_percentage=0.8,
    )

    ipt_tensor = torch.Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    base_out = base_layer(ipt_tensor)

    # Assert that test mode masked_layer produces different output as base layer
    masked_layer.train(False)
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out != base_out)

    # Assert that turning off masks gives same output as base layer
    masked_layer.use_masks = False
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out == base_out)


def test_gptconv1d_zero_ablate():
    layer = MagPruneGPTConv1D(
        10, 10, ablation="zero_ablate", mask_bias=False, prune_percentage=0.2
    )
    ipt = torch.ones(10)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))
    # Assert that max weight param is masked out
    to_ablate = torch.max(layer.weight)
    assert layer.weight_mask[layer.weight == to_ablate] == 0


def test_gptconv1d_random_ablate():
    layer = MagPruneGPTConv1D(
        10, 10, ablation="random_ablate", mask_bias=False, prune_percentage=0.2
    )
    ipt = torch.ones(10)
    rand_out = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))
    # Assert that max weight param is masked out
    to_ablate = torch.max(layer.weight)
    assert layer.weight_mask[layer.weight == to_ablate] == 0

    # Assert that random ablation is different than zero ablation
    layer = MagPruneGPTConv1D(
        10, 10, ablation="zero_ablate", mask_bias=False, prune_percentage=0.2
    )
    zero_out = layer(ipt)  # Weight mask computed during forward pass
    assert torch.all(rand_out != zero_out)


def test_gptconv1d_randomly_sampled_ablate():
    layer = MagPruneGPTConv1D(
        10, 10, ablation="randomly_sampled", mask_bias=False, prune_percentage=0.8
    )
    ipt = torch.ones(10)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))

    # Assert that the layer is masking out the same number of parameters as are in the subnetwork
    assert (
        torch.sum(~layer.weight_mask.bool())
        / torch.sum(torch.ones(layer.weight_mask.shape))
        == 1 - layer.prune_percentage
    )

    # Assert that no error is thrown if the subnetwork params contain >50% of parameters
    layer = MagPruneGPTConv1D(
        10, 10, ablation="randomly_sampled", mask_bias=False, prune_percentage=0.2
    )
    _ = layer(ipt)  # Weight mask computed during forward pass


def test_gptconv1d_complement_sampled_ablate():
    layer = MagPruneGPTConv1D(
        10, 10, ablation="complement_sampled", mask_bias=False, prune_percentage=0.8
    )
    ipt = torch.ones(10)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))

    # Assert that the layer is masking out the same number of parameters as are in the subnetwork
    assert (
        torch.sum(~layer.weight_mask.bool())
        / torch.sum(torch.ones(layer.weight_mask.shape))
        == 1 - layer.prune_percentage
    )

    # Assert that an error is thrown if the subnetwork params contain >50% of parameters
    layer = MagPruneGPTConv1D(
        10, 10, ablation="complement_sampled", mask_bias=False, prune_percentage=0.2
    )
    with pytest.raises(Exception):
        _ = layer(ipt)  # Weight mask computed during forward pass


def test_gptconv1d_masking():
    # Test that masking every param gives the zero vector
    layer = MagPruneGPTConv1D(
        10, 10, ablation="none", mask_bias=True, prune_percentage=1.0
    )
    ipt = torch.ones(10)
    layer.train(False)
    out = layer(ipt)
    assert torch.all(out == torch.zeros(out.shape))

    # Test that masking every weight param gives the bias vector
    layer = MagPruneGPTConv1D(
        10, 10, ablation="none", mask_bias=False, prune_percentage=1.0
    )
    ipt = torch.ones(10)
    layer.train(False)
    out = layer(ipt)
    assert torch.all(
        out == layer.bias
    )  # Bias is initialized at 0's anyway for GPTConv1D, so kind of redundant...


def test_gptconv1d_l0_calc():
    layer = MagPruneGPTConv1D(
        10, 15, ablation="none", mask_bias=True, prune_percentage=0.5
    )
    l_train_l0 = layer.calculate_l0()
    l_train_l0_max = layer.calculate_max_l0()

    # Assert that training L0 calculation is working
    assert l_train_l0.item().is_integer()  # Mask entries should be binary
    assert l_train_l0 != l_train_l0_max

    # Assert that test L0 calculation is working
    layer.train(False)
    l_test_l0 = layer.calculate_l0()
    l_test_l0_max = layer.calculate_max_l0()
    assert l_test_l0.item().is_integer()  # Mask entries should be binary
    assert l_test_l0 != l_test_l0_max
    assert l_test_l0 == l_train_l0


### Test MagPruneConv1d Layer


def test_conv1d_init():
    prune_percentage = 1.0
    layer = MagPruneConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="none",
        mask_bias=True,
        prune_percentage=prune_percentage,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "bias")

    layer = MagPruneConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="none",
        mask_bias=False,
        prune_percentage=prune_percentage,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "bias")

    layer = MagPruneConv1d(
        1,
        10,
        3,
        bias=False,
        ablation="none",
        mask_bias=False,
        prune_percentage=prune_percentage,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert layer.bias is None

    # Assert that you cannot mask a bias that isn't there
    with pytest.raises(Exception):
        layer = MagPruneConv1d(
            1,
            10,
            3,
            bias=False,
            ablation="none",
            mask_bias=True,
            prune_percentage=prune_percentage,
        )


def test_conv1d_init_percentage():
    # Assert that the number of parameters that would be masked for a given forward pass sample
    # is close to the expected number
    prune_percentage = 0.9
    layer = MagPruneConv1d(
        100, 100, 3, ablation="none", mask_bias=False, prune_percentage=prune_percentage
    )
    mask = layer._compute_mask("weight_mask_params")
    assert torch.sum(mask == 1) == (torch.sum(torch.ones(mask.shape)) * 0.1)

    prune_percentage = 0.1
    layer = MagPruneConv1d(
        100, 100, 3, ablation="none", mask_bias=False, prune_percentage=prune_percentage
    )
    mask = layer._compute_mask("weight_mask_params")
    assert torch.sum(mask == 1) == (torch.sum(torch.ones(mask.shape)) * 0.9)

    prune_percentage = 0.5
    layer = MagPruneConv1d(
        100, 100, 3, ablation="none", mask_bias=False, prune_percentage=prune_percentage
    )
    mask = layer._compute_mask("weight_mask_params")
    assert torch.sum(mask == 1) == (torch.sum(torch.ones(mask.shape)) * 0.5)


def test_conv1d_from_layer():
    base_layer = nn.Conv1d(1, 10, 3)
    masked_layer = MagPruneConv1d.from_layer(
        base_layer,
        ablation="none",
        mask_bias=True,
        prune_percentage=0.8,
    )

    ipt_tensor = torch.Tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0]])
    base_out = base_layer(ipt_tensor)

    # Assert that test and train mode outputs are the same as each other, and different from the base output
    masked_layer.train(False)
    masked_out_test = masked_layer(ipt_tensor)
    assert not torch.all(masked_out_test == base_out)

    # Assert that train mode masked_layer produces diff output as base layer
    masked_layer.train(True)
    masked_out_train = masked_layer(ipt_tensor)
    assert not torch.all(masked_out_train == base_out)

    assert torch.all(masked_out_train == masked_out_test)

    masked_layer = MagPruneConv1d.from_layer(
        base_layer,
        ablation="none",
        mask_bias=True,
        prune_percentage=0.0,
    )

    # Assert that test and train mode outputs are the same as each other, and the same as the base output
    masked_layer.train(False)
    masked_out_test = masked_layer(ipt_tensor)
    assert torch.all(masked_out_test == base_out)

    masked_layer.train(True)
    masked_out_train = masked_layer(ipt_tensor)
    assert torch.all(masked_out_train == base_out)
    assert torch.all(masked_out_train == masked_out_test)


def test_conv1d_use_mask():
    base_layer = nn.Conv1d(1, 10, 3)
    masked_layer = MagPruneConv1d.from_layer(
        base_layer,
        ablation="none",
        mask_bias=True,
        prune_percentage=0.8,
    )

    ipt_tensor = torch.Tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0]])
    base_out = base_layer(ipt_tensor)

    # Assert that test mode masked_layer produces different output as base layer
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out != base_out)

    # Assert that turning off masks gives same output as base layer
    masked_layer.use_masks = False
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out == base_out)


def test_conv1d_zero_ablate():
    layer = MagPruneConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="zero_ablate",
        mask_bias=False,
        prune_percentage=0.2,
    )
    ipt = torch.ones(1, 10)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))
    # Assert that max weight param is masked out
    to_ablate = torch.max(layer.weight)
    assert layer.weight_mask[layer.weight == to_ablate] == 0


def test_conv1d_random_ablate():
    layer = MagPruneConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="random_ablate",
        mask_bias=False,
        prune_percentage=0.2,
    )
    ipt = torch.ones(1, 10)
    rand_out = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))
    # Assert that max weight param is masked out
    to_ablate = torch.max(layer.weight)
    assert layer.weight_mask[layer.weight == to_ablate] == 0

    # Assert that random ablation is different than zero ablation
    layer = MagPruneConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="zero_ablate",
        mask_bias=False,
        prune_percentage=0.2,
    )
    layer.train(False)
    zero_out = layer(ipt)  # Weight mask computed during forward pass
    assert torch.all(rand_out != zero_out)


def test_conv1d_randomly_sampled_ablate():
    layer = MagPruneConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="randomly_sampled",
        mask_bias=False,
        prune_percentage=0.8,
    )
    ipt = torch.ones(1, 10)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))

    # Assert that the layer is masking out the same number of parameters as are in the subnetwork
    assert (
        torch.sum(~layer.weight_mask.bool())
        / torch.sum(torch.ones(layer.weight_mask.shape))
        == 1 - layer.prune_percentage
    )

    # Assert that no error is thrown if the subnetwork params contain >50% of parameters
    layer = MagPruneConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="randomly_sampled",
        mask_bias=False,
        prune_percentage=0.2,
    )
    _ = layer(ipt)  # Weight mask computed during forward pass


def test_conv1d_complement_sampled_ablate():
    layer = MagPruneConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="complement_sampled",
        mask_bias=False,
        prune_percentage=0.8,
    )
    ipt = torch.ones(1, 10)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))

    # Assert that the layer is masking out the same number of parameters as are in the subnetwork
    assert (
        torch.sum(~layer.weight_mask.bool())
        / torch.sum(torch.ones(layer.weight_mask.shape))
        == 1 - layer.prune_percentage
    )

    # Assert that an error is thrown if the subnetwork params contain >50% of parameters
    layer = MagPruneConv1d(
        1,
        10,
        3,
        bias=True,
        ablation="complement_sampled",
        mask_bias=False,
        prune_percentage=0.2,
    )
    with pytest.raises(Exception):
        _ = layer(ipt)  # Weight mask computed during forward pass


def test_conv1d_masking():
    # Test that masking every param gives the zero vector
    layer = MagPruneConv1d(
        1, 10, 3, bias=True, ablation="none", mask_bias=True, prune_percentage=1.0
    )
    ipt = torch.ones(1, 10)
    out = layer(ipt)
    assert torch.all(out == torch.zeros(out.shape))

    # Test that masking every weight param gives the bias vector
    layer = MagPruneConv1d(
        1, 10, 3, bias=True, ablation="none", mask_bias=False, prune_percentage=1.0
    )
    ipt = torch.ones(1, 10)
    out = layer(ipt)
    assert torch.all(out[:, 0] == layer.bias)


def test_conv1d_l0_calc():
    layer = MagPruneConv1d(
        1, 10, 3, bias=True, ablation="none", mask_bias=True, prune_percentage=1.0
    )
    l_train_l0 = layer.calculate_l0()
    l_train_l0_max = layer.calculate_max_l0()

    # Assert that training L0 calculation is working
    assert l_train_l0.item().is_integer()  # Mask entries should be binary
    assert l_train_l0 != l_train_l0_max

    # Assert that test L0 calculation is working
    layer.train(False)
    l_test_l0 = layer.calculate_l0()
    l_test_l0_max = layer.calculate_max_l0()
    assert l_test_l0.item().is_integer()  # Mask entries should be binary
    assert l_test_l0 != l_test_l0_max
    assert l_test_l0 == l_train_l0


### Test MagPruneConv2d Layer


def test_conv2d_init():
    prune_percentage = 1.0
    layer = MagPruneConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="none",
        mask_bias=True,
        prune_percentage=prune_percentage,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "bias")

    layer = MagPruneConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="none",
        mask_bias=False,
        prune_percentage=prune_percentage,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert hasattr(layer, "bias")

    layer = MagPruneConv2d(
        1,
        10,
        3,
        bias=False,
        ablation="none",
        mask_bias=False,
        prune_percentage=prune_percentage,
    )

    # Test that mask parameters have been initialized correctly
    assert hasattr(layer, "weight")
    assert layer.bias is None

    # Assert that you cannot mask a bias that isn't there
    with pytest.raises(Exception):
        layer = MagPruneConv2d(
            1,
            10,
            3,
            bias=False,
            ablation="none",
            mask_bias=True,
            prune_percentage=prune_percentage,
        )


def test_conv2d_init_percentage():
    # Assert that the number of parameters that would be masked for a given forward pass sample
    # is close to the expected number
    prune_percentage = 0.9
    layer = MagPruneConv2d(
        100, 100, 3, ablation="none", mask_bias=False, prune_percentage=prune_percentage
    )
    mask = layer._compute_mask("weight_mask_params")
    assert torch.sum(mask == 1) == (torch.sum(torch.ones(mask.shape)) * 0.1)

    prune_percentage = 0.1
    layer = MagPruneConv2d(
        100, 100, 3, ablation="none", mask_bias=False, prune_percentage=prune_percentage
    )
    mask = layer._compute_mask("weight_mask_params")
    assert torch.sum(mask == 1) == (torch.sum(torch.ones(mask.shape)) * 0.9)

    prune_percentage = 0.5
    layer = MagPruneConv2d(
        100, 100, 3, ablation="none", mask_bias=False, prune_percentage=prune_percentage
    )
    mask = layer._compute_mask("weight_mask_params")
    assert torch.sum(mask == 1) == (torch.sum(torch.ones(mask.shape)) * 0.5)


def test_conv2d_from_layer():
    base_layer = nn.Conv2d(1, 10, 3)
    masked_layer = MagPruneConv2d.from_layer(
        base_layer,
        ablation="none",
        mask_bias=True,
        prune_percentage=0.8,
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

    # Assert that test and train mode outputs are the same as each other, and different from the base output
    masked_layer.train(False)
    masked_out_test = masked_layer(ipt_tensor)
    assert not torch.all(masked_out_test == base_out)

    # Assert that train mode masked_layer produces diff output as base layer
    masked_layer.train(True)
    masked_out_train = masked_layer(ipt_tensor)
    assert not torch.all(masked_out_train == base_out)

    assert torch.all(masked_out_train == masked_out_test)

    masked_layer = MagPruneConv2d.from_layer(
        base_layer,
        ablation="none",
        mask_bias=True,
        prune_percentage=0.0,
    )

    # Assert that test and train mode outputs are the same as each other, and the same as the base output
    masked_layer.train(False)
    masked_out_test = masked_layer(ipt_tensor)
    assert torch.all(masked_out_test == base_out)

    masked_layer.train(True)
    masked_out_train = masked_layer(ipt_tensor)
    assert torch.all(masked_out_train == base_out)
    assert torch.all(masked_out_train == masked_out_test)


def test_conv2d_use_mask():
    base_layer = nn.Conv2d(1, 10, 3)
    masked_layer = MagPruneConv2d.from_layer(
        base_layer,
        ablation="none",
        mask_bias=True,
        prune_percentage=0.8,
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

    # Assert that test mode masked_layer produces different output as base layer
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out != base_out)

    # Assert that turning off masks gives same output as base layer
    masked_layer.use_masks = False
    masked_out = masked_layer(ipt_tensor)
    assert torch.all(masked_out == base_out)


def test_conv2d_zero_ablate():
    layer = MagPruneConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="zero_ablate",
        mask_bias=False,
        prune_percentage=0.2,
    )
    ipt = torch.ones(1, 10, 10)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))
    # Assert that max weight param is masked out
    to_ablate = torch.max(layer.weight)
    assert layer.weight_mask[layer.weight == to_ablate] == 0


def test_conv2d_random_ablate():
    layer = MagPruneConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="random_ablate",
        mask_bias=False,
        prune_percentage=0.2,
    )
    ipt = torch.ones(1, 10, 10)
    rand_out = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))
    # Assert that max weight param is masked out
    to_ablate = torch.max(layer.weight)
    assert layer.weight_mask[layer.weight == to_ablate] == 0

    # Assert that random ablation is different than zero ablation
    layer = MagPruneConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="zero_ablate",
        mask_bias=False,
        prune_percentage=0.2,
    )
    zero_out = layer(ipt)  # Weight mask computed during forward pass
    assert torch.all(rand_out != zero_out)


def test_conv2d_randomly_sampled_ablate():
    layer = MagPruneConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="randomly_sampled",
        mask_bias=False,
        prune_percentage=0.8,
    )
    ipt = torch.ones(1, 10, 10)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))

    # Assert that the layer is masking out the same number of parameters as are in the subnetwork
    assert (
        torch.sum(~layer.weight_mask.bool())
        / torch.sum(torch.ones(layer.weight_mask.shape))
        == 1 - layer.prune_percentage
    )

    # Assert that no error is thrown if the subnetwork params contain >50% of parameters
    layer = MagPruneConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="randomly_sampled",
        mask_bias=False,
        prune_percentage=0.2,
    )
    _ = layer(ipt)  # Weight mask computed during forward pass


def test_conv2d_complement_sampled_ablate():
    layer = MagPruneConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="complement_sampled",
        mask_bias=False,
        prune_percentage=0.8,
    )
    ipt = torch.ones(1, 10, 10)
    _ = layer(ipt)  # Weight mask computed during forward pass

    # Assert that mask is binary
    assert torch.all(torch.logical_or(layer.weight_mask == 1, layer.weight_mask == 0))

    # Assert that the layer is masking out the same number of parameters as are in the subnetwork
    assert (
        torch.sum(~layer.weight_mask.bool())
        / torch.sum(torch.ones(layer.weight_mask.shape))
        == 1 - layer.prune_percentage
    )

    # Assert that an error is thrown if the subnetwork params contain >50% of parameters
    layer = MagPruneConv2d(
        1,
        10,
        3,
        bias=True,
        ablation="complement_sampled",
        mask_bias=False,
        prune_percentage=0.2,
    )
    with pytest.raises(Exception):
        _ = layer(ipt)  # Weight mask computed during forward pass


def test_conv2d_masking():
    # Test that masking every param gives the zero vector
    layer = MagPruneConv2d(
        1, 10, 3, bias=True, ablation="none", mask_bias=True, prune_percentage=1.0
    )
    ipt = torch.ones(1, 10, 10)
    out = layer(ipt)
    assert torch.all(out == torch.zeros(out.shape))

    # Test that masking every weight param gives the bias vector
    layer = MagPruneConv2d(
        1, 10, 3, bias=True, ablation="none", mask_bias=False, prune_percentage=1.0
    )
    out = layer(ipt)
    assert torch.all(out[:, 0, 0] == layer.bias)


def test_conv2d_l0_calc():
    layer = MagPruneConv2d(
        1, 10, 3, bias=True, ablation="none", mask_bias=True, prune_percentage=1.0
    )
    l_train_l0 = layer.calculate_l0()
    l_train_l0_max = layer.calculate_max_l0()

    # Assert that training L0 calculation is working
    assert l_train_l0.item().is_integer()  # Mask entries should  be binary
    assert l_train_l0 != l_train_l0_max

    # Assert that test L0 calculation is working
    layer.train(False)
    l_test_l0 = layer.calculate_l0()
    l_test_l0_max = layer.calculate_max_l0()
    assert l_test_l0.item().is_integer()  # Mask entries should be binary
    assert l_test_l0 != l_test_l0_max
    assert l_test_l0 == l_train_l0
