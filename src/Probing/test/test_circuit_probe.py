import pytest
from ...Models.circuit_model import CircuitModel
from ...Probing.circuit_probe import CircuitProbe
from ...Probing.probe_configs import *
from ..residual_update_model import ResidualUpdateModel
from ...Models.model_configs import CircuitConfig
from transformers import BertModel, BertTokenizerFast
from ...Masking.mask_layer import MaskLayer
import torch
from torch.optim import AdamW
from copy import deepcopy


def create_test_probe():
    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 0.0,
        },
        target_layers=["encoder.layer.1.output.dense"],
        freeze_base=True,
        add_l0=True,
        l0_lambda=1e-8,
    )

    resid_config = ResidualUpdateModelConfig(
        "bert",
        target_layers=[1],
        mlp=True,
        attn=False,
        circuit=True,
        base=True,
        updates=True,
        stream=False,
    )

    probe_config = CircuitProbeConfig(
        probe_activations="mlp_update_1",
        circuit_config=circuit_config,
        resid_config=resid_config,
    )

    bert = BertModel.from_pretrained("prajjwal1/bert-tiny")

    probe = CircuitProbe(probe_config, bert)
    return probe


def test_circuit_probe_initialization():
    # Assert that you can initialize a CircuitProbe Model
    create_test_probe()


def test_incompatible_configuration_exceptions():
    # Assert that an error is thrown if configurations are incompatible with circuit probing
    bert = BertModel.from_pretrained("prajjwal1/bert-tiny")

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 0.0,
        },
        target_layers=["encoder.layer.1.output.dense"],
        freeze_base=False,
        add_l0=True,
        l0_lambda=1e-8,
    )

    resid_config = ResidualUpdateModelConfig(
        "bert", target_layers=[1], mlp=True, attn=False, circuit=True, base=True
    )

    probe_config = CircuitProbeConfig(
        probe_activations="mlp_1",
        circuit_config=circuit_config,
        resid_config=resid_config,
    )

    with pytest.raises(Exception):
        CircuitProbe(probe_config, bert)

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 0.0,
        },
        target_layers=["encoder.layer.1.output.dense"],
        freeze_base=True,
        add_l0=True,
        l0_lambda=1e-8,
    )

    resid_config = ResidualUpdateModelConfig(
        "bert", target_layers=[1, 2], mlp=True, attn=False, circuit=True, base=True
    )

    probe_config = CircuitProbeConfig(
        probe_activations="mlp_1",
        circuit_config=circuit_config,
        resid_config=resid_config,
    )

    with pytest.raises(Exception):
        CircuitProbe(probe_config, bert)

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 0.0,
        },
        target_layers=["encoder.layer.1.output.dense"],
        freeze_base=True,
        add_l0=True,
        l0_lambda=1e-8,
    )

    resid_config = ResidualUpdateModelConfig(
        "bert", target_layers=[1], mlp=False, attn=False, circuit=True, base=True
    )

    probe_config = CircuitProbeConfig(
        probe_activations="mlp_1",
        circuit_config=circuit_config,
        resid_config=resid_config,
    )

    with pytest.raises(Exception):
        CircuitProbe(probe_config, bert)


def test_circuit_probe_training():
    # Assert that everything is frozen/in eval mode except the target layer's mask parameters,
    # even when the overall model is in training mode

    probe = create_test_probe()
    probe.train()

    for m in probe.modules():
        if (
            issubclass(m.__class__, MaskLayer)
            or issubclass(m.__class__, CircuitProbe)
            or issubclass(m.__class__, ResidualUpdateModel)
            or issubclass(m.__class__, CircuitModel)
        ):
            assert m.training == True
        else:
            assert m.training == False

    probe.train(False)

    for m in probe.modules():
        assert m.training == False


def test_representation_matching_loss():
    # Assert that representation matching loss is correct
    probe = create_test_probe()

    labels = torch.Tensor([0, 1, 0, 1])
    good_updates = torch.Tensor(
        [[0.5, 0.1, 0.9], [0.05, 0.8, 0.6], [0.6, 0.09, 0.85], [0.1, 0.8, 0.55]]
    )
    bad_updates = torch.Tensor(
        [[0.5, 0.1, 0.9], [0.6, 0.09, 0.85], [0.05, 0.8, 0.6], [0.1, 0.8, 0.55]]
    )

    low_loss = probe._compute_representation_matching_loss(good_updates, labels)
    high_loss = probe._compute_representation_matching_loss(bad_updates, labels)
    assert low_loss < high_loss


def test_forward_pass():
    # Assert that forward passes work, and that labels assigned to words
    # that are in similar contexts produce lower loss than labels that are
    # assigned to words in different contexts

    probe = create_test_probe()
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")

    input = ["the cat sleeps on the mat", "the dog rests in the rug"]
    token_mask = torch.tensor(
        [[0, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 0]]
    ).bool()
    labels = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    encs = tokenizer(input, return_tensors="pt")
    out = probe(**encs, labels=labels, token_mask=token_mask)

    # Assert that residual stream updates are created in the right layer
    assert probe.config.probe_activations in probe.wrapped_model.vector_cache

    low_loss = out.loss

    input = ["the cat sleeps on the mat", "jump near the big red flag"]
    token_mask = torch.tensor(
        [[0, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 0]]
    ).bool()
    labels = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    encs = tokenizer(input, return_tensors="pt")
    out = probe(**encs, labels=labels, token_mask=token_mask)

    high_loss = out.loss

    assert low_loss < high_loss

    # Assert that # labels must be equal to # of relevant (i.e. not special, not subword) tokens
    with pytest.raises(Exception):
        input = ["the cat sleeps on the mat"]
        token_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0]]).bool()
        labels = torch.tensor([[0, 1, 2, 3, 4]])
        encs = tokenizer(input, return_tensors="pt")
        out = probe(**encs, labels=labels, token_mask=token_mask)

    with pytest.raises(Exception):
        input = ["the catcat sleeps on the mat"]
        token_mask = torch.tensor([[0, 1, 0, 1, 1, 1, 1, 1, 0]]).bool()
        labels = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
        encs = tokenizer(input, return_tensors="pt")
        out = probe(**encs, labels=labels, token_mask=token_mask)

    input = ["the catcat sleeps on the mat"]
    token_mask = torch.tensor([[0, 1, 0, 1, 1, 1, 1, 1, 0]]).bool()
    labels = torch.tensor([[0, 1, 2, 3, 4, 5]])
    encs = tokenizer(input, return_tensors="pt")
    out = probe(**encs, labels=labels, token_mask=token_mask)


def test_training_circuit_probe():
    # Assert that training circuit probe only updates the mask parameters
    probe = create_test_probe()
    original_probe = deepcopy(probe)
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")

    input_batch = [
        "the cat sleeps on the mat",
        "the dog rests in the rug",
    ]
    token_mask = torch.tensor(
        [[0, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 0]]
    ).bool()
    encs = tokenizer(input_batch, return_tensors="pt")
    labels = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])

    optimizer = AdamW(probe.parameters(), lr=5e-2)

    probe.train()

    outputs = probe(**encs, labels=labels, token_mask=token_mask)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    for name, param in probe.named_parameters():
        if "weight_mask_param" in name:
            assert ~torch.all(param == original_probe.state_dict()[name])
        else:
            assert torch.all(param == original_probe.state_dict()[name])
