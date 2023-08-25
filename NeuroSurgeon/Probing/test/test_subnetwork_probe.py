from copy import deepcopy

import pytest
import torch
from torch.optim import AdamW
from transformers import BertModel, BertTokenizerFast

from ...Masking.mask_layer import MaskLayer
from ...Models.circuit_model import CircuitModel
from ...Models.model_configs import CircuitConfig
from ...Probing.probe_configs import *
from ...Probing.subnetwork_probe import SubnetworkProbe
from ..residual_update_model import ResidualUpdateModel


def create_test_probe(labeling="sequence"):
    circuit_config = CircuitConfig(
        mask_method="hard_concrete",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "neuron",
            "mask_bias": False,
            "mask_init_percentage": 0.9,
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
        updates=False,
        stream=True,
    )

    probe_config = SubnetworkProbeConfig(
        probe_vectors="mlp_stream_1",
        n_classes=2,
        circuit_config=circuit_config,
        resid_config=resid_config,
        labeling=labeling,
    )

    bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
    probe = SubnetworkProbe(probe_config, bert)
    return probe


def test_subnetwork_probe_initialization():
    # Assert that you can initialize a SubnetworkProbe Model
    create_test_probe()


def test_incompatible_configuration_exceptions():
    # Assert that an error is thrown if configurations are incompatible with subnetwork probing
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
        "bert",
        target_layers=[1],
        mlp=True,
        attn=True,
        updates=False,
        stream=True,
    )

    probe_config = SubnetworkProbeConfig(
        probe_vectors="mlp_stream_1",
        n_classes=2,
        circuit_config=circuit_config,
        resid_config=resid_config,
    )

    with pytest.raises(Exception):
        SubnetworkProbe(probe_config, bert)

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
        target_layers=[1, 2],
        mlp=True,
        attn=False,
        updates=False,
        stream=True,
    )

    probe_config = SubnetworkProbeConfig(
        probe_vectors="mlp_stream_1",
        n_classes=2,
        circuit_config=circuit_config,
        resid_config=resid_config,
    )

    with pytest.raises(Exception):
        SubnetworkProbe(probe_config, bert)

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
        mlp=False,
        attn=False,
        updates=False,
        stream=True,
    )

    probe_config = SubnetworkProbeConfig(
        probe_vectors="mlp_stream_1",
        n_classes=2,
        circuit_config=circuit_config,
        resid_config=resid_config,
    )

    with pytest.raises(Exception):
        SubnetworkProbe(probe_config, bert)

    probe_config = SubnetworkProbeConfig(
        probe_vectors="mlp_stream_1",
        n_classes=2,
        circuit_config=circuit_config,
        resid_config=resid_config,
        labeling="error",
    )

    with pytest.raises(Exception):
        SubnetworkProbe(probe_config, bert)


def test_subnetwork_probe_training_state():
    # Assert that everything is frozen/in eval mode except the target layer's mask parameters,
    # even when the overall model is in training mode

    probe = create_test_probe()
    probe.train()

    for name, m in probe.named_modules():
        if (
            issubclass(m.__class__, MaskLayer)
            or issubclass(m.__class__, SubnetworkProbe)
            or issubclass(m.__class__, ResidualUpdateModel)
            or issubclass(m.__class__, CircuitModel)
            or "probe" in name
            or "loss" in name
        ):
            assert m.training == True
        else:
            assert m.training == False

    probe.train(False)

    for name, m in probe.named_modules():
        print(name)
        assert m.training == False


def test_forward_pass_sequence():
    # Assert that forward passes work when labels are at the sequence level

    probe = create_test_probe(labeling="sequence")
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")

    input = ["the cat sleeps on the mat", "the dog rests on the rug"]
    token_mask = torch.tensor(
        [[1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]]
    ).bool()
    labels = torch.tensor([[1], [0]])
    encs = tokenizer(input, return_tensors="pt")
    _ = probe(**encs, labels=labels, token_mask=token_mask)

    # Assert that residual stream updates are created in the right layer
    assert probe.config.probe_vectors in probe.wrapped_model.vector_cache

    # Assert that # labels must be equal to # of input examples
    with pytest.raises(Exception):
        input = ["the cat sleeps on the mat"]
        token_mask = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]]).bool()
        labels = torch.tensor([[0, 1]])
        encs = tokenizer(input, return_tensors="pt")
        out = probe(**encs, labels=labels, token_mask=token_mask)

    with pytest.raises(Exception):
        input = ["the cat sleeps on the mat"]
        token_mask = torch.tensor([[1, 0, 0, 1, 0, 0, 0, 0]]).bool()
        labels = torch.tensor([[1]])
        encs = tokenizer(input, return_tensors="pt")
        out = probe(**encs, labels=labels, token_mask=token_mask)


def test_forward_pass_token():
    # Assert that forward passes work when labels are at the token level

    probe = create_test_probe("token")
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")

    input = ["the cat sleeps on the mat", "the dog rests on the rug"]
    token_mask = torch.tensor(
        [[0, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 0]]
    ).bool()
    labels = torch.tensor([[1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0]])
    encs = tokenizer(input, return_tensors="pt")
    _ = probe(**encs, labels=labels, token_mask=token_mask)

    # Assert that residual stream updates are created in the right layer
    assert probe.config.probe_vectors in probe.wrapped_model.vector_cache

    # Assert that # labels must be equal to # of tokens in mask examples
    with pytest.raises(Exception):
        input = ["the cat sleeps on the mat"]
        token_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0]]).bool()
        labels = torch.tensor([[1]])
        encs = tokenizer(input, return_tensors="pt")
        out = probe(**encs, labels=labels, token_mask=token_mask)

    with pytest.raises(Exception):
        input = ["the cat sleeps on the mat"]
        token_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0]]).bool()
        labels = torch.tensor([[1, 0, 0, 0, 1, 1, 1, 0]])
        encs = tokenizer(input, return_tensors="pt")
        out = probe(**encs, labels=labels, token_mask=token_mask)


def test_training_subnetwork_probe():
    # Assert that training subnetwork probe only updates the mask parameters
    probe = create_test_probe(labeling="sequence")
    original_probe = deepcopy(probe)
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")

    input_batch = [
        "the cat sleeps on the mat",
        "the dog rests in the rug",
    ]
    token_mask = torch.tensor(
        [[1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]]
    ).bool()
    encs = tokenizer(input_batch, return_tensors="pt")
    labels = torch.tensor([[0], [1]])

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
        elif "probe" in name:
            assert ~torch.all(param == original_probe.state_dict()[name])
        else:
            assert torch.all(param == original_probe.state_dict()[name])
