import pytest
from ..circuit_model import CircuitModel
from ..circuit_probe import CircuitProbe
from ..residual_update_model import ResidualUpdateModel
from ..model_configs import *
from transformers import BertModel, BertTokenizerFast
from ...Masking.mask_layer import MaskLayer
import torch
from torch.optim import AdamW
from copy import deepcopy


def create_test_probe():
    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={"ablation": "none", "mask_bias": False, "mask_init_value": 0.0},
        target_layers=["encoder.layer.1.output.dense"],
        freeze_base=True,
        add_l0=True,
        l0_lambda=1e-8,
    )

    resid_config = ResidualUpdateModelConfig(
        "bert", target_layers=[1], mlp=True, attn=False, circuit=True, base=True
    )

    probe_config = CircuitProbeConfig(
        probe_updates="mlp_1", circuit_config=circuit_config, resid_config=resid_config
    )

    bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")

    probe = CircuitProbe(probe_config, bert, tokenizer)
    return probe


def test_circuit_probe_initialization():
    # Assert that you can initialize a CircuitProbe Model
    create_test_probe()


def test_incompatible_configuration_exceptions():
    # Assert that an error is thrown if configurations are incompatible with circuit probing
    bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={"ablation": "none", "mask_bias": False, "mask_init_value": 0.0},
        target_layers=["encoder.layer.1.output.dense"],
        freeze_base=False,
        add_l0=True,
        l0_lambda=1e-8,
    )

    resid_config = ResidualUpdateModelConfig(
        "bert", target_layers=[1], mlp=True, attn=False, circuit=True, base=True
    )

    probe_config = CircuitProbeConfig(
        probe_updates="mlp_1", circuit_config=circuit_config, resid_config=resid_config
    )

    with pytest.raises(Exception):
        CircuitProbe(probe_config, bert, tokenizer)

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={"ablation": "none", "mask_bias": False, "mask_init_value": 0.0},
        target_layers=["encoder.layer.1.output.dense"],
        freeze_base=True,
        add_l0=True,
        l0_lambda=1e-8,
    )

    resid_config = ResidualUpdateModelConfig(
        "bert", target_layers=[1, 2], mlp=True, attn=False, circuit=True, base=True
    )

    probe_config = CircuitProbeConfig(
        probe_updates="mlp_1", circuit_config=circuit_config, resid_config=resid_config
    )

    with pytest.raises(Exception):
        CircuitProbe(probe_config, bert, tokenizer)

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={"ablation": "none", "mask_bias": False, "mask_init_value": 0.0},
        target_layers=["encoder.layer.1.output.dense"],
        freeze_base=True,
        add_l0=True,
        l0_lambda=1e-8,
    )

    resid_config = ResidualUpdateModelConfig(
        "bert", target_layers=[1], mlp=False, attn=False, circuit=True, base=True
    )

    probe_config = CircuitProbeConfig(
        probe_updates="mlp_1", circuit_config=circuit_config, resid_config=resid_config
    )

    with pytest.raises(Exception):
        CircuitProbe(probe_config, bert, tokenizer)


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


def test_token_mask_creation():
    # Assert that token mask is correct, it ignores subwords and special tokens
    probe = create_test_probe()

    input_strings = ["This is a testtesttest", "This oneoneone is too"]

    mask = probe._create_token_mask(probe.tokenizer(input_strings)["input_ids"])

    ground_truth = torch.Tensor(
        [
            [False, True, True, True, True, False, False, False],
            [False, True, True, False, False, True, True, False],
        ]
    ).bool()

    assert torch.all(ground_truth == mask)


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
    input = ["the cat sleeps on the mat", "the dog rests in the rug"]
    labels = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    encs = probe.tokenizer(input, return_tensors="pt")
    out = probe(**encs, labels=labels)

    # Assert that residual stream updates are created in the right layer
    assert probe.config.probe_updates in probe.wrapped_model.residual_stream_updates

    low_loss = out.loss

    input = ["the cat sleeps on the mat", "jump near the big red flag"]
    labels = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    encs = probe.tokenizer(input, return_tensors="pt")
    out = probe(**encs, labels=labels)

    high_loss = out.loss

    assert low_loss < high_loss

    # Assert that # labels must be equal to # of relevant (i.e. not special, not subword) tokens
    with pytest.raises(Exception):
        input = ["the cat sleeps on the mat"]
        labels = torch.tensor([[0, 1, 2, 3, 4]])
        encs = probe.tokenizer(input, return_tensors="pt")
        out = probe(**encs, labels=labels)

    with pytest.raises(Exception):
        input = ["the catcat sleeps on the mat"]
        labels = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
        encs = probe.tokenizer(input, return_tensors="pt")
        out = probe(**encs, labels=labels)

    input = ["the catcat sleeps on the mat"]
    labels = torch.tensor([[0, 1, 2, 3, 4, 5]])
    encs = probe.tokenizer(input, return_tensors="pt")
    out = probe(**encs, labels=labels)


def test_training_circuit_probe():
    # Assert that training circuit probe only updates the mask parameters
    probe = create_test_probe()
    original_probe = deepcopy(probe)

    input_batch = [
        "the cat sleeps on the mat",
        "the dog rests in the rug",
    ]
    encs = probe.tokenizer(input_batch, return_tensors="pt")
    labels = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])

    optimizer = AdamW(probe.parameters(), lr=5e-2)

    probe.train()

    outputs = probe(**encs, labels=labels)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    for name, param in probe.named_parameters():
        if "weight_mask_param" in name:
            assert ~torch.all(param == original_probe.state_dict()[name])
        else:
            assert torch.all(param == original_probe.state_dict()[name])
