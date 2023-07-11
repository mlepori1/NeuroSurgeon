import pytest
import torch
import torch.nn as nn
from ..circuit_model import CircuitModel
from ..model_configs import CircuitConfig
from ...Masking.mask_layer import MaskLayer
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BertModel,
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)


def test_replace_layers_bert():
    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": True,
            "mask_init_value": 1.0,
        },
        target_layers=[],
        freeze_base=True,
        add_l0=False,
    )
    # Assert only correct layers are switched, that you cannot mask layers besides conv and linear layers
    ### BERT
    bert = BertModel.from_pretrained("prajjwal1/bert-tiny")

    circuit_config.target_layers = [
        "encoder.layer.0.attention.self.query",
        "encoder.layer.1.output.dense",
        "pooler.dense",
    ]
    circuit_model = CircuitModel(circuit_config, bert)
    circuit_modules = {}
    for name, mod in circuit_model.root_model.named_modules():
        circuit_modules[name] = mod

    clean_bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
    for name, mod in clean_bert.named_modules():
        if name not in circuit_config.target_layers and name != "":
            assert type(mod) == type(circuit_modules[name])
        elif name != "":  # Name for overall BertModel
            # Assert that all consituent layers in config.target_layers are MaskLayers
            assert issubclass(type(circuit_modules[name]), MaskLayer)
            # Assert that masks are right, and that weights are the same as underlying model
            assert torch.all(mod.weight == circuit_modules[name].weight)
            if hasattr(mod, "bias") and mod.bias is not None:
                assert torch.all(mod.bias == circuit_modules[name].bias)
            assert torch.all(
                circuit_modules[name].weight_mask_params
                == torch.full(
                    circuit_modules[name].weight_mask_params.shape,
                    circuit_config.mask_hparams["mask_init_value"],
                )
            )

    bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
    circuit_config.target_layers = [
        "encoder.layer.0.attention.self.query",
        "encoder.layer.1.output.dense",
        "pooler.dense",
        "encoder.layer.1.attention.output.LayerNorm",  # In Model, unsupported layer type
    ]
    with pytest.raises(Exception):
        circuit_model = CircuitModel(circuit_config, bert)

    bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
    circuit_config.target_layers = ["some_layer"]  # Not in model
    with pytest.raises(Exception):
        circuit_model = CircuitModel(circuit_config, bert)


def test_model_freezing_bert():
    # Test with freeze_base == True
    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": True,
            "mask_init_value": 1.0,
        },
        target_layers=[],
        freeze_base=True,
        add_l0=False,
    )

    bert = BertModel.from_pretrained("prajjwal1/bert-tiny")

    circuit_config.target_layers = [
        "encoder.layer.0.attention.self.query",
        "encoder.layer.1.output.dense",
        "pooler.dense",
    ]
    circuit_model = CircuitModel(circuit_config, bert)
    circuit_model.train(True)

    # Assert that only weight_mask_param and bias_mask_param require gradients
    # Assert that everything except mask_layers are in eval mode
    for name, mod in circuit_model.root_model.named_modules():
        if name in circuit_config.target_layers:
            assert mod.training == True
            assert mod.weight.requires_grad == False
            assert mod.weight_mask_params.requires_grad == True

            if hasattr(mod, "bias") and mod.bias is not None:
                assert mod.bias.requires_grad == False
                assert mod.bias_mask_params.requires_grad == True
        else:
            assert mod.training == False

    circuit_model.train(False)

    for name, mod in circuit_model.root_model.named_modules():
        if name in circuit_config.target_layers:
            assert mod.training == False
        else:
            assert mod.training == False

    # Test with freeze_base == False
    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": True,
            "mask_init_value": 1.0,
        },
        target_layers=[],
        freeze_base=False,
        add_l0=False,
    )

    bert = BertModel.from_pretrained("prajjwal1/bert-tiny")

    circuit_config.target_layers = [
        "encoder.layer.0.attention.self.query",
        "encoder.layer.1.output.dense",
        "pooler.dense",
    ]
    circuit_model = CircuitModel(circuit_config, bert)
    circuit_model.train(True)

    # Assert that everything is in train mode and requires_gradients
    for name, mod in circuit_model.root_model.named_modules():
        if name in circuit_config.target_layers:
            assert mod.training == True
            assert mod.weight.requires_grad == True
            assert mod.weight_mask_params.requires_grad == True

            if hasattr(mod, "bias") and mod.bias is not None:
                assert mod.bias.requires_grad == True
                assert mod.bias_mask_params.requires_grad == True
        else:
            assert mod.training == True

    circuit_model.train(False)

    for name, mod in circuit_model.root_model.named_modules():
        if name in circuit_config.target_layers:
            assert mod.training == False
        else:
            assert mod.training == False


def test_forward_pass_bert():
    # Assert that underlying model and masked model with positive mask weights
    # give the same output for eval mode, different for train mode

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": True,
            "mask_init_value": 1.0,
        },
        target_layers=[
            "bert.encoder.layer.0.attention.self.query",
            "bert.encoder.layer.1.output.dense",
            "bert.pooler.dense",
        ],
        freeze_base=True,
        add_l0=True,
    )

    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
    inputs = tokenizer("Hello, this is a test", return_tensors="pt")

    model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
    model.train(False)
    base_logits = model(**inputs).logits.cpu()

    model.train(True)
    circuit_model = CircuitModel(circuit_config, model)
    circuit_model.train(False)
    eval_circuit_logits = circuit_model(**inputs).logits.cpu()

    assert torch.all(eval_circuit_logits == base_logits)

    circuit_model.train(True)
    train_circuit_logits = circuit_model(**inputs).logits.cpu()

    assert not torch.all(train_circuit_logits == base_logits)


def test_use_mask_bert():
    # Assert that underlying model and masked model with negative mask weights
    # give different output for eval mode, but same output when use_masks is set to False
    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": True,
            "mask_init_value": -1.0,
        },
        target_layers=[
            "bert.encoder.layer.0.attention.self.query",
            "bert.encoder.layer.1.output.dense",
            "bert.pooler.dense",
        ],
        freeze_base=True,
        add_l0=True,
    )

    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
    inputs = tokenizer("Hello, this is a test", return_tensors="pt")

    model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
    model.train(False)
    base_logits = model(**inputs).logits.cpu()

    model.train(True)
    circuit_model = CircuitModel(circuit_config, model)
    circuit_model.train(False)
    use_masks_logits = circuit_model(**inputs).logits.cpu()

    assert torch.all(use_masks_logits != base_logits)

    circuit_model.use_masks = False
    do_not_use_masks_logits = circuit_model(**inputs).logits.cpu()

    assert not torch.all(do_not_use_masks_logits == base_logits)


def test_l0_calc_bert():
    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": True,
            "mask_init_value": 0.0,
        },
        target_layers=[
            "bert.encoder.layer.0.output.dense",
        ],
        freeze_base=True,
        add_l0=True,
    )

    model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
    circuit_model = CircuitModel(circuit_config, model)
    l0_0_init = circuit_model.compute_l0_statistics()["total_l0"]

    assert l0_0_init == 0

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": True,
            "mask_init_value": 1.0,
        },
        target_layers=[
            "bert.encoder.layer.0.output.dense",
        ],
        freeze_base=True,
        add_l0=True,
    )

    model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
    circuit_model = CircuitModel(circuit_config, model)
    l0_1_layer = circuit_model.compute_l0_statistics()["total_l0"]

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": True,
            "mask_init_value": 1.0,
        },
        target_layers=[
            "bert.encoder.layer.0.output.dense",
            "bert.encoder.layer.1.output.dense",
        ],
        freeze_base=True,
        add_l0=True,
    )

    model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
    circuit_model = CircuitModel(circuit_config, model)
    l0_2_layer = circuit_model.compute_l0_statistics()["total_l0"]

    assert 2 * l0_1_layer == l0_2_layer


def test_temperature():
    # Assert that different temperatures changes the output of train l0 calc
    # Assert that model outputs change accordingly
    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 1.0,
        },
        target_layers=[
            "bert.encoder.layer.0.output.dense",
        ],
        freeze_base=True,
        add_l0=True,
    )

    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
    inputs = tokenizer("Hello, this is a test", return_tensors="pt")
    model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")

    circuit_model = CircuitModel(circuit_config, model)
    l0_t_1 = circuit_model._compute_l0_loss()
    logits_t_1 = circuit_model(**inputs).logits.cpu()

    circuit_model.temperature = 2
    l0_t_2 = circuit_model._compute_l0_loss()
    logits_t_2 = circuit_model(**inputs).logits.cpu()

    assert l0_t_1 < l0_t_2
    assert torch.all(logits_t_1 != logits_t_2)

    circuit_model.train(False)
    logits_test = circuit_model(**inputs).logits.cpu()

    assert torch.all(logits_t_2 != logits_test)

    # Assert that train l0 calc == test l0 calc at high temperatures
    # Assert that model outputs are the same
    circuit_model.train(True)
    circuit_model.temperature = 1000

    test_l0 = circuit_model.compute_l0_statistics()["total_l0"]
    logits_high_temp = circuit_model(**inputs).logits.cpu()
    high_temp_l0 = circuit_model._compute_l0_loss()
    assert torch.all(logits_high_temp == logits_test)
    assert test_l0 == high_temp_l0


def test_ablate_mode_bert():
    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 1.0,
        },
        target_layers=[
            "bert.encoder.layer.0.output.dense",
        ],
        freeze_base=True,
        add_l0=True,
    )

    model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
    circuit_model = CircuitModel(circuit_config, model)
    base_l0 = circuit_model.compute_l0_statistics()["total_l0"]

    # Assert that masks are inverted if ablate mode is switched to zero_ablate or random_ablate
    circuit_model.set_ablate_mode("zero_ablate")
    zero_l0 = circuit_model.compute_l0_statistics()["total_l0"]
    assert zero_l0 == 0.0

    circuit_model.set_ablate_mode("random_ablate")
    rand_l0 = circuit_model.compute_l0_statistics()["total_l0"]
    assert rand_l0 == 0.0

    # Assert that switching back works as well.
    circuit_model.set_ablate_mode("none")
    back_l0 = circuit_model.compute_l0_statistics()["total_l0"]
    assert base_l0 == back_l0

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": -1.0,
        },
        target_layers=[
            "bert.encoder.layer.0.output.dense",
        ],
        freeze_base=True,
        add_l0=True,
    )

    model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
    circuit_model = CircuitModel(circuit_config, model)
    neg_mask_base_l0 = circuit_model.compute_l0_statistics()["total_l0"]
    assert neg_mask_base_l0 == zero_l0

    circuit_model.set_ablate_mode("zero_ablate")
    neg_mask_zero_l0 = circuit_model.compute_l0_statistics()["total_l0"]

    assert neg_mask_zero_l0 == base_l0


def test_compute_l0_loss_bert():
    # Assert that L0 loss is correct in train mode, returns 0 in eval mode
    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 0.75,
        },
        target_layers=[
            "bert.encoder.layer.0.output.dense",
        ],
        freeze_base=True,
        add_l0=True,
    )

    model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
    circuit_model = CircuitModel(circuit_config, model)
    train_l0_loss = circuit_model._compute_l0_loss()
    mod_dict = {item[0]: item[1] for item in circuit_model.named_modules()}
    hypothesized_l0_loss = len(
        mod_dict["root_model." + circuit_config.target_layers[0]].weight.reshape(-1)
    ) * nn.functional.sigmoid(torch.tensor([0.75]))
    assert train_l0_loss.detach() == pytest.approx(hypothesized_l0_loss.item(), 0.01)

    circuit_model.train(False)
    assert circuit_model._compute_l0_loss() == 0.0


def setup_dataset():
    dataset = load_dataset("glue", "cola")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(["sentence"])
    dataset = dataset.remove_columns(["idx"])
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.with_format("torch")
    dataset = dataset["train"].select(range(16))
    dataloader = DataLoader(dataset, shuffle=False, batch_size=8)
    return dataloader


def train_loop(model, dataloader):
    optimizer = AdamW(model.parameters(), lr=5e-2)
    num_training_steps = len(dataloader)
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for batch in dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)


def test_training_bert():
    dataloader = setup_dataset()

    # Check that the training runs with the circuit model without bugs with frozen underlying model
    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 0.0,
        },
        target_layers=[
            "bert.encoder.layer.0.output.dense",
        ],
        freeze_base=True,
        add_l0=True,
    )

    model = BertForSequenceClassification.from_pretrained(
        "prajjwal1/bert-tiny", num_labels=2
    )
    circuit_model = CircuitModel(circuit_config, model)

    mod_dict = {item[0]: item[1] for item in circuit_model.named_modules()}

    before_weight = torch.clone(
        mod_dict["root_model." + circuit_config.target_layers[0]].weight
    )
    before_mask = torch.clone(
        mod_dict["root_model." + circuit_config.target_layers[0]].weight_mask_params
    )

    train_loop(circuit_model, dataloader)

    # Assert that mask weights are different, but all underlying weights are the same
    mod_dict = {item[0]: item[1] for item in circuit_model.named_modules()}

    after_weight = torch.clone(
        mod_dict["root_model." + circuit_config.target_layers[0]].weight
    )
    after_mask = torch.clone(
        mod_dict["root_model." + circuit_config.target_layers[0]].weight_mask_params
    )

    assert torch.all(before_weight == after_weight)
    assert torch.all(before_mask != after_mask)

    # Assert that the trainer runs with the circuit model without bugs with unfrozen underlying model
    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 0.0,
        },
        target_layers=[
            "bert.encoder.layer.0.output.dense",
        ],
        freeze_base=False,
        add_l0=True,
    )

    model = BertForSequenceClassification.from_pretrained(
        "prajjwal1/bert-tiny", num_labels=2
    )
    circuit_model = CircuitModel(circuit_config, model)

    mod_dict = {item[0]: item[1] for item in circuit_model.named_modules()}

    before_weight = torch.clone(
        mod_dict["root_model." + circuit_config.target_layers[0]].weight
    )
    before_mask = torch.clone(
        mod_dict["root_model." + circuit_config.target_layers[0]].weight_mask_params
    )

    train_loop(circuit_model, dataloader)

    # Assert that all weights are different.
    mod_dict = {item[0]: item[1] for item in circuit_model.named_modules()}

    after_weight = torch.clone(
        mod_dict["root_model." + circuit_config.target_layers[0]].weight
    )
    after_mask = torch.clone(
        mod_dict["root_model." + circuit_config.target_layers[0]].weight_mask_params
    )

    assert torch.all(before_weight != after_weight)
    assert torch.all(before_mask != after_mask)


def test_lambda_effect_bert():
    # Assert that training with very high lambda drops L0 more than smaller lambda,
    # which drops more than 0.0 lambda. Test that L0 does drop from initialization for nonzero lambda
    dataloader = setup_dataset()

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 0.0,
        },
        target_layers=[
            "bert.encoder.layer.0.output.dense",
        ],
        freeze_base=True,
        add_l0=True,
        l0_lambda=10,
    )

    model = BertForSequenceClassification.from_pretrained(
        "prajjwal1/bert-tiny", num_labels=2
    )
    circuit_model = CircuitModel(circuit_config, model)

    train_loop(circuit_model, dataloader)

    lambda_10_l0 = circuit_model._compute_l0_loss()

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 0.0,
        },
        target_layers=[
            "bert.encoder.layer.0.output.dense",
        ],
        freeze_base=True,
        add_l0=True,
        l0_lambda=1e-7,
    )

    model = BertForSequenceClassification.from_pretrained(
        "prajjwal1/bert-tiny", num_labels=2
    )
    circuit_model = CircuitModel(circuit_config, model)

    train_loop(circuit_model, dataloader)

    lambda_smaller_l0 = circuit_model._compute_l0_loss()

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 0.0,
        },
        target_layers=[
            "bert.encoder.layer.0.output.dense",
        ],
        freeze_base=True,
        add_l0=False,
    )

    model = BertForSequenceClassification.from_pretrained(
        "prajjwal1/bert-tiny", num_labels=2
    )
    circuit_model = CircuitModel(circuit_config, model)

    train_loop(circuit_model, dataloader)

    lambda_no_l0 = circuit_model._compute_l0_loss()

    assert lambda_no_l0 > lambda_smaller_l0
    assert lambda_smaller_l0 > lambda_10_l0


def test_mask_init_effect_bert():
    # Assert that L0 for mask_init -1 < 0 < 1.
    dataloader = setup_dataset()

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": -1.0,
        },
        target_layers=[
            "bert.encoder.layer.0.output.dense",
        ],
        freeze_base=True,
        add_l0=True,
    )

    model = BertForSequenceClassification.from_pretrained(
        "prajjwal1/bert-tiny", num_labels=2
    )
    circuit_model = CircuitModel(circuit_config, model)

    train_loop(circuit_model, dataloader)

    mask_init_neg_1 = circuit_model._compute_l0_loss()

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 0.0,
        },
        target_layers=[
            "bert.encoder.layer.0.output.dense",
        ],
        freeze_base=True,
        add_l0=True,
    )

    model = BertForSequenceClassification.from_pretrained(
        "prajjwal1/bert-tiny", num_labels=2
    )
    circuit_model = CircuitModel(circuit_config, model)

    train_loop(circuit_model, dataloader)

    mask_init_zero = circuit_model._compute_l0_loss()

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 1.0,
        },
        target_layers=[
            "bert.encoder.layer.0.output.dense",
        ],
        freeze_base=True,
        add_l0=False,
    )

    model = BertForSequenceClassification.from_pretrained(
        "prajjwal1/bert-tiny", num_labels=2
    )
    circuit_model = CircuitModel(circuit_config, model)

    train_loop(circuit_model, dataloader)

    mask_init_1 = circuit_model._compute_l0_loss()

    assert mask_init_1 > mask_init_zero
    assert mask_init_zero > mask_init_neg_1


def test_trainer_compatibility_bert():
    # @todo
    pass
