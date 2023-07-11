import pytest
import torch
from ..circuit_model import CircuitModel
from ..model_configs import CircuitConfig
from ...Masking.mask_layer import MaskLayer
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import (
    GPT2Model,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    ResNetModel,
    ResNetForImageClassification,
    ViTModel,
    ViTForImageClassification,
    AutoImageProcessor,
    Trainer,
    TrainingArguments,
)


def test_replace_layers_gpt():
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
    ### GPT2
    gpt = GPT2Model.from_pretrained("sshleifer/tiny-gpt2")

    circuit_config.target_layers = ["h.0.attn.c_proj", "h.0.mlp.c_fc", "h.1.mlp.c_proj"]
    circuit_model = CircuitModel(circuit_config, gpt)
    circuit_modules = {}
    for name, mod in circuit_model.root_model.named_modules():
        circuit_modules[name] = mod

    clean_gpt = GPT2Model.from_pretrained("sshleifer/tiny-gpt2")
    for name, mod in clean_gpt.named_modules():
        if name not in circuit_config.target_layers and name != "":
            assert type(mod) == type(circuit_modules[name])
        elif name != "":  # Name for overall GPTModel
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

    gpt = GPT2Model.from_pretrained("sshleifer/tiny-gpt2")
    circuit_config.target_layers = [
        "h.0.attn.c_proj",
        "h.0.mlp.c_fc",
        "h.1.mlp.c_proj" "h.0.mlp.dropout",  # In Model, unsupported layer type
    ]
    with pytest.raises(Exception):
        circuit_model = CircuitModel(circuit_config, gpt)

    gpt = GPT2Model.from_pretrained("sshleifer/tiny-gpt2")
    circuit_config.target_layers = ["some_layer"]  # Not in model
    with pytest.raises(Exception):
        circuit_model = CircuitModel(circuit_config, gpt)


def test_replace_layers_resnet():
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
    ### ResNet
    rn = ResNetModel.from_pretrained("microsoft/resnet-18")

    circuit_config.target_layers = [
        "encoder.stages.0.layers.0.layer.0.convolution",
        "encoder.stages.3.layers.0.shortcut.convolution",
    ]
    circuit_model = CircuitModel(circuit_config, rn)
    circuit_modules = {}
    for name, mod in circuit_model.root_model.named_modules():
        circuit_modules[name] = mod

    clean_rn = ResNetModel.from_pretrained("microsoft/resnet-18")
    for name, mod in clean_rn.named_modules():
        if name not in circuit_config.target_layers and name != "":
            assert type(mod) == type(circuit_modules[name])
        elif name != "":  # Name for overall ResNetModel
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

    # Make sure unsupported layers result in an error, even if they exist in network
    rn = ResNetModel.from_pretrained("microsoft/resnet-18")
    circuit_config.target_layers = [
        "encoder.stages.0.layers.0.layer.0.convolution",
        "encoder.stages.3.layers.0.shortcut.convolution",
        "pooler",
    ]

    with pytest.raises(Exception):
        circuit_model = CircuitModel(circuit_config, rn)

    rn = ResNetModel.from_pretrained("microsoft/resnet-18")
    circuit_config.target_layers = [
        "encoder.stages.0.layers.0.layer.0.convolution",
        "encoder.stages.3.layers.0.shortcut.convolution",
        "encoder.stages.3.layers.1.layer.0.normalization",
    ]
    with pytest.raises(Exception):
        circuit_model = CircuitModel(circuit_config, rn)

    # Make sure layers that aren't in a model throw an error
    rn = ResNetModel.from_pretrained("microsoft/resnet-18")
    circuit_config.target_layers = ["some_layer"]  # Not in model
    with pytest.raises(Exception):
        circuit_model = CircuitModel(circuit_config, rn)


def test_replace_layers_vit():
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
    vit = ViTModel.from_pretrained("lysandre/tiny-vit-random")

    circuit_config.target_layers = [
        "encoder.layer.0.attention.attention.query",
        "encoder.layer.1.attention.output.dense",
    ]
    circuit_model = CircuitModel(circuit_config, vit)
    circuit_modules = {}
    for name, mod in circuit_model.root_model.named_modules():
        circuit_modules[name] = mod

    clean_vit = ViTModel.from_pretrained("lysandre/tiny-vit-random")
    for name, mod in clean_vit.named_modules():
        if name not in circuit_config.target_layers and name != "":
            assert type(mod) == type(circuit_modules[name])
        elif name != "":  # Name for overall ViTModel
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

    # Make sure unsupported layers result in an error, even if they exist in network
    vit = ViTModel.from_pretrained("lysandre/tiny-vit-random")
    circuit_config.target_layers = [
        "encoder.layer.0.attention.attention.query",
        "encoder.layer.1.attention.output.dense",
        "encoder.layer.6.layernorm_after",
    ]

    with pytest.raises(Exception):
        circuit_model = CircuitModel(circuit_config, vit)

    vit = ViTModel.from_pretrained("lysandre/tiny-vit-random")
    circuit_config.target_layers = [
        "encoder.layer.0.attention.attention.query",
        "encoder.layer.1.attention.output.dense",
        "encoder.layer.6.output.dropout",
    ]
    with pytest.raises(Exception):
        circuit_model = CircuitModel(circuit_config, vit)

    # Make sure layers that aren't in a model throw an error
    vit = ViTModel.from_pretrained("lysandre/tiny-vit-random")
    circuit_config.target_layers = ["some_layer"]  # Not in model
    with pytest.raises(Exception):
        circuit_model = CircuitModel(circuit_config, vit)


def test_forward_pass_gpt():
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
            "transformer.h.0.attn.c_proj",
            "transformer.h.0.mlp.c_fc",
            "transformer.h.1.mlp.c_proj",
        ],
        freeze_base=True,
        add_l0=True,
    )

    tokenizer = GPT2Tokenizer.from_pretrained("sshleifer/tiny-gpt2")
    inputs = tokenizer("Hello, this is a test", return_tensors="pt")

    model = GPT2ForSequenceClassification.from_pretrained("sshleifer/tiny-gpt2")
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


def test_forward_pass_resnet():
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
            "resnet.encoder.stages.0.layers.0.layer.0.convolution",
            "resnet.encoder.stages.3.layers.0.shortcut.convolution",
        ],
        freeze_base=True,
        add_l0=True,
    )

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    im_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
    inputs = im_processor(image, return_tensors="pt")

    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
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


def test_forward_pass_vit():
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
            "vit.encoder.layer.9.output.dense",
            "vit.encoder.layer.11.attention.attention.query",
        ],
        freeze_base=True,
        add_l0=True,
    )

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    im_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    inputs = im_processor(image, return_tensors="pt")

    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
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


def setup_gpt_dataset(tokenizer):
    dataset = load_dataset("glue", "cola")

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


def test_training_gpt():
    tokenizer = GPT2Tokenizer.from_pretrained(
        "sshleifer/tiny-gpt2", pad_token="<|endoftext|>"
    )

    dataloader = setup_gpt_dataset(tokenizer)

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
            "transformer.h.1.mlp.c_proj",
        ],
        freeze_base=True,
        add_l0=True,
    )
    model = GPT2ForSequenceClassification.from_pretrained(
        "sshleifer/tiny-gpt2", num_labels=2, pad_token_id=tokenizer.pad_token_id
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
            "transformer.h.1.mlp.c_proj",
        ],
        freeze_base=False,
        add_l0=True,
    )

    model = GPT2ForSequenceClassification.from_pretrained(
        "sshleifer/tiny-gpt2", num_labels=2, pad_token_id=tokenizer.pad_token_id
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


def setup_image_dataset(processor_name):
    dataset = load_dataset("beans")
    image_processor = AutoImageProcessor.from_pretrained(processor_name)

    def process(examples):
        return image_processor(
            examples["image"],
            return_tensors="pt",
        )

    dataset = dataset.map(process, batched=True)
    dataset = dataset.remove_columns(["image_file_path"])
    dataset = dataset.remove_columns(["image"])
    dataset = dataset.with_format("torch")
    dataset = dataset["train"].select(range(16))
    dataloader = DataLoader(dataset, shuffle=False, batch_size=8)
    return dataloader


def test_training_resnet():
    dataloader = setup_image_dataset("microsoft/resnet-18")

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
            "resnet.encoder.stages.0.layers.0.layer.0.convolution",
        ],
        freeze_base=True,
        add_l0=True,
    )
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
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
            "resnet.encoder.stages.0.layers.0.layer.0.convolution",
        ],
        freeze_base=False,
        add_l0=True,
    )

    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
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


def test_training_vit():
    dataloader = setup_image_dataset("lysandre/tiny-vit-random")

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
            "vit.encoder.layer.0.attention.attention.query",
        ],
        freeze_base=True,
        add_l0=True,
    )
    model = ViTForImageClassification.from_pretrained("lysandre/tiny-vit-random")
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
            "vit.encoder.layer.0.attention.attention.query",
        ],
        freeze_base=False,
        add_l0=True,
    )

    model = ViTForImageClassification.from_pretrained("lysandre/tiny-vit-random")
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
