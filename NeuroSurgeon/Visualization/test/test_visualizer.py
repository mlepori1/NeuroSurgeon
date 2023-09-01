import pytest
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
)

from ...Models.circuit_model import CircuitConfig, CircuitModel
from ..visualizer import Visualizer, VisualizerConfig


def setup_bert_dataset(examples):
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
    dataset = dataset["train"].select(range(examples))
    dataloader = DataLoader(dataset, shuffle=True, batch_size=8)
    return dataloader


def setup_gpt2_dataset(examples):
    dataset = load_dataset("glue", "cola")
    tokenizer = GPT2Tokenizer.from_pretrained(
        "sshleifer/tiny-gpt2", pad_token="<|endoftext|>"
    )

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
    dataset = dataset["train"].select(range(examples))
    dataloader = DataLoader(dataset, shuffle=True, batch_size=8)
    return dataloader


def train_loop(model, dataloader):
    optimizer = AdamW(model.parameters(), lr=5e-2)
    num_training_steps = len(dataloader)
    progress_bar = tqdm(range(num_training_steps))
    model.to("cuda")
    model.train(True)
    for batch in dataloader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)


def test_visualization_bert():
    dataloader = setup_bert_dataset(examples=400)

    # Train two models slightly differently
    # Train Model 1
    model_1 = BertForSequenceClassification.from_pretrained(
        "prajjwal1/bert-small", num_labels=2
    )
    target_layers = list(model_1.state_dict().keys())
    target_layers = [
        ".".join(target_layer.split(".")[:-1])
        for target_layer in target_layers
        if "layer" in target_layer
        and "weight" in target_layer
        and "LayerNorm" not in target_layer
    ]
    circuit_config_1 = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 0.01,
        },
        target_layers=target_layers,
        freeze_base=True,
        add_l0=True,
    )
    circuit_model_1 = CircuitModel(circuit_config_1, model_1)
    train_loop(circuit_model_1, dataloader)
    circuit_model_1.to("cpu")

    # Train Model 2
    model_2 = BertForSequenceClassification.from_pretrained(
        "prajjwal1/bert-small", num_labels=2
    )
    target_layers = list(model_2.state_dict().keys())
    target_layers = [
        ".".join(target_layer.split(".")[:-1])
        for target_layer in target_layers
        if "layer" in target_layer
        and "weight" in target_layer
        and "LayerNorm" not in target_layer
    ]
    circuit_config_2 = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": -0.01,
        },
        target_layers=target_layers,
        freeze_base=True,
        add_l0=True,
    )
    circuit_model_2 = CircuitModel(circuit_config_2, model_2)
    train_loop(circuit_model_2, dataloader)
    circuit_model_2.to("cpu")

    # Train Model 3
    model_3 = BertForSequenceClassification.from_pretrained(
        "prajjwal1/bert-small", num_labels=2
    )
    target_layers = list(model_3.state_dict().keys())
    target_layers = [
        ".".join(target_layer.split(".")[:-1])
        for target_layer in target_layers
        if "layer" in target_layer
        and "weight" in target_layer
        and "LayerNorm" not in target_layer
    ]
    circuit_config_3 = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "neuron",
            "mask_bias": False,
            "mask_init_value": 0.00,
        },
        target_layers=target_layers,
        freeze_base=True,
        add_l0=True,
    )
    circuit_model_3 = CircuitModel(circuit_config_3, model_3)
    train_loop(circuit_model_3, dataloader)
    circuit_model_3.to("cpu")

    # Visualize subnetworks in various configurations and see if everything looks right
    models = [circuit_model_1, circuit_model_2]

    # Test plotting layers
    Visualizer(
        VisualizerConfig(
            models,
            figsize=(10, 10),
            plot_granularity="layer",
            title="Layer by Layer Subnetwork Overlap",
            outfile="Visualization/test/figs/plot_layers_test.png",
            alpha=0.6,
        )
    ).plot()

    # Test plotting blocks
    Visualizer(
        VisualizerConfig(
            models,
            figsize=(10, 10),
            plot_granularity="block",
            model_architecture="bert",
            num_heads=8,
            hidden_size=512,
            title="Block Subnetwork Overlap",
            outfile="Visualization/test/figs/plot_blocks_test.png",
            alpha=0.6,
        )
    ).plot()

    # Test plotting tensors
    Visualizer(
        VisualizerConfig(
            models,
            figsize=(10, 30),
            plot_granularity="tensor",
            title="Node by Node Subnetwork Overlap",
            outfile="Visualization/test/figs/plot_nodes_test.png",
            alpha=0.6,
        )
    ).plot()

    # Test plotting tensors, not full model
    Visualizer(
        VisualizerConfig(
            models,
            figsize=(10, 30),
            plot_granularity="tensor",
            plot_full_network=False,
            title="Node by Node Subnetwork Overlap (Just Masked)",
            outfile="Visualization/test/figs/plot_masked_nodes_test.png",
            alpha=0.6,
        )
    ).plot()

    # Test plotting blocks, at neuron level
    Visualizer(
        VisualizerConfig(
            [circuit_model_3],
            figsize=(10, 30),
            plot_granularity="block",
            model_architecture="bert",
            num_heads=8,
            hidden_size=512,
            title="Neuron-Level Subnetwork",
            outfile="Visualization/test/figs/plot_neuron_blocks_test.png",
            alpha=0.6,
        )
    ).plot()


def test_bert_block_visualization():
    """Test block visualization, especially splitting by attention heads"""
    dataloader = setup_bert_dataset(examples=80)

    # Train two models slightly differently
    # Train Model 1
    model_1 = BertForSequenceClassification.from_pretrained(
        "prajjwal1/bert-small", num_labels=2
    )
    model_1.prune_heads({0: [0], 1: [0], 2: [0], 3: [0]})

    target_layers = list(model_1.state_dict().keys())
    target_layers = [
        ".".join(target_layer.split(".")[:-1])
        for target_layer in target_layers
        if "layer" in target_layer
        and "weight" in target_layer
        and "LayerNorm" not in target_layer
    ]
    circuit_config_1 = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 0.00,
        },
        target_layers=target_layers,
        freeze_base=True,
        add_l0=True,
    )
    circuit_model_1 = CircuitModel(circuit_config_1, model_1)
    train_loop(circuit_model_1, dataloader)
    circuit_model_1.to("cpu")

    # Train Model 2
    model_2 = BertForSequenceClassification.from_pretrained(
        "prajjwal1/bert-small", num_labels=2
    )
    target_layers = list(model_2.state_dict().keys())
    target_layers = [
        ".".join(target_layer.split(".")[:-1])
        for target_layer in target_layers
        if "layer" in target_layer
        and "weight" in target_layer
        and "LayerNorm" not in target_layer
    ]
    circuit_config_2 = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 0.00,
        },
        target_layers=target_layers,
        freeze_base=True,
        add_l0=True,
    )
    circuit_model_2 = CircuitModel(circuit_config_2, model_2)
    train_loop(circuit_model_2, dataloader)
    circuit_model_2.to("cpu")

    # Should work because of head pruning
    Visualizer(
        VisualizerConfig(
            [circuit_model_1],
            figsize=(10, 15),
            plot_granularity="block",
            model_architecture="bert",
            num_heads=7,
            hidden_size=448,
            title="Block Subnetwork",
            outfile="Visualization/test/figs/bert_pruned_blocks.png",
            alpha=0.6,
        )
    ).plot()

    # Should fail because of head pruning
    with pytest.raises(Exception):
        Visualizer(
            VisualizerConfig(
                [circuit_model_1],
                figsize=(10, 15),
                plot_granularity="block",
                model_architecture="bert",
                num_heads=model_1.config.num_attention_heads,
                hidden_size=model_1.config.hidden_size,
                title="Block Subnetwork",
                outfile="Visualization/test/figs/bert_should_fail.png",
                alpha=0.6,
            )
        ).plot()

    Visualizer(
        VisualizerConfig(
            [circuit_model_2],
            figsize=(10, 15),
            plot_granularity="block",
            model_architecture="bert",
            num_heads=model_2.config.num_attention_heads,
            hidden_size=model_2.config.hidden_size,
            title="Block Subnetwork",
            outfile="Visualization/test/figs/bert_unpruned_blocks.png",
            alpha=0.6,
        )
    ).plot()

    # Should fail because of head pruning
    with pytest.raises(Exception):
        Visualizer(
            VisualizerConfig(
                [circuit_model_2],
                figsize=(10, 15),
                plot_granularity="block",
                model_architecture="bert",
                num_heads=7,
                hidden_size=448,
                title="Block Subnetwork",
                outfile="Visualization/test/figs/bert_should_fail.png",
                alpha=0.6,
            )
        ).plot()


def test_gpt2_block_visualization():
    """Test with GPT because of different configuration of attention weights"""
    dataloader = setup_gpt2_dataset(examples=64)

    # Train two models slightly differently
    # Train Model 1

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token="<|endoftext|>")

    model_1 = GPT2ForSequenceClassification.from_pretrained(
        "gpt2", num_labels=2, pad_token_id=tokenizer.pad_token_id
    )

    model_1.prune_heads(
        {
            0: [0],
            1: [0],
            2: [0],
            3: [0],
            4: [0],
            5: [0],
            6: [0],
            7: [0],
            8: [0],
            9: [0],
            10: [0],
            11: [0],
        }
    )

    target_layers = list(model_1.state_dict().keys())
    target_layers = [
        ".".join(target_layer.split(".")[:-1])
        for target_layer in target_layers
        if ".h." in target_layer
        and "weight" in target_layer
        and "ln" not in target_layer
    ]
    circuit_config_1 = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 0.00,
        },
        target_layers=target_layers,
        freeze_base=True,
        add_l0=True,
    )
    circuit_model_1 = CircuitModel(circuit_config_1, model_1)
    train_loop(circuit_model_1, dataloader)
    circuit_model_1.to("cpu")

    # Train Model 2
    model_2 = GPT2ForSequenceClassification.from_pretrained(
        "gpt2", num_labels=2, pad_token_id=tokenizer.pad_token_id
    )
    target_layers = list(model_2.state_dict().keys())
    target_layers = [
        ".".join(target_layer.split(".")[:-1])
        for target_layer in target_layers
        if ".h." in target_layer
        and "weight" in target_layer
        and "ln" not in target_layer
    ]
    circuit_config_2 = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": 0.00,
        },
        target_layers=target_layers,
        freeze_base=True,
        add_l0=True,
    )
    circuit_model_2 = CircuitModel(circuit_config_2, model_2)
    train_loop(circuit_model_2, dataloader)
    circuit_model_2.to("cpu")

    # Should work because of head pruning
    Visualizer(
        VisualizerConfig(
            [circuit_model_1],
            figsize=(25, 80),
            plot_granularity="block",
            model_architecture="gpt2",
            num_heads=11,
            hidden_size=704,
            title="Block Subnetwork",
            outfile="Visualization/test/figs/gpt_pruned_blocks.png",
            alpha=0.6,
        )
    ).plot()

    # Should fail because of head pruning
    with pytest.raises(Exception):
        Visualizer(
            VisualizerConfig(
                [circuit_model_1],
                figsize=(25, 80),
                plot_granularity="block",
                model_architecture="gpt2",
                num_heads=model_1.config.num_attention_heads,
                hidden_size=model_1.config.hidden_size,
                title="Block Subnetwork",
                outfile="Visualization/test/figs/gpt_should_fail.png",
                alpha=0.6,
            )
        ).plot()

    Visualizer(
        VisualizerConfig(
            [circuit_model_2],
            figsize=(25, 80),
            plot_granularity="block",
            model_architecture="gpt2",
            num_heads=model_2.config.num_attention_heads,
            hidden_size=model_2.config.hidden_size,
            title="Block Subnetwork",
            outfile="Visualization/test/figs/gpt_unpruned_blocks.png",
            alpha=0.6,
        )
    ).plot()

    # Should fail because of head pruning
    with pytest.raises(Exception):
        Visualizer(
            VisualizerConfig(
                [circuit_model_2],
                figsize=(25, 80),
                plot_granularity="block",
                model_architecture="gpt2",
                num_heads=11,
                hidden_size=704,
                title="Block Subnetwork",
                outfile="Visualization/test/figs/gpt_should_fail.png",
                alpha=0.6,
            )
        ).plot()
