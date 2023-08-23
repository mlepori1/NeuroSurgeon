from ..visualizer import Visualizer, VisualizerConfig
from ...Models.circuit_model import CircuitModel, CircuitConfig
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW


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
    dataset = dataset["train"].select(range(600))
    dataloader = DataLoader(dataset, shuffle=True, batch_size=8)
    return dataloader


def train_loop(model, dataloader):
    optimizer = AdamW(model.parameters(), lr=5e-2)
    num_training_steps = len(dataloader)
    progress_bar = tqdm(range(num_training_steps))
    model.to("cuda")
    model.train()
    for batch in dataloader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)


def test_training_bert():
    dataloader = setup_dataset()

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

    # Visualize subnetworks and see if everything looks right
    state_dicts = [circuit_model_1.state_dict(), circuit_model_2.state_dict()]
    Visualizer(
        VisualizerConfig(
            state_dicts,
            figsize=(10, 10),
            condense_layers=True,
            title="Layer by Layer Subnetwork Overlap",
            outfile="test_plot_layers.png",
            alpha=0.6,
        )
    ).plot()
    Visualizer(
        VisualizerConfig(
            state_dicts,
            figsize=(10, 30),
            condense_layers=False,
            title="Node by Node Subnetwork Overlap",
            outfile="test_plot_nodes.png",
            alpha=0.6,
        )
    ).plot()
    Visualizer(
        VisualizerConfig(
            state_dicts,
            figsize=(10, 30),
            condense_layers=False,
            plot_full_network=False,
            title="Node by Node Subnetwork Overlap (Just Masked)",
            outfile="test_plot_masked_nodes.png",
            alpha=0.6,
        )
    ).plot()
