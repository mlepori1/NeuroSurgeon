from ..Models.circuit_model import CircuitModel
from .residual_update_model import ResidualUpdateModel
from ..Probing.probe_configs import SubnetworkProbeConfig
from transformers import PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput
import torchmetrics
import torch
import torch.nn as nn


class SubnetworkProbe(nn.Module):
    def __init__(
        self,
        config: SubnetworkProbeConfig,
        model: nn.Module,
    ):
        super().__init__()
        self.config = config
        self._validate_configs()

        self.hidden_size = model.config.hidden_size
        # First create a CircuitModel
        self.wrapped_model = CircuitModel(self.config.circuit_config, model)
        # Then wrap it to get intermediate activations
        self.wrapped_model = ResidualUpdateModel(
            self.config.resid_config, self.wrapped_model
        )

        self.probe = self.create_probe()
        self.loss = nn.CrossEntropyLoss()

    def _validate_configs(self):
        # Ensure that wrapper configs specify valid behavior for circuit probing
        circuit_config = self.config.circuit_config
        resid_config = self.config.resid_config

        # Subnetwork Probing should only be performed when underlying model is frozen
        assert circuit_config.freeze_base

        # Subnetwork Probing probes particular residual stream updates
        assert len(resid_config.target_layers) == 1

        # Subnetwork probing must operate on a residal stream update or the residual stream itself
        assert (resid_config.mlp and not resid_config.attn) or (
            resid_config.attn and not resid_config.mlp
        )

        # Labeling must be either "sequence" or "token", corresponding to the probing task
        assert self.config.labeling in ["sequence", "token"]

    def create_probe(self):
        input_size = self.hidden_size
        if self.config.intermediate_size != -1:
            return nn.Sequential(
                nn.Linear(input_size, self.config.intermediate_size),
                nn.ReLU(),
                nn.Linear(self.config.intermediate_size, self.config.n_classes),
            )
        else:
            return nn.Sequential(nn.Linear(input_size, self.config.n_classes))

    def train(self, train_bool: bool = True):
        self.training = train_bool
        for module in self.children():
            module.train(train_bool)

    def forward(
        self, input_ids=None, labels=None, token_mask=None, return_dict=True, **kwargs
    ):
        # Must provide a token mask, which is a boolean mask for each input denoting which
        # residual streams to compute loss over

        # Call model forward pass, get out the correct activations
        _ = self.wrapped_model(input_ids=input_ids, **kwargs)
        updates = self.wrapped_model.vector_cache[self.config.probe_vectors]

        # Get one residual stream activation per label using mask indexing,
        # collapsing a batch of strings into a list of labels and residual stream activations
        token_mask = token_mask.reshape(-1)

        # Must be one label for each sequence if labeling==sequence
        if self.config.labeling == "sequence":
            assert torch.sum(token_mask) == len(input_ids)

        updates = updates.reshape(-1, self.hidden_size)
        updates = updates[token_mask]

        if labels is not None and self.config.labeling == "token":
            # Get rid of padding on labels (which is only used for batching)
            labels = labels[labels != -1]

        labels = labels.reshape(-1)
        assert len(updates) == len(
            labels
        )  # Ensure that there is only one update per label

        logits = self.probe(updates)

        loss = None
        if labels is not None:
            loss = self.loss(logits, labels)

        if not return_dict:
            return (loss,) + (updates,) if loss is not None else updates

        output = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=updates,
            attentions=None,
        )
        output.labels = labels
        return output
