import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

from ..Models.circuit_model import CircuitModel
from ..Probing.probe_configs import SubnetworkProbeConfig
from .residual_update_model import ResidualUpdateModel


class SubnetworkProbe(nn.Module):
    """This reimplements the technique introduced in Cao et al. 2021 (https://arxiv.org/abs/2104.03514).
    Probing introduces a linear layer or MLP to extract information from intermediate representations in a model.
    One can train the probe to classify inputs at the token or sequence level, using either intermediate
    updates or intermediate activations. Cao et al. introduced subnetwork probing, which optimizes a binary mask
    and a linear probe at the same time, resulting in low-complexity probes. This class implements this technique
    by introducing probing layers into a CircuitModel. One can also use this class to perform regular probing by
    specifying that no layers get masked in the CircuitModel config.

    :param config: A config file determining the behavior of the subnetwork probe
    :type config: SubnetworkProbeConfig
    :param model: The model to probe. Currently, it supports ViT, GPT2,
        GPTNeoX, BERT, RoBERTa, MPNet, ConvBERT, Ernie, and Electra models.
    :type model: nn.Module
    """

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

        self.probe = self._create_probe()
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

    def _create_probe(self):
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
        """Forward pass of the model

        :param input_ids: input tensors, defaults to None
        :type input_ids: torch.Tensor, optional
        :param labels: probing labels, defaults to None
        :type labels: torch.Tensor, optional
        :param token_mask: A mask defining which updates/residual stream entries should be mapped to labels, defaults to None
        :type token_mask: torch.Tensor, optional
        :param return_dict: Whether to return an output dictionary or a tuple, defaults to True
        :type return_dict: bool, optional
        :return: An object that contains output predictions, loss, etc. (dictionary)
        :rtype: SequenceClassifierOutput or Tuple
        """

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
