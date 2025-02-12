import torch
import torch.nn as nn
import torchmetrics
from transformers.modeling_outputs import SequenceClassifierOutput

from ..Models.circuit_model import CircuitModel
from .probe_configs import CircuitProbeConfig
from .residual_update_model import ResidualUpdateModel


class CircuitProbe(nn.Module):
    """This implements the circuit probing technique introduced in Lepori et al. 2024 (https://openreview.net/pdf?id=gUNeyiLNxr)
    This method learns a binary mask over parameters using a contrastive loss funciton, as described in the main paper.
    We also implement an experimental version of this method that links circuit probing and linear probing

    :param config: A config file determining the behavior of the circuit probe
    :type config: CircuitProbeConfig
    :param model: The model to probe. Currently, it supports ViT, GPT2,
        GPTNeoX, BERT, RoBERTa, MPNet, ConvBERT, Ernie, and Electra models.
    :type model: nn.Module
    """
    def __init__(
        self,
        config: CircuitProbeConfig,
        model: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.label_pad_idx = -1000

        self._validate_configs()

        # First create a CircuitModel
        self.wrapped_model = CircuitModel(self.config.circuit_config, model)
        # Then wrap it to get intermediate activations
        self.wrapped_model = ResidualUpdateModel(
            self.config.resid_config, self.wrapped_model
        )

    def _validate_configs(self):
        # Ensure that wrapper configs specify valid behavior for circuit probing
        circuit_config = self.config.circuit_config
        resid_config = self.config.resid_config

        # Circuit Probing should only be performed when underlying model is frozen
        assert circuit_config.freeze_base

        # Circuit Probing probes particular residual stream updates
        assert len(resid_config.target_layers) == 1

        # Circuit probing must operate on a residal stream update
        assert (resid_config.mlp and not resid_config.attn) or (
            resid_config.attn and not resid_config.mlp
        )

    def _compute_contrastive_loss(self, updates, labels, hidden_states=None):
        loss = None

        # 1. Create representational similarity matrix between update vectors using cosine sim
        # If hidden_states is None, this is just pairwise within updates
        rsm = torchmetrics.functional.pairwise_cosine_similarity(updates, hidden_states)

        # 2. Create ideal representational similarity matrix using labels
        labels_row = torch.repeat_interleave(labels, len(labels), dim=0)
        labels_col = labels.repeat(len(labels))
        # All members of the same class are perfectly similar, otherwise perfectly dissimilar
        concept_rsm = labels_row == labels_col
        concept_rsm = concept_rsm.reshape(len(labels), len(labels))

        # 3. Compute Soft Nearest Neighbors loss according to concept RSM
        inv_identity_matrix = ~torch.eye(
            concept_rsm.shape[0], dtype=torch.bool, device=concept_rsm.device
        )
        # Only sum over pairs of the same class, but not the exact same datapoint
        # (because that gives the loss function an unhelpful advantage)
        numerator_pairs = inv_identity_matrix * concept_rsm
        # Denominator includes every pair except i == j
        denominator_pairs = inv_identity_matrix

        # Create dissimiliarity matrix for computing loss
        rdm = 1 - rsm

        # Compute Soft Nearest Neighbors
        EPSILON = 1e-5
        numerator = (
            torch.sum(torch.exp(-rdm) * numerator_pairs, dim=1) + EPSILON
        )  # If no same class pairs, this is equivalent to not computing loss over that class
        denominator = torch.sum(torch.exp(-rdm) * denominator_pairs, dim=1) + EPSILON
        loss = -torch.sum(torch.log(numerator / denominator)) / len(numerator)

        return loss

    def train(self, train_bool: bool = True):
        self.training = train_bool
        self.wrapped_model.train(train_bool)

    def forward(
        self, input_ids=None, labels=None, token_mask=None, return_dict=True, **kwargs
    ):
        # Must provide a token mask, which is a boolean mask for each input denoting which
        # residual streams to compute loss over

        # If using the linear_probe loss, get raw model hidden states
        if self.config.loss == "linear_probe":

            # First get model state variables
            train_bool = self.training
            use_masks_bool = self.wrapped_model.wrapped_model.use_masks_bool
            self.train(False)
            self.wrapped_model.wrapped_model.use_masks(False)

            # Call model forward pass, get out the raw activations
            _ = self.wrapped_model(input_ids=input_ids, **kwargs)
            unmasked_updates = self.wrapped_model.vector_cache[
                self.config.probe_vectors
            ]

            # Get one residual stream update per label using mask indexing,
            # collapsing a batch of strings into a list of labels and residual stream updates
            token_mask = token_mask.reshape(-1)
            unmasked_updates = unmasked_updates.reshape(
                -1, self.wrapped_model.wrapped_model.wrapped_model.config.hidden_size
            )
            unmasked_updates = unmasked_updates[token_mask]
            unmasked_updates = unmasked_updates.detach()

            # Reset state of model
            self.wrapped_model.wrapped_model.use_masks(use_masks_bool)
            self.train(train_bool)

        # Call model forward pass, get out the correct activations
        _ = self.wrapped_model(**kwargs)
        updates = self.wrapped_model.vector_cache[self.config.probe_vectors]

        # Get one residual stream update per label using mask indexing,
        # collapsing a batch of strings into a list of labels and residual stream updates
        token_mask = token_mask.reshape(-1)
        updates = updates.reshape(
            -1, self.wrapped_model.wrapped_model.wrapped_model.config.hidden_size
        )
        updates = updates[token_mask]

        if labels is not None:
            labels = labels[
                labels != self.label_pad_idx
            ]  # Gets rid of label padding before computing representation matching loss
            labels = labels.reshape(-1)
            assert len(updates) == len(
                labels
            )  # Ensure that there is only one update per label

        loss = None

        if labels is not None:
            if self.config.loss == "contrastive":
                # Compute soft NN Loss
                loss = self._compute_contrastive_loss(updates, labels)
            elif self.config.loss == "linear_probe":
                # Compute linear probe loss, which is a variation of contrastive
                loss = self._compute_contrastive_loss(
                    updates, labels, hidden_states=unmasked_updates
                )

        # Add in L0 Regularization to keep mask small
        if self.config.circuit_config.add_l0:
            loss += (
                self.config.circuit_config.l0_lambda
                * self.wrapped_model.wrapped_model._compute_l0_loss()
            )

        if not return_dict:
            return (loss,) + (updates,) if loss is not None else updates

        output = SequenceClassifierOutput(
            loss=loss,
            logits=None,
            hidden_states=updates,
            attentions=None,
        )
        output.labels = labels
        return output
