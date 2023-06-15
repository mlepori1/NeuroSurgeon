from .circuit_model import CircuitModel
from .residual_update_model import ResidualUpdateModel
from .model_configs import CircuitProbeConfig
from transformers import PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput
import torchmetrics
import torch
import torch.nn as nn
import numpy as np


class CircuitProbe(nn.Module):
    def __init__(
        self,
        config: CircuitProbeConfig,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerFast,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        # Must be PreTrainedTokenizerFast for subword masking functionality
        assert issubclass(self.tokenizer.__class__, PreTrainedTokenizerFast)

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

    def _create_token_mask(self, input_ids=None):
        # Labels are only provided at the word level, not the subword level
        # Must identify the entries in seq_len tokens that correspond to the first token per word
        # Extract those update vectors for representation matching

        # Create a mask to get rid of subword tokens
        strings = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        strings = [st.split(" ") for st in strings]
        encs = self.tokenizer(
            strings,
            is_split_into_words=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        offset_map = encs["offset_mapping"]
        not_subword = offset_map[:, :, 0] == 0

        # Compute mask to get rid of special tokens
        not_special = ~encs["special_tokens_mask"].bool()

        # Elementwise multiplication of masks to get first token for each word
        token_mask = not_subword * not_special
        token_mask = token_mask.bool()

        return token_mask

    def _compute_representation_matching_loss(self, updates, labels):
        loss = None

        # 1. Create representational similarity matrix between update vectors using cosine sim
        rsm = torchmetrics.functional.pairwise_cosine_similarity(updates)

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

    def forward(self, input_ids=None, labels=None, return_dict=True, **kwargs):
        token_mask = self._create_token_mask(input_ids)

        # Call model forward pass, get out the correct activations
        _ = self.wrapped_model(input_ids=input_ids, **kwargs)
        updates = self.wrapped_model.residual_stream_updates[self.config.probe_updates]

        # Get one residual stream update per label using mask indexing,
        # collapsing a batch of strings into a list of labels and residual stream updates
        token_mask = token_mask.reshape(-1)
        updates = updates.reshape(
            -1, self.wrapped_model.model.root_model.config.hidden_size
        )
        updates = updates[token_mask]

        labels = labels.reshape(-1)
        assert len(updates) == len(
            labels
        )  # Ensure that there is only one update per label

        loss = None

        # Compute Representation Matching Loss

        loss = self._compute_representation_matching_loss(updates, labels)

        # Add in L0 Regularization to keep mask small
        if self.config.circuit_config.add_l0:
            loss += (
                self.config.circuit_config.l0_lambda
                * self.wrapped_model.model._compute_l0_loss()
            )

        if not return_dict:
            return (loss,) + (updates,) if loss is not None else updates

        return SequenceClassifierOutput(
            loss=loss,
            logits=None,
            hidden_states=updates,
            attentions=None,
        )
