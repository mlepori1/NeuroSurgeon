from .circuit_model import CircuitModel
from .residual_update_model import ResidualUpdateModel
from .model_configs import CircuitProbeConfig
from transformers import PreTrainedTokenizer
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
        tokenizer: PreTrainedTokenizer,
    ):
        super.__init__()
        self.config = config
        self.tokenizer = tokenizer

        # First create a CircuitModel
        self.wrapped_model = CircuitModel(self.config.circuit_config, model)
        # Then wrap it to get intermediate activations
        self.wrapped_model = ResidualUpdateModel(
            self.config.resid_config, self.wrapped_model
        )

    def _create_token_mask(self, input_ids=None):
        # Labels are only provided at the word level, not the subword level
        # Must identify the entries in seq_len tokens that correspond to the first token per word
        # Extract those update vectors for representation matching

        # Create a mask to get rid of subword tokens
        strings = self.tokenizer.batch_decode(input_ids)
        strings = [st.split(" ") for st in strings]
        encs = self.tokenizer(
            strings, is_split_into_words=True, return_offsets_mapping=True
        )
        offset_map = np.stack(
            [np.asarray(offsets) for offsets in encs["offset_mapping"]]
        )
        not_subword = offset_map[:, :, 0] == 0

        # Compute mask to get rid of special tokens
        not_special = torch.stack(
            [
                ~torch.Tensor(self.tokenizer.get_special_tokens_mask(ids))
                for ids in input_ids
            ]
        )

        # Elementwise multiplication of masks to get first token for each word
        token_mask = not_subword * not_special
        token_mask = token_mask.bool()

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

        # Compute Soft Nearest Neighbors
        EPSILON = 1e-5
        numerator = (
            torch.sum(torch.exp(-rsm) * numerator_pairs, dim=1) + EPSILON
        )  # If no same class pairs, this is equivalent to not computing loss over that class
        denominator = torch.sum(torch.exp(-rsm) * denominator_pairs, dim=1) + EPSILON
        loss = -torch.sum(torch.log(numerator / denominator)) / len(numerator)

        return loss

    def forward(self, input_ids=None, return_dict=True, **kwargs):
        token_mask = self._create_token_mask(input_ids)

        # Call model forward pass, get out the correct activations
        _ = self.wrapped_model(input_ids, **kwargs)
        updates = self.wrapped_model.residual_stream_updates[self.config.target_layer]

        # Get one residual stream update per label using mask indexing,
        # collapsing a batch of strings into a list of labels and residual stream updates
        token_mask = token_mask.reshape(-1)
        updates = updates.reshape(-1, self.wrapped_model.model.hidden_size)
        updates = updates[token_mask]

        labels = labels.reshape(-1)
        assert len(updates) == len(
            labels
        )  # Ensure that there is only one update per label

        loss = None

        # Compute Representation Matching Loss

        loss = self._compute_representation_matching_loss(updates, labels)

        # Add in L0 Regularization to keep mask small
        loss += self.lamb * self.compute_l0()

        if not return_dict:
            return (loss,) + (updates,) if loss is not None else updates

        return SequenceClassifierOutput(
            loss=loss,
            logits=None,
            hidden_states=updates,
            attentions=None,
        )
