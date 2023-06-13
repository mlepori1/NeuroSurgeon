from circuit_model import CircuitModel
from residual_update_model import ResidualUpdateModel
from transformers import PretrainedModel
import torchmetrics
import torch


class CircuitProbe(PretrainedModel):
    def __init__(self, config, model, tokenizer):
        self.tokenizer = tokenizer
        self.wrapped_model = CircuitModel(
            self._config_2_circuit_model_config(config), model
        )
        self.wrapped_model = ResidualUpdateModel(
            self._config_2_residual_update_config(config), self.wrapped_model
        )

    def _config_2_circuit_model_config(self):
        # @ToDo Must create a circuit_probe config
        # @ToDo Must fill out
        pass

    def _config_2_residual_update_config(self):
        # @ToDo Must fill out
        pass

    def forward(self, input_ids=None, return_dict=True, **kwargs):
        # Labels are only provided at the word level, not the subword level
        # Must identify the entries in seq_len tokens that correspond to the first token per word
        # Extract those update vectors for representation matching

        # Compute a mask over subwords (which start with ## for BERT)
        not_subword = torch.stack(
            [
                torch.tensor(
                    [
                        not tok.startswith(self.config.subword_identifer)
                        for tok in self.tokenizer.convert_ids_to_tokens(ids)
                    ]
                )
                for ids in input_ids
            ]
        )
        # Compute mask over special tokens
        not_special = torch.stack(
            [
                ~torch.Tensor(self.tokenizer.get_special_tokens_mask(ids))
                for ids in input_ids
            ]
        )
        # Elementwise multiplication of masks is what we want
        token_mask = not_subword * not_special
        token_mask = token_mask.bool()

        # @ToDo call model forward pass, get out the correct activations

        # Get one residual stream update per label using mask indexing
        token_mask = token_mask.reshape(-1)
        outputs = outputs.reshape(-1, self.wrapped_model.model.hidden_size)
        outputs = outputs[token_mask]

        labels = labels.reshape(-1)
        assert len(outputs) == len(
            labels
        )  # Ensure that there is only one update per label

        loss = None

        if self.config.rm_loss == "soft_NN":
            # 1. Create representational similarity matrix between update vectors using cosine sim
            rsm = torchmetrics.functional.pairwise_cosine_similarity(outputs)

            # 2. Create ideal representational similarity matrix using labels
            labels_row = torch.repeat_interleave(labels, len(labels), dim=0)
            labels_col = labels.repeat(len(labels))
            # All members of the same class are perfectly similar, otherwise perfectly dissimilar
            concept_rsm = labels_row == labels_col
            concept_rsm = concept_rsm.reshape(len(labels), len(labels))

            # 3. Compute Soft Nearest Neighbors loss according to concept RSM
            identity_matrix = ~torch.eye(
                concept_rsm.shape[0], dtype=torch.bool, device=concept_rsm.device
            )
            # Only sum over pairs of the same class, but not the exact same point
            numerator_pairs = identity_matrix * concept_rsm
            # Denominator includes every pair except i == j
            denominator_pairs = identity_matrix

            # Compute Soft Nearest Neighbors
            EPSILON = 1e-5
            numerator = (
                torch.sum(torch.exp(-rsm) * numerator_pairs, dim=1) + EPSILON
            )  # If no same class pairs, this is equivalent to not computing loss over that class
            denominator = (
                torch.sum(torch.exp(-rsm) * denominator_pairs, dim=1) + EPSILON
            )
            loss = -torch.sum(torch.log(numerator / denominator)) / len(numerator)

            # Add in L0 Regularization to keep mask small
            loss += self.lamb * self.compute_l0()

        if not return_dict:
            return (loss,) + (outputs,) if loss is not None else outputs

        return SequenceClassifierOutput(
            loss=loss,
            logits=None,
            hidden_states=outputs,
            attentions=None,
        )
