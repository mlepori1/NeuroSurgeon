import warnings

import torch
import torch.nn as nn

from ..Models.circuit_model import CircuitModel
from .probe_configs import ResidualUpdateModelConfig


class ResidualUpdateModel(nn.Module):
    """A simple wrapped that adds hooks into a model to return intermediate hidden states
    or updates from particular MLP or Attention layers. This is loosely modeled off of the
    Transformer Lens library from Neel Nanda (https://github.com/neelnanda-io/TransformerLens).
    This is a minimal implementation that is built to subnetwork based pruning efforts. Currently,
    it supports ViT, GPT2, GPTNeoX, BERT, RoBERTa, MPNet, ConvBERT, Ernie, and Electra models.
    On every forward pass, the specied updates/activations are placed in a vector_cache dictionary.


    :param config: A configuration object defining which updates/activations to store
    :type config: ResidualUpdateModelConfig
    :param model: A transformer model to wrap
    :type model: nn.Module (should be one of the model architectures stated above)
    """

    def __init__(
        self,
        config: ResidualUpdateModelConfig,
        model: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.wrapped_model = model
        self.vector_cache = {}
        self.hooks = []

        if issubclass(self.wrapped_model.__class__, CircuitModel):
            self.to_hook = self.wrapped_model.wrapped_model
        else:
            self.to_hook = self.wrapped_model

        if self.config.model_type == "gpt":
            # This applies to GPT-2 and GPT Neo in the transformers repo
            if hasattr(self.to_hook, "transformer"):
                self.to_hook = self.to_hook.transformer
            self._add_gpt_hooks()

        elif self.config.model_type == "gpt_neox":
            if hasattr(self.to_hook, "gpt_neox"):
                self.to_hook = self.to_hook.gpt_neox
            self._add_gpt_neox_hooks()

        elif self.config.model_type == "bert":
            # This applies to BERT and RoBERTa-style models in the transformers repo
            if hasattr(self.to_hook, "bert"):
                self.to_hook = self.to_hook.bert
            elif hasattr(self.to_hook, "roberta"):
                self.to_hook = self.to_hook.roberta
            elif hasattr(self.to_hook, "mpnet"):
                self.to_hook = self.to_hook.mpnet
            elif hasattr(self.to_hook, "ernie"):
                self.to_hook = self.to_hook.ernie
            elif hasattr(self.to_hook, "electra"):
                self.to_hook = self.to_hook.electra
            elif hasattr(self.to_hook, "convbert"):
                self.to_hook = self.to_hook.convbert
            else:
                warnings.warn(
                    "ResidualUpdateModel only supports BERT, RoBERTa, MPNet, Ernie, Electra, and ConvBERT bert-style models right now"
                )
            self._add_bert_hooks()

        elif self.config.model_type == "vit":
            # This applies to ViT models in the transformers repo
            if hasattr(self.to_hook, "vit"):
                self.to_hook = self.to_hook.vit
            self._add_vit_hooks()
        else:
            raise ValueError("model type must be one of [gpt, bert, vit]")

    def _add_bert_hooks(self):
        for i in self.config.target_layers:
            if self.config.attn:
                if self.config.updates:
                    self.hooks.append(
                        self.to_hook.encoder.layer[
                            i
                        ].attention.output.dense.register_forward_hook(
                            self._get_activation("attn_update_" + str(i))
                        )
                    )
                if self.config.stream:
                    self.hooks.append(
                        self.to_hook.encoder.layer[
                            i
                        ].attention.output.register_forward_hook(
                            self._get_activation("attn_stream_" + str(i))
                        )
                    )
            if self.config.mlp:
                if self.config.updates:
                    self.hooks.append(
                        self.to_hook.encoder.layer[
                            i
                        ].output.dense.register_forward_hook(
                            self._get_activation("mlp_update_" + str(i))
                        )
                    )
                if self.config.stream:
                    self.hooks.append(
                        self.to_hook.encoder.layer[i].output.register_forward_hook(
                            self._get_activation("mlp_stream_" + str(i))
                        )
                    )

    def _add_gpt_hooks(self):
        for i in self.config.target_layers:
            if self.config.attn:
                if self.config.updates:
                    self.hooks.append(
                        self.to_hook.h[i].attn.register_forward_hook(
                            self._get_activation("attn_update_" + str(i))
                        )
                    )
                if self.config.stream:
                    self.hooks.append(
                        self.to_hook.h[i].attn.register_forward_hook(
                            self._get_activation("attn_stream_" + str(i))
                        )
                    )
                    self.hooks.append(
                        self.to_hook.h[i].register_forward_hook(
                            self._add_hidden_state("attn_stream_" + str(i))
                        )
                    )
            if self.config.mlp:
                if self.config.updates:
                    self.hooks.append(
                        self.to_hook.h[i].mlp.register_forward_hook(
                            self._get_activation("mlp_update_" + str(i))
                        )
                    )
                if self.config.stream:
                    self.hooks.append(
                        self.to_hook.h[i].register_forward_hook(
                            self._get_activation("mlp_stream_" + str(i))
                        )
                    )

    def _add_gpt_neox_hooks(self):
        for i in self.config.target_layers:
            if self.config.attn:
                if self.config.updates:
                    self.hooks.append(
                        self.to_hook.layers[i].attention.register_forward_hook(
                            self._get_activation("attn_update_" + str(i))
                        )
                    )
                if self.config.stream:
                    warnings.warn(
                        "For GPTNeoX, only track Attn stream if use_parallel_residual=False"
                    )
                    self.hooks.append(
                        self.to_hook.layers[i].attention.register_forward_hook(
                            self._get_activation("attn_stream_" + str(i))
                        )
                    )
                    self.hooks.append(
                        self.to_hook.h[i].register_forward_hook(
                            self._add_hidden_state("attn_stream_" + str(i))
                        )
                    )
            if self.config.mlp:
                if self.config.updates:
                    self.hooks.append(
                        self.to_hook.layers[i].mlp.register_forward_hook(
                            self._get_activation("mlp_update_" + str(i))
                        )
                    )
                if self.config.stream:
                    self.hooks.append(
                        self.to_hook.layers[i].register_forward_hook(
                            self._get_activation("mlp_stream_" + str(i))
                        )
                    )

    def _add_vit_hooks(self):
        for i in self.config.target_layers:
            if self.config.attn:
                if self.config.updates:
                    self.hooks.append(
                        self.to_hook.encoder.layer[i].attention.register_forward_hook(
                            self._get_activation("attn_update_" + str(i))
                        )
                    )
                if self.config.stream:
                    self.hooks.append(
                        self.to_hook.encoder.layer[i].attention.register_forward_hook(
                            self._get_activation("attn_stream_" + str(i))
                        )
                    )
                    self.hooks.append(
                        self.to_hook.encoder.layer[i].register_forward_hook(
                            self._add_hidden_state("attn_stream_" + str(i))
                        )
                    )
            if self.config.mlp:
                if self.config.updates:
                    self.hooks.append(
                        self.to_hook.encoder.layer[
                            i
                        ].output.dense.register_forward_hook(
                            self._get_activation("mlp_update_" + str(i))
                        )
                    )
                if self.config.stream:
                    self.to_hook.encoder.layer[i].output.register_forward_hook(
                        self._get_activation("mlp_stream_" + str(i))
                    )

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        return self.wrapped_model(**kwargs)

    def train(self, train_bool: bool = True):
        self.training = train_bool
        self.wrapped_model.train(train_bool)

    def _get_activation(self, name):
        # Credit to Jack Merullo for this code
        def hook(module, input, output):
            if "update" in name:
                if self.config.model_type == "bert":
                    self.vector_cache[name] = torch.clone(output)
                elif self.config.model_type == "gpt" and "attn" in name:
                    self.vector_cache[name] = torch.clone(output[0])
                elif self.config.model_type == "gpt" and "mlp" in name:
                    self.vector_cache[name] = torch.clone(output)
                elif self.config.model_type == "gpt_neox" and "attn" in name:
                    self.vector_cache[name] = torch.clone(output[0])
                elif self.config.model_type == "gpt_neox" and "mlp" in name:
                    self.vector_cache[name] = torch.clone(output)
                elif self.config.model_type == "vit" and "attn" in name:
                    self.vector_cache[name] = torch.clone(output[0])
                elif self.config.model_type == "vit" and "mlp" in name:
                    self.vector_cache[name] = torch.clone(output)
            elif "stream" in name:
                if self.config.model_type == "bert":
                    self.vector_cache[name] = torch.clone(output)
                elif self.config.model_type == "gpt" and "attn" in name:
                    self.vector_cache[name] = torch.clone(output[0])
                elif self.config.model_type == "gpt" and "mlp" in name:
                    self.vector_cache[name] = torch.clone(output[0])
                elif self.config.model_type == "gpt_neox" and "attn" in name:
                    # Be careful! Need use_parallel_residual = False for this to work!
                    self.vector_cache[name] = torch.clone(output[0])
                elif self.config.model_type == "gpt_neox" and "mlp" in name:
                    self.vector_cache[name] = torch.clone(output[0])
                elif self.config.model_type == "vit" and "attn" in name:
                    self.vector_cache[name] = torch.clone(output[0])
                elif self.config.model_type == "vit" and "mlp" in name:
                    self.vector_cache[name] = torch.clone(output)

        return hook

    def _add_hidden_state(self, name):
        # Credit to Jack Merullo for this code
        def hook(module, input, output):
            if "stream" in name:
                if self.config.model_type == "gpt" and "attn" in name:
                    self.vector_cache[name] += torch.clone(input[0])
                elif self.config.model_type == "gpt_neox" and "attn" in name:
                    self.vector_cache[name] += torch.clone(input[0])
                elif self.config.model_type == "vit" and "attn" in name:
                    self.vector_cache[name] += torch.clone(input[0])

        return hook
