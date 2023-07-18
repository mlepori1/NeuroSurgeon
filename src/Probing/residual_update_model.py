import torch.nn as nn
from .probe_configs import ResidualUpdateModelConfig
from transformers import (
    BertPreTrainedModel,
    RobertaPreTrainedModel,
    MPNetPreTrainedModel,
    XLMRobertaPreTrainedModel,
    ErniePreTrainedModel,
    ElectraPreTrainedModel,
    ConvBertPreTrainedModel,
)


class ResidualUpdateModel(nn.Module):
    def __init__(
        self,
        config: ResidualUpdateModelConfig,
        model: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.model = model
        self.circuit = self.config.circuit
        self.base = self.config.base
        self.residual_stream_activations = {}
        self.hooks = []

        if self.circuit:
            self.to_hook = self.model.root_model
        else:
            self.to_hook = self.model

        if self.config.model_type == "gpt":
            # This applies to GPT-2 and GPT Neo in the transformers repo
            if not self.base:
                self.to_hook = self.to_hook.transformer
            self._add_gpt_hooks()

        elif self.config.model_type == "gpt_neox":
            if not self.base:
                self.to_hook = self.to_hook.gpt_neox
            self._add_gpt_neox_hooks()

        elif self.config.model_type == "bert":
            # This applies to BERT and RoBERTa-style models in the transformers repo
            if not self.base:
                if issubclass(self.to_hook.__class__, BertPreTrainedModel):
                    self.to_hook = self.to_hook.bert
                elif issubclass(self.to_hook.__class__, RobertaPreTrainedModel):
                    self.to_hook = self.to_hook.roberta
                elif issubclass(self.to_hook.__class__, MPNetPreTrainedModel):
                    self.to_hook = self.to_hook.mpnet
                elif issubclass(self.to_hook.__class__, XLMRobertaPreTrainedModel):
                    self.to_hook = self.to_hook.roberta
                elif issubclass(self.to_hook.__class__, ErniePreTrainedModel):
                    self.to_hook = self.to_hook.ernie
                elif issubclass(self.to_hook.__class__, ElectraPreTrainedModel):
                    self.to_hook = self.to_hook.electra
                elif issubclass(self.to_hook.__class__, ConvBertPreTrainedModel):
                    self.to_hook = self.to_hook.convbert
                else:
                    raise ValueError(
                        "We only support BERT, RoBERTa, MPNet, XLM-Roberta, Ernie, ConvBERT, and Electra models currently"
                    )
            self._add_bert_hooks()

        elif self.config.model_type == "vit":
            # This applies to ViT models in the transformers repo
            if not self.base:
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
                            self._get_activation("attn_" + str(i))
                        )
                    )
                elif self.config.residual:
                    self.hooks.append(
                        self.to_hook.encoder.layer[
                            i
                        ].attention.output.register_forward_hook(
                            self._get_activation("attn_" + str(i))
                        )                        
                    )
            if self.config.mlp:
                if self.config.updates:
                    self.hooks.append(
                        self.to_hook.encoder.layer[i].output.dense.register_forward_hook(
                            self._get_activation("mlp_" + str(i))
                        )
                    )
                elif self.config.residual:
                    self.hooks.append(
                        self.to_hook.encoder.layer[i].output.register_forward_hook(
                            self._get_activation("mlp_" + str(i))
                        )
                    )
    def _add_gpt_hooks(self):
        for i in self.config.target_layers:
            if self.config.attn:
                if self.config.updates:
                    self.hooks.append(
                        self.to_hook.h[i].attn.register_forward_hook(
                            self._get_activation("attn_" + str(i))
                        )
                    )
                elif self.config.residual:
                    self.hooks.append(
                        self.to_hook.h[i].attn.register_forward_hook(
                            self._get_activation("attn_" + str(i))
                        )
                    )
                    self.hooks.append(
                        self.to_hook.h[i].register_forward_hook(
                            self._add_hidden_state("attn_" + str(i))
                        )
                    )            
            if self.config.mlp:
                if self.config.updates:
                    self.hooks.append(
                        self.to_hook.h[i].mlp.register_forward_hook(
                            self._get_activation("mlp_" + str(i))
                        )
                    )
                elif self.config.residual:
                    self.hooks.append(
                        self.to_hook.h[i].register_forward_hook(
                            self._get_activation("mlp_" + str(i))
                        )
                    )
    def _add_gpt_neox_hooks(self):
        for i in self.config.target_layers:
            if self.config.attn:
                if self.config.updates:
                    self.hooks.append(
                        self.to_hook.layers[i].attention.register_forward_hook(
                            self._get_activation("attn_" + str(i))
                        )
                    )
                elif self.config.residual:
                    self.hooks.append(
                        self.to_hook.layers[i].attention.register_forward_hook(
                            self._get_activation("attn_" + str(i))
                        )
                    )
                    self.hooks.append(
                        self.to_hook.h[i].register_forward_hook(
                            self._add_hidden_state("attn_" + str(i))
                        )
                    )             
            if self.config.mlp:
                if self.config.updates:
                    self.hooks.append(
                        self.to_hook.layers[i].mlp.register_forward_hook(
                            self._get_activation("mlp_" + str(i))
                        )
                    )
                elif self.config.residual:
                    self.hooks.append(
                        self.to_hook.layers[i].register_forward_hook(
                            self._get_activation("mlp_" + str(i))
                        )
                    )    
    def _add_vit_hooks(self):
        for i in self.config.target_layers:
            if self.config.attn:
                if self.config.updates:
                    self.hooks.append(
                        self.to_hook.encoder.layer[i].attention.register_forward_hook(
                            self._get_activation("attn_" + str(i))
                        )
                    )
                elif self.config.residual:
                    self.hooks.append(
                        self.to_hook.encoder.layer[i].attention.register_forward_hook(
                            self._get_activation("attn_" + str(i))
                        )
                    )
                    self.hooks.append(
                        self.to_hook.encoder.layer[i](
                            self._add_hidden_state("attn_" + str(i))
                        )
                    )
            if self.config.mlp:
                if self.config.updates:
                    self.hooks.append(
                        self.to_hook.encoder.layer[i].output.dense.register_forward_hook(
                            self._get_activation("mlp_" + str(i))
                        )
                    )
                elif self.config.residual:
                        self.to_hook.encoder.layer[i].output.register_forward_hook(
                            self._get_activation("mlp_" + str(i))
                        )
    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def train(self, train_bool: bool = True):
        self.training = train_bool
        self.model.train(train_bool)

    def _get_activation(self, name):
        # Credit to Jack Merullo for this code
        def hook(module, input, output):
            if self.config.updates:
                if self.config.model_type == "bert":
                    self.residual_stream_activations[name] = output
                elif self.config.model_type == "gpt" and "attn" in name:
                    self.residual_stream_activations[name] = output[0]
                elif self.config.model_type == "gpt" and "mlp" in name:
                    self.residual_stream_activations[name] = output
                elif self.config.model_type == "gpt_neox" and "attn" in name:
                    self.residual_stream_activations[name] = output[0]
                elif self.config.model_type == "gpt_neox" and "mlp" in name:
                    self.residual_stream_activations[name] = output
                elif self.config.model_type == "vit" and "attn" in name:
                    self.residual_stream_activations[name] = output[0]
                elif self.config.model_type == "vit" and "mlp" in name:
                    self.residual_stream_activations[name] = output
            elif self.config.residual:
                if self.config.model_type == "bert":
                    self.residual_stream_activations[name] = output
                elif self.config.model_type == "gpt" and "attn" in name:
                    self.residual_stream_activations[name] = output[0] 
                elif self.config.model_type == "gpt" and "mlp" in name:
                    self.residual_stream_activations[name] = output[0]
                elif self.config.model_type == "gpt_neox" and "attn" in name:
                    # Be careful! Need use_parallel_residual = False for this to work!
                    self.residual_stream_activations[name] = output[0]
                elif self.config.model_type == "gpt_neox" and "mlp" in name:
                    self.residual_stream_activations[name] = output[0]
                elif self.config.model_type == "vit" and "attn" in name:
                    self.residual_stream_activations[name] = output[0]
                elif self.config.model_type == "vit" and "mlp" in name:
                    self.residual_stream_activations[name] = output

        return hook

    def _add_hidden_state(self, name):
        # Credit to Jack Merullo for this code
        def hook(module, input, output):
            if self.config.residual:
                if self.config.model_type == "gpt" and "attn" in name:
                    self.residual_stream_activations[name] += input[0]
                elif self.config.model_type == "gpt_neox" and "attn" in name:
                    self.residual_stream_activations[name] += input[0]
                elif self.config.model_type == "vit" and "attn" in name:
                    self.residual_stream_activations[name] += input[0]


        return hook
