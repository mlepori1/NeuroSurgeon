import torch.nn as nn
from .model_configs import ResidualUpdateModelConfig


class ResidualUpdateModel(nn.Module):
    def __init__(self, config: ResidualUpdateModelConfig, model: nn.Module):
        super().__init__()
        self.config = config
        self.model = model
        self.residual_stream_updates = {}
        self.hooks = []

        if self.config.model_type == "gpt":
            # This applies to GPT-2 and GPT Neo in the transformers repo
            self._add_gpt_hooks()

        elif self.config.model_type == "bert":
            # This applies to BERT and RoBERTa in the transformers repo
            self._add_bert_hooks()

        elif self.config.model_type == "vit":
            # This applies to ViT models in the transformers repo
            self._add_vit_hooks()

    def _add_bert_hooks(self):
        for i in self.config.target_layers:
            if self.config.attn:
                self.hooks.append(
                    self.model.encoder.layer[
                        i
                    ].attention.output.dense.register_forward_hook(
                        self._get_activation("attn_" + str(i))
                    )
                )
            if self.config.mlp:
                self.hooks.append(
                    self.model.encoder.layer[i].output.dense.register_forward_hook(
                        self._get_activation("mlp_" + str(i))
                    )
                )

    def _add_gpt_hooks(self):
        for i in self.config.target_layers:
            if self.config.attn:
                self.hooks.append(
                    self.model.h[i].attn.register_forward_hook(
                        self._get_activation("attn_" + str(i))
                    )
                )
            if self.config.mlp:
                self.hooks.append(
                    self.model.h[i].mlp.register_forward_hook(
                        self._get_activation("mlp_" + str(i))
                    )
                )

    def _add_vit_hooks(self):
        for i in self.config.target_layers:
            if self.config.attn:
                self.hooks.append(
                    self.model.encoder.layer[i].attention.register_forward_hook(
                        self._get_activation("attn_" + str(i))
                    )
                )
            if self.config.mlp:
                self.hooks.append(
                    self.model.encoder.layer[i].output.dense.register_forward_hook(
                        self._get_activation("mlp_" + str(i))
                    )
                )

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def _get_activation(self, name):
        # Credit to Jack Merullo for this code
        def hook(module, input, output):
            if self.config.model_type == "bert":
                self.residual_stream_updates[name] = output
            elif self.config.model_type == "gpt" and "attn" in name:
                self.residual_stream_updates[name] = output[0]
            elif self.config.model_type == "gpt" and "mlp" in name:
                self.residual_stream_updates[name] = output
            elif self.config.model_type == "vit" and "attn" in name:
                self.residual_stream_updates[name] = output[0]
            elif self.config.model_type == "vit" and "mlp" in name:
                self.residual_stream_updates[name] = output

        return hook
