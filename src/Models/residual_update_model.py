from transformers import PretrainedModel
from model_configs import ResidualUpdateModelConfig

class ResidualUpdateModel(PretrainedModel):

    def __init__(self, config: ResidualUpdateModelConfig, model:PretrainedModel):

        self.config = config
        self.model = model
        self.hooks = []
        if self.config.model_type == "gpt":
            # This applies to GPT-2 and GPT Neo in the transformers repo
            if self.config.base:
                self._add_gpt2_hooks()
            else:
                self._add_gpt2_clf_hooks()
        elif self.config.model_type == "bert":
            # This applies to BERT and RoBERTa in the transformers repo
            if self.config.base:
                self._add_bert_hooks()
            else:
                self._add_bert_clf_hooks()

    def _add_gpt2_clf_hooks(self):
        for i in self.config.target_layers:
            if self.config.attn:
                self.hooks.append(self.model.transformer.h[i].attn.register_forward_hook(self._get_activation('attn_'+str(i))))
            if self.config.mlp:
                self.hooks.append(self.model.transformer.h[i].mlp.register_forward_hook(self._get_activation('mlp_'+str(i))))

    def _add_gpt2_hooks(self):
        for i in self.config.target_layers:
            if self.config.attn:
                self.hooks.append(self.model.h[i].attn.register_forward_hook(self._get_activation('attn_'+str(i))))
            if self.config.mlp:
                self.hooks.append(self.model.h[i].mlp.register_forward_hook(self._get_activation('mlp_'+str(i))))

    def _add_bert_hooks(self):
        for i in self.config.target_layers:
            if self.config.attn:
                self.hooks.append(self.model.encoder.layer[i].attention.output.dense.register_forward_hook(self._get_activation('attn_'+str(i))))
            if self.config.mlp:
                self.hooks.append(self.model.encoder.layer[i].output.dense.register_forward_hook(self._get_activation('mlp_'+str(i))))

    def _add_bert_clf_hooks(self):
        for i in self.config.target_layers:
            if self.config.attn:
                self.hooks.append(self.model.bert.encoder.layer[i].attention.output.dense.register_forward_hook(self._get_activation('attn_'+str(i))))
            if self.config.mlp:
                self.hooks.append(self.model.bert.encoder.layer[i].output.dense.register_forward_hook(self._get_activation('mlp_'+str(i))))

    def __getattr__(self, name):
        # Override this to call functions from the underlying model if not found in this class
        return getattr(self.model, name)
    
    def _get_activation(self, name):
        #Credit to Jack Merullo for this code
        def hook(module, input, output):
            if "attn" in name:
                self.model.activations_[name] = output[0].detach()
            if "mlp" in name:
                self.model.activations_[name] = output.detach()

        return hook