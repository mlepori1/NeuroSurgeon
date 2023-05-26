import pytest
from ..circuit_model import CircuitModel
from ..model_configs import CircuitConfig
from ...Masking.mask_layer import MaskLayer
from datasets import load_dataset
from transformers import BertModel, BertForSequenceClassification, BertTokenizer,\
                        GPT2Model, GPT2ForSequenceClassification, GPT2Tokenizer,\
                        ResNetModel, ResNetForImageClassification,\
                        ViTModel, ViTForImageClassification, AutoImageProcessor,\
                        Trainer, TrainingArguments


def test_replace_layers():

    circuit_config = CircuitConfig(
            mask_method="continuous_sparsification", 
            mask_hparams={
                "ablation" : "none",
                "mask_bias" : True,
                "mask_init_value" : 0.0
            },
            target_layers=[],
            freeze_base=True,
            add_l0=False,
        )
    # Assert only correct layers are switched, that you cannot mask non-conv and linear layers
    ## BERT
    bert = BertModel.from_pretrained("bert-base-uncased")

    circuit_config.target_layers = [
        "encoder.layer.7.attention.self.query",
        "encoder.layer.11.output.dense",
        "pooler.dense"
    ]
    circuit_model = CircuitModel(circuit_config, bert)
    circuit_modules = {}
    for name, mod in circuit_model.root_model.named_modules(): 
        circuit_modules[name] = mod 

    clean_bert = BertModel.from_pretrained("bert-base-uncased")
    for name, mod in clean_bert.named_modules():
        if name not in circuit_config.target_layers and name != '':
            assert type(mod) == type(circuit_modules[name])
        elif name != '': # Name for overall BertModel
            assert issubclass(type(circuit_modules[name]), MaskLayer)

    bert = BertModel.from_pretrained("bert-base-uncased")
    circuit_config.target_layers = [
            "encoder.layer.7.attention.self.query",
            "encoder.layer.11.output.dense",
            "pooler.dense",
            "encoder.layer.10.attention.output.LayerNorm" # In Model, unsupported layer type
        ]  
    with pytest.raises(Exception):
        circuit_model = CircuitModel(circuit_config, bert)

    circuit_config.target_layers = [
        "some_layer" # Not in model
    ]
    with pytest.raises(Exception):
        circuit_model = CircuitModel(circuit_config, bert)

    ## GPT2
    gpt = GPT2Model.from_pretrained("sshleifer/tiny-gpt2")

    circuit_config.target_layers = [
        "h.0.attn.c_proj",
        "h.0.mlp.c_fc",
        "h.1.mlp.c_proj"
    ]
    circuit_model = CircuitModel(circuit_config, gpt)
    circuit_modules = {}
    for name, mod in circuit_model.root_model.named_modules(): 
        circuit_modules[name] = mod 

    clean_gpt = GPT2Model.from_pretrained("sshleifer/tiny-gpt2")
    for name, mod in clean_gpt.named_modules():
        if name not in circuit_config.target_layers and name != '':
            assert type(mod) == type(circuit_modules[name])
        elif name != '': # Name for overall GPTModel
            assert issubclass(type(circuit_modules[name]), MaskLayer)

    gpt = GPT2Model.from_pretrained("sshleifer/tiny-gpt2")
    circuit_config.target_layers = [
            "h.0.attn.c_proj",
            "h.0.mlp.c_fc",
            "h.1.mlp.c_proj"
            "h.0.mlp.dropout" # In Model, unsupported layer type
        ]  
    with pytest.raises(Exception):
        circuit_model = CircuitModel(circuit_config, gpt)

    circuit_config.target_layers = [
        "some_layer" # Not in model
    ]
    with pytest.raises(Exception):
        circuit_model = CircuitModel(circuit_config, gpt)

    ## ResNet

    ## ViT

    # Assert that masks are right, and that weights are the same as underlying model
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    pass

def test_model_freezing():
    # Assert that only weight_mask_param and bias_mask_param require gradients
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    # Assert that everything except mask_layers are in eval mode
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    pass

def test_train_mode_toggle():
    # Assert that, for unfrozen underlying model, train works like normal
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    # Assert that, for frozen model, train only toggles mask_layers
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    pass

def test_forward_pass():
    # Assert that underlying model and masked model with positive mask weights
    # give the same output for eval mode, different for train mode
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    # Assert that add_L0 only works for models that return loss
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    pass

def test_l0_calc():
    # Assert that train L0 calc is working
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    # Assert that test L0 calc is working
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    pass

def test_temperature():
    # Assert that different temperatures changes the output of train l0 calc
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    # Assert that model outputs change accordingly
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    # Assert that train l0 calc == test l0 calc at high temperatures 
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    # Assert that model outputs are the same
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    pass

def test_ablate_mode():
    # Assert that masks are inverted if ablate mode is switched to zero_ablate or random_ablate
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    # Assert that switching back works as well.
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    pass

def test_compute_l0_loss():
    # Assert that L0 loss is correct in train mode, returns 0 in eval mode
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    pass

def test_trainer_compatibility():
    # Assert that the trainer runs with the circuit model without bugs with frozen underlying model
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    # Assert that mask weights are different, but all underlying weights are the same
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT

    # Assert that the trainer runs with the circuit model without bugs with unfrozen underlying model
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    # Assert that all weights are different.
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    pass

def test_lambda_effect():
    # Assert that training with very high lambda drops L0 more than smaller lambda,
    # which drops more than 0.0 lambda. Test that L0 does drop from initialization for nonzero lambda
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    pass

def test_mask_init_effect():
    # Assert that L0 for mask_init -1 < 0 < 1. Assert that L0 for all drops when you include L0 regularization
    ## BERT

    ## GPT2

    ## ResNet

    ## ViT
    pass