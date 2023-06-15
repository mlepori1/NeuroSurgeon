import pytest
import torch
from datasets import load_dataset
from transformers import (
    BertModel,
    BertForMaskedLM,
    GPT2Model,
    GPT2LMHeadModel,
    ViTModel,
    ViTForMaskedImageModeling,
    BertTokenizer,
    GPT2Tokenizer,
    AutoImageProcessor,
)
from ..residual_update_model import ResidualUpdateModel
from ..model_configs import ResidualUpdateModelConfig


def test_base_model_initializations():
    # Test that residual models can be initialized for different model types
    bert_config = ResidualUpdateModelConfig("bert", [0, 1], True, True)
    bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
    resid_model = ResidualUpdateModel(bert_config, bert)

    gpt_config = ResidualUpdateModelConfig("gpt", [0, 1], True, True)
    gpt = GPT2Model.from_pretrained("sshleifer/tiny-gpt2")
    resid_model = ResidualUpdateModel(gpt_config, gpt)

    vit_config = ResidualUpdateModelConfig("vit", [0, 1], True, True)
    vit = ViTModel.from_pretrained("lysandre/tiny-vit-random")
    resid_model = ResidualUpdateModel(vit_config, vit)

    # Assert that one cannot add a hook to a layer that doesn't exist
    with pytest.raises(Exception):
        bert_config = ResidualUpdateModelConfig("bert", [24], True, True)
        bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
        resid_model = ResidualUpdateModel(bert_config, bert)

    # Assert that one cannot specify the wrong model type
    with pytest.raises(Exception):
        wrong_config = ResidualUpdateModelConfig("gpt", [0], True, True)
        bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
        resid_model = ResidualUpdateModel(wrong_config, bert)


def test_generative_model_initializations():
    # Test that residual models can be initialized for different model types
    bert_config = ResidualUpdateModelConfig("bert", [0, 1], True, True, False, False)
    bert = BertForMaskedLM.from_pretrained("prajjwal1/bert-tiny")
    resid_model = ResidualUpdateModel(bert_config, bert)

    gpt_config = ResidualUpdateModelConfig("gpt", [0, 1], True, True, False, False)
    gpt = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
    resid_model = ResidualUpdateModel(gpt_config, gpt)

    vit_config = ResidualUpdateModelConfig("vit", [0, 1], True, True, False, False)
    vit = ViTForMaskedImageModeling.from_pretrained("lysandre/tiny-vit-random")
    resid_model = ResidualUpdateModel(vit_config, vit)


def test_model_forward_pass():
    # Test that you can run a forward pass on models
    bert_config = ResidualUpdateModelConfig("bert", [0, 1], True, True)
    bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
    resid_model = ResidualUpdateModel(bert_config, bert)

    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
    ipt = tokenizer("This is a test", return_tensors="pt")
    resid_model(**ipt)


def test_residual_stream_dict_updates():
    # Test that running a forward pass updates the residual_stream_updates dict correctly
    bert_config = ResidualUpdateModelConfig("bert", [0, 1], True, True)
    bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
    resid_model = ResidualUpdateModel(bert_config, bert)

    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
    ipt = tokenizer("This is a test", return_tensors="pt")
    resid_model(**ipt)

    assert "mlp_0" in resid_model.residual_stream_updates.keys()
    assert "attn_0" in resid_model.residual_stream_updates.keys()
    assert "mlp_1" in resid_model.residual_stream_updates.keys()
    assert "attn_1" in resid_model.residual_stream_updates.keys()

    assert resid_model.residual_stream_updates["mlp_0"].shape == (1, 6, 128)
    assert resid_model.residual_stream_updates["mlp_0"].requires_grad == True

    assert resid_model.residual_stream_updates["attn_0"].shape == (1, 6, 128)
    assert resid_model.residual_stream_updates["attn_0"].requires_grad == True


def test_self_consistency_bert():
    # Assert that you can recover the next layer from the previous layer + the residual stream updates
    bert_config = ResidualUpdateModelConfig("bert", [1], True, True)
    bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
    resid_model = ResidualUpdateModel(bert_config, bert)
    resid_model.eval()

    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
    ipt = tokenizer("This is a test", return_tensors="pt")
    output = resid_model(**ipt, return_dict=True, output_hidden_states=True)
    pre_layer_residual_stream = output.hidden_states[1]
    post_layer_residual_stream = output.hidden_states[2]

    attn_update = resid_model.residual_stream_updates["attn_1"]
    mlp_update = resid_model.residual_stream_updates["mlp_1"]

    # Need to do the layernorms in the right place
    mid_stream = resid_model.model.encoder.layer[1].attention.output.LayerNorm(
        pre_layer_residual_stream + attn_update
    )
    post_stream = resid_model.model.encoder.layer[1].output.LayerNorm(
        mid_stream + mlp_update
    )

    assert torch.all(post_stream == post_layer_residual_stream)


def test_self_consistency_gpt():
    # Assert that you can recover the next layer from the previous layer + the residual stream updates
    gpt_config = ResidualUpdateModelConfig("gpt", [0], True, True)
    gpt = GPT2Model.from_pretrained("sshleifer/tiny-gpt2")
    resid_model = ResidualUpdateModel(gpt_config, gpt)
    resid_model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("sshleifer/tiny-gpt2")
    ipt = tokenizer("This is a test", return_tensors="pt")
    output = resid_model(**ipt, return_dict=True, output_hidden_states=True)

    pre_layer_residual_stream = output.hidden_states[0]
    post_layer_residual_stream = output.hidden_states[1]

    attn_update = resid_model.residual_stream_updates["attn_0"]
    mlp_update = resid_model.residual_stream_updates["mlp_0"]

    mid_stream = pre_layer_residual_stream + attn_update
    post_stream = mid_stream + mlp_update

    assert torch.all(post_stream == post_layer_residual_stream)


def test_self_consistency_vit():
    # Assert that you can recover the next layer from the previous layer + the residual stream updates
    vit_config = ResidualUpdateModelConfig("vit", [0], True, True)
    vit = ViTModel.from_pretrained("lysandre/tiny-vit-random")
    resid_model = ResidualUpdateModel(vit_config, vit)
    resid_model.eval()

    im_processor = AutoImageProcessor.from_pretrained("lysandre/tiny-vit-random")
    dataset = load_dataset("beans")
    image = dataset["test"]["image"][0]
    ipt = im_processor(image, return_tensors="pt")
    output = resid_model(**ipt, return_dict=True, output_hidden_states=True)

    pre_layer_residual_stream = output.hidden_states[0]
    post_layer_residual_stream = output.hidden_states[1]

    attn_update = resid_model.residual_stream_updates["attn_0"]
    mlp_update = resid_model.residual_stream_updates["mlp_0"]

    mid_stream = pre_layer_residual_stream + attn_update
    post_stream = mid_stream + mlp_update

    assert torch.all(post_stream == post_layer_residual_stream)
