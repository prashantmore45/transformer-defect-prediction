"""Tests for sdp.model.classifier. A tiny random encoder is used: no network, no GPU."""

import copy

import pytest
import torch
from transformers import RobertaConfig, RobertaModel

from sdp.data.taxonomy import NUM_LEAVES
from sdp.model.classifier import CodeBERTClassifier

_HIDDEN = 32
_PAD_ID = 1


def _tiny_encoder() -> RobertaModel:
    config = RobertaConfig(
        vocab_size=100,
        hidden_size=_HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=70,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=_PAD_ID,
        type_vocab_size=1,
    )
    return RobertaModel(config, add_pooling_layer=False)


def test_logits_have_one_score_per_class_and_are_finite():
    torch.manual_seed(0)
    model = CodeBERTClassifier(_tiny_encoder())
    ids = torch.randint(2, 100, (3, 10))
    logits = model(ids, torch.ones_like(ids))
    assert logits.shape == (3, NUM_LEAVES)
    assert torch.isfinite(logits).all()


def test_num_labels_is_configurable():
    torch.manual_seed(0)
    model = CodeBERTClassifier(_tiny_encoder(), num_labels=4)
    ids = torch.randint(2, 100, (2, 8))
    assert model(ids, torch.ones_like(ids)).shape == (2, 4)


def test_padding_does_not_change_logits():
    torch.manual_seed(0)
    model = CodeBERTClassifier(_tiny_encoder())
    model.eval()
    ids = torch.randint(2, 100, (1, 6))
    padded = torch.cat([ids, torch.full((1, 4), _PAD_ID)], dim=1)
    padded_mask = torch.tensor([[1] * 6 + [0] * 4])
    with torch.no_grad():
        plain = model(ids, torch.ones_like(ids))
        with_padding = model(padded, padded_mask)
    assert torch.allclose(plain, with_padding, atol=1e-5)


def test_head_initialization():
    torch.manual_seed(0)
    encoder = _tiny_encoder()
    model = CodeBERTClassifier(encoder)
    weight = model.classifier.weight
    assert weight.std().item() == pytest.approx(encoder.config.initializer_range, abs=0.01)
    assert abs(weight.mean().item()) < 0.01
    assert torch.count_nonzero(model.classifier.bias) == 0


def test_gradients_reach_head_and_encoder():
    torch.manual_seed(0)
    model = CodeBERTClassifier(_tiny_encoder())
    ids = torch.randint(2, 100, (2, 8))
    logits = model(ids, torch.ones_like(ids))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert model.classifier.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())


def test_gradient_checkpointing_matches_plain_training():
    torch.manual_seed(0)
    plain = CodeBERTClassifier(_tiny_encoder(), dropout=0.0)
    checkpointed = copy.deepcopy(plain)
    checkpointed.enable_gradient_checkpointing()
    assert checkpointed.encoder.is_gradient_checkpointing
    assert not plain.encoder.is_gradient_checkpointing

    ids = torch.randint(2, 100, (2, 8))
    mask = torch.ones_like(ids)
    labels = torch.tensor([0, 1])
    results = []
    for model in (plain, checkpointed):
        model.train()
        logits = model(ids, mask)
        torch.nn.functional.cross_entropy(logits, labels).backward()
        first_query = model.encoder.encoder.layer[0].attention.self.query.weight
        results.append((logits.detach(), model.classifier.weight.grad, first_query.grad))

    for plain_value, checkpointed_value in zip(*results, strict=True):
        assert torch.allclose(plain_value, checkpointed_value, atol=1e-5)


def test_rejects_bad_arguments():
    with pytest.raises(ValueError, match="num_labels"):
        CodeBERTClassifier(_tiny_encoder(), num_labels=1)
    with pytest.raises(ValueError, match="dropout"):
        CodeBERTClassifier(_tiny_encoder(), dropout=1.0)


def test_num_parameters_counts_trainable_only():
    model = CodeBERTClassifier(_tiny_encoder())
    assert model.num_parameters() == sum(p.numel() for p in model.parameters())
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    assert model.num_parameters(trainable_only=True) == _HIDDEN * NUM_LEAVES + NUM_LEAVES


def test_head_dropout_is_active_only_in_train_mode():
    torch.manual_seed(0)
    model = CodeBERTClassifier(_tiny_encoder(), dropout=0.9)
    ids = torch.randint(2, 100, (2, 8))
    mask = torch.ones_like(ids)
    model.train()
    assert not torch.allclose(model(ids, mask), model(ids, mask))
    model.eval()
    assert torch.allclose(model(ids, mask), model(ids, mask))
