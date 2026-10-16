"""Tests for sdp.training.checkpoint. A tiny random encoder is used: no network, no GPU."""

# pandas must be imported before torch in this process. Importing it afterwards makes
# pytest's fault handler print a (non-fatal) Windows access-violation traceback while
# pyarrow loads.
import pandas  # noqa: F401
import torch
from transformers import RobertaConfig, RobertaModel

from sdp.data.taxonomy import TAXONOMY_VERSION
from sdp.evaluation.metrics import LEAF_NAMES
from sdp.model.classifier import CodeBERTClassifier
from sdp.training.checkpoint import load_best, load_last, save_best, save_last


def _tiny_encoder() -> RobertaModel:
    config = RobertaConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=70,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=1,
        type_vocab_size=1,
    )
    return RobertaModel(config, add_pooling_layer=False)


def _setup(seed: int):
    torch.manual_seed(seed)
    model = CodeBERTClassifier(_tiny_encoder(), dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (1 + step))
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return model, optimizer, scheduler, scaler


def test_last_checkpoint_round_trip_restores_everything(tmp_path):
    model, optimizer, scheduler, scaler = _setup(0)
    ids = torch.randint(2, 100, (4, 8))
    for _ in range(2):
        loss = model(ids, torch.ones_like(ids)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    path = tmp_path / "last.pt"
    stopper = {"best": float("-inf"), "best_epoch": 0, "bad_epochs": 1}
    history = [{"epoch": 1, "val_macro_f1": 0.25, "is_best": True}]
    save_last(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=3,
        stopper=stopper,
        config={"learning_rate": 1e-3, "precision": "fp32"},
        history=history,
    )
    assert path.exists()
    assert not (tmp_path / "last.pt.tmp").exists()

    state = load_last(path)
    assert state["epoch"] == 3
    assert state["stopper"] == stopper
    assert state["history"] == history
    assert state["config"] == {"learning_rate": 1e-3, "precision": "fp32"}

    other_model, other_optimizer, other_scheduler, _ = _setup(99)
    other_model.load_state_dict(state["model"])
    other_optimizer.load_state_dict(state["optimizer"])
    other_scheduler.load_state_dict(state["scheduler"])
    for (name, mine), (_, theirs) in zip(
        model.state_dict().items(), other_model.state_dict().items(), strict=True
    ):
        assert torch.equal(mine, theirs), name
    assert other_scheduler.last_epoch == scheduler.last_epoch
    assert other_scheduler.get_last_lr() == scheduler.get_last_lr()
    first_parameter = next(iter(optimizer.state_dict()["state"]))
    assert other_optimizer.state_dict()["state"][first_parameter]["step"] == (
        optimizer.state_dict()["state"][first_parameter]["step"]
    )


def test_best_checkpoint_is_self_describing(tmp_path):
    path = tmp_path / "best.pt"
    meta = {
        "label_names": list(LEAF_NAMES),
        "taxonomy_version": TAXONOMY_VERSION,
        "best_epoch": 3,
        "val_macro_f1": 0.5,
    }
    save_best(path, {"weight": torch.arange(4.0)}, meta)
    loaded = load_best(path)
    assert loaded["meta"] == meta
    assert torch.equal(loaded["model_state"]["weight"], torch.arange(4.0))
    assert not (tmp_path / "best.pt.tmp").exists()
