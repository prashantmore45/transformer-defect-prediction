"""Fine-tune CodeBERT on the frozen 9-class dataset (Milestone 7.4).

Usage (from the repository root, laptop plugged in):

    python scripts/train_codebert.py                      # the real run (configs/train.yaml)
    python scripts/train_codebert.py --resume             # continue after an interruption
    python scripts/train_codebert.py --smoke              # 1-2 minute end-to-end check

Everything is written to runs/<run_name>/ (git-ignored): config.json, metrics.csv,
train.log, last.pt (resumable state), best.pt (best weights plus metadata),
val_report_best.csv and summary.json. The test split is never read.

Smoke mode trains for one epoch on a small random subset (the train size is chosen so the
final accumulation group is partial) into runs/smoke, which is wiped on each fresh smoke run.
"""

import argparse
import random
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, Subset

from sdp.data.dataset import SourceCodeDataset
from sdp.model.classifier import CodeBERTClassifier
from sdp.training.class_weights import compute_class_weights, count_classes
from sdp.training.config import TrainConfig
from sdp.training.trainer import make_logger, run_training

SMOKE_TRAIN_SAMPLES = 70
SMOKE_VAL_SAMPLES = 32


def _subset(dataset: Dataset, size: int, seed: int) -> Subset:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="train.yaml (default: repo)")
    parser.add_argument("--resume", action="store_true", help="continue from runs/<run>/last.pt")
    parser.add_argument("--smoke", action="store_true", help="quick end-to-end check")
    parser.add_argument("--run-name", default=None, help="override run_name")
    parser.add_argument("--max-epochs", type=int, default=None, help="override max_epochs")
    parser.add_argument("--stop-after-epoch", type=int, default=None, help="simulate interruption")
    parser.add_argument("--log-every", type=int, default=100, help="optimizer steps between logs")
    args = parser.parse_args()

    transformers.logging.set_verbosity_error()
    cfg = TrainConfig.from_yaml(args.config)
    if args.smoke:
        cfg = replace(cfg, run_name="smoke", max_epochs=1)
    if args.run_name is not None:
        cfg = replace(cfg, run_name=args.run_name)
    if args.max_epochs is not None:
        cfg = replace(cfg, max_epochs=args.max_epochs)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; training runs on the local GPU only.")
    if cfg.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("bf16 is not supported on this GPU; set precision to fp16 in the config.")
    device = torch.device("cuda")

    run_dir = cfg.run_dir
    if args.smoke and not args.resume and cfg.run_name == "smoke" and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(run_dir / "train.log")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    train_path = cfg.cache_path / "train.parquet"
    labels = pd.read_parquet(train_path, columns=["label"])["label"].to_numpy()
    counts = count_classes(labels)
    weights = torch.tensor(
        compute_class_weights(counts, cfg.class_weight_power), dtype=torch.float32, device=device
    )
    log(f"class counts {counts.tolist()}; weights {[round(float(w), 3) for w in weights.tolist()]}")

    train_dataset: Dataset = SourceCodeDataset(train_path)
    val_dataset: Dataset = SourceCodeDataset(cfg.cache_path / "val.parquet")
    if args.smoke:
        train_dataset = _subset(train_dataset, SMOKE_TRAIN_SAMPLES, cfg.seed)
        val_dataset = _subset(val_dataset, SMOKE_VAL_SAMPLES, cfg.seed)

    model = CodeBERTClassifier.from_pretrained(
        cfg.model_name,
        dropout=cfg.dropout,
        gradient_checkpointing=cfg.gradient_checkpointing,
        local_files_only=True,
    ).to(device)
    pad_token_id = int(model.encoder.config.pad_token_id)

    summary = run_training(
        cfg=cfg,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        class_counts=counts,
        weights=weights,
        pad_token_id=pad_token_id,
        device=device,
        resume=args.resume,
        log=log,
        stop_after_epoch=args.stop_after_epoch,
        log_every=args.log_every,
    )
    log(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
