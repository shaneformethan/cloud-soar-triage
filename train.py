"""
Training script for the multi-modal SOAR triage model (Section III).

Training configuration (Table II — shared across all model variants):
  Optimizer  : Adam
  LR         : {1e-4, 5e-4, 1e-3}  (grid-searched; default 1e-3)
  Batch size : {32, 64, 128}        (default 64)
  Max epochs : 100 (early stopping, patience = 10)
  LR decay   : ReduceLROnPlateau (factor 0.5, patience 5)
  Weight init: Xavier uniform (done in model __init__)
  Loss       : CrossEntropy

Usage:
    python train.py [--config config/default.yaml] [--device cpu|cuda]
    python train.py --variant arch_a    # train ablation variant
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import load_splits, TriageDataset, collate_batch
from src.models.model import (
    MultiModalTriageModel, ModelConfig, MaskedModalityModel, MODALITY_SUBSETS,
)
from src.baselines.arch_variants import AblationModel


# ─────────────────────────────────────────────────────────────────────────────
# Default training config
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TRAIN_CFG = {
    # 5e-4 (within the draft's {1e-4, 5e-4, 1e-3} grid) is the selected value:
    # it trains the cross-modal attention fusion stably, whereas 1e-3 makes the
    # z_fuse attention oscillate and underfit.
    "lr":              5e-4,
    "batch_size":      64,
    "max_epochs":      100,
    "patience":        10,
    "lr_factor":       0.5,
    "lr_patience":     5,
    "grad_clip":       1.0,
    "weight_decay":    1e-4,    # AdamW L2 regularization (curbs overfitting)
    "label_smoothing": 0.05,    # softens targets, improves calibration
    "processed_dir":   "data/processed",
    "checkpoint_dir":  "checkpoints",
    "log_every":       5,   # log every N epochs
}


def compute_class_weights(
    loader: DataLoader, n_classes: int = 4, device: str = "cpu"
) -> torch.Tensor:
    """
    Inverse-frequency class weights for CrossEntropyLoss, computed from the
    training split's severity_label distribution.

    Rationale: the benchmark's severity classes are imbalanced (e.g.
    "informational" and "medium" are under-represented relative to "high"
    and "critical"). Without weighting, the cross-entropy loss is dominated
    by the majority classes and the model learns to never predict the
    minority classes (recall = 0), which is exactly the failure mode seen
    in the unweighted runs. Weighting follows the same "balanced" principle
    already used for the RandomForest/XGBoost baselines
    (src/baselines/traditional.py).
    """
    counts = np.zeros(n_classes, dtype=np.float64)
    for c in loader.dataset.clusters:
        counts[int(c.severity_label)] += 1
    counts = np.maximum(counts, 1.0)  # avoid div-by-zero for absent classes
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    grad_clip: float,
) -> tuple[float, float]:
    """Returns (avg_loss, weighted_f1)."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        iam_tokens  = batch["iam_tokens"].to(device)
        iam_lengths = batch["iam_lengths"].to(device)
        flow_vec    = batch["flow_vec"].to(device)
        ti_vec      = batch["ti_vec"].to(device)
        labels      = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(iam_tokens, iam_lengths, flow_vec, ti_vec)
        loss   = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    n = len(all_labels)
    avg_loss = total_loss / max(n, 1)
    wf1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    return avg_loss, wf1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Returns (avg_loss, weighted_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        iam_tokens  = batch["iam_tokens"].to(device)
        iam_lengths = batch["iam_lengths"].to(device)
        flow_vec    = batch["flow_vec"].to(device)
        ti_vec      = batch["ti_vec"].to(device)
        labels      = batch["label"].to(device)

        logits = model(iam_tokens, iam_lengths, flow_vec, ti_vec)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    n = len(all_labels)
    return total_loss / max(n, 1), float(
        f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: str,
    checkpoint_name: str = "best_model.pt",
) -> dict:
    """
    Full training loop with early stopping and LR decay.

    Returns dict with training history.
    """
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / checkpoint_name

    class_weights = compute_class_weights(train_loader, n_classes=4, device=device)
    print(f"  Class weights (info/med/high/crit): "
          f"{[round(w, 3) for w in class_weights.cpu().tolist()]}")
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg.get("label_smoothing", 0.0),
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",          # maximize val F1
        factor=cfg["lr_factor"],
        patience=cfg["lr_patience"],
    )

    best_val_f1 = -1.0
    patience_counter = 0
    history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}

    print(f"Training on {device} | "
          f"params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg["grad_clip"]
        )
        vl_loss, vl_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl_f1)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(vl_loss)
        history["val_f1"].append(vl_f1)

        if epoch % cfg["log_every"] == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:>3}/{cfg['max_epochs']} | "
                f"loss {tr_loss:.4f}/{vl_loss:.4f} | "
                f"F1 {tr_f1:.4f}/{vl_f1:.4f} | "
                f"lr {lr:.2e} | {elapsed:.1f}s"
            )

        # Checkpoint
        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            patience_counter = 0
            _save_checkpoint(model, cfg, epoch, vl_f1, str(ckpt_path))
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {cfg['patience']} epochs).")
                break

    print(f"  Best val F1: {best_val_f1:.4f} -> saved to {ckpt_path}")
    history["best_val_f1"] = best_val_f1
    history["checkpoint_path"] = str(ckpt_path)
    return history


def _model_cfg_dict(mc: ModelConfig) -> dict:
    return {
        "d": mc.d, "dropout": mc.dropout, "n_classes": mc.n_classes,
        "iam_n_layers": mc.iam_n_layers, "iam_n_heads": mc.iam_n_heads,
        "iam_max_seq_len": mc.iam_max_seq_len,
        "iam_use_positional_encoding": mc.iam_use_positional_encoding,
        "flow_n_layers": mc.flow_n_layers, "ti_n_layers": mc.ti_n_layers,
        "ti_hidden_dim": mc.ti_hidden_dim, "fusion_strategy": mc.fusion_strategy,
        "fusion_n_heads": mc.fusion_n_heads, "cls_n_layers": mc.cls_n_layers,
    }


def _save_checkpoint(
    model: nn.Module, cfg: dict, epoch: int, val_f1: float, path: str
) -> None:
    cfg_dict = {}
    extra = {}
    if isinstance(model, MultiModalTriageModel):
        cfg_dict = _model_cfg_dict(model.cfg)
    elif isinstance(model, MaskedModalityModel):
        cfg_dict = _model_cfg_dict(model.cfg)
        extra["active_modalities"] = sorted(model.active_modalities)
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg_dict,
        "epoch": epoch,
        "val_f1": val_f1,
        "train_cfg": cfg,
        **extra,
    }, path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multi-modal SOAR triage model")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--lr", type=float, default=DEFAULT_TRAIN_CFG["lr"])
    p.add_argument("--batch_size", type=int, default=DEFAULT_TRAIN_CFG["batch_size"])
    p.add_argument("--max_epochs", type=int, default=DEFAULT_TRAIN_CFG["max_epochs"])
    p.add_argument("--patience", type=int, default=DEFAULT_TRAIN_CFG["patience"])
    p.add_argument("--weight_decay", type=float,
                   default=DEFAULT_TRAIN_CFG["weight_decay"])
    p.add_argument("--label_smoothing", type=float,
                   default=DEFAULT_TRAIN_CFG["label_smoothing"])
    p.add_argument("--log_every", type=int,
                   default=DEFAULT_TRAIN_CFG["log_every"])
    p.add_argument(
        "--variant", default="proposed",
        choices=(
            ["proposed", "iam_priority"]
            + list(AblationModel.VARIANTS.keys())
            + list(MODALITY_SUBSETS.keys())
        ),
        help="Model variant to train (default: proposed model)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (Section IV: 5 seeds per configuration)"
    )
    p.add_argument(
        "--processed_dir", default=DEFAULT_TRAIN_CFG["processed_dir"],
        help="Path to processed benchmark pkl files"
    )
    p.add_argument(
        "--checkpoint_dir", default=DEFAULT_TRAIN_CFG["checkpoint_dir"]
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Update config
    cfg = {**DEFAULT_TRAIN_CFG}
    cfg["lr"]             = args.lr
    cfg["batch_size"]     = args.batch_size
    cfg["max_epochs"]     = args.max_epochs
    cfg["patience"]       = args.patience
    cfg["weight_decay"]   = args.weight_decay
    cfg["label_smoothing"] = args.label_smoothing
    cfg["log_every"]      = args.log_every
    cfg["processed_dir"]  = args.processed_dir
    cfg["checkpoint_dir"] = args.checkpoint_dir

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    # Reproducibility (Section IV: 5 seeds per configuration)
    torch.manual_seed(args.seed)

    print(f"=== Training variant: {args.variant} (seed={args.seed}) ===")
    train_loader, val_loader, _ = load_splits(
        cfg["processed_dir"],
        batch_size=cfg["batch_size"],
    )

    # Build model
    if args.variant == "proposed":
        model = MultiModalTriageModel(ModelConfig())
        ckpt_name = "proposed_best.pt"
    elif args.variant == "iam_priority":
        model = MultiModalTriageModel(ModelConfig(fusion_strategy="iam_priority"))
        ckpt_name = "iam_priority_best.pt"
    elif args.variant in MODALITY_SUBSETS:
        model = MaskedModalityModel(MODALITY_SUBSETS[args.variant], ModelConfig())
        ckpt_name = f"{args.variant}_best.pt"
    else:
        model = AblationModel(variant=args.variant)
        ckpt_name = f"{args.variant}_best.pt"

    if args.seed != 42:
        ckpt_name = ckpt_name.replace("_best.pt", f"_seed{args.seed}_best.pt")

    model.to(device)
    history = train(model, train_loader, val_loader, cfg, device, ckpt_name)
    print("Training complete.")
