"""
60-trial random search for the proposed model's architecture hyperparameters
(Table I, tab:hp_proposed; Bergstra & Bengio, b24).

Search space (architecture):
    d (embedding dim)        : {128, 256, 512}
    IAM encoder layers       : {2, 4, 6}
    Attention heads h         : {4, 8, 16}   (must divide d)
    Flow encoder layers       : {1, 2, 3}
    TI encoder layers         : {1, 2}
    Classification head layers: {1, 2}
    Dropout rate              : {0.1, 0.2, 0.3}

Training hyperparameters (sampled jointly, shared with Table I "Training"
block):
    Learning rate : {1e-4, 5e-4, 1e-3}
    Batch size    : {32, 64, 128}

NOTE on IAM sequence length: Table I also lists {32, 64, 128} as a candidate
for IAM sequence length. Changing this value requires rebuilding the
processed benchmark (build_benchmark.py) with a different
benchmark.iam_max_seq_len, since IAM token sequences are pre-padded/truncated
at build time. That dimension is therefore explored separately as the
sensitivity analysis in Section IV-C (15/60-day baseline windows x 32/128
sequence-length bounds), not as part of this architecture search. This script
fixes iam_max_seq_len=64 (the default benchmark build).

NOTE on tau_h / tau_l: these SOAR routing thresholds are NOT part of this
search. They are fit directly on the validation set at the 3% FNR ceiling by
SOARRouter.fit_thresholds() (see src/soar/integration.py), which is run
during full evaluation (main.py) for every trained model.

Each trial trains the proposed model (ModelConfig with sampled
hyperparameters, fusion_strategy="zfuse") with a REDUCED training budget
(max_epochs=15, patience=3) to keep the 60-trial search tractable, and is
scored by weighted F1 on the validation split. The best configuration is
written to config/best_hp.json; the final reported result should then be
obtained by training that configuration with the full budget
(max_epochs=100, patience=10) via train.py.

Usage:
    python scripts/random_search.py [--trials 60] [--seed 42] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_splits
from src.models.model import ModelConfig, MultiModalTriageModel
from train import train, DEFAULT_TRAIN_CFG


# ─────────────────────────────────────────────────────────────────────────────
# Search space
# ─────────────────────────────────────────────────────────────────────────────

D_CHOICES = [128, 256, 512]
IAM_LAYERS_CHOICES = [2, 4, 6]
HEADS_CHOICES = [4, 8, 16]
FLOW_LAYERS_CHOICES = [1, 2, 3]
TI_LAYERS_CHOICES = [1, 2]
CLS_LAYERS_CHOICES = [1, 2]
DROPOUT_CHOICES = [0.1, 0.2, 0.3]

LR_CHOICES = [1e-4, 5e-4, 1e-3]
BATCH_SIZE_CHOICES = [32, 64, 128]

SEARCH_MAX_EPOCHS = 15
SEARCH_PATIENCE = 3


def sample_trial(rng: random.Random) -> dict:
    """Sample one hyperparameter configuration, respecting d % heads == 0."""
    d = rng.choice(D_CHOICES)
    valid_heads = [h for h in HEADS_CHOICES if d % h == 0]
    return {
        "d": d,
        "iam_n_layers": rng.choice(IAM_LAYERS_CHOICES),
        "n_heads": rng.choice(valid_heads),
        "flow_n_layers": rng.choice(FLOW_LAYERS_CHOICES),
        "ti_n_layers": rng.choice(TI_LAYERS_CHOICES),
        "cls_n_layers": rng.choice(CLS_LAYERS_CHOICES),
        "dropout": rng.choice(DROPOUT_CHOICES),
        "lr": rng.choice(LR_CHOICES),
        "batch_size": rng.choice(BATCH_SIZE_CHOICES),
    }


def trial_to_model_config(trial: dict) -> ModelConfig:
    return ModelConfig(
        d=trial["d"],
        dropout=trial["dropout"],
        iam_n_layers=trial["iam_n_layers"],
        iam_n_heads=trial["n_heads"],
        flow_n_layers=trial["flow_n_layers"],
        ti_n_layers=trial["ti_n_layers"],
        fusion_n_heads=trial["n_heads"],
        fusion_strategy="zfuse",
        cls_n_layers=trial["cls_n_layers"],
    )


def main() -> None:
    p = argparse.ArgumentParser(description="60-trial random search (Table I)")
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--checkpoint_dir", default="checkpoints/random_search")
    p.add_argument("--out", default="config/best_hp.json")
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    rng = random.Random(args.seed)

    print(f"=== 60-trial random search (n_trials={args.trials}, seed={args.seed}) ===")
    train_loader, val_loader, _ = load_splits(args.processed_dir, batch_size=64)

    best_f1 = -1.0
    best_trial = None
    log = []

    for i in range(1, args.trials + 1):
        trial = sample_trial(rng)
        torch.manual_seed(args.seed * 1000 + i)

        model_cfg = trial_to_model_config(trial)
        model = MultiModalTriageModel(model_cfg)
        model.to(device)

        # batch size is search-specific, so rebuild the train loader if needed
        loader = train_loader
        if trial["batch_size"] != 64:
            loader, _, _ = load_splits(args.processed_dir, batch_size=trial["batch_size"])

        cfg = {
            **DEFAULT_TRAIN_CFG,
            "lr": trial["lr"],
            "batch_size": trial["batch_size"],
            "max_epochs": SEARCH_MAX_EPOCHS,
            "patience": SEARCH_PATIENCE,
            "checkpoint_dir": args.checkpoint_dir,
            "log_every": SEARCH_MAX_EPOCHS,  # quiet
        }

        t0 = time.time()
        history = train(model, loader, val_loader, cfg, device, f"trial_{i:02d}.pt")
        elapsed = time.time() - t0
        val_f1 = history["best_val_f1"]

        log.append({"trial": i, **trial, "val_f1": val_f1, "seconds": round(elapsed, 1)})
        print(f"[trial {i:02d}/{args.trials}] val_f1={val_f1:.4f} "
              f"({elapsed:.1f}s) | {trial}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_trial = trial
            print(f"  ↑ new best (val_f1={best_f1:.4f})")

    print(f"\n=== Best configuration (val_f1={best_f1:.4f}) ===")
    print(json.dumps(best_trial, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"best_trial": best_trial, "best_val_f1": best_f1, "log": log}, f, indent=2)
    print(f"\nSaved best config + full trial log to {out_path}")
    print("Next: train this configuration with the full budget via train.py "
          "(max_epochs=100, patience=10) for the reported result.")


if __name__ == "__main__":
    main()
