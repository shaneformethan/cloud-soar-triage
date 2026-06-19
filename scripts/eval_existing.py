"""
Lean evaluation: report the comparison table for whatever deep-model
checkpoints already exist in ``checkpoints/``, plus the always-fast traditional
ML baselines (RandomForest, XGBoost) and DeepCASE.

Unlike ``main.py`` (which trains every one of the 17 configurations when a
checkpoint is missing), this script only *evaluates* models that are already
trained.  Use it to demonstrate results from a curated subset without paying for
a full 17-config sweep on CPU.

Usage:
    python scripts/eval_existing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_splits
from src.models.model import load_model
from src.baselines.arch_variants import AblationModel
from src.baselines.traditional import RandomForestBaseline, XGBoostBaseline
from src.calibration.temperature import TemperatureScaler
from src.soar.integration import SOARRouter
from src.evaluation.metrics import full_evaluation_report
from main import (
    collect_logits_and_labels, collect_numpy_arrays, evaluate_deep_model,
)

CKPT_DIR = Path("checkpoints")
DEVICE = "cpu"


def _load_any(name: str):
    """Load a checkpoint as the right class (proposed/masked or ablation)."""
    path = CKPT_DIR / f"{name}_best.pt"
    if not path.exists():
        return None
    if name in AblationModel.VARIANTS:
        mdl = AblationModel(name)
        state = torch.load(str(path), map_location=DEVICE, weights_only=False)
        mdl.load_state_dict(state["model_state"])
        mdl.to(DEVICE).eval()
        return mdl
    mdl, _ = load_model(str(path), DEVICE)
    return mdl


def _deep_split_f1(mdl, loader) -> float:
    """Weighted F1 of a deep model's argmax predictions on a loader."""
    logits, labels, _ = collect_logits_and_labels(mdl, loader, DEVICE)
    preds = logits.argmax(dim=-1).numpy()
    return float(f1_score(labels.numpy(), preds, average="weighted", zero_division=0))


def main() -> None:
    import os
    os.chdir(Path(__file__).parent.parent)

    tr, va, te = load_splits("data/processed", batch_size=64)
    results = {}
    split_f1 = {}   # name -> (train, val, test) weighted F1, for overfitting check

    # All checkpoints present in the folder.
    names = sorted(p.name.replace("_best.pt", "") for p in CKPT_DIR.glob("*_best.pt"))
    print(f"Found checkpoints: {names}\n")
    for name in names:
        mdl = _load_any(name)
        if mdl is None:
            continue
        results[name] = evaluate_deep_model(mdl, tr, va, te, DEVICE, name)
        split_f1[name] = (_deep_split_f1(mdl, tr), _deep_split_f1(mdl, va),
                          _deep_split_f1(mdl, te))

    # Traditional baselines (fast).
    train_np, val_np, test_np = (collect_numpy_arrays(x) for x in (tr, va, te))
    for nm, cls in [("trad_rf", RandomForestBaseline), ("trad_xgb", XGBoostBaseline)]:
        try:
            clf = cls()
            clf.fit(train_np["iam_tokens"], train_np["iam_lengths"],
                    train_np["flow_vecs"], train_np["ti_vecs"], train_np["labels"])
            test_proba = clf.predict_proba(test_np["iam_tokens"], test_np["iam_lengths"],
                                           test_np["flow_vecs"], test_np["ti_vecs"])
            val_proba = clf.predict_proba(val_np["iam_tokens"], val_np["iam_lengths"],
                                          val_np["flow_vecs"], val_np["ti_vecs"])
            tau_h, tau_l = SOARRouter().fit_thresholds(val_proba, val_np["labels"])
            rep = full_evaluation_report(probs=test_proba, labels=test_np["labels"],
                                         tau_h=tau_h, tau_l=tau_l,
                                         scenario_classes=test_np["scenarios"])
            rep["model_name"] = nm
            results[nm] = rep
            train_pred = clf.predict(train_np["iam_tokens"], train_np["iam_lengths"],
                                     train_np["flow_vecs"], train_np["ti_vecs"])
            val_pred = clf.predict(val_np["iam_tokens"], val_np["iam_lengths"],
                                   val_np["flow_vecs"], val_np["ti_vecs"])
            split_f1[nm] = (
                float(f1_score(train_np["labels"], train_pred, average="weighted", zero_division=0)),
                float(f1_score(val_np["labels"], val_pred, average="weighted", zero_division=0)),
                rep.get("weighted_f1", 0.0),
            )
        except Exception as e:
            print(f"  {nm} failed: {e}")

    # Summary table.
    print(f"\n{'='*72}")
    print("  SUMMARY: test set")
    print(f"  {'Model':<18}{'W-F1':>8}{'Macro-F1':>10}{'WC':>8}{'FPsup':>8}{'FNR':>8}")
    print(f"  {'-'*68}")
    for nm, r in sorted(results.items(), key=lambda kv: -kv[1].get("weighted_f1", 0)):
        print(f"  {nm:<18}{r.get('weighted_f1',0):>8.4f}{r.get('macro_f1',0):>10.4f}"
              f"{r.get('analyst_workload_compression',0):>8.1%}"
              f"{r.get('fp_suppression_rate',float('nan')):>8.1%}"
              f"{r.get('fnr',float('nan')):>8.3f}")
    print(f"{'='*72}")

    # Overfitting check: train vs val vs test weighted F1.
    print(f"\n{'='*72}")
    print("  OVERFITTING CHECK (weighted F1 per split; small train-test gap = OK)")
    print(f"  {'Model':<18}{'train':>8}{'val':>8}{'test':>8}{'train-test':>12}")
    print(f"  {'-'*60}")
    for nm in sorted(split_f1, key=lambda k: -split_f1[k][2]):
        tr_f1, va_f1, te_f1 = split_f1[nm]
        gap = tr_f1 - te_f1
        flag = "  <-- check" if gap > 0.05 else ""
        print(f"  {nm:<18}{tr_f1:>8.4f}{va_f1:>8.4f}{te_f1:>8.4f}{gap:>12.4f}{flag}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
