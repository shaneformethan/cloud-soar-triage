"""
Resumable focused multi-seed runner for the key comparison
(proposed vs concat/avg-pool fusion vs RandomForest/XGBoost), reported as
mean +/- std over seeds.

Design for an unreliable machine: results are written to JSON *after every
(seed, model)*, and on restart any (seed, model) already recorded is skipped.
A shutdown therefore only costs the run in progress. Models are trained on the
fixed benchmark in data/processed (so the test set is identical across seeds);
only the training seed (init + shuffling) varies, which isolates model
differences -- the right design for "is the gap significant?".

Usage:
    python scripts/multiseed_focused.py --seeds 42 43 44 45 46 --max_epochs 12
    python scripts/multiseed_focused.py            # resume / finish
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_splits
from src.models.model import MultiModalTriageModel, ModelConfig
from src.baselines.arch_variants import AblationModel
from src.baselines.traditional import RandomForestBaseline, XGBoostBaseline
from src.soar.integration import SOARRouter
from src.evaluation.metrics import full_evaluation_report
from train import train, DEFAULT_TRAIN_CFG
from main import evaluate_deep_model, collect_numpy_arrays

DEEP = {"proposed", "arch_e", "arch_f"}
METRICS = ["weighted_f1", "macro_f1", "analyst_workload_compression",
           "fp_suppression_rate", "fnr", "mean_time_to_triage_ms",
           "weighted_f1_single", "weighted_f1_dual", "weighted_f1_tri"]


def _slim(report: dict) -> dict:
    return {k: report.get(k) for k in METRICS if k in report}


def _build(variant: str):
    if variant == "proposed":
        return MultiModalTriageModel(ModelConfig())
    return AblationModel(variant=variant)   # arch_e / arch_f


def run_deep(variant, seed, args, tr, va, te, ckpt_dir):
    torch.manual_seed(seed)
    model = _build(variant).to(args.device)
    cfg = {**DEFAULT_TRAIN_CFG, "max_epochs": args.max_epochs,
           "patience": args.patience, "checkpoint_dir": str(ckpt_dir)}
    name = f"{variant}_seed{seed}.pt"
    train(model, tr, va, cfg, args.device, name)
    state = torch.load(str(ckpt_dir / name), map_location=args.device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(args.device)
    return _slim(evaluate_deep_model(model, tr, va, te, args.device, f"{variant}[{seed}]"))


def run_ml(variant, seed, train_np, val_np, test_np):
    cls = RandomForestBaseline if variant == "trad_rf" else XGBoostBaseline
    clf = cls(random_state=seed)
    clf.fit(train_np["iam_tokens"], train_np["iam_lengths"],
            train_np["flow_vecs"], train_np["ti_vecs"], train_np["labels"])
    vp = clf.predict_proba(val_np["iam_tokens"], val_np["iam_lengths"],
                           val_np["flow_vecs"], val_np["ti_vecs"])
    tp = clf.predict_proba(test_np["iam_tokens"], test_np["iam_lengths"],
                           test_np["flow_vecs"], test_np["ti_vecs"])
    tau_h, tau_l = SOARRouter().fit_thresholds(vp, val_np["labels"])
    return _slim(full_evaluation_report(probs=tp, labels=test_np["labels"],
                                        tau_h=tau_h, tau_l=tau_l,
                                        scenario_classes=test_np["scenarios"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--variants", nargs="+",
                   default=["proposed", "arch_e", "arch_f", "trad_rf", "trad_xgb"])
    p.add_argument("--max_epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--checkpoint_dir", default="checkpoints/ms")
    p.add_argument("--out", default="results/multiseed_focused.json")
    args = p.parse_args()

    import os
    os.chdir(Path(__file__).parent.parent)
    ckpt_dir = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load any prior results. Keyed results[variant][str(seed)] = metrics.
    results = json.loads(out.read_text()) if out.exists() else {}
    for v in args.variants:
        results.setdefault(v, {})

    tr, va, te = load_splits(args.processed_dir, batch_size=64)
    train_np = val_np = test_np = None
    if any(v in ("trad_rf", "trad_xgb") for v in args.variants):
        train_np, val_np, test_np = (collect_numpy_arrays(x) for x in (tr, va, te))

    for seed in args.seeds:
        for variant in args.variants:
            if str(seed) in results[variant]:
                print(f"skip {variant} seed={seed} (done)")
                continue
            print(f"\n{'='*60}\n  {variant}  seed={seed}\n{'='*60}")
            if variant in DEEP:
                rep = run_deep(variant, seed, args, tr, va, te, ckpt_dir)
            else:
                rep = run_ml(variant, seed, train_np, val_np, test_np)
            results[variant][str(seed)] = rep
            out.write_text(json.dumps(results, indent=2, default=str))  # incremental save

    # Aggregate.
    print(f"\n{'='*72}\n  MULTI-SEED SUMMARY (mean +/- std over {len(args.seeds)} seeds)")
    print(f"  {'Model':<12}{'W-F1':>16}{'Macro-F1':>16}{'tri W-F1':>16}")
    print(f"  {'-'*60}")
    def ms(v, key):
        xs = [results[v][s].get(key) for s in results[v] if results[v][s].get(key) is not None]
        if not xs: return float('nan'), 0.0
        return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)
    for v in sorted(args.variants, key=lambda v: -ms(v, "weighted_f1")[0]):
        m, sd = ms(v, "weighted_f1"); mm, sdm = ms(v, "macro_f1"); mt, sdt = ms(v, "weighted_f1_tri")
        print(f"  {v:<12}{m:.4f}+/-{sd:.4f}{mm:>9.4f}+/-{sdm:.4f}{mt:>8.4f}+/-{sdt:.4f}")
    print(f"{'='*72}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
