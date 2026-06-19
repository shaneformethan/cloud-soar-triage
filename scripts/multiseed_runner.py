"""
Multi-seed experiment runner (Section IV): all 17 configurations x 5 seeds
= 85 training runs, results reported as mean +/- std.

The 17 configurations are:
    1  proposed            (zfuse fusion, tri-modal)
    2  iam_priority        (IAM-prioritized fusion, tri-modal)
    3-8  arch_a .. arch_f  (architectural ablations, tri-modal)
    9-14 iam_only, flow_only, ti_only, iam_flow, iam_ti, flow_ti
         (modality-subset ablations)
    15 trad_rf             (Random Forest, flattened features)
    16 trad_xgb            (XGBoost, flattened features)
    17 deepcase            (DeepCASE, adapted to flow modality)

For deep variants (1-14), each seed trains a fresh model via train.py and the
resulting checkpoint is evaluated with the same pipeline as main.py
(temperature scaling, SOAR threshold fitting at the 3% FNR ceiling, full
report). For the non-deep baselines (15-17), each "seed" re-fits the
estimator with a different random_state, since these models have no
torch-level seed.

Usage:
    python scripts/multiseed_runner.py [--seeds 42 43 44 45 46]
                                        [--max_epochs 100] [--device cpu]
                                        [--checkpoint_dir checkpoints/multiseed]
                                        [--out results/multiseed_results.json]

NOTE: This is a long-running script (85 training runs at full budget). For a
quick correctness check, pass --max_epochs with a small value (e.g. 5) and/or
--seeds with a single seed.
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
from src.models.model import (
    MultiModalTriageModel, ModelConfig, MaskedModalityModel, MODALITY_SUBSETS,
)
from src.baselines.arch_variants import AblationModel
from src.baselines.traditional import RandomForestBaseline, XGBoostBaseline
from src.baselines.deepcase import DeepCASEBaseline
from src.soar.integration import SOARRouter
from src.evaluation.metrics import full_evaluation_report
from train import train, DEFAULT_TRAIN_CFG
from main import evaluate_deep_model, collect_numpy_arrays


DEEP_VARIANTS = (
    ["proposed", "iam_priority"]
    + list(AblationModel.VARIANTS.keys())
    + list(MODALITY_SUBSETS.keys())
)
ML_VARIANTS = ["trad_rf", "trad_xgb", "deepcase"]
ALL_VARIANTS = DEEP_VARIANTS + ML_VARIANTS  # 14 + 3 = 17


def build_deep_model(variant: str) -> torch.nn.Module:
    if variant == "proposed":
        return MultiModalTriageModel(ModelConfig())
    if variant == "iam_priority":
        return MultiModalTriageModel(ModelConfig(fusion_strategy="iam_priority"))
    if variant in MODALITY_SUBSETS:
        return MaskedModalityModel(MODALITY_SUBSETS[variant], ModelConfig())
    return AblationModel(variant=variant)


def ckpt_name_for(variant: str, seed: int) -> str:
    base = f"{variant}_best.pt"
    if seed != 42:
        base = base.replace("_best.pt", f"_seed{seed}_best.pt")
    return base


def one_hot(preds: np.ndarray, n_classes: int = 4) -> np.ndarray:
    probs = np.zeros((len(preds), n_classes), dtype=np.float32)
    probs[np.arange(len(preds)), preds] = 1.0
    return probs


def run_deep_variant(
    variant: str, seed: int, args, train_loader, val_loader, test_loader, device,
) -> dict:
    torch.manual_seed(seed)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_path = ckpt_dir / ckpt_name_for(variant, seed)

    model = build_deep_model(variant)
    model.to(device)

    # Always train fresh when this (variant, seed) has not been recorded yet:
    # a checkpoint left by a disconnected run may be only partially trained, so
    # we never trust it. Completed (variant, seed) pairs are skipped earlier via
    # the results JSON, so we never redo finished work.
    cfg = {
        **DEFAULT_TRAIN_CFG,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "checkpoint_dir": str(ckpt_dir),
    }
    train(model, train_loader, val_loader, cfg, device, ckpt_path.name)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])

    model.to(device)
    report = evaluate_deep_model(
        model, train_loader, val_loader, test_loader, device,
        model_name=f"{variant} (seed={seed})",
    )
    return report


def run_ml_variant(
    variant: str, seed: int, train_np: dict, val_np: dict, test_np: dict,
) -> dict:
    router = SOARRouter()

    if variant == "trad_rf":
        clf = RandomForestBaseline(random_state=seed)
        clf.fit(train_np["iam_tokens"], train_np["iam_lengths"],
                train_np["flow_vecs"], train_np["ti_vecs"], train_np["labels"])
        val_proba = clf.predict_proba(val_np["iam_tokens"], val_np["iam_lengths"],
                                        val_np["flow_vecs"], val_np["ti_vecs"])
        test_proba = clf.predict_proba(test_np["iam_tokens"], test_np["iam_lengths"],
                                         test_np["flow_vecs"], test_np["ti_vecs"])
    elif variant == "trad_xgb":
        clf = XGBoostBaseline(random_state=seed)
        clf.fit(train_np["iam_tokens"], train_np["iam_lengths"],
                train_np["flow_vecs"], train_np["ti_vecs"], train_np["labels"])
        val_proba = clf.predict_proba(val_np["iam_tokens"], val_np["iam_lengths"],
                                        val_np["flow_vecs"], val_np["ti_vecs"])
        test_proba = clf.predict_proba(test_np["iam_tokens"], test_np["iam_lengths"],
                                         test_np["flow_vecs"], test_np["ti_vecs"])
    elif variant == "deepcase":
        clf = DeepCASEBaseline()
        torch.manual_seed(seed)
        clf.fit(train_np["flow_vecs"], train_np["labels"])
        val_proba = one_hot(clf.predict(val_np["flow_vecs"]))
        test_proba = one_hot(clf.predict(test_np["flow_vecs"]))
    else:
        raise ValueError(f"Unknown ML variant: {variant!r}")

    tau_h, tau_l = router.fit_thresholds(val_proba, val_np["labels"])
    report = full_evaluation_report(
        probs=test_proba,
        labels=test_np["labels"],
        tau_h=tau_h,
        tau_l=tau_l,
        scenario_classes=test_np["scenarios"],
    )
    return report


METRIC_KEYS = [
    "weighted_f1",
    "analyst_workload_compression",
    "fp_suppression_rate",
    "fnr",
    "mean_time_to_triage_ms",
]


def aggregate(reports: list[dict]) -> dict:
    agg = {}
    for key in METRIC_KEYS:
        vals = [r.get(key) for r in reports if r.get(key) is not None]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if vals:
            agg[key] = {
                "mean": float(statistics.mean(vals)),
                "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-seed runner: 17 configs x 5 seeds")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--max_epochs", type=int, default=DEFAULT_TRAIN_CFG["max_epochs"])
    p.add_argument("--patience", type=int, default=DEFAULT_TRAIN_CFG["patience"])
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--checkpoint_dir", default="checkpoints/multiseed")
    p.add_argument("--out", default="results/multiseed_results.json")
    p.add_argument(
        "--variants", nargs="+", default=ALL_VARIANTS, choices=ALL_VARIANTS,
        help="Subset of configurations to run (default: all 17)",
    )
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = load_splits(args.processed_dir, batch_size=64)
    train_np = val_np = test_np = None
    if any(v in ML_VARIANTS for v in args.variants):
        train_np = collect_numpy_arrays(train_loader)
        val_np = collect_numpy_arrays(val_loader)
        test_np = collect_numpy_arrays(test_loader)

    all_results: dict[str, list[dict]] = {v: [] for v in args.variants}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Resume: reload any results already saved, and skip those (variant, seed) ──
    done: set[tuple[str, int]] = set()
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            for variant, reports in prev.get("raw", {}).items():
                if variant not in all_results:
                    all_results[variant] = []
                for r in reports:
                    all_results[variant].append(r)
                    if r.get("seed") is not None:
                        done.add((variant, int(r["seed"])))
            print(f"Resuming: {len(done)} (variant, seed) runs already done.")
        except Exception as e:
            print(f"Could not read prior results ({e}); starting fresh.")

    def _save() -> None:
        summary = {v: aggregate(reps) for v, reps in all_results.items()}
        out_path.write_text(json.dumps(
            {"summary": summary, "raw": all_results}, indent=2, default=str))

    n_total = len(args.variants) * len(args.seeds)
    run_idx = 0
    for variant in args.variants:
        for seed in args.seeds:
            run_idx += 1
            if (variant, seed) in done:
                print(f"[{run_idx}/{n_total}] skip {variant} seed={seed} (done)")
                continue
            print(f"\n{'='*70}")
            print(f"[{run_idx}/{n_total}] variant={variant} seed={seed}")
            print(f"{'='*70}")

            if variant in ML_VARIANTS:
                report = run_ml_variant(variant, seed, train_np, val_np, test_np)
            else:
                report = run_deep_variant(
                    variant, seed, args, train_loader, val_loader, test_loader, device
                )
            report["seed"] = seed
            all_results[variant].append(report)
            _save()   # incremental save after every run (survives disconnects)

    # ── Aggregate mean +/- std across seeds ─────────────────────────────
    summary = {}
    for variant, reports in all_results.items():
        summary[variant] = aggregate(reports)

    print(f"\n{'='*78}")
    print("  SUMMARY: mean +/- std over seeds (Section IV)")
    print(f"{'='*78}")
    header = f"  {'Config':<16} {'W-F1':>16} {'WC':>16} {'FNR':>14}"
    print(header)
    print(f"  {'-'*74}")
    for variant in args.variants:
        s = summary[variant]
        wf1 = s.get("weighted_f1", {"mean": float("nan"), "std": 0.0})
        wc  = s.get("analyst_workload_compression", {"mean": float("nan"), "std": 0.0})
        fnr = s.get("fnr", {"mean": float("nan"), "std": 0.0})
        print(
            f"  {variant:<16} "
            f"{wf1['mean']:.4f}+/-{wf1['std']:.4f} "
            f"{wc['mean']:.1%}+/-{wc['std']:.1%} "
            f"{fnr['mean']:.3f}+/-{fnr['std']:.3f}"
        )
    print(f"{'='*78}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "raw": all_results}, f, indent=2, default=str)
    print(f"Saved full results to {out_path}")


if __name__ == "__main__":
    main()
