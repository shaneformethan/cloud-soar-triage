"""
Main entry point: full evaluation pipeline (Section III-F + IV).

Runs the proposed model and all baselines against the held-out test set,
then prints a consolidated report covering:
  - Analyst workload compression (at 3% FNR ceiling)
  - FP suppression rate
  - Mean time-to-triage
  - Weighted F1 / per-class metrics
  - ECE before/after temperature scaling
  - Per-scenario breakdown (single/dual/tri)

Usage:
    python main.py                   # full evaluation (proposed + all baselines)
    python main.py --smoke           # quick smoke-test (2 epochs, tiny dataset)
    python main.py --variant arch_a  # evaluate specific ablation variant
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import load_splits, TriageDataset, collate_batch
from src.models.model import (
    MultiModalTriageModel, ModelConfig, load_model,
    MaskedModalityModel, MODALITY_SUBSETS,
)
from src.calibration.temperature import TemperatureScaler
from src.soar.integration import SOARRouter
from src.evaluation.metrics import (
    full_evaluation_report, print_report, mean_time_to_triage
)
from src.baselines.arch_variants import AblationModel
from src.baselines.traditional import RandomForestBaseline, XGBoostBaseline
from src.baselines.deepcase import DeepCASEBaseline
from train import train, DEFAULT_TRAIN_CFG


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def collect_logits_and_labels(
    model: torch.nn.Module,
    loader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Collect all logits, labels, and scenario classes from a DataLoader."""
    model.eval()
    all_logits, all_labels, all_scenarios = [], [], []

    with torch.no_grad():
        for batch in loader:
            iam  = batch["iam_tokens"].to(device)
            lens = batch["iam_lengths"].to(device)
            flow = batch["flow_vec"].to(device)
            ti   = batch["ti_vec"].to(device)
            lbls = batch["label"]

            logits = model(iam, lens, flow, ti)
            all_logits.append(logits.cpu())
            all_labels.append(lbls)
            all_scenarios.extend(batch.get("scenario", []))

    return (
        torch.cat(all_logits),
        torch.cat(all_labels),
        all_scenarios,
    )


def collect_numpy_arrays(loader) -> dict:
    """Collect all batch arrays as numpy for sklearn baselines."""
    iam_tokens_list, iam_lengths_list, flow_list, ti_list, labels_list, sc_list = \
        [], [], [], [], [], []
    for batch in loader:
        iam_tokens_list.append(batch["iam_tokens"].numpy())
        iam_lengths_list.append(batch["iam_lengths"].numpy())
        flow_list.append(batch["flow_vec"].numpy())
        ti_list.append(batch["ti_vec"].numpy())
        labels_list.append(batch["label"].numpy())
        sc_list.extend(batch.get("scenario", []))

    return {
        "iam_tokens":  np.concatenate(iam_tokens_list),
        "iam_lengths": np.concatenate(iam_lengths_list),
        "flow_vecs":   np.concatenate(flow_list),
        "ti_vecs":     np.concatenate(ti_list),
        "labels":      np.concatenate(labels_list),
        "scenarios":   sc_list,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_deep_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: str,
    model_name: str = "proposed",
) -> dict:
    print(f"\n{'-'*62}")
    print(f"  Evaluating: {model_name}")

    # 1. Collect val logits for calibration + threshold optimization
    val_logits, val_labels, _ = collect_logits_and_labels(model, val_loader, device)

    # 2. Temperature scaling (Section III-C)
    scaler = TemperatureScaler(model)
    T = scaler.fit(val_logits, val_labels)
    ece_before, ece_after = scaler.ece_report()
    print(f"  Calibration: T={T:.4f} | ECE {ece_before:.4f} -> {ece_after:.4f}")

    # 3. SOAR threshold optimization on val (Section III-D)
    val_probs = scaler.predict_proba(val_logits).detach().numpy()
    router = SOARRouter()
    tau_h, tau_l = router.fit_thresholds(val_probs, val_labels.numpy())
    print(f"  Thresholds : tau_h={tau_h:.2f}, tau_l={tau_l:.2f}")

    # 4. Evaluate on test set
    test_logits, test_labels, test_scenarios = collect_logits_and_labels(
        model, test_loader, device
    )
    test_probs = scaler.predict_proba(test_logits).detach().numpy()
    test_labels_np = test_labels.numpy()

    # 5. MTTD measurement
    test_batches = []
    for batch in test_loader:
        test_batches.append((
            batch["iam_tokens"].to(device),
            batch["iam_lengths"].to(device),
            batch["flow_vec"].to(device),
            batch["ti_vec"].to(device),
        ))

    def forward_fn(*args):
        with torch.no_grad():
            return model(*args)

    mttd = mean_time_to_triage(forward_fn, test_batches)
    print(f"  MTTD       : {mttd['mean_ms']:.3f} +/- {mttd['std_ms']:.3f} ms")

    # 6. Full report
    report = full_evaluation_report(
        probs=test_probs,
        labels=test_labels_np,
        tau_h=tau_h,
        tau_l=tau_l,
        scenario_classes=test_scenarios,
        mttd_ms=mttd["mean_ms"],
        ece_before=ece_before,
        ece_after=ece_after,
    )
    report["model_name"] = model_name
    print_report(report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test (2 epochs, no checkpoint needed)
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test(processed_dir: str, device: str) -> None:
    print("=== SMOKE TEST (2 epochs) ===")
    train_loader, val_loader, test_loader = load_splits(
        processed_dir, batch_size=32
    )

    # Proposed model
    model = MultiModalTriageModel(ModelConfig())
    model.to(device)

    cfg = {**DEFAULT_TRAIN_CFG, "max_epochs": 2, "patience": 99, "log_every": 1,
           "checkpoint_dir": "/tmp/cst_smoke"}
    train(model, train_loader, val_loader, cfg, device, "smoke_proposed.pt")

    # Forward pass dim check
    model.eval()
    batch = next(iter(test_loader))
    with torch.no_grad():
        logits = model(
            batch["iam_tokens"].to(device),
            batch["iam_lengths"].to(device),
            batch["flow_vec"].to(device),
            batch["ti_vec"].to(device),
        )
    assert logits.shape[-1] == 4, f"Logits shape error: {logits.shape}"
    print(f"  Logits shape: {logits.shape} OK")
    print(f"  Params: {model.count_parameters():,}")
    print("  Smoke test PASSED.")


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation (all variants)
# ─────────────────────────────────────────────────────────────────────────────

def full_evaluation(processed_dir: str, checkpoint_dir: str, device: str) -> None:
    print("=== FULL EVALUATION ===")
    train_loader, val_loader, test_loader = load_splits(
        processed_dir, batch_size=64
    )

    results = {}

    # ── Proposed model ──────────────────────────────────────────────────
    ckpt = Path(checkpoint_dir) / "proposed_best.pt"
    if ckpt.exists():
        model, _ = load_model(str(ckpt), device)
    else:
        print("  No checkpoint found; training proposed model for 100 epochs…")
        model = MultiModalTriageModel(ModelConfig())
        model.to(device)
        cfg = {**DEFAULT_TRAIN_CFG, "checkpoint_dir": checkpoint_dir,
               "processed_dir": processed_dir}
        train(model, train_loader, val_loader, cfg, device, "proposed_best.pt")
        model, _ = load_model(str(Path(checkpoint_dir) / "proposed_best.pt"), device)

    results["proposed"] = evaluate_deep_model(
        model, train_loader, val_loader, test_loader, device, "proposed"
    )

    # ── Ablation variants ────────────────────────────────────────────────
    for variant in AblationModel.VARIANTS:
        ckpt = Path(checkpoint_dir) / f"{variant}_best.pt"
        if ckpt.exists():
            abl = AblationModel(variant)
            state = torch.load(str(ckpt), map_location=device, weights_only=False)
            abl.load_state_dict(state["model_state"])
        else:
            print(f"  Training {variant}…")
            abl = AblationModel(variant)
            abl.to(device)
            cfg = {**DEFAULT_TRAIN_CFG, "checkpoint_dir": checkpoint_dir,
                   "max_epochs": 50}
            train(abl, train_loader, val_loader, cfg, device, f"{variant}_best.pt")
            state = torch.load(
                str(Path(checkpoint_dir) / f"{variant}_best.pt"),
                map_location=device, weights_only=False,
            )
            abl.load_state_dict(state["model_state"])
        abl.to(device)
        results[variant] = evaluate_deep_model(
            abl, train_loader, val_loader, test_loader, device, variant
        )

    # ── Modality ablation (6 subsets + IAM-prioritized fusion) ──────────
    for variant in list(MODALITY_SUBSETS.keys()) + ["iam_priority"]:
        ckpt = Path(checkpoint_dir) / f"{variant}_best.pt"
        if variant == "iam_priority":
            cfg_obj = ModelConfig(fusion_strategy="iam_priority")
            build_fn = lambda: MultiModalTriageModel(cfg_obj)
        else:
            active = MODALITY_SUBSETS[variant]
            build_fn = lambda active=active: MaskedModalityModel(active, ModelConfig())

        if ckpt.exists():
            mdl = build_fn()
            state = torch.load(str(ckpt), map_location=device, weights_only=False)
            mdl.load_state_dict(state["model_state"])
        else:
            print(f"  Training {variant}…")
            mdl = build_fn()
            mdl.to(device)
            cfg = {**DEFAULT_TRAIN_CFG, "checkpoint_dir": checkpoint_dir,
                   "processed_dir": processed_dir}
            train(mdl, train_loader, val_loader, cfg, device, f"{variant}_best.pt")
            state = torch.load(
                str(Path(checkpoint_dir) / f"{variant}_best.pt"),
                map_location=device, weights_only=False,
            )
            mdl.load_state_dict(state["model_state"])
        mdl.to(device)
        results[variant] = evaluate_deep_model(
            mdl, train_loader, val_loader, test_loader, device, variant
        )

    # ── Traditional ML ───────────────────────────────────────────────────
    train_np = collect_numpy_arrays(train_loader)
    val_np   = collect_numpy_arrays(val_loader)
    test_np  = collect_numpy_arrays(test_loader)

    for name, cls in [("trad_rf", RandomForestBaseline), ("trad_xgb", XGBoostBaseline)]:
        print(f"\n{'-'*62}")
        print(f"  Evaluating: {name}")
        try:
            clf = cls()
            clf.fit(
                train_np["iam_tokens"], train_np["iam_lengths"],
                train_np["flow_vecs"], train_np["ti_vecs"],
                train_np["labels"],
            )
            proba = clf.predict_proba(
                test_np["iam_tokens"], test_np["iam_lengths"],
                test_np["flow_vecs"], test_np["ti_vecs"],
            )
            router = SOARRouter()
            val_proba = clf.predict_proba(
                val_np["iam_tokens"], val_np["iam_lengths"],
                val_np["flow_vecs"], val_np["ti_vecs"],
            )
            tau_h, tau_l = router.fit_thresholds(val_proba, val_np["labels"])

            report = full_evaluation_report(
                probs=proba,
                labels=test_np["labels"],
                tau_h=tau_h,
                tau_l=tau_l,
                scenario_classes=test_np["scenarios"],
            )
            report["model_name"] = name
            print_report(report)
            results[name] = report
        except Exception as e:
            print(f"  {name} failed: {e}")

    # ── DeepCASE (external baseline, adapted to flow modality) ───────────
    print(f"\n{'-'*62}")
    print("  Evaluating: deepcase")
    try:
        deepcase = DeepCASEBaseline()
        deepcase.fit(train_np["flow_vecs"], train_np["labels"])

        def _one_hot(preds: np.ndarray, n_classes: int = 4) -> np.ndarray:
            probs = np.zeros((len(preds), n_classes), dtype=np.float32)
            probs[np.arange(len(preds)), preds] = 1.0
            return probs

        val_preds = deepcase.predict(val_np["flow_vecs"])
        router = SOARRouter()
        tau_h, tau_l = router.fit_thresholds(_one_hot(val_preds), val_np["labels"])

        test_preds = deepcase.predict(test_np["flow_vecs"])
        report = full_evaluation_report(
            probs=_one_hot(test_preds),
            labels=test_np["labels"],
            tau_h=tau_h,
            tau_l=tau_l,
            scenario_classes=test_np["scenarios"],
        )
        report["model_name"] = "deepcase"
        print_report(report)
        results["deepcase"] = report
    except Exception as e:
        print(f"  deepcase failed: {e}")

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Proposed vs Baselines (test set)")
    print(f"{'='*70}")
    print(f"  {'Model':<20} {'W-F1':>8} {'WC':>8} {'FPR':>8} {'FNR':>8} {'MTTD ms':>10}")
    print(f"  {'-'*66}")
    for name, r in results.items():
        wf1  = r.get("weighted_f1", 0)
        wc   = r.get("analyst_workload_compression", 0)
        fpr  = r.get("fp_suppression_rate", float("nan"))
        fnr  = r.get("fnr", float("nan"))
        mttd = r.get("mean_time_to_triage_ms", float("nan"))
        print(f"  {name:<20} {wf1:>8.4f} {wc:>8.1%} {fpr:>8.1%} "
              f"{fnr:>8.3f} {mttd if mttd else float('nan'):>10.3f}")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-modal SOAR Triage — main entry point"
    )
    p.add_argument("--smoke", action="store_true",
                   help="Run 2-epoch smoke test only")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument(
        "--variant", default=None,
        choices=(
            list(AblationModel.VARIANTS.keys())
            + list(MODALITY_SUBSETS.keys())
            + ["iam_priority"]
        ),
        help="Evaluate a single ablation variant only (currently informational; "
             "full_evaluation() always runs all variants)"
    )
    p.add_argument(
        "--output", default=None,
        help="Simpan output evaluasi ke file (misal: results/eval.txt)"
    )
    return p.parse_args()


class _Tee:
    """Menulis ke stdout dan file secara bersamaan."""
    def __init__(self, filepath: str):
        self._file = open(filepath, "w", encoding="utf-8")
        self._stdout = sys.stdout
        # When stdout is redirected/piped (e.g. PowerShell `| Tee-Object`),
        # Windows often falls back to a legacy codepage (cp1252) that can't
        # encode the box-drawing/Greek characters (─, τ, ✓, …) used in the
        # report output. Force UTF-8 so printing never crashes.
        if hasattr(self._stdout, "reconfigure"):
            try:
                self._stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    def write(self, msg):
        try:
            self._stdout.write(msg)
        except UnicodeEncodeError:
            enc = getattr(self._stdout, "encoding", None) or "ascii"
            self._stdout.write(msg.encode(enc, errors="replace").decode(enc, errors="replace"))
        self._file.write(msg)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


if __name__ == "__main__":
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.")
        device = "cpu"

    os.chdir(Path(__file__).parent)   # ensure relative paths work

    # Output otomatis ke file (disimpan di results/ agar root tetap rapi)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"results/eval_{timestamp}.txt" if not args.smoke else None
    output_path = args.output or default_output

    tee = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        tee = _Tee(output_path)
        sys.stdout = tee

    try:
        if args.smoke:
            smoke_test(args.processed_dir, device)
        else:
            full_evaluation(args.processed_dir, args.checkpoint_dir, device)
    finally:
        if tee:
            sys.stdout = tee._stdout
            tee.close()
            print(f"\nHasil evaluasi disimpan ke: {output_path}")
