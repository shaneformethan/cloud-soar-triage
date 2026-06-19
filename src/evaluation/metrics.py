"""
Evaluation metrics (Section III-F).

Three primary outcomes:
  1. Analyst workload compression
       Proportion of incidents resolved without direct analyst involvement
       (auto-dispatched or informational), reported at fixed 3% FNR ceiling.

  2. False positive suppression rate
       Fraction of generated alerts correctly classified as non-actionable
       (informational / low-confidence), against ground-truth labels.
       Target: ≥ 70% (conservative lower bound vs. production SOC FP rates [b1]).

  3. Mean time-to-triage (MTTD)
       Average wall-clock time from alert ingestion to routing decision,
       measured in milliseconds. Captures pipeline latency, not analyst time.

Supporting metrics:
  - Weighted F1, per-class precision/recall/F1 (model quality)
  - ECE before/after temperature scaling (calibration quality)
  - Confusion matrix
  - Per-scenario-class breakdown (single/dual/tri)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

SEVERITY_NAMES = ["informational", "medium", "high", "critical"]
FNR_CEILING = 0.03      # Section III-F
FP_SUPPRESSION_TARGET = 0.70  # Section III-F


# ─────────────────────────────────────────────────────────────────────────────
# Primary metrics
# ─────────────────────────────────────────────────────────────────────────────

def analyst_workload_compression(
    probs: np.ndarray,        # (N, 4) calibrated probs
    labels: np.ndarray,       # (N,)
    tau_h: float,             # auto-dispatch threshold
    tau_l: float,             # deferral threshold (unused here, kept for API)
    fnr_ceiling: float = FNR_CEILING,
) -> dict[str, float]:
    """
    Compute analyst workload compression at the given thresholds.

    Workload compression = fraction auto-dispatched (conf ≥ τ_h)
    subject to FNR constraint: if FNR > fnr_ceiling, report None.

    Returns dict with keys:
      workload_compression, fnr, meets_fnr_constraint, auto_rate
    """
    preds = probs.argmax(axis=1)
    conf  = probs.max(axis=1)

    auto_mask    = conf >= tau_h
    true_pos     = labels >= 1           # non-informational incidents
    fn_mask      = auto_mask & (preds == 0) & true_pos
    fnr          = fn_mask.sum() / max(true_pos.sum(), 1)
    auto_rate    = float(auto_mask.mean())
    meets        = fnr <= fnr_ceiling

    return {
        "workload_compression":  auto_rate if meets else 0.0,
        "auto_dispatch_rate":    auto_rate,
        "fnr":                   float(fnr),
        "meets_fnr_constraint":  meets,
    }


def fp_suppression_rate(
    preds: np.ndarray,    # (N,)  predicted severity labels
    labels: np.ndarray,   # (N,)  ground-truth severity labels
) -> float:
    """
    False positive suppression rate: fraction of true informational alerts
    (label=0) correctly predicted as informational.

    Targets ≥ 70% per Section III-F.
    """
    true_info = labels == 0
    if true_info.sum() == 0:
        return float("nan")
    correctly_suppressed = ((preds == 0) & true_info).sum()
    return float(correctly_suppressed / true_info.sum())


def mean_time_to_triage(
    forward_fn: Callable,
    batches: list,
    device: str = "cpu",
    n_warmup: int = 3,
) -> dict[str, float]:
    """
    Measure mean time-to-triage in milliseconds.

    Runs `forward_fn(*batch)` for each batch, records wall-clock time,
    and reports per-sample latency statistics.

    Parameters
    ----------
    forward_fn : callable
        Model forward function accepting batch args and returning logits.
    batches    : list of argument tuples, one per batch.
    n_warmup   : number of warmup batches before timing starts.

    Returns
    -------
    dict with mean_ms, std_ms, min_ms, max_ms, n_samples
    """
    if hasattr(forward_fn, "__self__"):
        forward_fn.__self__.eval()

    # Warmup
    for i, batch in enumerate(batches[:n_warmup]):
        _ = forward_fn(*batch)

    latencies_ms = []
    n_samples = 0

    for batch in batches[n_warmup:]:
        # Determine batch size from first tensor argument
        first = next((b for b in batch if hasattr(b, "__len__")), None)
        bs = len(first) if first is not None else 1

        t0 = time.perf_counter()
        _ = forward_fn(*batch)
        t1 = time.perf_counter()

        per_sample_ms = (t1 - t0) * 1000.0 / max(bs, 1)
        latencies_ms.extend([per_sample_ms] * bs)
        n_samples += bs

    arr = np.array(latencies_ms)
    return {
        "mean_ms": float(arr.mean()) if len(arr) else 0.0,
        "std_ms":  float(arr.std())  if len(arr) else 0.0,
        "min_ms":  float(arr.min())  if len(arr) else 0.0,
        "max_ms":  float(arr.max())  if len(arr) else 0.0,
        "n_samples": n_samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Supporting metrics
# ─────────────────────────────────────────────────────────────────────────────

def classification_metrics(
    preds: np.ndarray,    # (N,)
    labels: np.ndarray,   # (N,)
    scenario_classes: list[str] | None = None,
) -> dict[str, float | dict]:
    """
    Weighted F1, per-class metrics, confusion matrix.
    Optionally breaks down by scenario class (single/dual/tri).
    """
    weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    macro_f1    = float(f1_score(labels, preds, average="macro", zero_division=0))
    report      = classification_report(
        labels, preds,
        labels=list(range(len(SEVERITY_NAMES))),
        target_names=SEVERITY_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=list(range(4))).tolist()

    result: dict = {
        "weighted_f1": weighted_f1,
        "macro_f1":    macro_f1,
        "per_class":   {k: v for k, v in report.items() if k in SEVERITY_NAMES},
        "confusion_matrix": cm,
    }

    if scenario_classes is not None:
        for sc in ("single", "dual", "tri"):
            mask = np.array([c == sc for c in scenario_classes])
            if mask.sum() > 0:
                result[f"weighted_f1_{sc}"] = float(
                    f1_score(labels[mask], preds[mask],
                             average="weighted", zero_division=0)
                )

    return result


def full_evaluation_report(
    probs: np.ndarray,          # (N, 4) calibrated probs
    labels: np.ndarray,         # (N,)
    tau_h: float,
    tau_l: float,
    scenario_classes: list[str] | None = None,
    mttd_ms: float | None = None,
    ece_before: float | None = None,
    ece_after: float | None = None,
) -> dict:
    """
    Assemble the complete evaluation report as a single dict.
    """
    preds = probs.argmax(axis=1)

    workload = analyst_workload_compression(probs, labels, tau_h, tau_l)
    fp_rate  = fp_suppression_rate(preds, labels)
    clf      = classification_metrics(preds, labels, scenario_classes)

    report = {
        # Primary metrics (Section III-F)
        "analyst_workload_compression": workload["workload_compression"],
        "auto_dispatch_rate":           workload["auto_dispatch_rate"],
        "fnr":                          workload["fnr"],
        "meets_fnr_3pct_ceiling":       workload["meets_fnr_constraint"],
        "fp_suppression_rate":          fp_rate,
        "meets_fp_suppression_70pct":   fp_rate >= FP_SUPPRESSION_TARGET,
        # MTTD
        "mean_time_to_triage_ms":       mttd_ms,
        # Calibration
        "ece_before_scaling":           ece_before,
        "ece_after_scaling":            ece_after,
        # Classification quality
        **clf,
        # Thresholds used
        "tau_h": tau_h,
        "tau_l": tau_l,
    }
    return report


def print_report(report: dict) -> None:
    """Pretty-print the evaluation report."""
    sep = "-" * 62
    print(f"\n{'='*62}")
    print("  EVALUATION REPORT - Multi-Modal SOAR Triage")
    print(f"{'='*62}")

    print(f"\n  PRIMARY METRICS (Section III-F)")
    print(sep)
    wc = report.get("analyst_workload_compression", 0)
    print(f"  Analyst workload compression : {wc:.1%}  "
          f"(at FNR <= 3%: {report.get('meets_fnr_3pct_ceiling')})")
    print(f"  Auto-dispatch rate           : {report.get('auto_dispatch_rate', 0):.1%}")
    print(f"  FNR at tau_h={report.get('tau_h'):.2f}               : {report.get('fnr', 0):.3f}")
    fp = report.get("fp_suppression_rate", float("nan"))
    print(f"  FP suppression rate          : {fp:.1%}  "
          f"(>=70%: {report.get('meets_fp_suppression_70pct')})")
    mttd = report.get("mean_time_to_triage_ms")
    if mttd is not None:
        print(f"  Mean time-to-triage          : {mttd:.3f} ms")

    print(f"\n  CALIBRATION (ECE)")
    print(sep)
    print(f"  ECE before temperature scaling : {report.get('ece_before_scaling')}")
    print(f"  ECE after  temperature scaling : {report.get('ece_after_scaling')}")

    print(f"\n  CLASSIFICATION QUALITY")
    print(sep)
    print(f"  Weighted F1 : {report.get('weighted_f1', 0):.4f}")
    print(f"  Macro F1    : {report.get('macro_f1', 0):.4f}")

    per = report.get("per_class", {})
    print(f"\n  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*47}")
    for cls in SEVERITY_NAMES:
        m = per.get(cls, {})
        print(f"  {cls:<15} {m.get('precision', 0):>10.4f} "
              f"{m.get('recall', 0):>10.4f} {m.get('f1-score', 0):>10.4f}")

    for sc in ("single", "dual", "tri"):
        key = f"weighted_f1_{sc}"
        if key in report:
            print(f"\n  Weighted F1 ({sc:>6} scenario): {report[key]:.4f}")

    print(f"\n{'='*62}\n")
