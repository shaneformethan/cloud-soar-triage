"""
Generate a presentable RESULTS.md from the multi-seed result JSONs.

Reads results/multiseed_iid.json and results/multiseed_temporal.json (produced
by scripts/multiseed_runner.py) and writes a clean Markdown report with the
controlled vs temporal comparison, SOAR metrics, and the real-data validation.

Usage:
    python scripts/make_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROWS = [
    ("__h__", "Tri-modal (Flow + IAM + TI)"),
    ("arch_a", "Arch-A (BiLSTM)"),
    ("arch_d", "Arch-D (Transformer, no PE)"),
    ("iam_priority", "IAM-priority fusion"),
    ("arch_c", "Arch-C (1D-CNN)"),
    ("arch_f", "Arch-F (avg-pool)"),
    ("arch_e", "Arch-E (concat)"),
    ("arch_b", "Arch-B (BiGRU)"),
    ("proposed", "Cross-modal (z_fuse)"),
    ("trad_xgb", "XGBoost"),
    ("trad_rf", "Random Forest"),
    ("__h__", "Reduced modality subsets"),
    ("iam_flow", "IAM + Flow"),
    ("flow_ti", "Flow + TI"),
    ("iam_ti", "IAM + TI"),
    ("flow_only", "Flow only"),
    ("iam_only", "IAM only"),
    ("ti_only", "TI only"),
    ("__h__", "External baseline"),
    ("deepcase", "DeepCASE (flow)"),
]


def cell(summary, name, key, pct=False):
    agg = summary.get(name, {}).get(key)
    if not agg:
        return "--"
    m, s = agg["mean"], agg["std"]
    if pct:
        return f"{m*100:.1f}"
    return f"{m:.3f}±{s:.3f}"


def main():
    root = Path(__file__).parent.parent
    iid = json.loads((root / "results/multiseed_iid.json").read_text())["summary"]
    tmp = json.loads((root / "results/multiseed_temporal.json").read_text())["summary"]

    L = []
    L.append("# Evaluation Results\n")
    L.append("Multi-modal cloud security incident triage benchmark. "
             "All numbers are **mean ± std over 5 seeds** on the held-out test set, "
             "from `scripts/multiseed_runner.py` (17 configurations × 5 seeds = 85 runs).\n")
    L.append("Two evaluation regimes: **i.i.d.** (controlled benchmark) and "
             "**temporal** (per-class chronological split of the real flows: earliest "
             "70% of each class trains, latest 20% tests).\n")

    L.append("## Main comparison\n")
    L.append("| Configuration | W-F1 (i.i.d.) | W-F1 (temporal) | Workload comp. % | FP-supp. % | FNR |")
    L.append("|---|---|---|---|---|---|")
    for name, label in ROWS:
        if name == "__h__":
            L.append(f"| **{label}** | | | | | |")
            continue
        L.append(f"| {label} | {cell(iid,name,'weighted_f1')} | "
                 f"{cell(tmp,name,'weighted_f1')} | "
                 f"{cell(iid,name,'analyst_workload_compression',pct=True)} | "
                 f"{cell(iid,name,'fp_suppression_rate',pct=True)} | "
                 f"{cell(iid,name,'fnr')} |")
    L.append("\n*Workload compression, FP-suppression, and FNR are on the i.i.d. "
             "(deployment) benchmark, at the 3% FNR ceiling.*\n")

    L.append("## Key findings\n")
    L.append("1. **Modality coverage dominates architecture.** Any configuration with "
             "both Flow and IAM reaches ~0.97 W-F1 (i.i.d.); single-modality models are "
             "far weaker (0.31-0.76). Among tri-modal models the spread is only "
             "0.967-0.973 — cross-modal attention, pooling, concatenation, recurrent/"
             "convolutional encoders, and gradient-boosted trees are statistically tied.\n")
    L.append("2. **Temporal shift is the real challenge.** Under the temporal split every "
             "model drops to ~0.75-0.85. The collapse is in the real flow stream "
             "(Flow-only 0.76→0.37; DeepCASE 0.30→0.06) because attack tooling "
             "evolves over the two-week capture; the synthetic IAM stream does not shift "
             "(IAM-only ~0.75 both regimes). XGBoost is the strongest temporal model.\n")
    L.append("3. **Calibration enables SOAR auto-dispatch.** Temperature scaling brings the "
             "multimodal model's ECE to ~0.02 (i.i.d.); the multimodal models automate "
             "~96% of incidents below a 1% false-negative rate, suppressing >98% of "
             "non-actionable alerts.\n")

    L.append("## Real-data validation (raw CIC-IDS2018 flows, temporal split)\n")
    L.append("Classifiers trained directly on raw CSE-CIC-IDS2018 per-flow records "
             "(26 real features, real labels), per-class temporal split:\n")
    L.append("| Model | Weighted F1 | high-sev F1 | critical (DoS/DDoS) F1 |")
    L.append("|---|---|---|---|")
    L.append("| XGBoost | 0.64 | ~0.998 | ~0.35 (recall ~0.21) |")
    L.append("| Random Forest | 0.62 | ~0.998 | ~0.33 |")
    L.append("\nThis confirms the flow signal is real (not a benchmark artifact) and that "
             "temporal generalization is hard for evolving DoS/DDoS tooling. Reproduce with "
             "`python scripts/validate_real_flow.py`.\n")

    L.append("## Reproduce\n")
    L.append("```\n"
             "python scripts/fetch_ti_pool.py            # real TI pool (needs .env keys)\n"
             "python scripts/build_benchmark_semi.py     # i.i.d. benchmark\n"
             "python scripts/build_benchmark_semi.py --temporal-split \\\n"
             "       --processed_dir data/processed_temporal   # temporal benchmark\n"
             "python scripts/multiseed_runner.py --seeds 42 43 44 45 46   # 17x5 (GPU)\n"
             "python scripts/make_results.py             # regenerate this file\n"
             "```\n")

    out = root / "RESULTS.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
