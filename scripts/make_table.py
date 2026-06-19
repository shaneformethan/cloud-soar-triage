"""
Generate the LaTeX §IV results table from multiseed_runner JSON output(s).

Produces a "controlled (i.i.d.) vs temporal" weighted-F1 comparison table ready
to paste into draft.tex. Either column may be omitted if its JSON is missing.

Usage:
    # both columns
    python scripts/make_table.py --iid results/multiseed_results.json \
                                 --temporal results/multiseed_temporal.json
    # only what you have
    python scripts/make_table.py --iid path/to/multiseed_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Display order + human-readable labels (internal name -> (label, modalities)).
ROWS = [
    ("__h1__", "Tri-modal (Flow + IAM + TI)", ""),
    ("arch_a",       "Arch-A (BiLSTM)",            "F+I+T"),
    ("arch_d",       "Arch-D (Transf., no PE)",    "F+I+T"),
    ("iam_priority", "IAM-priority fusion",        "F+I+T"),
    ("arch_c",       "Arch-C (1D-CNN)",            "F+I+T"),
    ("arch_f",       "Arch-F (avg-pool)",          "F+I+T"),
    ("arch_e",       "Arch-E (concat)",            "F+I+T"),
    ("arch_b",       "Arch-B (BiGRU)",             "F+I+T"),
    ("proposed",     "Cross-modal ($z_\\text{fuse}$)", "F+I+T"),
    ("trad_xgb",     "Trad-B (XGBoost)",           "F+I+T"),
    ("trad_rf",      "Trad-A (Random Forest)",     "F+I+T"),
    ("__h2__", "Reduced modality subsets", ""),
    ("iam_flow", "IAM + Flow", "F+I"),
    ("flow_ti",  "Flow + TI",  "F+T"),
    ("iam_ti",   "IAM + TI",   "I+T"),
    ("flow_only","Flow only",  "F"),
    ("iam_only", "IAM only",   "I"),
    ("ti_only",  "TI only",    "T"),
    ("__h3__", "", ""),
    ("deepcase", "DeepCASE (external)", "F"),
]


def cell(summary: dict | None, key: str) -> str:
    if summary is None:
        return "--"
    agg = summary.get(key, {}).get("weighted_f1")
    if not agg:
        return "--"
    return f"${agg['mean']:.3f}{{\\pm}}{agg['std']:.3f}$"


def load(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"% (warning: {path} not found)")
        return None
    return json.loads(p.read_text()).get("summary", {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iid", default="results/multiseed_results.json")
    ap.add_argument("--temporal", default="results/multiseed_temporal.json")
    args = ap.parse_args()

    iid = load(args.iid)
    tmp = load(args.temporal)

    print("\\begin{table}[t]")
    print("\\caption{Weighted F1 (mean $\\pm$ std over 5 seeds): controlled "
          "i.i.d.\\ benchmark vs.\\ per-class temporal split.}")
    print("\\label{tab:results}")
    print("\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}")
    print("\\begin{tabular}{llcc}")
    print("\\toprule")
    print("\\textbf{Configuration} & \\textbf{Mod.} & "
          "\\textbf{W-F1 (i.i.d.)} & \\textbf{W-F1 (temporal)} \\\\")
    print("\\midrule")
    for name, label, mod in ROWS:
        if name.startswith("__h"):
            if label:
                print(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{label}}}}} \\\\")
            else:
                print("\\midrule")
            continue
        print(f"{label} & {mod} & {cell(iid, name)} & {cell(tmp, name)} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    main()
