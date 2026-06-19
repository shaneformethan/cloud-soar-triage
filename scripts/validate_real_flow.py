"""
Real-data validation of the flow modality.

Trains classifiers DIRECTLY on real CSE-CIC-IDS2018 per-flow records (real
features, real labels), mapping each flow's attack label to the paper's
four-class severity scheme, under a genuine **temporal** split: within each
severity class, the earliest 70% of flows (by timestamp) are training, the next
10% validation, and the latest 20% test. This grounds the flow component of the
benchmark in real data and tests temporal generalization (train on earlier
attack instances, evaluate on later ones), addressing the i.i.d.-split
limitation.

Usage:
    python scripts/validate_real_flow.py [--per_class 40000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.benchmark.severity_mapping import cic2018_label_to_severity

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Curated informative numeric feature subset (present in CSE-CIC-IDS2018 CSVs).
FEATURES = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Mean",
    "Bwd Pkt Len Max", "Bwd Pkt Len Mean", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd Header Len", "Bwd Header Len", "Pkt Len Mean", "Pkt Len Std",
    "SYN Flag Cnt", "ACK Flag Cnt", "Init Fwd Win Byts", "Init Bwd Win Byts",
    "Down/Up Ratio",
]


def load_real_flows(data_dir: Path, per_class: int, seed: int = 42):
    """Load real per-flow records with severity + timestamp, capped per class."""
    rng = np.random.default_rng(seed)
    csv_files = sorted(data_dir.glob("*.csv"))
    want = set(FEATURES) | {"Label", "Timestamp"}
    feat_cols = FEATURES
    buckets: dict[int, list] = {0: [], 1: [], 2: [], 3: []}
    times: dict[int, list] = {0: [], 1: [], 2: [], 3: []}

    for f in csv_files:
        print(f"[real-flow] {f.name} …", flush=True)
        df = pd.read_csv(f, usecols=lambda c: c in want, low_memory=False)
        df = df[df["Label"].astype(str).str.lower() != "label"]
        if "Timestamp" not in df.columns:
            continue
        ts = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
        sev = df["Label"].map(cic2018_label_to_severity)
        X = df[[c for c in feat_cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        mask = sev.notna() & ts.notna() & X.notna().all(axis=1)
        X, sevv, tsv = X[mask].to_numpy(np.float32), sev[mask].astype(int).to_numpy(), ts[mask].astype("int64").to_numpy()
        for s in (0, 1, 2, 3):
            idx = np.where(sevv == s)[0]
            if len(idx) == 0:
                continue
            cap = max(per_class // len(csv_files), 200)
            if len(idx) > cap:
                idx = rng.choice(idx, size=cap, replace=False)
            buckets[s].append(X[idx]); times[s].append(tsv[idx])

    feats, sevs, tstamps = [], [], []
    for s in (0, 1, 2, 3):
        if buckets[s]:
            feats.append(np.concatenate(buckets[s]));
            sevs.append(np.full(sum(len(a) for a in buckets[s]), s))
            tstamps.append(np.concatenate(times[s]))
    return (np.concatenate(feats), np.concatenate(sevs),
            np.concatenate(tstamps), feat_cols)


def temporal_split(X, y, t):
    """Per-class chronological split 70/10/20 by timestamp."""
    tr, va, te = [], [], []
    for s in np.unique(y):
        idx = np.where(y == s)[0]
        idx = idx[np.argsort(t[idx])]            # sort this class by time
        n = len(idx); a, b = int(n * 0.7), int(n * 0.8)
        tr += idx[:a].tolist(); va += idx[a:b].tolist(); te += idx[b:].tolist()
    return np.array(tr), np.array(va), np.array(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=40000)
    ap.add_argument("--data_dir", default="data/raw/cic_ids2018")
    args = ap.parse_args()
    import os
    os.chdir(Path(__file__).parent.parent)

    X, y, t, cols = load_real_flows(Path(args.data_dir), args.per_class)
    # median-impute any residual gaps (none after mask, but safe)
    X = np.nan_to_num(X, nan=0.0)
    tr, va, te = temporal_split(X, y, t)
    print(f"\n[real-flow] features={X.shape[1]} | train={len(tr)} val={len(va)} test={len(te)}")
    print(f"[real-flow] class counts (test): {np.bincount(y[te], minlength=4).tolist()}")

    names = {"RandomForest": RandomForestClassifier(
                n_estimators=300, max_depth=None, class_weight="balanced",
                n_jobs=-1, random_state=42)}
    if HAS_XGB:
        names["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.9,
            objective="multi:softprob", num_class=4, n_jobs=-1, random_state=42,
            eval_metric="mlogloss")

    print(f"\n{'='*60}\n  REAL CIC-IDS2018 FLOW — temporal split (test set)\n{'='*60}")
    for nm, clf in names.items():
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        wf1 = f1_score(y[te], p, average="weighted")
        mf1 = f1_score(y[te], p, average="macro")
        print(f"\n  {nm}:  Weighted-F1 = {wf1:.4f}   Macro-F1 = {mf1:.4f}")
        print(classification_report(y[te], p,
              target_names=["info", "medium", "high", "critical"],
              digits=4, zero_division=0))


if __name__ == "__main__":
    main()
