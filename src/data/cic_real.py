"""
Real CSE-CIC-IDS2018 flow pool builder.

Parses the raw CIC-IDS2018 CSVs and produces, per severity class, a pool of
real 12-dimensional aggregated flow vectors following the Section III-B spec:

    [ total_bytes, total_packets,
      proto_tcp_ratio, proto_udp_ratio, proto_icmp_ratio,
      dst_port_entropy, src_port_entropy,
      flow_iat_mean, flow_iat_std, flow_iat_min, flow_iat_max,
      flow_duration_mean ]

Because the public CSE-CIC-IDS2018 CSVs ship *without* source/destination IP
columns, and because real attacks occupy only a handful of distinct 5-minute
wall-clock windows, grouping strictly by (label, time-bucket) yields too few
buckets to train on.  We instead model a 5-minute bucket as a *random aggregate
of real flows of one attack family*: per severity class we collect that class's
real per-flow records and bootstrap ``n_buckets`` aggregate vectors, each built
from a random subset of flows.  Every bucket is therefore a genuine aggregate of
real CIC traffic, with natural variation and (because subsets are resampled)
effectively no duplicate vectors across train/val/test.  Each flow's label is
mapped to severity via :func:`severity_mapping.cic2018_label_to_severity`.

The resulting pool is cached to disk so the ~10M-row parse runs only once.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark.severity_mapping import cic2018_label_to_severity

FLOW_DIM = 12

# Minimal column set we actually need (keeps the 10M-row parse light).
_USECOLS = [
    "Dst Port", "Protocol", "Timestamp", "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Min", "Flow IAT Max",
    "Label",
]
_NUMERIC = [
    "Dst Port", "Protocol", "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Min", "Flow IAT Max",
]


def _entropy(series: pd.Series) -> float:
    """Shannon entropy (bits) of a value distribution."""
    counts = series.value_counts(normalize=True).to_numpy()
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    return float(-(counts * np.log2(counts)).sum())


# Per-flow numeric columns retained for bootstrap aggregation.
_FLOW_COLS = [
    "Dst Port", "Protocol", "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Min", "Flow IAT Max",
]


def _aggregate_subset(sub: np.ndarray) -> np.ndarray:
    """
    Aggregate a subset of per-flow records (columns ordered as _FLOW_COLS) into
    the raw 12-dim bucket vector.
    """
    n = max(len(sub), 1)
    dst_port = sub[:, 0]
    proto = sub[:, 1]
    flow_dur = sub[:, 2]
    tfp, tbp = sub[:, 3], sub[:, 4]
    tlf, tlb = sub[:, 5], sub[:, 6]
    iat_mean, iat_std, iat_min, iat_max = sub[:, 7], sub[:, 8], sub[:, 9], sub[:, 10]

    vec = np.zeros(FLOW_DIM, dtype=np.float64)
    vec[0] = np.clip(tlf, 0, None).sum() + np.clip(tlb, 0, None).sum()  # total bytes
    vec[1] = np.clip(tfp, 0, None).sum() + np.clip(tbp, 0, None).sum()  # total packets
    vec[2] = float((proto == 6).sum()) / n   # TCP ratio
    vec[3] = float((proto == 17).sum()) / n  # UDP ratio
    vec[4] = float((proto == 1).sum()) / n   # ICMP ratio
    vec[5] = _entropy(pd.Series(dst_port))   # dst port entropy
    vec[6] = 0.0                             # src port entropy (no column)
    vec[7] = float(np.mean(iat_mean))
    vec[8] = float(np.mean(iat_std))
    vec[9] = float(np.mean(iat_min))
    vec[10] = float(np.mean(iat_max))
    vec[11] = float(np.mean(flow_dur))
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def build_flow_pool(
    data_dir: str | Path,
    bucket_minutes: int = 5,
    cache_path: str | Path | None = None,
    rebuild: bool = False,
    n_buckets_per_sev: int = 2500,
    flows_per_bucket: tuple[int, int] = (40, 200),
    max_flows_per_sev: int = 150_000,
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """
    Build (or load from cache) a per-severity pool of RAW 12-dim flow vectors by
    bootstrap-aggregating real CIC per-flow records.

    Returns
    -------
    dict[int, np.ndarray]
        severity index (0-3) -> array of shape (n_buckets_per_sev, 12).
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists() and not rebuild:
        with open(cache_path, "rb") as f:
            pool = pickle.load(f)
        print(f"[cic_real] Loaded cached flow pool from {cache_path} "
              f"({ {k: len(v) for k, v in pool.items()} }).")
        return pool

    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    rng = np.random.default_rng(seed)
    flows: dict[int, list[np.ndarray]] = defaultdict(list)

    for f in csv_files:
        print(f"[cic_real] Parsing {f.name} …")
        try:
            df = pd.read_csv(f, usecols=lambda c: c in _USECOLS, low_memory=False)
        except Exception as e:
            print(f"[cic_real]   skipped ({e})")
            continue
        if "Label" not in df.columns or "Timestamp" not in df.columns:
            print("[cic_real]   missing Label/Timestamp, skipping")
            continue

        df = df[df["Label"].astype(str).str.lower() != "label"]
        for col in _FLOW_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=[c for c in _FLOW_COLS if c in df.columns])
        df["_sev"] = df["Label"].map(cic2018_label_to_severity)
        df = df.dropna(subset=["_sev"])
        if df.empty:
            continue
        df["_sev"] = df["_sev"].astype(int)

        feats = df[_FLOW_COLS].to_numpy(dtype=np.float64)
        sev = df["_sev"].to_numpy()
        for s in np.unique(sev):
            rows = feats[sev == s]
            # Reservoir-style cap: keep a random subsample if we have plenty.
            cur = sum(len(a) for a in flows[int(s)])
            if cur < max_flows_per_sev:
                take = min(len(rows), max_flows_per_sev - cur)
                if take < len(rows):
                    rows = rows[rng.choice(len(rows), size=take, replace=False)]
                flows[int(s)].append(rows)

    raw = {s: np.concatenate(v, axis=0) for s, v in flows.items() if v}
    print(f"[cic_real] Per-flow counts: { {k: len(v) for k, v in raw.items()} }")

    lo, hi = flows_per_bucket
    pool: dict[int, np.ndarray] = {}
    for s, mat in raw.items():
        buckets = np.empty((n_buckets_per_sev, FLOW_DIM), dtype=np.float64)
        for b in range(n_buckets_per_sev):
            m = int(rng.integers(lo, hi + 1))
            idx = rng.integers(0, len(mat), size=m)   # bootstrap with replacement
            buckets[b] = _aggregate_subset(mat[idx])
        pool[s] = buckets
    print(f"[cic_real] Built flow pool: { {k: len(v) for k, v in pool.items()} }")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(pool, f)
        print(f"[cic_real] Cached flow pool -> {cache_path}")
    return pool


def build_flow_pool_temporal(
    data_dir: str | Path,
    cache_path: str | Path | None = None,
    rebuild: bool = False,
    n_buckets_per_sev: int = 2500,
    flows_per_bucket: tuple[int, int] = (40, 200),
    max_flows_per_sev: int = 150_000,
    seed: int = 42,
) -> dict[str, dict[int, np.ndarray]]:
    """
    Like :func:`build_flow_pool`, but partitions each severity class's real
    flows **chronologically** before bootstrap-aggregating: the earliest 70% of
    a class's flows (by timestamp) feed the train buckets, the next 10% the
    validation buckets, and the latest 20% the test buckets. Each split's
    buckets are therefore aggregates of temporally-disjoint real flows, giving a
    true temporal split of the real-data component (train on earlier attack
    instances, evaluate on later ones).

    Returns ``{split: {sev: (n_buckets, 12)}}``.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists() and not rebuild:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    rng = np.random.default_rng(seed)
    # Per class: collect (features, timestamp) for chronological ordering.
    flows: dict[int, list[np.ndarray]] = defaultdict(list)
    stamps: dict[int, list[np.ndarray]] = defaultdict(list)

    for f in csv_files:
        print(f"[cic_real:temporal] Parsing {f.name} …")
        try:
            df = pd.read_csv(f, usecols=lambda c: c in _USECOLS, low_memory=False)
        except Exception as e:
            print(f"  skipped ({e})"); continue
        if "Label" not in df.columns or "Timestamp" not in df.columns:
            continue
        df = df[df["Label"].astype(str).str.lower() != "label"]
        for col in _FLOW_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        ts = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
        df = df.assign(_ts=ts)
        df = df.dropna(subset=[c for c in _FLOW_COLS if c in df.columns] + ["_ts"])
        df["_sev"] = df["Label"].map(cic2018_label_to_severity)
        df = df.dropna(subset=["_sev"])
        if df.empty:
            continue
        df["_sev"] = df["_sev"].astype(int)
        feats = df[_FLOW_COLS].to_numpy(np.float64)
        tcol = df["_ts"].astype("int64").to_numpy()
        sev = df["_sev"].to_numpy()
        for s in np.unique(sev):
            m = sev == s
            cur = sum(len(a) for a in flows[int(s)])
            if cur < max_flows_per_sev:
                rows, tt = feats[m], tcol[m]
                take = min(len(rows), max_flows_per_sev - cur)
                if take < len(rows):
                    sel = rng.choice(len(rows), size=take, replace=False)
                    rows, tt = rows[sel], tt[sel]
                flows[int(s)].append(rows); stamps[int(s)].append(tt)

    lo, hi = flows_per_bucket
    out = {"train": {}, "val": {}, "test": {}}
    nb = {"train": int(n_buckets_per_sev * 0.7),
          "val": max(int(n_buckets_per_sev * 0.1), 1),
          "test": int(n_buckets_per_sev * 0.2)}
    for s in flows:
        mat = np.concatenate(flows[s]); tt = np.concatenate(stamps[s])
        order = np.argsort(tt)                       # chronological
        mat = mat[order]
        n = len(mat); a, b = int(n * 0.7), int(n * 0.8)
        slices = {"train": mat[:a], "val": mat[a:b], "test": mat[b:]}
        for split, sl in slices.items():
            if len(sl) == 0:
                continue
            buckets = np.empty((nb[split], FLOW_DIM), dtype=np.float64)
            for bk in range(nb[split]):
                m = int(rng.integers(lo, hi + 1))
                idx = rng.integers(0, len(sl), size=m)
                buckets[bk] = _aggregate_subset(sl[idx])
            out[split][s] = buckets
    print(f"[cic_real:temporal] pool sizes: "
          f"{ {sp: {k: len(v) for k, v in out[sp].items()} for sp in out} }")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(out, f)
    return out


def fit_minmax_scaler(pool: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Fit a global min-max scaler over the union of all severity pools."""
    allv = np.concatenate([v for v in pool.values() if len(v)], axis=0)
    return {"min": allv.min(axis=0), "max": allv.max(axis=0)}


def apply_minmax(vec: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    mn, mx = scaler["min"], scaler["max"]
    rng = np.where(mx - mn > 0, mx - mn, 1.0)
    return np.clip((vec - mn) / rng, 0.0, 1.0).astype(np.float32)
