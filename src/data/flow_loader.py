"""
CSE-CIC-IDS2018 network flow loader.

Reads raw CSV files, aggregates per (src_ip, dst_ip, 5-min bucket),
and produces 12-dimensional feature vectors matching Section III-B spec:

  [total_bytes, total_packets,
   proto_tcp_ratio, proto_udp_ratio, proto_icmp_ratio,
   dst_port_entropy, src_port_entropy,
   flow_iat_mean, flow_iat_std, flow_iat_min, flow_iat_max,
   flow_duration_mean]

All values are min-max normalised to [0, 1] using training-set statistics.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

FLOW_DIM = 12

# Column aliases: maps our canonical names to the variants seen in different
# releases of the CIC-IDS2018 preprocessed CSVs.
_COL_ALIASES: dict[str, list[str]] = {
    "src_ip":        ["Src IP", "Source IP", "src_ip"],
    "dst_ip":        ["Dst IP", "Destination IP", "dst_ip"],
    "src_port":      ["Src Port", "Source Port", "src_port"],
    "dst_port":      ["Dst Port", "Destination Port", "dst_port"],
    "protocol":      ["Protocol", "protocol"],
    "timestamp":     ["Timestamp", "timestamp", "Flow Timestamp"],
    "fwd_bytes":     ["TotLen Fwd Pkts", "Total Length of Fwd Packets",
                      "Total Fwd Packets Length"],
    "bwd_bytes":     ["TotLen Bwd Pkts", "Total Length of Bwd Packets",
                      "Total Bwd Packets Length"],
    "fwd_pkts":      ["Tot Fwd Pkts", "Total Fwd Packets"],
    "bwd_pkts":      ["Tot Bwd Pkts", "Total Backward Packets"],
    "flow_iat_mean": ["Flow IAT Mean"],
    "flow_iat_std":  ["Flow IAT Std"],
    "flow_iat_min":  ["Flow IAT Min"],
    "flow_iat_max":  ["Flow IAT Max"],
    "flow_duration": ["Flow Duration"],
    "label":         ["Label", "label"],
}


def _resolve_col(df: pd.DataFrame, canonical: str) -> str | None:
    for alias in _COL_ALIASES[canonical]:
        if alias in df.columns:
            return alias
    return None


def _port_entropy(port_series: pd.Series) -> float:
    counts = port_series.value_counts(normalize=True).values
    return float(scipy_entropy(counts, base=2)) if len(counts) > 1 else 0.0


class FlowLoader:
    """
    Loads CIC-IDS2018 CSVs → aggregated 12-dim flow vectors per 5-min bucket.

    Usage:
        loader = FlowLoader(data_dir, bucket_minutes=5)
        df = loader.load()          # returns raw per-flow DataFrame
        buckets = loader.aggregate(df)  # returns list of FlowBucket
    """

    def __init__(self, data_dir: str | Path, bucket_minutes: int = 5):
        self.data_dir = Path(data_dir)
        self.bucket_seconds = bucket_minutes * 60
        self._scaler: dict[str, tuple[float, float]] | None = None  # (min, max)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load(self) -> pd.DataFrame:
        """Read all CSVs in data_dir, return unified DataFrame."""
        csv_files = sorted(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_dir}. "
                "Download CIC-IDS2018 first or use dummy mode."
            )
        frames = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                df = self._rename_columns(df)
                frames.append(df)
            except Exception as e:
                print(f"Warning: skipping {f.name}: {e}")
        return pd.concat(frames, ignore_index=True)

    def aggregate(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = True,
    ) -> list[dict]:
        """
        Aggregate per-flow records into 5-min buckets.

        Returns list of dicts, each representing one EventCluster's flow side:
          {
            "bucket_key": str,          # "{src_ip}_{dst_ip}_{bucket_ts}"
            "bucket_start": pd.Timestamp,
            "src_ip": str,
            "dst_ip": str,
            "flow_vec": np.ndarray,     # shape (12,), dtype float32
            "attack_tactic": str|None,  # via cic_label_to_tactic
            "raw_label": str,
          }
        """
        from src.benchmark.attack_stage import cic_label_to_tactic

        df = df.copy()
        df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["_ts"])
        df["_bucket"] = (
            df["_ts"].astype("int64") // (self.bucket_seconds * 1_000_000_000)
        ) * self.bucket_seconds * 1_000_000_000
        df["_bucket"] = pd.to_datetime(df["_bucket"])

        # Fill missing IP columns with placeholders
        for col in ["src_ip", "dst_ip"]:
            if col not in df.columns:
                df[col] = "0.0.0.0"

        groups = df.groupby(["src_ip", "dst_ip", "_bucket"])

        raw_features: list[tuple[str, pd.Timestamp, str, str, np.ndarray, str]] = []
        for (src_ip, dst_ip, bucket_ts), grp in groups:
            vec = self._extract_12dim(grp)
            # Majority-vote label within bucket
            raw_label = grp["label"].mode()[0] if "label" in grp.columns else "Benign"
            key = f"{src_ip}_{dst_ip}_{int(bucket_ts.timestamp())}"
            raw_features.append((key, bucket_ts, str(src_ip), str(dst_ip),
                                  vec, str(raw_label)))

        if not raw_features:
            return []

        vecs = np.stack([r[4] for r in raw_features], axis=0)  # (N, 12)
        if fit_scaler:
            self._fit_scaler(vecs)
        vecs_norm = self._normalise(vecs)

        results = []
        for i, (key, ts, src, dst, _, raw_label) in enumerate(raw_features):
            results.append({
                "bucket_key":    key,
                "bucket_start":  ts,
                "src_ip":        src,
                "dst_ip":        dst,
                "flow_vec":      vecs_norm[i].astype(np.float32),
                "attack_tactic": cic_label_to_tactic(raw_label),
                "raw_label":     raw_label,
            })
        return results

    def save_scaler(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._scaler, f)

    def load_scaler(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self._scaler = pickle.load(f)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        for canonical, aliases in _COL_ALIASES.items():
            for alias in aliases:
                if alias in df.columns and canonical not in df.columns:
                    rename_map[alias] = canonical
                    break
        return df.rename(columns=rename_map)

    def _extract_12dim(self, grp: pd.DataFrame) -> np.ndarray:
        """
        Extract raw (un-normalised) 12-dim vector from a bucket group.

        Dimensions (Section III-B):
          0  total_bytes
          1  total_packets
          2  proto_tcp_ratio
          3  proto_udp_ratio
          4  proto_icmp_ratio
          5  dst_port_entropy  (bits)
          6  src_port_entropy  (bits)
          7  flow_iat_mean
          8  flow_iat_std
          9  flow_iat_min
          10 flow_iat_max
          11 flow_duration_mean
        """
        vec = np.zeros(FLOW_DIM, dtype=np.float64)

        fwd = grp["fwd_bytes"].clip(lower=0) if "fwd_bytes" in grp else pd.Series([0])
        bwd = grp["bwd_bytes"].clip(lower=0) if "bwd_bytes" in grp else pd.Series([0])
        fpk = grp["fwd_pkts"].clip(lower=0) if "fwd_pkts" in grp else pd.Series([0])
        bpk = grp["bwd_pkts"].clip(lower=0) if "bwd_pkts" in grp else pd.Series([0])

        vec[0] = float((fwd.sum() + bwd.sum()))
        vec[1] = float((fpk.sum() + bpk.sum()))

        if "protocol" in grp.columns:
            total = max(len(grp), 1)
            vec[2] = (grp["protocol"] == 6).sum() / total   # TCP
            vec[3] = (grp["protocol"] == 17).sum() / total  # UDP
            vec[4] = (grp["protocol"] == 1).sum() / total   # ICMP

        if "dst_port" in grp.columns:
            vec[5] = _port_entropy(grp["dst_port"])
        if "src_port" in grp.columns:
            vec[6] = _port_entropy(grp["src_port"])

        for i, col in enumerate(
            ["flow_iat_mean", "flow_iat_std", "flow_iat_min",
             "flow_iat_max", "flow_duration"]
        ):
            if col in grp.columns:
                vec[7 + i] = float(grp[col].mean())

        return vec

    def _fit_scaler(self, vecs: np.ndarray) -> None:
        mins = vecs.min(axis=0)
        maxs = vecs.max(axis=0)
        self._scaler = {"min": mins, "max": maxs}

    def _normalise(self, vecs: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            return vecs
        mn, mx = self._scaler["min"], self._scaler["max"]
        rng = np.where(mx - mn > 0, mx - mn, 1.0)
        return np.clip((vecs - mn) / rng, 0.0, 1.0)
