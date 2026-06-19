"""
Chronological 70 / 10 / 20 split (Section III-E).

Splits are always done by sorting EventClusters on bucket_start (wall-clock
time), never by random index. This prevents temporal data leakage: the model
never sees future events during training or validation.

Target ratios:
  train : 70%
  val   : 10%
  test  : 20%
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.benchmark.aligner import EventCluster


@dataclass
class BenchmarkSplit:
    train: list[EventCluster]
    val:   list[EventCluster]
    test:  list[EventCluster]

    # ------------------------------------------------------------------ #
    # Convenience statistics                                               #
    # ------------------------------------------------------------------ #

    def ratios(self) -> dict[str, float]:
        total = len(self.train) + len(self.val) + len(self.test)
        if total == 0:
            return {"train": 0.0, "val": 0.0, "test": 0.0}
        return {
            "train": len(self.train) / total,
            "val":   len(self.val)   / total,
            "test":  len(self.test)  / total,
        }

    def scenario_composition(self, split: str) -> dict[str, float]:
        """Return fraction of each scenario class in the given split."""
        data = getattr(self, split)
        if not data:
            return {}
        counts: dict[str, int] = {}
        for c in data:
            counts[c.scenario_class] = counts.get(c.scenario_class, 0) + 1
        return {k: v / len(data) for k, v in counts.items()}

    def severity_distribution(self, split: str) -> dict[int, float]:
        data = getattr(self, split)
        if not data:
            return {}
        counts: dict[int, int] = {}
        for c in data:
            counts[c.severity_label] = counts.get(c.severity_label, 0) + 1
        return {k: v / len(data) for k, v in counts.items()}

    def print_summary(self) -> None:
        total = len(self.train) + len(self.val) + len(self.test)
        r = self.ratios()
        print(f"Total clusters : {total}")
        print(f"  train        : {len(self.train):>6}  ({r['train']:.1%})")
        print(f"  val          : {len(self.val):>6}  ({r['val']:.1%})")
        print(f"  test         : {len(self.test):>6}  ({r['test']:.1%})")
        for split in ("train", "val", "test"):
            sc = self.scenario_composition(split)
            print(f"  {split} scenario: "
                  + " ".join(f"{k}={v:.1%}" for k, v in sorted(sc.items())))


def chronological_split(
    clusters: list[EventCluster],
    train_frac: float = 0.70,
    val_frac: float = 0.10,
) -> BenchmarkSplit:
    """
    Sort EventClusters by bucket_start and slice into train/val/test.

    Parameters
    ----------
    clusters    : unsorted list of EventCluster
    train_frac  : fraction for training set  (default 0.70)
    val_frac    : fraction for validation set (default 0.10)
                  test gets the remainder     (default 0.20)
    """
    if not clusters:
        return BenchmarkSplit(train=[], val=[], test=[])

    sorted_clusters = sorted(clusters, key=lambda c: c.bucket_start)
    n = len(sorted_clusters)

    n_train = round(n * train_frac)
    n_val   = round(n * val_frac)
    n_test  = n - n_train - n_val

    # Clamp to avoid empty splits on tiny datasets
    n_train = max(n_train, 1)
    n_val   = max(n_val, 1)
    n_test  = max(n - n_train - n_val, 1)
    # Re-adjust train if clamping caused overshoot
    n_train = n - n_val - n_test

    train = sorted_clusters[:n_train]
    val   = sorted_clusters[n_train : n_train + n_val]
    test  = sorted_clusters[n_train + n_val :]

    return BenchmarkSplit(train=train, val=val, test=test)
