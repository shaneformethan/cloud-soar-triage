"""
PyTorch Dataset and DataLoader utilities for the benchmark.

Loads pre-built EventCluster pickles and wraps them in a Dataset
that returns (iam_tokens, iam_lengths, flow_vec, ti_vec, label, scenario_class).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.benchmark.aligner import EventCluster


class TriageDataset(Dataset):
    """PyTorch Dataset wrapping a list of EventCluster objects."""

    def __init__(self, clusters: list[EventCluster]):
        self.clusters = clusters

    def __len__(self) -> int:
        return len(self.clusters)

    def __getitem__(self, idx: int) -> dict:
        c = self.clusters[idx]
        return {
            "iam_tokens":  torch.tensor(c.iam_seq, dtype=torch.long),      # (64, 5)
            "iam_lengths": torch.tensor(c.iam_seq_len, dtype=torch.long),  # scalar
            "flow_vec":    torch.tensor(c.flow_vec, dtype=torch.float32),  # (12,)
            "ti_vec":      torch.tensor(c.ti_vec, dtype=torch.float32),    # (16,)
            "label":       torch.tensor(c.severity_label, dtype=torch.long),
            "scenario":    c.scenario_class,                                # str
        }


def load_splits(
    processed_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load pre-built benchmark pkl files and return (train, val, test) DataLoaders.
    """
    processed_dir = Path(processed_dir)
    loaders = []
    for split in ("train", "val", "test"):
        path = processed_dir / f"benchmark_{split}.pkl"
        with open(path, "rb") as f:
            clusters: list[EventCluster] = pickle.load(f)
        ds = TriageDataset(clusters)
        shuffle = (split == "train")
        loaders.append(
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=(split == "train"),
            )
        )
    return tuple(loaders)  # type: ignore[return-value]


def collate_batch(batch: list[dict]) -> dict:
    """
    Custom collate_fn that handles the non-tensor 'scenario' field.
    """
    keys = [k for k in batch[0] if k != "scenario"]
    result = {k: torch.stack([item[k] for item in batch]) for k in keys}
    result["scenario"] = [item["scenario"] for item in batch]
    return result
