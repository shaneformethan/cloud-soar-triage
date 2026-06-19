"""
Dummy data generators for all three modalities.

Used when mode=dummy so the full pipeline can be exercised without
downloading any dataset or calling any API.

All generators are deterministic given a seed, and all output dimensions
match the Section III-B spec exactly:
  - IAM  : (seq_len, 5)  with seq_len <= 64
  - Flow : (12,)
  - TI   : (16,)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from src.benchmark.attack_stage import (
    ALL_ATTACK_TACTICS,
    TACTIC_TO_SEVERITY_IDX,
    combine_tactic_severity,
)
from src.utils.circularity_guard import IamGenerationContext

IAM_EVENT_DIM = 5
IAM_MAX_SEQ = 64
FLOW_DIM = 12
TI_DIM = 16


# ------------------------------------------------------------------ #
# Per-modality generators                                             #
# ------------------------------------------------------------------ #

def dummy_flow_vec(rng: np.random.Generator, anomalous: bool) -> np.ndarray:
    """Return shape (12,) float32 flow vector."""
    base = rng.random(FLOW_DIM).astype(np.float32)
    if anomalous:
        # Push 3 random dims toward 1 to simulate attack traffic
        idxs = rng.choice(FLOW_DIM, size=3, replace=False)
        base[idxs] = rng.uniform(0.7, 1.0, size=3).astype(np.float32)
    return base


def dummy_iam_seq(
    ctx: IamGenerationContext,
) -> np.ndarray:
    """
    Return shape (target_event_count, 5) int32 IAM sequence.
    Uses ONLY information in IamGenerationContext (tactic + session metadata).
    """
    rng = np.random.default_rng(ctx.rng_seed)
    n = min(max(ctx.target_event_count, 1), IAM_MAX_SEQ)

    # Per-column vocab sizes MUST match src.models.encoders.IAMEncoder.IAM_VOCAB_SIZES
    # = [256, 10, 6, 5, 2] (action_type, resource_type, principal_role,
    # src_ip_region, baseline_flag). Each nn.Embedding has vocab_size + 1
    # entries (padding_idx=0), so valid token values are 0..vocab_size.
    COL_VOCAB_SIZES = [256, 10, 6, 5]  # baseline_flag (col 4) handled separately
    cols = [
        rng.integers(0, v, size=n, dtype=np.int32) for v in COL_VOCAB_SIZES
    ]
    seq = np.stack(cols, axis=1)  # (n, 4)

    # is_in_baseline: 1 if action index appeared earlier in sequence
    history: set[int] = set()
    baseline_flags = np.zeros((n, 1), dtype=np.int32)
    for i in range(n):
        aidx = int(seq[i, 0])
        baseline_flags[i, 0] = int(aidx in history)
        history.add(aidx)

    seq = np.concatenate([seq, baseline_flags], axis=1)  # (n, 5)

    if ctx.attack_tactic is not None:
        # Anomalous sessions: concentrate actions in a smaller range to
        # simulate repeated suspicious API calls
        seq[:, 0] = rng.integers(0, 8, size=n, dtype=np.int32)

    return seq


def dummy_ti_vec(rng: np.random.Generator, anomalous: bool) -> np.ndarray:
    """Return shape (16,) float32 TI vector."""
    v = rng.random(TI_DIM).astype(np.float32)
    # Binarise binary fields
    binary_idxs = [1, 5, 7, 9, 11, 12, 13]
    threshold = 0.3 if anomalous else 0.8
    for i in binary_idxs:
        v[i] = float(v[i] > threshold)
    if anomalous:
        v[0] = rng.uniform(0.6, 1.0)   # high reputation score
        v[2] = rng.uniform(0.6, 1.0)   # high abuse confidence
    return v.astype(np.float32)


# ------------------------------------------------------------------ #
# Full benchmark generator                                            #
# ------------------------------------------------------------------ #

def generate_dummy_benchmark(
    n_buckets: int,
    scenario_ratios: dict[str, float],
    seed: int = 42,
) -> list[dict]:
    """
    Generate n_buckets synthetic EventCluster dicts matching the full
    pipeline output schema.

    Scenario composition follows the ratios from config (benign/single/dual/tri).
    Chronological ordering is preserved by incrementing bucket timestamps.

    Returns
    -------
    List of dicts with keys:
      bucket_key, bucket_start, src_ip, dst_ip,
      flow_vec (12,), iam_seq (<=64,5), iam_seq_len,
      ti_vec (16,), severity_label (int 0-3),
      scenario_class (str), active_modalities (list[str])
    """
    rng = np.random.default_rng(seed)
    t0 = datetime(2018, 2, 14, 8, 0, 0, tzinfo=timezone.utc)
    results = []

    # Pre-compute scenario class assignments to hit exact ratios
    scenario_classes = _sample_scenario_classes(n_buckets, scenario_ratios, rng)

    for i, sc in enumerate(scenario_classes):
        bucket_start = t0 + timedelta(minutes=5 * i)
        src_ip = f"10.0.{rng.integers(0,256)}.{rng.integers(1,255)}"
        dst_ip = f"172.16.{rng.integers(0,256)}.{rng.integers(1,255)}"
        bucket_key = f"{src_ip}_{dst_ip}_{int(bucket_start.timestamp())}"

        active = _sample_active_modalities(sc, rng)

        # Choose a tactic for the bucket (for active modalities). "benign"
        # buckets have no active modalities, so no tactic is ever attached
        # to any modality and severity_label resolves to 0 ("informational")
        # below.
        tactic = ALL_ATTACK_TACTICS[int(rng.integers(0, len(ALL_ATTACK_TACTICS)))]

        flow_tactic = tactic if "flow" in active else None
        iam_tactic = tactic if "iam" in active else None
        ti_anomalous = "ti" in active

        # --- Flow vector (label NOT passed to IAM generator below) ---
        flow_vec = dummy_flow_vec(rng, anomalous="flow" in active)

        # --- IAM sequence (receives tactic, NOT severity/flow features) ---
        iam_ctx = IamGenerationContext(
            attack_tactic=iam_tactic,
            session_src_ip=src_ip,
            session_start_ts=bucket_start,
            session_duration_s=float(rng.uniform(10, 300)),
            target_event_count=int(rng.integers(4, IAM_MAX_SEQ + 1)),
            rng_seed=seed * 10000 + i,
        )
        iam_seq = dummy_iam_seq(iam_ctx)

        # --- TI vector ---
        ti_vec = dummy_ti_vec(rng, anomalous=ti_anomalous)

        # --- Severity: derived AFTER all modalities are built ---
        # Uses per-modality tactics, NOT flow feature vector
        tactics_for_label = [flow_tactic, iam_tactic,
                              tactic if ti_anomalous else None]
        severity_label = combine_tactic_severity(tactics_for_label)

        results.append({
            "bucket_key":       bucket_key,
            "bucket_start":     bucket_start,
            "src_ip":           src_ip,
            "dst_ip":           dst_ip,
            "flow_vec":         flow_vec,
            "iam_seq":          iam_seq,
            "iam_seq_len":      iam_seq.shape[0],
            "ti_vec":           ti_vec,
            "severity_label":   severity_label,
            "scenario_class":   sc,
            "active_modalities": list(active),
        })

    return results


# ------------------------------------------------------------------ #
# Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _sample_scenario_classes(
    n: int,
    ratios: dict[str, float],
    rng: np.random.Generator,
) -> list[str]:
    """
    Interleave scenario class labels uniformly across the timeline so that
    every chronological segment (train/val/test) has the same target ratio.

    Strategy: build one tile of length = LCM of ratio denominators, then
    repeat and trim. This avoids block-assignment which would cause val/test
    to contain only the tail class after a chronological split.
    """
    # Build a single period tile proportional to target ratios
    tile: list[str] = []
    for sc, ratio in ratios.items():
        count = max(1, round(10 * ratio))
        tile.extend([sc] * count)

    # Repeat tile to cover n, with a small per-tile shuffle for variety
    full: list[str] = []
    while len(full) < n:
        shuffled_tile = tile.copy()
        rng.shuffle(shuffled_tile)
        full.extend(shuffled_tile)

    return full[:n]


def _sample_active_modalities(
    scenario_class: str, rng: np.random.Generator
) -> set[str]:
    all_mods = ["flow", "iam", "ti"]
    if scenario_class == "benign":
        # No modality carries an attack signal -> severity 0 ("informational")
        return set()
    if scenario_class == "tri":
        return set(all_mods)
    if scenario_class == "dual":
        chosen = rng.choice(all_mods, size=2, replace=False)
        return set(chosen.tolist())
    # single
    return {all_mods[int(rng.integers(0, 3))]}
