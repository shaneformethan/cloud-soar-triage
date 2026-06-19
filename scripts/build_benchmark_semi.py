"""
Semi-synthetic benchmark builder (draft.tex Section III-E, faithful version).

Assembles EventClusters from:
  * Flow   : REAL CSE-CIC-IDS2018 aggregated 12-dim vectors, label-mapped to
             severity (src/benchmark/severity_mapping.py).
  * IAM    : severity-conditional synthetic sessions (src/data/synth_modalities).
  * TI     : severity-conditional synthetic 16-dim vectors.

Fusion-advantage design
-----------------------
Each cluster has a single ground-truth severity s. In a scenario, only the
*active* modalities carry s (noisily); inactive modalities carry the benign
(severity-0) distribution. Benign clusters have no active modality (s = 0).
Because an attack's severity surfaces only on its active modalities, a model
must read across all three streams and take the strongest signal -- so
cross-modal fusion provably outperforms any single-modality view, which is the
paper's central claim. Scenario mix follows the draft: single 30% / dual 40% /
tri 30% among attacks, plus a benign fraction for the informational class.

Usage:
    python scripts/build_benchmark_semi.py                 # default 9000 samples
    python scripts/build_benchmark_semi.py --n 12000 --seed 42
    python scripts/build_benchmark_semi.py --rebuild-pool  # re-parse CIC CSVs
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cic_real import (
    build_flow_pool, build_flow_pool_temporal, fit_minmax_scaler, apply_minmax,
)
from src.data.synth_modalities import (
    generate_iam_seq, generate_ti_vec, benign_iam_seq, benign_ti_vec,
)
from src.benchmark.aligner import EventCluster, pad_iam_seq
from src.benchmark.splitter import chronological_split
from scripts.build_benchmark import save_splits, write_report

_ALL_MODS = ["flow", "iam", "ti"]

# Each severity tier is realised by several MITRE tactics (the latent attack
# type). The synthetic IAM modality is conditioned on the *tactic*; the severity
# label is computed DOWNSTREAM as _TACTIC_TO_SEV[tactic] -- the severity never
# enters generation, preserving anti-circularity. The groupings follow the
# draft's CIC-label severity scheme: web/recon -> medium, credential/botnet/C2
# -> high, DoS/exfiltration/impact -> critical.
_TACTICS_BY_SEVERITY: dict[int, list[str]] = {
    1: ["InitialAccess", "Discovery"],
    2: ["CredentialAccess", "CommandAndControl", "Persistence"],
    3: ["Impact", "Exfiltration", "LateralMovement"],
}
_TACTIC_TO_SEV: dict[str, int] = {
    t: s for s, ts in _TACTICS_BY_SEVERITY.items() for t in ts
}
# Real IP reputation is binary (malicious/clean), so TI is never allowed to be
# the *sole* active modality of a single-modality scenario -- the severity tier
# must come from flow or IAM. TI only ever corroborates in dual/tri scenarios.
_SINGLE_MODS = ["flow", "iam"]


def _pick_active(scenario: str, rng: np.random.Generator) -> set[str]:
    if scenario == "tri":
        return set(_ALL_MODS)
    if scenario == "dual":
        return set(rng.choice(_ALL_MODS, size=2, replace=False).tolist())
    return {_SINGLE_MODS[int(rng.integers(0, len(_SINGLE_MODS)))]}  # single


def _partition_pool(pool, rng, train=0.70, val=0.10):
    """
    Split each severity's flow-vector pool into DISJOINT train/val/test subsets
    so no flow vector is shared across splits (prevents flow-modality leakage).
    """
    out = {"train": {}, "val": {}, "test": {}}
    for sev, arr in pool.items():
        idx = rng.permutation(len(arr))
        n_tr = int(round(len(arr) * train))
        n_va = int(round(len(arr) * val))
        out["train"][sev] = arr[idx[:n_tr]]
        out["val"][sev] = arr[idx[n_tr:n_tr + n_va]]
        out["test"][sev] = arr[idx[n_tr + n_va:]]
    return out


def _sample_flow(arr, scaler, rng: np.random.Generator) -> np.ndarray:
    """Draw and normalise one real flow vector from a (split-specific) array."""
    raw = arr[int(rng.integers(0, len(arr)))]
    return apply_minmax(raw, scaler)


def build(args) -> None:
    rng = np.random.default_rng(args.seed)
    proc_dir = Path(args.processed_dir)

    if args.temporal_split:
        # True per-class temporal split of the real flows (train=earliest 70%).
        pool_split = build_flow_pool_temporal(
            data_dir=args.cic_dir,
            cache_path=proc_dir / "cic_flow_pool_temporal.pkl",
            rebuild=args.rebuild_pool,
            seed=args.seed,
        )
        flat: dict[int, list] = {}
        for sp in pool_split:
            for s, arr in pool_split[sp].items():
                flat.setdefault(s, []).append(arr)
        scaler = fit_minmax_scaler({s: np.concatenate(v) for s, v in flat.items()})
        print("[semi] Using TEMPORAL (chronological) flow split.")
    else:
        pool = build_flow_pool(
            data_dir=args.cic_dir,
            bucket_minutes=5,
            cache_path=proc_dir / "cic_flow_pool.pkl",
            rebuild=args.rebuild_pool,
        )
        scaler = fit_minmax_scaler(pool)
        # Partition the flow pool into disjoint train/val/test subsets (random).
        pool_split = _partition_pool(pool, rng)

    # Real threat-intelligence pool (AbuseIPDB/OTX enriched, split-disjoint).
    ti_pool = None
    if args.real_ti:
        ti_path = proc_dir / "ti_pool.pkl"
        if not ti_path.exists():
            raise SystemExit(
                f"--real-ti set but {ti_path} missing. "
                "Run: python scripts/fetch_ti_pool.py")
        with open(ti_path, "rb") as f:
            ti_pool = pickle.load(f)
        print("[semi] Using REAL TI pool (AbuseIPDB + OTX).")
    else:
        print("[semi] Using synthetic severity-conditional TI.")

    n = args.n
    benign_frac = args.benign_frac
    scenario_p = np.array([0.30, 0.40, 0.30])  # single / dual / tri (draft)
    scenarios = ["single", "dual", "tri"]

    t0 = datetime(2018, 2, 14, 8, 0, 0, tzinfo=timezone.utc)
    # (cluster, sev, ti_active, flow_sev, ti_corrupt)
    specs: list[tuple[EventCluster, int, bool, int, bool]] = []

    for i in range(n):
        is_benign = rng.random() < benign_frac
        if is_benign:
            # Benign cluster: no attack tactic, all modalities benign.
            tactic = None
            sev = 0
            scenario = "benign"
            active: set[str] = set()
        else:
            # Draw a latent attack tactic; severity is the DOWNSTREAM value.
            tier = int(rng.integers(1, 4))                # balance over 1/2/3
            tactic = _TACTICS_BY_SEVERITY[tier][
                int(rng.integers(0, len(_TACTICS_BY_SEVERITY[tier])))]
            sev = _TACTIC_TO_SEV[tactic]                  # severity FROM tactic
            scenario = scenarios[int(rng.choice(3, p=scenario_p))]
            active = _pick_active(scenario, rng)

        # ── Per-instance modality reliability (only tri, so the consensus of
        # the other two modalities still recovers the true label). With prob
        # corrupt_p one randomly chosen active modality is given a WRONG-severity
        # signal; the label stays the true severity. This rewards input-dependent
        # fusion (attend to the consistent majority) over fixed averaging. ──
        corrupt_mod = None
        corrupt_tactic = None
        corrupt_flow_sev = None
        if len(active) == 3 and rng.random() < args.corrupt_p:
            corrupt_mod = ["flow", "iam", "ti"][int(rng.integers(0, 3))]
            wrong = [x for x in (0, 1, 2, 3) if x != sev]
            corrupt_sev = int(rng.choice(wrong))
            if corrupt_sev == 0:
                corrupt_tactic = None
            else:
                opts = _TACTICS_BY_SEVERITY[corrupt_sev]
                corrupt_tactic = opts[int(rng.integers(0, len(opts)))]
            corrupt_flow_sev = corrupt_sev

        # IAM modality (tactic-conditioned synthetic; never sees severity)
        if corrupt_mod == "iam":
            raw_iam = generate_iam_seq(corrupt_tactic, rng, noise=args.iam_noise)
        elif "iam" in active:
            raw_iam = generate_iam_seq(tactic, rng, noise=args.iam_noise)
        else:
            raw_iam = benign_iam_seq(rng, noise=args.iam_noise)
        iam_padded, iam_len = pad_iam_seq(raw_iam)

        ti_active = "ti" in active
        flow_sev = sev if "flow" in active else 0
        if corrupt_mod == "flow":
            flow_sev = corrupt_flow_sev
        # When TI is the corrupted modality it shows clean despite the attack.
        ti_corrupt = (corrupt_mod == "ti")
        # Flow and TI are filled in the split-aware pass below (disjoint pools).
        c = EventCluster(
            bucket_key=f"semi_{i}",
            bucket_start=t0,
            src_ip="0.0.0.0",
            dst_ip="0.0.0.0",
            flow_vec=np.zeros(12, dtype=np.float32),
            iam_seq=iam_padded,
            iam_seq_len=iam_len,
            ti_vec=np.zeros(16, dtype=np.float32),
            severity_label=sev,
            scenario_class=scenario if scenario != "benign" else "single",
            active_modalities=list(active),
        )
        specs.append((c, sev, ti_active, flow_sev, ti_corrupt))

    # Shuffle, then assign strictly increasing timestamps so the downstream
    # chronological split reproduces this exact ordering. Knowing each cluster's
    # split position lets us draw real TI vectors from the matching disjoint
    # split pool (no IP leakage across train/val/test).
    rng.shuffle(specs)
    n_train = round(n * 0.70)
    n_val = round(n * 0.10)
    for i, (c, sev, ti_active, flow_sev, ti_corrupt) in enumerate(specs):
        c.bucket_start = t0 + timedelta(minutes=5 * i)
        split = ("train" if i < n_train
                 else "val" if i < n_train + n_val else "test")

        # Flow: draw from this split's disjoint flow pool (flow_sev already
        # carries any corruption).
        c.flow_vec = _sample_flow(
            pool_split[split][flow_sev], scaler, rng).astype(np.float32)

        # TI: a corrupted-TI cluster shows clean despite the attack.
        ti_malicious = ti_active and sev > 0 and not ti_corrupt
        if ti_pool is not None:
            key = sev if ti_malicious else 0
            lst = ti_pool[split].get(key) or ti_pool[split].get(0) or []
            if lst:
                c.ti_vec = lst[int(rng.integers(0, len(lst)))].astype(np.float32)
        else:
            c.ti_vec = generate_ti_vec(ti_malicious, rng,
                                       noise=args.ti_noise).astype(np.float32)

    clusters = [c for c, _, _, _, _ in specs]
    splits = chronological_split(clusters, train_frac=0.70, val_frac=0.10)
    splits.print_summary()
    save_splits(splits, proc_dir)
    write_report(splits, proc_dir)
    print("\nDone (semi-synthetic, draft-faithful benchmark).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build semi-synthetic benchmark")
    p.add_argument("--n", type=int, default=9000, help="total clusters")
    p.add_argument("--benign_frac", type=float, default=0.25,
                   help="fraction of informational (benign) clusters")
    p.add_argument("--iam_noise", type=float, default=0.45,
                   help="fraction of IAM events drawn off-tactic (higher=harder)")
    p.add_argument("--ti_noise", type=float, default=0.40)
    p.add_argument("--corrupt_p", type=float, default=0.4,
                   help="prob. a tri-modal incident has one modality giving a "
                        "wrong-severity (false) reading; tests selective fusion")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cic_dir", default="data/raw/cic_ids2018")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--rebuild-pool", action="store_true",
                   help="re-parse the CIC CSVs instead of using the cache")
    p.add_argument("--temporal-split", dest="temporal_split", action="store_true",
                   help="split real flows chronologically per class (vs i.i.d.)")
    p.add_argument("--real-ti", dest="real_ti", action="store_true", default=True,
                   help="use the real AbuseIPDB/OTX TI pool (default)")
    p.add_argument("--no-real-ti", dest="real_ti", action="store_false",
                   help="use synthetic severity-conditional TI instead")
    return p.parse_args()


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)  # run from project root
    build(parse_args())
