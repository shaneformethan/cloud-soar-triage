"""
Benchmark builder entry point.

Usage:
    # Dummy mode (no downloads needed):
    python scripts/build_benchmark.py --config config/dummy.yaml

    # Real mode (after downloading datasets):
    python scripts/build_benchmark.py --config config/default.yaml

Output:
    data/processed/benchmark_train.pkl
    data/processed/benchmark_val.pkl
    data/processed/benchmark_test.pkl
    data/processed/build_report.txt
"""

import argparse
import pickle
import sys
import os
import time
from pathlib import Path

import yaml

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env if present
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dummy(cfg: dict, out_dir: Path) -> None:
    from src.data.dummy import generate_dummy_benchmark
    from src.benchmark.aligner import EventCluster, pad_iam_seq
    from src.benchmark.splitter import chronological_split

    n_buckets = cfg.get("dummy", {}).get("n_buckets", 2000)
    seed = cfg.get("random_seed", 42)
    ratios = cfg["benchmark"]["scenario"]

    print(f"[dummy] Generating {n_buckets} synthetic event clusters...")
    t0 = time.time()
    raw = generate_dummy_benchmark(
        n_buckets=n_buckets,
        scenario_ratios={
            "benign": ratios.get("benign_ratio", 0.0),
            "single": ratios["single_ratio"],
            "dual":   ratios["dual_ratio"],
            "tri":    ratios["tri_ratio"],
        },
        seed=seed,
    )

    clusters = []
    for c in raw:
        padded, seq_len = pad_iam_seq(c["iam_seq"])
        clusters.append(EventCluster(
            bucket_key=c["bucket_key"],
            bucket_start=c["bucket_start"],
            src_ip=c["src_ip"],
            dst_ip=c["dst_ip"],
            flow_vec=c["flow_vec"],
            iam_seq=padded,
            iam_seq_len=seq_len,
            ti_vec=c["ti_vec"],
            severity_label=c["severity_label"],
            scenario_class=c["scenario_class"],
            active_modalities=c["active_modalities"],
        ))

    split_cfg = cfg["benchmark"]["split"]
    splits = chronological_split(
        clusters,
        train_frac=split_cfg["train"],
        val_frac=split_cfg["val"],
    )
    elapsed = time.time() - t0
    print(f"[dummy] Built in {elapsed:.1f}s")
    return splits


def build_real(cfg: dict, out_dir: Path) -> None:
    from src.data.flow_loader import FlowLoader
    from src.data.iam_loader import IamDistributions
    from src.data.threat_intel import ThreatIntelEnricher
    from src.benchmark.scenario_builder import ScenarioBuilder
    from src.benchmark.aligner import BucketAligner
    from src.benchmark.splitter import chronological_split

    seed = cfg.get("random_seed", 42)
    ratios = cfg["benchmark"]["scenario"]
    split_cfg = cfg["benchmark"]["split"]
    raw_dir = Path(cfg["data"]["raw_dir"])
    proc_dir = Path(cfg["data"]["processed_dir"])

    # --- Flow ---
    print("[real] Loading CIC-IDS2018 flow records...")
    flow_loader = FlowLoader(
        data_dir=raw_dir / "cic_ids2018",
        bucket_minutes=cfg["benchmark"]["bucket_minutes"],
    )
    flow_df = flow_loader.load()
    flow_buckets = flow_loader.aggregate(flow_df, fit_scaler=True)
    flow_loader.save_scaler(proc_dir / "flow_scaler.pkl")
    print(f"[real] {len(flow_buckets)} flow buckets loaded.")

    # --- IAM distributions ---
    dist_cache = Path(cfg["data"]["cloudtrail"]["distributions_cache"])
    iam_dist = IamDistributions()
    if dist_cache.exists():
        print("[real] Loading cached IAM distributions...")
        iam_dist.load(dist_cache)
    else:
        cloudtrail_dir = raw_dir / "cloudtrail"
        json_files = list(cloudtrail_dir.rglob("*.json")) if cloudtrail_dir.exists() else []
        if json_files:
            print(f"[real] Analysing CloudTrail corpus ({len(json_files)} files)...")
            iam_dist.fit_from_cloudtrail(cloudtrail_dir)
        else:
            print("[real] CloudTrail folder empty -- using MITRE ATT&CK seed distributions.")
            iam_dist.fit_from_seeds()
        dist_cache.parent.mkdir(parents=True, exist_ok=True)
        iam_dist.save(dist_cache)

    # --- Threat intel ---
    ti_cfg = cfg["data"]["threat_intel"]
    ti_enricher = ThreatIntelEnricher(
        abuseipdb_key=ti_cfg.get("abuseipdb_key", ""),
        otx_key=ti_cfg.get("otx_key", ""),
        cache_dir=ti_cfg["cache_dir"],
        cache_ttl_hours=ti_cfg.get("cache_ttl_hours", 24),
    )

    # --- Scenario builder ---
    builder = ScenarioBuilder(seed=seed)
    assignments = builder.assign(
        flow_buckets,
        ratios={
            "single": ratios["single_ratio"],
            "dual":   ratios["dual_ratio"],
            "tri":    ratios["tri_ratio"],
        },
    )

    # --- Aligner ---
    print("[real] Aligning modalities into EventClusters...")
    aligner = BucketAligner(iam_dist, ti_enricher, global_seed=seed)
    clusters = aligner.align(flow_buckets, assignments)
    print(f"[real] {len(clusters)} EventClusters built.")

    # --- Chronological split ---
    splits = chronological_split(
        clusters,
        train_frac=split_cfg["train"],
        val_frac=split_cfg["val"],
    )
    return splits


def save_splits(splits, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        path = out_dir / f"benchmark_{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(getattr(splits, name), f)
        print(f"  Saved {len(getattr(splits, name))} clusters -> {path}")


def write_report(splits, out_dir: Path) -> None:
    report_path = out_dir / "build_report.txt"
    lines = ["=== Benchmark Build Report ===\n"]
    lines.append(f"Total clusters : "
                 f"{len(splits.train)+len(splits.val)+len(splits.test)}\n")
    r = splits.ratios()
    for s in ("train", "val", "test"):
        comp = splits.scenario_composition(s)
        sev  = splits.severity_distribution(s)
        lines.append(
            f"\n--- {s.upper()} ({r[s]:.1%}, n={len(getattr(splits, s))}) ---\n"
        )
        lines.append(
            "  Scenario: "
            + " ".join(f"{k}={v:.1%}" for k, v in sorted(comp.items()))
            + "\n"
        )
        sev_names = ["info", "med", "high", "crit"]
        lines.append(
            "  Severity: "
            + " ".join(f"{sev_names[k]}={v:.1%}"
                       for k, v in sorted(sev.items()))
            + "\n"
        )
    with open(report_path, "w") as f:
        f.writelines(lines)
    print(f"\n  Report -> {report_path}")
    print("".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/dummy.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["data"]["processed_dir"])
    mode = cfg.get("mode", "dummy")

    print(f"=== Building benchmark (mode={mode}) ===")
    if mode == "dummy":
        splits = build_dummy(cfg, out_dir)
    else:
        splits = build_real(cfg, out_dir)

    splits.print_summary()
    save_splits(splits, out_dir)
    write_report(splits, out_dir)
    print("\nDone.")
