"""
Test (c): komposisi scenario class & chronological split sesuai target.

Verifies:
  - Split ratios ≈ 70/10/20 (±2pp tolerance)
  - Scenario class composition ≈ 30/40/30 (±3pp tolerance)
  - Chronological ordering preserved: max(train) < min(val) < min(test)
  - No data leakage across splits (bucket_keys are disjoint)
"""

import numpy as np
import pytest
from datetime import datetime, timezone

from src.data.dummy import generate_dummy_benchmark
from src.benchmark.aligner import EventCluster, pad_iam_seq
from src.benchmark.splitter import chronological_split, BenchmarkSplit

SCENARIO_RATIOS = {"single": 0.30, "dual": 0.40, "tri": 0.30}
N_BUCKETS = 1000
SEED = 42

# Tolerance for ratio checks (percentage points)
SPLIT_TOL = 0.02
# val/test splits are smaller (~100–200 samples), so per-class variance
# is naturally higher. 5pp tolerance is sufficient to detect real breakage.
SCENARIO_TOL = 0.05


# ------------------------------------------------------------------ #
# Fixture: build benchmark once for all tests                         #
# ------------------------------------------------------------------ #

def _build_split() -> BenchmarkSplit:
    raw = generate_dummy_benchmark(
        n_buckets=N_BUCKETS, scenario_ratios=SCENARIO_RATIOS, seed=SEED
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
    return chronological_split(clusters, train_frac=0.70, val_frac=0.10)


_SPLIT = _build_split()


# ------------------------------------------------------------------ #
# Split ratio tests                                                    #
# ------------------------------------------------------------------ #

def test_split_total_count():
    total = len(_SPLIT.train) + len(_SPLIT.val) + len(_SPLIT.test)
    assert total == N_BUCKETS, f"Total clusters {total} != {N_BUCKETS}"

def test_train_ratio():
    r = _SPLIT.ratios()["train"]
    assert abs(r - 0.70) <= SPLIT_TOL, f"Train ratio {r:.3f} not ≈ 0.70"

def test_val_ratio():
    r = _SPLIT.ratios()["val"]
    assert abs(r - 0.10) <= SPLIT_TOL, f"Val ratio {r:.3f} not ≈ 0.10"

def test_test_ratio():
    r = _SPLIT.ratios()["test"]
    assert abs(r - 0.20) <= SPLIT_TOL, f"Test ratio {r:.3f} not ≈ 0.20"

def test_no_empty_splits():
    assert len(_SPLIT.train) > 0, "Train split is empty"
    assert len(_SPLIT.val)   > 0, "Val split is empty"
    assert len(_SPLIT.test)  > 0, "Test split is empty"


# ------------------------------------------------------------------ #
# Chronological ordering tests                                         #
# ------------------------------------------------------------------ #

def test_train_is_chronologically_before_val():
    max_train = max(c.bucket_start for c in _SPLIT.train)
    min_val   = min(c.bucket_start for c in _SPLIT.val)
    assert max_train <= min_val, (
        f"Temporal leakage: latest train bucket ({max_train}) "
        f"> earliest val bucket ({min_val})"
    )

def test_val_is_chronologically_before_test():
    max_val  = max(c.bucket_start for c in _SPLIT.val)
    min_test = min(c.bucket_start for c in _SPLIT.test)
    assert max_val <= min_test, (
        f"Temporal leakage: latest val bucket ({max_val}) "
        f"> earliest test bucket ({min_test})"
    )

def test_train_internally_ordered():
    ts = [c.bucket_start for c in _SPLIT.train]
    assert ts == sorted(ts), "Train split is not chronologically ordered"

def test_val_internally_ordered():
    ts = [c.bucket_start for c in _SPLIT.val]
    assert ts == sorted(ts), "Val split is not chronologically ordered"

def test_test_internally_ordered():
    ts = [c.bucket_start for c in _SPLIT.test]
    assert ts == sorted(ts), "Test split is not chronologically ordered"


# ------------------------------------------------------------------ #
# No data leakage across splits                                        #
# ------------------------------------------------------------------ #

def test_bucket_keys_disjoint():
    train_keys = {c.bucket_key for c in _SPLIT.train}
    val_keys   = {c.bucket_key for c in _SPLIT.val}
    test_keys  = {c.bucket_key for c in _SPLIT.test}
    assert not train_keys & val_keys,  "Duplicate keys between train and val"
    assert not train_keys & test_keys, "Duplicate keys between train and test"
    assert not val_keys   & test_keys, "Duplicate keys between val and test"


# ------------------------------------------------------------------ #
# Scenario class composition tests                                     #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_scenario_composition_single(split_name):
    comp = _SPLIT.scenario_composition(split_name)
    ratio = comp.get("single", 0.0)
    assert abs(ratio - 0.30) <= SCENARIO_TOL, (
        f"{split_name} single ratio {ratio:.3f} not ≈ 0.30"
    )

@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_scenario_composition_dual(split_name):
    comp = _SPLIT.scenario_composition(split_name)
    ratio = comp.get("dual", 0.0)
    assert abs(ratio - 0.40) <= SCENARIO_TOL, (
        f"{split_name} dual ratio {ratio:.3f} not ≈ 0.40"
    )

@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_scenario_composition_tri(split_name):
    comp = _SPLIT.scenario_composition(split_name)
    ratio = comp.get("tri", 0.0)
    assert abs(ratio - 0.30) <= SCENARIO_TOL, (
        f"{split_name} tri ratio {ratio:.3f} not ≈ 0.30"
    )

def test_all_scenario_classes_present_in_all_splits():
    for split_name in ("train", "val", "test"):
        comp = _SPLIT.scenario_composition(split_name)
        for sc in ("single", "dual", "tri"):
            assert sc in comp, \
                f"Scenario class '{sc}' missing from {split_name} split"


# ------------------------------------------------------------------ #
# Severity distribution sanity                                         #
# ------------------------------------------------------------------ #

def test_all_severity_classes_present_in_train():
    dist = _SPLIT.severity_distribution("train")
    assert len(dist) >= 2, \
        f"Train set has only {len(dist)} severity class(es): {dist}"

def test_severity_labels_in_range():
    for split_name in ("train", "val", "test"):
        for c in getattr(_SPLIT, split_name):
            assert 0 <= c.severity_label <= 3, \
                f"severity_label={c.severity_label} out of 0–3 in {split_name}"
