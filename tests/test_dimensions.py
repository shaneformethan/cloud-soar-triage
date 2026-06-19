"""
Test (a): semua dimensi vektor sesuai spec Section III-B.

  IAM  : sequence shape (≤64, 5) — padded ke (64, 5) dalam EventCluster
  Flow : shape (12,)
  TI   : shape (16,)
  Severity label : int 0–3
"""

import numpy as np
import pytest
from datetime import datetime, timezone

from src.data.dummy import (
    dummy_flow_vec,
    dummy_iam_seq,
    dummy_ti_vec,
    generate_dummy_benchmark,
    IAM_EVENT_DIM,
    IAM_MAX_SEQ,
    FLOW_DIM,
    TI_DIM,
)
from src.benchmark.aligner import pad_iam_seq, EventCluster
from src.utils.circularity_guard import IamGenerationContext

SCENARIO_RATIOS = {"single": 0.30, "dual": 0.40, "tri": 0.30}
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ #
# Per-modality dimension tests                                         #
# ------------------------------------------------------------------ #

def test_flow_vec_dim():
    vec = dummy_flow_vec(RNG, anomalous=False)
    assert vec.shape == (FLOW_DIM,), f"Expected ({FLOW_DIM},), got {vec.shape}"
    assert vec.dtype == np.float32

def test_flow_vec_dim_anomalous():
    vec = dummy_flow_vec(RNG, anomalous=True)
    assert vec.shape == (FLOW_DIM,), f"Expected ({FLOW_DIM},), got {vec.shape}"

def test_flow_dim_constant():
    """FLOW_DIM must equal 12 per paper spec."""
    assert FLOW_DIM == 12

def test_ti_vec_dim():
    vec = dummy_ti_vec(RNG, anomalous=False)
    assert vec.shape == (TI_DIM,), f"Expected ({TI_DIM},), got {vec.shape}"
    assert vec.dtype == np.float32

def test_ti_vec_dim_anomalous():
    vec = dummy_ti_vec(RNG, anomalous=True)
    assert vec.shape == (TI_DIM,), f"Expected ({TI_DIM},), got {vec.shape}"

def test_ti_dim_constant():
    """TI_DIM must equal 16 per paper spec."""
    assert TI_DIM == 16

@pytest.mark.parametrize("n_events", [1, 4, 32, 64])
def test_iam_seq_dim_various_lengths(n_events):
    ctx = IamGenerationContext(
        attack_tactic="Discovery",
        session_src_ip="10.0.0.1",
        session_start_ts=datetime(2018, 2, 14, tzinfo=timezone.utc),
        session_duration_s=60.0,
        target_event_count=n_events,
        rng_seed=n_events,
    )
    seq = dummy_iam_seq(ctx)
    assert seq.shape == (n_events, IAM_EVENT_DIM), \
        f"Expected ({n_events},{IAM_EVENT_DIM}), got {seq.shape}"
    assert seq.dtype == np.int32

def test_iam_max_seq_len_constant():
    """IAM_MAX_SEQ must equal 64 per paper spec."""
    assert IAM_MAX_SEQ == 64

def test_iam_event_dim_constant():
    """IAM_EVENT_DIM must equal 5 per design decision."""
    assert IAM_EVENT_DIM == 5

def test_iam_seq_capped_at_64():
    """Sequences requesting > 64 events must be capped."""
    ctx = IamGenerationContext(
        attack_tactic=None,
        session_src_ip="10.0.0.1",
        session_start_ts=datetime(2018, 2, 14, tzinfo=timezone.utc),
        session_duration_s=300.0,
        target_event_count=999,  # intentionally over cap
        rng_seed=1,
    )
    seq = dummy_iam_seq(ctx)
    assert seq.shape[0] <= IAM_MAX_SEQ, \
        f"IAM sequence length {seq.shape[0]} exceeds IAM_MAX_SEQ={IAM_MAX_SEQ}"


# ------------------------------------------------------------------ #
# Padding tests                                                        #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("n", [1, 10, 64])
def test_pad_iam_seq_shape(n):
    raw = np.zeros((n, IAM_EVENT_DIM), dtype=np.int32)
    padded, length = pad_iam_seq(raw)
    assert padded.shape == (IAM_MAX_SEQ, IAM_EVENT_DIM)
    assert length == n
    # Padding region must be zeros
    assert np.all(padded[n:] == 0)

def test_pad_iam_seq_truncates_over_64():
    raw = np.ones((100, IAM_EVENT_DIM), dtype=np.int32)
    padded, length = pad_iam_seq(raw)
    assert padded.shape == (IAM_MAX_SEQ, IAM_EVENT_DIM)
    assert length == IAM_MAX_SEQ


# ------------------------------------------------------------------ #
# EventCluster validation                                             #
# ------------------------------------------------------------------ #

def test_event_cluster_validate_dims():
    """EventCluster.validate_dims() must pass for a well-formed cluster."""
    clusters = generate_dummy_benchmark(
        n_buckets=10, scenario_ratios=SCENARIO_RATIOS, seed=0
    )
    for c_dict in clusters:
        padded, seq_len = pad_iam_seq(c_dict["iam_seq"])
        cluster = EventCluster(
            bucket_key=c_dict["bucket_key"],
            bucket_start=c_dict["bucket_start"],
            src_ip=c_dict["src_ip"],
            dst_ip=c_dict["dst_ip"],
            flow_vec=c_dict["flow_vec"],
            iam_seq=padded,
            iam_seq_len=seq_len,
            ti_vec=c_dict["ti_vec"],
            severity_label=c_dict["severity_label"],
            scenario_class=c_dict["scenario_class"],
            active_modalities=c_dict["active_modalities"],
        )
        cluster.validate_dims()  # must not raise


# ------------------------------------------------------------------ #
# Value range tests                                                    #
# ------------------------------------------------------------------ #

def test_flow_vec_normalised():
    """Dummy flow vectors must be in [0, 1]."""
    for _ in range(20):
        vec = dummy_flow_vec(RNG, anomalous=bool(RNG.integers(0, 2)))
        assert vec.min() >= 0.0 and vec.max() <= 1.0, \
            f"flow_vec out of [0,1]: min={vec.min()}, max={vec.max()}"

def test_ti_vec_normalised():
    """Dummy TI vectors must be in [0, 1]."""
    for _ in range(20):
        vec = dummy_ti_vec(RNG, anomalous=bool(RNG.integers(0, 2)))
        assert vec.min() >= 0.0 and vec.max() <= 1.0, \
            f"ti_vec out of [0,1]: min={vec.min()}, max={vec.max()}"

def test_severity_label_range():
    clusters = generate_dummy_benchmark(
        n_buckets=100, scenario_ratios=SCENARIO_RATIOS, seed=7
    )
    for c in clusters:
        assert 0 <= c["severity_label"] <= 3, \
            f"severity_label={c['severity_label']} out of 0–3"
