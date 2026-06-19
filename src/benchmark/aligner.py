"""
5-minute bucket aligner: joins Flow + IAM + TI into EventCluster objects.

Each EventCluster = one training sample:
  - flow_vec  : (12,)     float32
  - iam_seq   : (64, 5)   int32   (padded; iam_seq_len = actual length)
  - ti_vec    : (16,)     float32
  - severity_label : int  0–3
  - scenario_class : str  single|dual|tri

Anti-circularity control (Section III-E):
  The flow bucket's raw_label / severity_label is NEVER passed to
  IamGenerationContext. The IAM generator receives only the MITRE tactic
  (derived by attack_stage.py) and session metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.benchmark.attack_stage import (
    combine_tactic_severity,
    tactic_to_severity_idx,
)
from src.utils.circularity_guard import IamGenerationContext

IAM_MAX_SEQ = 64
IAM_EVENT_DIM = 5
FLOW_DIM = 12
TI_DIM = 16


@dataclass
class EventCluster:
    """One multi-modal triage sample."""
    bucket_key:       str
    bucket_start:     datetime
    src_ip:           str
    dst_ip:           str
    flow_vec:         np.ndarray   # shape (12,)  float32
    iam_seq:          np.ndarray   # shape (64, 5) int32  (zero-padded)
    iam_seq_len:      int          # actual event count before padding (1–64)
    ti_vec:           np.ndarray   # shape (16,)  float32
    severity_label:   int          # 0=informational … 3=critical
    scenario_class:   str          # "single" | "dual" | "tri"
    active_modalities: list[str]   # which modalities carry attack signal

    def validate_dims(self) -> None:
        assert self.flow_vec.shape == (FLOW_DIM,), \
            f"flow_vec shape {self.flow_vec.shape} != ({FLOW_DIM},)"
        assert self.iam_seq.shape == (IAM_MAX_SEQ, IAM_EVENT_DIM), \
            f"iam_seq shape {self.iam_seq.shape} != ({IAM_MAX_SEQ},{IAM_EVENT_DIM})"
        assert 1 <= self.iam_seq_len <= IAM_MAX_SEQ, \
            f"iam_seq_len={self.iam_seq_len} out of range"
        assert self.ti_vec.shape == (TI_DIM,), \
            f"ti_vec shape {self.ti_vec.shape} != ({TI_DIM},)"
        assert 0 <= self.severity_label <= 3, \
            f"severity_label={self.severity_label} out of range"
        assert self.scenario_class in ("single", "dual", "tri"), \
            f"unknown scenario_class={self.scenario_class}"


def pad_iam_seq(seq: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Pad raw IAM sequence to (IAM_MAX_SEQ, IAM_EVENT_DIM) with zeros.
    Returns (padded_array, original_length).
    """
    n = seq.shape[0]
    if n > IAM_MAX_SEQ:
        seq = seq[:IAM_MAX_SEQ]
        n = IAM_MAX_SEQ
    padded = np.zeros((IAM_MAX_SEQ, IAM_EVENT_DIM), dtype=np.int32)
    padded[:n] = seq
    return padded, n


class BucketAligner:
    """
    Aligns the three modalities into EventCluster objects.

    Usage (real-data mode):
        aligner = BucketAligner(iam_distributions, ti_enricher, global_seed)
        clusters = aligner.align(flow_buckets, scenario_assignments)

    The aligner receives:
      - flow_buckets   : output of FlowLoader.aggregate()
      - scenario_assignments : list[dict] from ScenarioBuilder.assign()
        Each entry specifies which modalities are "active" (carry signal)
        and the iam_tactic that should be used for generation.

    Severity label is assigned here, after all modalities are generated,
    using combine_tactic_severity() — NOT from the raw flow label.
    """

    def __init__(self, iam_distributions, ti_enricher, global_seed: int = 42):
        self.iam_dist = iam_distributions
        self.ti_enricher = ti_enricher
        self.global_seed = global_seed

    def align(
        self,
        flow_buckets: list[dict],
        scenario_assignments: list[dict],
    ) -> list[EventCluster]:
        """
        Parameters
        ----------
        flow_buckets : list[dict]
            From FlowLoader.aggregate(). Each dict has:
            bucket_key, bucket_start, src_ip, dst_ip, flow_vec,
            attack_tactic, raw_label.

        scenario_assignments : list[dict]
            From ScenarioBuilder.assign(). Each dict has:
            bucket_key, scenario_class, active_modalities,
            iam_tactic (may differ from flow tactic in single/dual scenarios).

        Returns
        -------
        list[EventCluster]
        """
        # Index scenario assignments by bucket_key
        assign_idx = {a["bucket_key"]: a for a in scenario_assignments}
        clusters = []

        for i, fb in enumerate(flow_buckets):
            key = fb["bucket_key"]
            assign = assign_idx.get(key)
            if assign is None:
                continue

            active = set(assign["active_modalities"])
            iam_tactic = assign["iam_tactic"]    # tactic for IAM, NOT severity
            flow_tactic = assign["flow_tactic"]  # tactic for flow modality
            ti_anomalous = "ti" in active

            # --- Flow vector (already normalised by FlowLoader) ---
            flow_vec = fb["flow_vec"].astype(np.float32)

            # --- IAM sequence ---
            # IamGenerationContext has NO severity_label, NO raw_label,
            # NO flow_vec — only tactic + session metadata.
            iam_ctx = IamGenerationContext(
                attack_tactic=iam_tactic,
                session_src_ip=fb["src_ip"],
                session_start_ts=fb["bucket_start"],
                session_duration_s=300.0,
                target_event_count=int(
                    np.random.default_rng(self.global_seed + i)
                    .integers(4, IAM_MAX_SEQ + 1)
                ),
                rng_seed=self.global_seed * 10000 + i,
            )
            raw_iam = self.iam_dist.sample_sequence(iam_ctx)
            iam_padded, iam_len = pad_iam_seq(raw_iam)

            # --- TI vector ---
            ti_vec = self.ti_enricher.enrich_bucket([fb["src_ip"], fb["dst_ip"]])
            if not ti_anomalous:
                # Suppress TI signal for benign TI modalities
                ti_vec = ti_vec * 0.2  # attenuate rather than zero for realism

            # --- Severity label ---
            # Derived from per-modality tactics AFTER all modalities generated.
            # This is the ONLY place severity is computed.
            ti_tactic = assign.get("ti_tactic") if ti_anomalous else None
            severity = combine_tactic_severity(
                [flow_tactic, iam_tactic, ti_tactic]
            )

            clusters.append(EventCluster(
                bucket_key=key,
                bucket_start=fb["bucket_start"],
                src_ip=fb["src_ip"],
                dst_ip=fb["dst_ip"],
                flow_vec=flow_vec,
                iam_seq=iam_padded,
                iam_seq_len=iam_len,
                ti_vec=ti_vec,
                severity_label=severity,
                scenario_class=assign["scenario_class"],
                active_modalities=list(active),
            ))

        return clusters
