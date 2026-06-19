"""
SOAR integration layer (Section III-D).

Replaces static rule evaluation in the playbook dispatch pipeline.

Three-outcome routing per incident cluster:
  confidence ≥ τ_h  → AUTO_DISPATCH  (automated playbook execution)
  τ_l ≤ conf < τ_h  → REDUCED_REVIEW (reduced-effort analyst workflow)
  confidence < τ_l  → MANUAL_TRIAGE  (full manual analyst handling)

Thresholds τ_h and τ_l are selected on the held-out validation split by
minimizing analyst queue depth (= AUTO_DISPATCH fraction) subject to the
3% false negative rate (FNR) ceiling specified in Section III-F.

Severity label → playbook mapping is maintained as a separate configuration
artifact; the model's scoring and ranking function is kept independent of
which specific playbook gets executed (design rationale: Section III-D).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F


class RoutingDecision(str, Enum):
    AUTO_DISPATCH  = "auto_dispatch"
    REDUCED_REVIEW = "reduced_review"
    MANUAL_TRIAGE  = "manual_triage"


@dataclass
class TriageResult:
    """Result of routing a single incident cluster."""
    severity_pred:   int            # 0=informational … 3=critical
    confidence:      float          # max-class calibrated probability
    routing:         RoutingDecision
    playbook_id:     str | None     # None if MANUAL_TRIAGE
    probs:           np.ndarray     # shape (4,)


# Severity index → default playbook mapping (Section III-D).
# Configurable without model retraining.
DEFAULT_PLAYBOOK_MAP: dict[int, str] = {
    0: "PB-INFO-01",    # informational → log & close
    1: "PB-MED-01",     # medium → investigate & contain
    2: "PB-HIGH-01",    # high → isolate & escalate
    3: "PB-CRIT-01",    # critical → immediate incident response
}


class SOARRouter:
    """
    Applies threshold-based routing to calibrated model outputs.

    Parameters
    ----------
    tau_h : float
        Auto-dispatch confidence threshold (default 0.80 from Table I search).
    tau_l : float
        Deferral confidence threshold (default 0.50).
    playbook_map : dict
        Maps severity index → playbook identifier string.
    """

    def __init__(
        self,
        tau_h: float = 0.80,
        tau_l: float = 0.50,
        playbook_map: dict[int, str] | None = None,
    ):
        assert 0 < tau_l < tau_h < 1.0, "Must satisfy 0 < τ_l < τ_h < 1"
        self.tau_h = tau_h
        self.tau_l = tau_l
        self.playbook_map = playbook_map or DEFAULT_PLAYBOOK_MAP

    def route_batch(
        self,
        probs: torch.Tensor | np.ndarray,   # (B, 4) calibrated probabilities
    ) -> list[TriageResult]:
        """Route a batch of incidents."""
        if isinstance(probs, torch.Tensor):
            probs_np = probs.detach().cpu().numpy()
        else:
            probs_np = np.array(probs)

        results = []
        for row in probs_np:
            pred_class = int(np.argmax(row))
            conf = float(row[pred_class])

            if conf >= self.tau_h:
                routing = RoutingDecision.AUTO_DISPATCH
                playbook = self.playbook_map.get(pred_class)
            elif conf >= self.tau_l:
                routing = RoutingDecision.REDUCED_REVIEW
                playbook = self.playbook_map.get(pred_class)
            else:
                routing = RoutingDecision.MANUAL_TRIAGE
                playbook = None

            results.append(TriageResult(
                severity_pred=pred_class,
                confidence=conf,
                routing=routing,
                playbook_id=playbook,
                probs=row.copy(),
            ))
        return results

    # ------------------------------------------------------------------ #
    # Threshold optimization                                               #
    # ------------------------------------------------------------------ #

    def fit_thresholds(
        self,
        probs: np.ndarray,      # (N, 4) calibrated probs on val set
        labels: np.ndarray,     # (N,)   ground-truth severity labels
        fnr_ceiling: float = 0.03,
        tau_h_candidates: list[float] | None = None,
        tau_l_candidates: list[float] | None = None,
    ) -> tuple[float, float]:
        """
        Grid-search τ_h and τ_l to minimize analyst queue depth
        (fraction not auto-dispatched) subject to FNR ≤ fnr_ceiling.

        FNR definition: fraction of true non-informational incidents
        (severity ≥ 1) that are incorrectly auto-dispatched as informational.

        Returns (tau_h, tau_l) best found.
        """
        if tau_h_candidates is None:
            tau_h_candidates = [0.70, 0.80, 0.90]
        if tau_l_candidates is None:
            tau_l_candidates = [0.40, 0.50, 0.60]

        best_tau_h = self.tau_h
        best_tau_l = self.tau_l
        best_auto_rate = -1.0

        for tau_h in tau_h_candidates:
            for tau_l in tau_l_candidates:
                if tau_l >= tau_h:
                    continue
                # FNR: critical/high incidents sent to auto-dispatch as benign
                preds = probs.argmax(axis=1)
                conf  = probs.max(axis=1)
                auto_mask = conf >= tau_h

                # False negatives: true positive (label≥1) but predicted
                # informational (pred=0) AND auto-dispatched
                true_pos_mask = labels >= 1
                fn_mask = auto_mask & (preds == 0) & true_pos_mask
                fnr = fn_mask.sum() / max(true_pos_mask.sum(), 1)

                if fnr > fnr_ceiling:
                    continue   # constraint violated

                auto_rate = auto_mask.mean()
                if auto_rate > best_auto_rate:
                    best_auto_rate = auto_rate
                    best_tau_h = tau_h
                    best_tau_l = tau_l

        self.tau_h = best_tau_h
        self.tau_l = best_tau_l
        return best_tau_h, best_tau_l

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def compute_routing_stats(
        self,
        results: list[TriageResult],
        labels: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute workload and false positive metrics over routed results.

        Returns
        -------
        dict with keys:
          auto_dispatch_rate    – fraction auto-dispatched
          reduced_review_rate   – fraction reduced review
          manual_triage_rate    – fraction fully manual
          analyst_workload_reduction – 1 - manual_triage_rate
          fp_suppression_rate   – fraction of true informational correctly
                                  classified as non-actionable
          fnr_at_thresholds     – false negative rate under current thresholds
        """
        n = len(results)
        auto    = sum(1 for r in results if r.routing == RoutingDecision.AUTO_DISPATCH)
        reduced = sum(1 for r in results if r.routing == RoutingDecision.REDUCED_REVIEW)
        manual  = sum(1 for r in results if r.routing == RoutingDecision.MANUAL_TRIAGE)

        preds  = np.array([r.severity_pred for r in results])
        confs  = np.array([r.confidence    for r in results])

        # FP suppression: true informational (label=0) → pred=0 (correctly suppressed)
        true_info = labels == 0
        fp_suppressed = (preds == 0) & true_info
        fp_suppression = fp_suppressed.sum() / max(true_info.sum(), 1)

        # FNR: true positives (label≥1) auto-dispatched as informational
        true_pos = labels >= 1
        fn = ((confs >= self.tau_h) & (preds == 0) & true_pos)
        fnr = fn.sum() / max(true_pos.sum(), 1)

        return {
            "auto_dispatch_rate":         auto / n,
            "reduced_review_rate":        reduced / n,
            "manual_triage_rate":         manual / n,
            "analyst_workload_reduction": (auto + reduced) / n,
            "fp_suppression_rate":        float(fp_suppression),
            "fnr_at_thresholds":          float(fnr),
            "tau_h":                      self.tau_h,
            "tau_l":                      self.tau_l,
        }
