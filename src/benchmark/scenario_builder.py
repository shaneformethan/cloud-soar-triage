"""
Scenario class builder.

Assigns each flow bucket a scenario class (single/dual/tri) and decides
which modalities carry attack signal for that bucket.

Key invariant (anti-circularity):
  The IAM tactic for a bucket is chosen from the MITRE tactic space,
  not from the flow bucket's severity label. When a modality is "benign"
  (not in active_modalities), its tactic is set to None so the IAM
  generator samples from the benign distribution.

Composition targets (Section III-E):
  single : 30%
  dual   : 40%
  tri    : 30%
"""

from __future__ import annotations

import numpy as np

from src.benchmark.attack_stage import ALL_ATTACK_TACTICS


_ALL_MODALITIES = ["flow", "iam", "ti"]


class ScenarioBuilder:
    """
    Assigns scenario classes and per-modality tactic tags to flow buckets.

    Usage:
        builder = ScenarioBuilder(seed=42)
        assignments = builder.assign(flow_buckets, ratios)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def assign(
        self,
        flow_buckets: list[dict],
        ratios: dict[str, float] | None = None,
    ) -> list[dict]:
        """
        Parameters
        ----------
        flow_buckets : list[dict]
            Each dict must have: bucket_key, attack_tactic (from flow record).

        ratios : dict  {scenario_class: fraction}
            Defaults to {"single": 0.30, "dual": 0.40, "tri": 0.30}.

        Returns
        -------
        list[dict] with keys:
            bucket_key, scenario_class, active_modalities,
            flow_tactic, iam_tactic, ti_tactic
        """
        if ratios is None:
            ratios = {"single": 0.30, "dual": 0.40, "tri": 0.30}

        n = len(flow_buckets)
        scenario_labels = self._draw_scenario_labels(n, ratios)
        results = []

        for fb, sc in zip(flow_buckets, scenario_labels):
            flow_tactic = fb.get("attack_tactic")  # from MITRE mapping, not severity
            active = self._pick_active_modalities(sc)

            # Each active modality gets the bucket's tactic.
            # Each benign modality gets None (samples benign distribution).
            # DESIGN DECISION: all active modalities share the same tactic
            # for simplicity; a more complex model could give each modality
            # an independent tactic sampled from the same attack campaign.
            iam_tactic = flow_tactic if "iam" in active else None
            ti_tactic  = flow_tactic if "ti"  in active else None
            eff_flow_tactic = flow_tactic if "flow" in active else None

            results.append({
                "bucket_key":       fb["bucket_key"],
                "scenario_class":   sc,
                "active_modalities": list(active),
                "flow_tactic":      eff_flow_tactic,
                "iam_tactic":       iam_tactic,
                "ti_tactic":        ti_tactic,
            })

        return results

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _draw_scenario_labels(
        self, n: int, ratios: dict[str, float]
    ) -> list[str]:
        """
        Draw scenario class labels hitting exact integer counts.
        Remainder goes to the last class.
        """
        labels: list[str] = []
        keys = list(ratios.keys())
        remaining = n
        for k in keys[:-1]:
            count = round(n * ratios[k])
            labels.extend([k] * count)
            remaining -= count
        labels.extend([keys[-1]] * remaining)
        self.rng.shuffle(labels)
        return labels

    def _pick_active_modalities(self, scenario_class: str) -> set[str]:
        if scenario_class == "tri":
            return set(_ALL_MODALITIES)
        if scenario_class == "dual":
            chosen = self.rng.choice(_ALL_MODALITIES, size=2, replace=False)
            return set(chosen.tolist())
        # single
        return {_ALL_MODALITIES[int(self.rng.integers(0, 3))]}
