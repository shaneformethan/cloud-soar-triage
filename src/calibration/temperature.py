"""
Temperature scaling calibration (Section III-C).

Post-hoc calibration via a single scalar T > 0, optimized on the held-out
validation split by minimizing Expected Calibration Error (ECE) via L-BFGS.

Calibrated probability for class i:
    p̂_i = exp(z_i / T) / Σ_j exp(z_j / T)

References: Guo et al. 2017 [b17], Section III-C.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS


class TemperatureScaler(nn.Module):
    """
    Wraps a trained model and applies temperature scaling.

    Usage:
        scaler = TemperatureScaler(model)
        scaler.fit(val_loader, device)          # optimizes T on validation set
        probs = scaler.predict_proba(batch)     # calibrated probabilities
        ece_before, ece_after = scaler.ece_report()
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self._ece_before: float | None = None
        self._ece_after: float | None = None

    # ------------------------------------------------------------------ #
    # Fitting                                                              #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        logits_all: torch.Tensor,   # (N, C) — pre-collected logits from val set
        labels_all: torch.Tensor,   # (N,)   — ground-truth labels
        n_bins: int = 15,
    ) -> float:
        """
        Optimize T by minimizing NLL on the validation logits using L-BFGS.

        Parameters
        ----------
        logits_all : pre-computed validation logits (not softmaxed)
        labels_all : ground-truth labels
        n_bins     : bins for ECE computation

        Returns T (the optimized temperature scalar).
        """
        self._ece_before = compute_ece(
            F.softmax(logits_all, dim=-1), labels_all, n_bins
        )

        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            scaled = logits_all / self.temperature.clamp(min=1e-4)
            loss = F.cross_entropy(scaled, labels_all)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            scaled_logits = logits_all / self.temperature.clamp(min=1e-4)
            self._ece_after = compute_ece(
                F.softmax(scaled_logits, dim=-1), labels_all, n_bins
            )

        return float(self.temperature.item())

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Return temperature-scaled logits."""
        return logits / self.temperature.clamp(min=1e-4)

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated softmax probabilities."""
        return F.softmax(self.forward(logits), dim=-1)

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def ece_report(self) -> tuple[float, float]:
        """Return (ECE before, ECE after) calibration."""
        if self._ece_before is None or self._ece_after is None:
            raise RuntimeError("Call fit() before ece_report().")
        return self._ece_before, self._ece_after

    @property
    def T(self) -> float:
        return float(self.temperature.item())


# ─────────────────────────────────────────────────────────────────────────────
# ECE computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(
    probs: torch.Tensor,   # (N, C) — softmax probabilities
    labels: torch.Tensor,  # (N,)
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).

    Bins predictions by max confidence, then computes the weighted average
    gap between confidence and accuracy across bins.
    """
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = confidences.gt(lo) & confidences.le(hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc  = accuracies[mask].float().mean()
        ece += mask.float().mean() * (bin_conf - bin_acc).abs()

    return float(ece.item())
