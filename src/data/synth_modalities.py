"""
Tactic-conditioned synthetic generators for the IAM and TI modalities
(Section III-E of draft.tex).

ANTI-CIRCULARITY (Section III-E, and the repo's circularity_guard design):
    The synthetic generators are conditioned on the MITRE **attack tactic**
    only -- never on the severity label.  The severity label is a *downstream*
    deterministic function of the tactic (attack_stage.tactic_to_severity_idx),
    computed separately.  A model therefore has to learn
    IAM-behaviour -> tactic -> severity, which is ordinary supervised learning
    through a latent cause, not label leakage.  Two clusters that share a tactic
    produce IAM/TI from the same distribution regardless of anything else.

Design principle (the paper's thesis):
    Each modality carries the incident *partially and noisily*; only the active
    modalities of a scenario carry the attack tactic, inactive modalities are
    benign.  No single stream is sufficient, but together they disambiguate the
    attack.

IAM token layout (5 ints/event), matching encoders.IAMEncoder.IAM_VOCAB_SIZES
= [256, 10, 6, 5, 2]:
    [action_type_idx, resource_type_idx, principal_role_idx,
     src_ip_region_idx, is_in_baseline]
"""

from __future__ import annotations

import numpy as np

IAM_EVENT_DIM = 5
IAM_MAX_SEQ = 64
TI_DIM = 16

# Per-tactic action-index band within 1..63 (also the bin range the
# RandomForest/XGBoost histogram baseline reads). Bands deliberately OVERLAP so
# the tactic is only partially decodable from IAM alone (realistic ambiguity).
# ``None`` is the benign / legitimate-access distribution.
_TACTIC_BAND: dict[str | None, tuple[int, int]] = {
    None:                  (1, 18),
    "InitialAccess":       (14, 30),
    "Discovery":           (20, 36),
    "CredentialAccess":    (30, 46),
    "CommandAndControl":   (38, 52),
    "Persistence":         (34, 50),
    "Impact":              (46, 62),
    "Exfiltration":        (50, 64),
    "LateralMovement":     (42, 58),
}
# Stable per-tactic index used to tilt the non-action columns (resource /
# principal / region / novelty) WITHOUT referencing severity.
_TACTIC_ORDER = list(_TACTIC_BAND.keys())


def _tactic_idx(tactic: str | None) -> int:
    return _TACTIC_ORDER.index(tactic) if tactic in _TACTIC_BAND else 0


def generate_iam_seq(
    tactic: str | None,
    rng: np.random.Generator,
    noise: float = 0.35,
) -> np.ndarray:
    """
    Generate a tactic-conditioned IAM event sequence, shape (n, 5) int32.

    ``noise`` is the fraction of events whose action index is drawn from the
    full 1..63 range instead of the tactic band, controlling how decodable the
    tactic is from the IAM modality alone.
    """
    ti = _tactic_idx(tactic)
    # Sequence length grows mildly with tactic index (benign shortest), with
    # heavy overlap so length is not a giveaway.
    base = 6 + ti * 4
    n = int(np.clip(rng.normal(base + 10, 12), 4, IAM_MAX_SEQ))

    lo, hi = _TACTIC_BAND.get(tactic, _TACTIC_BAND[None])
    band = rng.integers(lo, hi, size=n)
    rand = rng.integers(1, 64, size=n)
    use_rand = rng.random(n) < noise
    actions = np.where(use_rand, rand, band).astype(np.int32)

    # Resource type (vocab 10): tilt by tactic index (not severity).
    res_center = 1 + int(round(ti / max(len(_TACTIC_ORDER) - 1, 1) * 7))
    resources = np.clip(np.round(rng.normal(res_center, 1.8, size=n)), 1, 9).astype(np.int32)

    principals = np.clip(np.round(rng.normal(1 + ti * 0.5, 1.2, size=n)), 1, 5).astype(np.int32)
    regions = rng.integers(1, 5, size=n).astype(np.int32)

    # Novelty flag: attack tactics (ti>0) are more likely to be flagged novel.
    p_novel = 0.10 if tactic is None else min(0.30 + 0.05 * ti, 0.8)
    history: set[int] = set()
    baseline = np.zeros(n, dtype=np.int32)
    for i in range(n):
        a = int(actions[i])
        if a in history and rng.random() > p_novel:
            baseline[i] = 1
        history.add(a)

    return np.stack([actions, resources, principals, regions, baseline], axis=1)


def benign_iam_seq(rng: np.random.Generator, noise: float = 0.35) -> np.ndarray:
    """Benign IAM session (tactic = None: legitimate access)."""
    return generate_iam_seq(None, rng, noise=noise)


# 16-dim TI layout (see data/threat_intel.py); these indices are binary flags.
_TI_BINARY_IDX = [1, 5, 7, 9, 11, 12, 13]


def generate_ti_vec(
    malicious: bool,
    rng: np.random.Generator,
    noise: float = 0.30,
) -> np.ndarray:
    """
    Generate a 16-dim threat-intelligence vector.

    Real IP reputation is effectively binary (an IP is flagged or it is not), so
    the synthetic fallback mirrors that: ``malicious`` controls a high/low
    reputation regime.  It does NOT encode a severity tier (the severity tier is
    carried by the flow / IAM modalities).
    """
    m = 0.8 if malicious else 0.1
    v = np.clip(rng.normal(m, noise, size=TI_DIM), 0.0, 1.0).astype(np.float32)
    for i in _TI_BINARY_IDX:
        p = np.clip(m + rng.normal(0.0, noise), 0.05, 0.95)
        v[i] = float(rng.random() < p)
    return v


def benign_ti_vec(rng: np.random.Generator, noise: float = 0.30) -> np.ndarray:
    """Benign TI vector (clean IP)."""
    return generate_ti_vec(False, rng, noise=noise)
