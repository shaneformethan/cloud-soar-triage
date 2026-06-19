"""
MITRE ATT&CK for Cloud stage model.

Maps CIC-IDS2018 attack category labels → MITRE tactic tags,
then MITRE tactic tags → 4-class severity.

This is the ONLY component that sees the raw CIC label. Its output
(tactic tag) is what flows to the IAM generator — the severity label
is produced separately and never enters the IAM generation path.

References: MITRE ATT&CK for Cloud [b16], Section III-E.
"""

from __future__ import annotations

# DESIGN DECISION: CIC-IDS2018 label → MITRE ATT&CK Cloud tactic mapping.
# Source: manual alignment of CIC attack categories against
# https://attack.mitre.org/matrices/enterprise/cloud/
# Update this table if you relabel the dataset differently.
CIC_LABEL_TO_TACTIC: dict[str, str | None] = {
    "Benign":                       None,
    "BENIGN":                       None,
    "DoS slowloris":                "Impact",
    "DoS Slowhttptest":             "Impact",
    "DoS Hulk":                     "Impact",
    "DoS GoldenEye":                "Impact",
    "Heartbleed":                   "InitialAccess",
    "Web Attack \x96 Brute Force": "CredentialAccess",
    "Web Attack – Brute Force":     "CredentialAccess",
    "Web Attack \x96 XSS":         "InitialAccess",
    "Web Attack – XSS":             "InitialAccess",
    "Web Attack \x96 Sql Injection":"InitialAccess",
    "Web Attack – Sql Injection":   "InitialAccess",
    "Infiltration":                 "LateralMovement",
    "Bot":                          "CommandAndControl",
    "DDoS":                         "Impact",
    "FTP-Patator":                  "CredentialAccess",
    "SSH-Patator":                  "CredentialAccess",
}

# DESIGN DECISION: MITRE tactic → 4-class severity index.
# 0=informational, 1=medium, 2=high, 3=critical
# Rationale: tactics with direct data access / persistence / lateral movement
# are critical; credential attacks are high; initial foothold / DoS are high;
# reconnaissance is medium.
TACTIC_TO_SEVERITY_IDX: dict[str | None, int] = {
    None:                  0,  # informational
    "Discovery":           1,  # medium
    "DefenseEvasion":      1,  # medium
    "InitialAccess":       2,  # high
    "Impact":              2,  # high
    "CredentialAccess":    2,  # high
    "Execution":           2,  # high
    "Persistence":         2,  # high
    "Collection":          2,  # high
    "LateralMovement":     3,  # critical
    "CommandAndControl":   3,  # critical
    "PrivilegeEscalation": 3,  # critical
    "Exfiltration":        3,  # critical
}

SEVERITY_NAMES: list[str] = ["informational", "medium", "high", "critical"]

# All cloud-relevant MITRE tactics (used by scenario builder).
ALL_ATTACK_TACTICS: list[str] = [
    t for t in TACTIC_TO_SEVERITY_IDX if t is not None
]


def cic_label_to_tactic(raw_label: str) -> str | None:
    """Return the MITRE tactic for a CIC-IDS2018 label string."""
    label = raw_label.strip()
    tactic = CIC_LABEL_TO_TACTIC.get(label)
    if tactic is None and label.lower() not in ("benign", ""):
        # Unknown label: treat as medium-severity discovery rather than silently
        # dropping, so unknown attack types don't vanish from the benchmark.
        return "Discovery"
    return tactic


def tactic_to_severity_idx(tactic: str | None) -> int:
    """Return 0–3 severity index for a MITRE tactic (or None = benign)."""
    return TACTIC_TO_SEVERITY_IDX.get(tactic, 1)


def severity_idx_to_name(idx: int) -> str:
    return SEVERITY_NAMES[idx]


def combine_tactic_severity(tactics: list[str | None]) -> int:
    """
    Given a list of per-modality tactics (some may be None = benign),
    return the combined severity index as the maximum across modalities.

    Used by scenario_builder to assign the cluster-level label after
    all modalities are generated independently.
    """
    return max(tactic_to_severity_idx(t) for t in tactics)
