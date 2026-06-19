"""
Draft-faithful CSE-CIC-IDS2018 attack-class -> 4-class severity mapping
(Section III-E of draft.tex, "Dataset Sources and Characteristics").

The draft specifies the mapping verbatim:

    benign traffic                         -> Informational (0)
    web attacks (XSS, SQL injection,
        brute-force web) and infiltration  -> Medium        (1)
    brute-force SSH/FTP, Heartbleed,
        and botnet                         -> High          (2)
    DoS/DDoS attacks                       -> Critical       (3)

This is the ONLY place the real CIC label is turned into a severity label.
Unlike the older tactic-based path in ``attack_stage.py`` (which was written
against 2017-style label strings and routed everything through a MITRE tactic
table), this mapping is keyed on the *actual* CSE-CIC-IDS2018 label strings and
follows the draft's four-tier scheme exactly, so the flow modality carries a
genuine, learnable signal about severity.

Reference: alert-priority classifications in the SOC triage literature [b2] and
MITRE ATT&CK for Enterprise impact ratings [b26], as cited in the draft.
"""

from __future__ import annotations

# 0=informational, 1=medium, 2=high, 3=critical
SEVERITY_NAMES: list[str] = ["informational", "medium", "high", "critical"]

# Canonical CSE-CIC-IDS2018 label strings -> severity index.
# Keys are normalised (stripped, lower-cased) before lookup so minor casing /
# spelling variants across CSV releases still resolve.
_CIC2018_LABEL_TO_SEVERITY: dict[str, int] = {
    # --- Informational (benign) ---
    "benign": 0,

    # --- Medium: web attacks + infiltration ---
    "brute force -web": 1,
    "brute force -xss": 1,
    "sql injection": 1,
    "infilteration": 1,   # CSE-CIC-IDS2018 uses this (mis)spelling
    "infiltration": 1,

    # --- High: credential brute force + botnet (+ Heartbleed if present) ---
    "ftp-bruteforce": 2,
    "ssh-bruteforce": 2,
    "bot": 2,
    "heartbleed": 2,

    # --- Critical: DoS / DDoS ---
    "dos attacks-hulk": 3,
    "dos attacks-slowhttptest": 3,
    "dos attacks-goldeneye": 3,
    "dos attacks-slowloris": 3,
    "ddos attacks-loic-http": 3,
    "ddos attack-loic-udp": 3,
    "ddos attack-hoic": 3,
}


def cic2018_label_to_severity(raw_label: str) -> int | None:
    """
    Map a raw CSE-CIC-IDS2018 label string to a severity index 0-3.

    Returns ``None`` for unrecognised strings (e.g. the stray repeated-header
    rows that read ``"Label"``), so callers can drop them rather than silently
    assigning a wrong class.
    """
    if raw_label is None:
        return None
    key = str(raw_label).strip().lower()
    if key in ("", "label", "nan"):
        return None
    sev = _CIC2018_LABEL_TO_SEVERITY.get(key)
    if sev is not None:
        return sev
    # Fall back on coarse keyword matching for unseen variants so a new attack
    # family is never silently dropped.
    if "ddos" in key or "dos" in key:
        return 3
    if "bot" in key or "bruteforce" in key or "heartbleed" in key:
        return 2
    if "infilt" in key or "sql" in key or "xss" in key or "web" in key:
        return 1
    return None
