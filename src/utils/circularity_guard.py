"""
Anti-circularity enforcement for IAM sequence generation.

The core constraint from Section III-E:
  IAM sequences are generated using ONLY the attack-stage model output
  (MITRE tactic) and session structure. The flow record's ground-truth
  severity label must never reach the IAM generator.

This is enforced structurally: IamGenerationContext is a frozen dataclass
whose fields are audited by unit tests (test_anti_circularity.py).
Any attempt to add severity_label, raw_attack_category, or flow_features
to this class will cause those tests to fail immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime

import numpy as np

# Fields that must NEVER appear in IamGenerationContext.
# Unit tests check this set against the actual dataclass fields.
FORBIDDEN_FIELDS: frozenset[str] = frozenset(
    {"severity_label", "severity_index", "raw_attack_category", "cic_label",
     "flow_features", "flow_vec", "ground_truth", "label"}
)


@dataclass(frozen=True)
class IamGenerationContext:
    """
    Everything the IAM sequence generator is allowed to see.

    Permitted:
      - MITRE ATT&CK tactic (derived from flow features via attack_stage.py,
        NOT from the severity label)
      - Session-level metadata: source IP, timing, event count budget

    Prohibited (enforced by test_anti_circularity.py):
      - severity_label / severity_index
      - raw CIC-IDS2018 attack category string
      - flow feature vector
    """
    attack_tactic: str | None    # MITRE tactic tag, e.g. "Impact"; None = benign
    session_src_ip: str          # source IP address of the session
    session_start_ts: datetime   # wall-clock start of the 5-min bucket
    session_duration_s: float    # duration in seconds (≤ 300 for a 5-min bucket)
    target_event_count: int      # how many IAM events to generate (1–64)
    rng_seed: int                # reproducible seed for this bucket's generator


def audit_context_fields() -> None:
    """
    Raise AssertionError if any forbidden field name appears in
    IamGenerationContext. Called by unit tests and at import time in CI.
    """
    actual = {f.name for f in fields(IamGenerationContext)}
    violations = actual & FORBIDDEN_FIELDS
    assert not violations, (
        f"CIRCULARITY VIOLATION: IamGenerationContext contains forbidden "
        f"field(s): {violations}. These fields would leak ground-truth labels "
        f"into the IAM generator."
    )


# Run the audit at import time so violations surface immediately.
audit_context_fields()
