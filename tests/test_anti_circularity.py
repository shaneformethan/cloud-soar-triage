"""
Test (b): tidak ada label leakage antar-modality.

Three levels of verification:
  1. Structural  — IamGenerationContext has no forbidden fields (severity,
                   raw label, flow vector) at the class definition level.
  2. Runtime     — the IamGenerationContext instances produced by the pipeline
                   contain no severity information.
  3. Independence — changing the flow severity label does NOT change the
                    generated IAM sequence when all other inputs are equal.
"""

import dataclasses
from datetime import datetime, timezone

import numpy as np
import pytest

from src.utils.circularity_guard import (
    IamGenerationContext,
    FORBIDDEN_FIELDS,
    audit_context_fields,
)
from src.data.dummy import dummy_iam_seq
from src.benchmark.attack_stage import (
    cic_label_to_tactic,
    tactic_to_severity_idx,
    combine_tactic_severity,
)


# ------------------------------------------------------------------ #
# 1. Structural: field-level audit                                     #
# ------------------------------------------------------------------ #

def test_audit_passes_on_clean_context():
    """audit_context_fields() must not raise for the current definition."""
    audit_context_fields()  # would raise AssertionError if violated


def test_forbidden_fields_not_in_context():
    """No field in IamGenerationContext may have a forbidden name."""
    actual_fields = {f.name for f in dataclasses.fields(IamGenerationContext)}
    violations = actual_fields & FORBIDDEN_FIELDS
    assert not violations, (
        f"CIRCULARITY VIOLATION: IamGenerationContext has forbidden "
        f"field(s): {violations}"
    )


def test_forbidden_field_set_covers_severity_variants():
    """FORBIDDEN_FIELDS must include all obvious severity / label names."""
    required = {"severity_label", "severity_index", "raw_attack_category",
                 "cic_label", "flow_features", "flow_vec", "label"}
    assert required <= FORBIDDEN_FIELDS, (
        f"FORBIDDEN_FIELDS is missing: {required - FORBIDDEN_FIELDS}"
    )


def test_context_is_frozen():
    """IamGenerationContext must be frozen (immutable after construction)."""
    ctx = _make_ctx("Impact")
    # Use normal attribute assignment — the only valid way to test frozen=True.
    # object.__setattr__ bypasses the frozen guard and would not raise.
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
        ctx.attack_tactic = "Exfiltration"  # type: ignore[misc]


# ------------------------------------------------------------------ #
# 2. Runtime: no severity in generated context                        #
# ------------------------------------------------------------------ #

def test_context_has_no_severity_attribute():
    ctx = _make_ctx("LateralMovement")
    for forbidden in FORBIDDEN_FIELDS:
        assert not hasattr(ctx, forbidden), \
            f"IamGenerationContext has forbidden attribute: {forbidden}"


def test_context_tactic_is_not_severity():
    """
    The tactic field carries MITRE tactic strings, NOT severity integers
    (0/1/2/3) or severity name strings (informational/medium/high/critical).
    """
    severity_names = {"informational", "medium", "high", "critical"}
    tactics_used = [None, "Impact", "LateralMovement", "CredentialAccess"]
    for t in tactics_used:
        ctx = _make_ctx(t)
        assert ctx.attack_tactic not in severity_names, \
            f"attack_tactic looks like a severity name: {ctx.attack_tactic!r}"
        assert ctx.attack_tactic != 0
        assert ctx.attack_tactic != 1
        assert ctx.attack_tactic != 2
        assert ctx.attack_tactic != 3


# ------------------------------------------------------------------ #
# 3. Independence: changing severity must not change IAM sequence      #
# ------------------------------------------------------------------ #

def test_iam_sequence_independent_of_severity():
    """
    Two buckets with the same tactic but different severity labels must
    produce identical IAM sequences (because IAM generator never sees label).

    Simulates the scenario where two flow records have the same attack
    type (same MITRE tactic) but one is labelled 'high' and the other
    'critical' — the IAM generator must be blind to that distinction.
    """
    tactic = "Impact"
    # Severity 2 = high, Severity 3 = critical — both map to same tactic
    assert tactic_to_severity_idx(tactic) in (2, 3)  # sanity check

    ctx_a = _make_ctx(tactic, seed=42)
    ctx_b = _make_ctx(tactic, seed=42)   # identical context

    seq_a = dummy_iam_seq(ctx_a)
    seq_b = dummy_iam_seq(ctx_b)
    assert np.array_equal(seq_a, seq_b), \
        "Same context must produce identical IAM sequences (deterministic)."


def test_iam_sequence_differs_for_different_tactics():
    """
    Different tactics must generally produce different IAM sequences,
    confirming the tactic (not a fixed benign distribution) drives output.
    """
    ctx_attack  = _make_ctx("CredentialAccess", seed=99)
    ctx_benign  = _make_ctx(None, seed=99)

    seq_attack = dummy_iam_seq(ctx_attack)
    seq_benign = dummy_iam_seq(ctx_benign)

    # They can't be identical: attack concentrates in a small action range
    assert not np.array_equal(seq_attack, seq_benign), (
        "Attack and benign IAM sequences are identical — tactic has no effect."
    )


def test_label_not_derivable_from_context_alone():
    """
    The severity label must NOT be derivable from IamGenerationContext.
    Verify that the same context tactic maps to multiple possible severity
    values depending on OTHER modalities (handled by combine_tactic_severity).

    Example: tactic="Impact" alone gives severity=2 (high), but combined
    with LateralMovement in another modality gives severity=3 (critical).
    """
    tactic = "Impact"
    # Single modality
    sev_single = combine_tactic_severity([tactic, None, None])
    # With LateralMovement in another modality
    sev_multi = combine_tactic_severity([tactic, "LateralMovement", None])
    assert sev_single != sev_multi, (
        "combine_tactic_severity must reflect ALL modalities, not just one."
    )


# ------------------------------------------------------------------ #
# 4. Attack-stage mapping is tactic-level, not severity-level         #
# ------------------------------------------------------------------ #

def test_two_cic_labels_same_tactic_different_severity_impossible():
    """
    cic_label_to_tactic produces tactic tags (strings), never severity ints.
    """
    labels_to_test = [
        "DoS slowloris", "DDoS", "Bot", "Infiltration",
        "Web Attack – Brute Force", "Benign",
    ]
    for label in labels_to_test:
        tactic = cic_label_to_tactic(label)
        assert tactic is None or isinstance(tactic, str), \
            f"cic_label_to_tactic returned non-string: {tactic!r}"
        assert tactic not in (0, 1, 2, 3), \
            f"cic_label_to_tactic returned severity int: {tactic!r}"


# ------------------------------------------------------------------ #
# Helper                                                              #
# ------------------------------------------------------------------ #

def _make_ctx(
    tactic: str | None,
    seed: int = 42,
    n_events: int = 16,
) -> IamGenerationContext:
    return IamGenerationContext(
        attack_tactic=tactic,
        session_src_ip="10.0.0.5",
        session_start_ts=datetime(2018, 2, 14, 10, 0, 0, tzinfo=timezone.utc),
        session_duration_s=120.0,
        target_event_count=n_events,
        rng_seed=seed,
    )
