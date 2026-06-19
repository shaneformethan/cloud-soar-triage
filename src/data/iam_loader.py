"""
flaws.cloud CloudTrail IAM loader + sequence generator.

Two phases:
  1. Analysis   : parse JSON CloudTrail logs → build per-tactic distributions
                  of (eventName, eventSource) pairs → save to disk.
  2. Generation : given IamGenerationContext (NO severity label) → sample
                  IAM event sequence → return np.ndarray shape (seq_len, 5).

Each event token is a 5-dim integer vector:
  [action_type_idx, resource_type_idx, principal_role_idx,
   src_ip_region_idx, is_in_baseline]

DESIGN DECISION: 5 features per IAM event token.
  - Categoricals stored as integer indices; the downstream transformer
    encoder handles embedding internally.
  - is_in_baseline (0 or 1): whether eventName appeared in the principal's
    30-day history within the corpus.  In generation mode this is inferred
    from whether the sampled eventName is in the principal's running history.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from src.utils.circularity_guard import IamGenerationContext

IAM_EVENT_DIM = 5
IAM_MAX_SEQ = 64

# Map MITRE tactic → set of IAM event name prefixes commonly seen in corpus.
# DESIGN DECISION: manual mapping based on ATT&CK for Cloud technique list.
# These are seeds; actual distributions are learned from the flaws.cloud corpus.
_TACTIC_SEED_EVENTS: dict[str | None, list[str]] = {
    None: [
        "GetObject", "ListBuckets", "DescribeInstances", "GetUser",
        "ListRoles", "GetBucketAcl", "DescribeSecurityGroups",
    ],
    "Discovery": [
        "ListBuckets", "DescribeInstances", "ListRoles", "GetUser",
        "DescribeVpcs", "ListUsers", "GetAccountAuthorizationDetails",
        "DescribeSecurityGroups", "ListPolicies",
    ],
    "CredentialAccess": [
        "GetSecretValue", "ListSecrets", "GetParameter",
        "CreateAccessKey", "UpdateAccessKey", "GetSessionToken",
        "AssumeRole", "ListAccessKeys",
    ],
    "InitialAccess": [
        "ConsoleLogin", "AssumeRoleWithSAML", "AssumeRoleWithWebIdentity",
        "GetFederationToken", "SwitchRole",
    ],
    "PrivilegeEscalation": [
        "AttachUserPolicy", "PutUserPolicy", "AttachRolePolicy",
        "PutRolePolicy", "CreateRole", "AddUserToGroup",
        "AttachGroupPolicy", "CreatePolicyVersion",
    ],
    "LateralMovement": [
        "AssumeRole", "SwitchRole", "CreateInstanceProfile",
        "AddRoleToInstanceProfile", "PassRole",
    ],
    "Persistence": [
        "CreateUser", "CreateAccessKey", "UpdateLoginProfile",
        "AddUserToGroup", "CreateRole", "PutRolePolicy",
    ],
    "Exfiltration": [
        "GetObject", "CopyObject", "CreatePresignedUrl",
        "PutBucketAcl", "PutObjectAcl", "ModifySnapshotAttribute",
    ],
    "CommandAndControl": [
        "RunInstances", "SendCommand", "InvokeFunction",
        "CreateFunction", "UpdateFunctionCode",
    ],
    "Impact": [
        "DeleteBucket", "DeleteObject", "DeleteUser",
        "TerminateInstances", "DeleteSnapshot", "PutBucketPolicy",
    ],
    "DefenseEvasion": [
        "DeleteTrail", "UpdateTrail", "StopLogging",
        "DeleteFlowLogs", "PutEventSelectors",
    ],
    "Execution": [
        "RunInstances", "SendCommand", "InvokeFunction", "StartInstances",
    ],
    "Collection": [
        "GetObject", "ListObjects", "CopyObject", "GetBucketVersioning",
    ],
}

_SERVICES: list[str] = [
    "s3", "iam", "sts", "ec2", "lambda", "secretsmanager",
    "ssm", "kms", "cloudtrail", "other",
]
_PRINCIPAL_TYPES: list[str] = [
    "IAMUser", "AssumedRole", "Root", "AWSService", "FederatedUser", "other",
]
_REGIONS: list[str] = [
    "us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "other",
]


def _service_from_source(event_source: str) -> str:
    svc = event_source.split(".")[0].lower()
    return svc if svc in _SERVICES else "other"


def _region_idx(region: str) -> int:
    return _REGIONS.index(region) if region in _REGIONS else len(_REGIONS) - 1


class IamDistributions:
    """
    Per-tactic probability distributions over IAM action vocabulary,
    learned from the flaws.cloud CloudTrail corpus.
    """

    def __init__(self) -> None:
        # vocab: action_name → int index
        self.action_vocab: dict[str, int] = {}
        # per-tactic counts: tactic_key → array of counts over vocab
        self._counts: dict[str | None, np.ndarray] = {}
        self._probs: dict[str | None, np.ndarray] = {}
        self._built = False

    # ------------------------------------------------------------------ #
    # Analysis phase                                                       #
    # ------------------------------------------------------------------ #

    def fit_from_cloudtrail(self, cloudtrail_dir: Path) -> None:
        """
        Parse all CloudTrail JSON files and build per-tactic distributions.
        """
        from src.benchmark.attack_stage import cic_label_to_tactic

        raw_counts: dict[str | None, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Seed vocab with known tactic events first
        for events in _TACTIC_SEED_EVENTS.values():
            for e in events:
                if e not in self.action_vocab:
                    self.action_vocab[e] = len(self.action_vocab)

        json_files = list(cloudtrail_dir.rglob("*.json"))
        print(f"[IAMLoader] Parsing {len(json_files)} CloudTrail JSON files…")

        for jf in json_files:
            try:
                with open(jf, encoding="utf-8") as fh:
                    data = json.load(fh)
                records = data.get("Records", []) if isinstance(data, dict) else []
                for rec in records:
                    event_name = rec.get("eventName", "Unknown")
                    if event_name not in self.action_vocab:
                        self.action_vocab[event_name] = len(self.action_vocab)
                    # Assign tactic based on event name patterns
                    tactic = _infer_tactic_from_event(event_name)
                    raw_counts[tactic][event_name] += 1
            except Exception:
                continue

        self._build_probs(raw_counts)

    def fit_from_seeds(self) -> None:
        """
        Build distributions purely from the _TACTIC_SEED_EVENTS table.
        Used in dummy mode when no CloudTrail corpus is available.
        """
        for events in _TACTIC_SEED_EVENTS.values():
            for e in events:
                if e not in self.action_vocab:
                    self.action_vocab[e] = len(self.action_vocab)

        raw_counts: dict[str | None, dict[str, int]] = {}
        for tactic, events in _TACTIC_SEED_EVENTS.items():
            raw_counts[tactic] = {e: 1 for e in events}

        self._build_probs(raw_counts)

    def _build_probs(
        self, raw_counts: dict[str | None, dict[str, int]]
    ) -> None:
        V = len(self.action_vocab)
        for tactic, counts in raw_counts.items():
            arr = np.zeros(V, dtype=np.float64)
            for name, cnt in counts.items():
                idx = self.action_vocab.get(name)
                if idx is not None:
                    arr[idx] = cnt
            # Laplace smoothing so every action has nonzero probability
            arr += 1.0
            self._counts[tactic] = arr
            self._probs[tactic] = arr / arr.sum()
        self._built = True

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"vocab": self.action_vocab,
                         "probs": self._probs}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.action_vocab = data["vocab"]
        self._probs = data["probs"]
        self._built = True

    # ------------------------------------------------------------------ #
    # Generation phase                                                     #
    # ------------------------------------------------------------------ #

    def sample_sequence(self, ctx: IamGenerationContext) -> np.ndarray:
        """
        Sample an IAM event sequence for the given context.

        Parameters
        ----------
        ctx : IamGenerationContext
            Contains ONLY tactic + session metadata. NO severity label.

        Returns
        -------
        seq : np.ndarray, shape (target_event_count, IAM_EVENT_DIM), int32
            Padded to (IAM_MAX_SEQ, IAM_EVENT_DIM) by callers if needed.
        """
        assert self._built, "Call fit_from_cloudtrail() or fit_from_seeds() first."

        rng = np.random.default_rng(ctx.rng_seed)
        n = min(max(ctx.target_event_count, 1), IAM_MAX_SEQ)

        # Tactic probability vector (fall back to None = benign if unseen)
        probs = self._probs.get(ctx.attack_tactic, self._probs.get(None))
        if probs is None:
            probs = np.ones(len(self.action_vocab)) / len(self.action_vocab)

        action_idxs = rng.choice(len(self.action_vocab), size=n, p=probs)
        inv_vocab = {v: k for k, v in self.action_vocab.items()}

        seq = np.zeros((n, IAM_EVENT_DIM), dtype=np.int32)
        history: set[int] = set()

        for i, aidx in enumerate(action_idxs):
            event_name = inv_vocab.get(aidx, "Unknown")
            service = _service_from_source(_event_to_source(event_name))
            resource_idx = _SERVICES.index(service) if service in _SERVICES else len(_SERVICES) - 1
            principal_idx = rng.integers(0, len(_PRINCIPAL_TYPES))
            region_idx = _region_idx(
                ctx.session_src_ip.split(".")[:1][0]  # crude region proxy
                if ctx.session_src_ip else "other"
            )
            is_in_baseline = int(aidx in history)
            history.add(aidx)

            seq[i] = [aidx, resource_idx, principal_idx, region_idx, is_in_baseline]

        return seq  # shape (n, 5)


# ------------------------------------------------------------------ #
# Module-level helpers                                                #
# ------------------------------------------------------------------ #

def _infer_tactic_from_event(event_name: str) -> str | None:
    """Heuristic: map a CloudTrail eventName to a MITRE tactic."""
    name = event_name.lower()
    if any(k in name for k in ("deletetrail", "stoplog", "deleteflow")):
        return "DefenseEvasion"
    if any(k in name for k in ("attachpolicy", "putpolicy", "createpolicy",
                                "addusertogroup", "createuser", "createrole")):
        return "PrivilegeEscalation"
    if any(k in name for k in ("assumerolewith", "getfederationtoken")):
        return "InitialAccess"
    if any(k in name for k in ("assumerole", "switchrole")):
        return "LateralMovement"
    if any(k in name for k in ("getsecret", "getparameter", "createaccesskey",
                                "getsessiontoken")):
        return "CredentialAccess"
    if any(k in name for k in ("deleteobject", "deletebucket", "terminateinstance",
                                "deleteuser")):
        return "Impact"
    if any(k in name for k in ("runinstances", "sendcommand", "invokefunction")):
        return "Execution"
    if any(k in name for k in ("getobject", "copyobject", "listobjects")):
        return "Collection"
    if any(k in name for k in ("list", "describe", "get", "head")):
        return "Discovery"
    return None


def _event_to_source(event_name: str) -> str:
    """Best-effort mapping from eventName prefix to AWS service source."""
    name = event_name.lower()
    if name.startswith(("getobject", "putobject", "listbucket", "deleteobject",
                         "copyobject", "createbucket")):
        return "s3.amazonaws.com"
    if name.startswith(("createuser", "attachpolicy", "createpolicy",
                         "createaccesskey", "addusertogroup")):
        return "iam.amazonaws.com"
    if name.startswith("assumerole") or name.startswith("getsessiontoken"):
        return "sts.amazonaws.com"
    if name.startswith(("runinstances", "describeinstance", "terminateinstance")):
        return "ec2.amazonaws.com"
    if name.startswith(("invoke", "createfunction", "updatefunction")):
        return "lambda.amazonaws.com"
    if name.startswith("getsecret"):
        return "secretsmanager.amazonaws.com"
    return "other.amazonaws.com"
