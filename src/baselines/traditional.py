"""
Traditional ML baselines: Random Forest (Trad-A) and XGBoost (Trad-B).
(Table III, Section III-E of draft.)

Feature representation: 92-dimensional flattened multi-modal feature vector
  = 64-dim IAM histogram  (action-type frequency counts, normalized)
  + 12-dim flow vector
  + 16-dim TI vector
Total: 64 + 12 + 16 = 92 dimensions (as specified in Table III footnote).

IAM histogram: counts of each action-type index (0–63) within the sequence,
normalized by sequence length. This removes sequential structure but
preserves frequency information — consistent with how traditional ML
is typically applied to log data.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


IAM_HISTOGRAM_DIM = 64   # number of histogram bins (action-type vocab)
FLOW_DIM = 12
TI_DIM = 16
FEATURE_DIM = IAM_HISTOGRAM_DIM + FLOW_DIM + TI_DIM   # = 92


def build_flat_features(
    iam_tokens: np.ndarray,    # (B, 64, 5) int
    iam_lengths: np.ndarray,   # (B,)
    flow_vecs: np.ndarray,     # (B, 12)
    ti_vecs: np.ndarray,       # (B, 16)
) -> np.ndarray:
    """
    Construct the 92-dim flattened feature vector for each sample.

    IAM histogram: counts of action_type_idx (column 0 of iam_tokens),
    normalized by iam_lengths. Shape: (B, 64).
    """
    B = iam_tokens.shape[0]
    iam_hist = np.zeros((B, IAM_HISTOGRAM_DIM), dtype=np.float32)

    for i in range(B):
        n = int(iam_lengths[i])
        action_idxs = iam_tokens[i, :n, 0]   # column 0 = action_type_idx
        for idx in action_idxs:
            bucket = min(int(idx), IAM_HISTOGRAM_DIM - 1)
            iam_hist[i, bucket] += 1.0
        if n > 0:
            iam_hist[i] /= n   # normalize by sequence length

    features = np.concatenate([iam_hist, flow_vecs, ti_vecs], axis=1)
    assert features.shape == (B, FEATURE_DIM), \
        f"Feature dim mismatch: expected ({B},{FEATURE_DIM}), got {features.shape}"
    return features


class RandomForestBaseline:
    """
    Trad-A: Random Forest classifier on 92-dim flattened multi-modal features.

    Hyperparameter search space (Table IV in draft):
      n_estimators : {100, 200, 500}
      max_depth    : {10, 20, None}
      max_features : {"sqrt", "log2"}
      class_weight : "balanced"
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = 20,
        max_features: str = "sqrt",
        random_state: int = 42,
    ):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(
        self,
        iam_tokens: np.ndarray,
        iam_lengths: np.ndarray,
        flow_vecs: np.ndarray,
        ti_vecs: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        X = build_flat_features(iam_tokens, iam_lengths, flow_vecs, ti_vecs)
        self.clf.fit(X, labels)

    def predict(
        self,
        iam_tokens: np.ndarray,
        iam_lengths: np.ndarray,
        flow_vecs: np.ndarray,
        ti_vecs: np.ndarray,
    ) -> np.ndarray:
        X = build_flat_features(iam_tokens, iam_lengths, flow_vecs, ti_vecs)
        return self.clf.predict(X).astype(np.int64)

    def predict_proba(
        self,
        iam_tokens: np.ndarray,
        iam_lengths: np.ndarray,
        flow_vecs: np.ndarray,
        ti_vecs: np.ndarray,
    ) -> np.ndarray:
        X = build_flat_features(iam_tokens, iam_lengths, flow_vecs, ti_vecs)
        return self.clf.predict_proba(X).astype(np.float32)


class XGBoostBaseline:
    """
    Trad-B: XGBoost classifier on 92-dim flattened multi-modal features.

    Hyperparameter search space (Table IV in draft):
      n_estimators  : {100, 200, 500}
      max_depth     : {3, 6, 9}
      learning_rate : {0.01, 0.1, 0.3}
      subsample     : {0.7, 0.9, 1.0}
      class_weight  : balanced (scale_pos_weight per class)
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.9,
        random_state: int = 42,
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost not installed. Run: pip install xgboost"
            )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self.clf: "xgb.XGBClassifier | None" = None
        # Maps encoded (0..k-1) class indices back to the original
        # SEVERITY_NAMES indices (0..3). Populated in fit(), since the
        # dummy benchmark may not contain every severity class (e.g. no
        # "informational" samples), and XGBoost's multi:softprob requires
        # labels to be a contiguous 0..k-1 range matching np.unique(y).
        self.label_encoder = LabelEncoder()

    def fit(
        self,
        iam_tokens: np.ndarray,
        iam_lengths: np.ndarray,
        flow_vecs: np.ndarray,
        ti_vecs: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        X = build_flat_features(iam_tokens, iam_lengths, flow_vecs, ti_vecs)

        # Re-encode labels to a contiguous 0..k-1 range as required by
        # XGBoost's multi:softprob objective.
        y_enc = self.label_encoder.fit_transform(labels)
        n_classes = len(self.label_encoder.classes_)

        self.clf = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=self.random_state,
            n_jobs=-1,
        )

        # Balanced weighting via sample_weight
        classes, counts = np.unique(y_enc, return_counts=True)
        weight_map = {c: len(y_enc) / (len(classes) * cnt)
                      for c, cnt in zip(classes, counts)}
        sample_weight = np.array([weight_map[lb] for lb in y_enc])
        self.clf.fit(X, y_enc, sample_weight=sample_weight)

    def predict(
        self,
        iam_tokens: np.ndarray,
        iam_lengths: np.ndarray,
        flow_vecs: np.ndarray,
        ti_vecs: np.ndarray,
    ) -> np.ndarray:
        X = build_flat_features(iam_tokens, iam_lengths, flow_vecs, ti_vecs)
        y_enc = self.clf.predict(X)
        return self.label_encoder.inverse_transform(y_enc).astype(np.int64)

    def predict_proba(
        self,
        iam_tokens: np.ndarray,
        iam_lengths: np.ndarray,
        flow_vecs: np.ndarray,
        ti_vecs: np.ndarray,
    ) -> np.ndarray:
        """
        Returns (B, 4) probabilities over all SEVERITY_NAMES classes, even if
        some classes (e.g. "informational") were absent from the training
        labels and therefore absent from the underlying XGBoost model.
        Missing classes get probability 0.
        """
        X = build_flat_features(iam_tokens, iam_lengths, flow_vecs, ti_vecs)
        proba_enc = self.clf.predict_proba(X).astype(np.float32)  # (B, k)

        n_total_classes = 4  # len(SEVERITY_NAMES)
        proba_full = np.zeros((proba_enc.shape[0], n_total_classes), dtype=np.float32)
        for enc_idx, orig_idx in enumerate(self.label_encoder.classes_):
            proba_full[:, int(orig_idx)] = proba_enc[:, enc_idx]
        return proba_full
