"""
DeepCASE baseline (flow-modality adaptation, Section III-E / Table III).

Original DeepCASE [Van Ede et al., b6] uses sequential context modeling over
alert chains to predict whether an alert warrants escalation. For this
benchmark we adapt it to the flow-modality input: network flow vectors are
treated as the sequential alert representation, and DeepCASE's LSTM context
encoder + DBSCAN clustering is replicated on the 12-dim flow feature vectors.

Adaptation choices:
  - Input: 12-dim aggregate flow vectors (same as FlowEncoder input)
  - Context encoder: LSTM with configurable embedding dim and window size
  - Clustering: DBSCAN on LSTM hidden states (epsilon, min_samples from Table II)
  - Severity assignment: majority-vote severity label within each cluster

Since ground-truth severity labels exist in our benchmark (unlike the original
SOC setting), we evaluate via cluster purity and classification accuracy on the
four-class severity labels. Hyperparameter search follows Table II in the draft.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class DeepCASEContextEncoder(nn.Module):
    """
    LSTM context encoder for sequential flow vectors.

    Input:
        flow_seq : (B, W, 12)  — window of W consecutive flow feature vectors
    Output:
        context  : (B, embed_dim)
    """

    def __init__(
        self,
        input_dim: int = 12,
        embed_dim: int = 128,
        window_size: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=embed_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, flow_seq: torch.Tensor) -> torch.Tensor:
        """flow_seq: (B, W, 12) → context: (B, embed_dim)"""
        _, (h_n, _) = self.lstm(flow_seq)
        out = h_n[-1]          # last layer hidden state: (B, embed_dim)
        return self.norm(out)


class DeepCASEBaseline:
    """
    DeepCASE-adapted baseline for the SOAR triage benchmark.

    Usage:
        baseline = DeepCASEBaseline(embed_dim=128, eps=0.3, min_samples=10)
        baseline.fit(train_flow_vecs, train_labels)  # learns cluster → class map
        preds = baseline.predict(test_flow_vecs)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        window_size: int = 20,
        eps: float = 0.3,
        min_samples: int = 10,
        device: str = "cpu",
    ):
        self.encoder = DeepCASEContextEncoder(
            embed_dim=embed_dim,
            window_size=window_size,
        ).to(device)
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        self.scaler = StandardScaler()
        self.cluster_labels: dict[int, int] = {}  # cluster_id → severity label
        self.device = device
        self._fitted = False

    def fit(
        self,
        flow_vecs: np.ndarray,   # (N, 12) float32
        labels: np.ndarray,      # (N,)    int 0–3
        batch_size: int = 256,
    ) -> None:
        """Encode all training flow vectors, cluster, assign labels."""
        contexts = self._encode_all(flow_vecs, batch_size)
        scaled = self.scaler.fit_transform(contexts)
        cluster_ids = self.dbscan.fit_predict(scaled)

        # Majority-vote label per cluster
        for cid in set(cluster_ids):
            if cid == -1:
                continue  # noise → will predict by nearest cluster or default
            mask = cluster_ids == cid
            # noinspection PyUnresolvedReferences
            majority = int(np.bincount(labels[mask]).argmax())
            self.cluster_labels[cid] = majority

        # For noise points: use majority class of training set
        self._default_label = int(np.bincount(labels).argmax())
        self._scaled_centers = {
            cid: scaled[cluster_ids == cid].mean(axis=0)
            for cid in self.cluster_labels
        }
        self._fitted = True
        print(
            f"[DeepCASE] Fitted: {len(self.cluster_labels)} clusters, "
            f"noise_points={int((cluster_ids == -1).sum())}"
        )

    def predict(
        self,
        flow_vecs: np.ndarray,   # (N, 12)
        batch_size: int = 256,
    ) -> np.ndarray:
        """Predict severity labels for test flow vectors."""
        assert self._fitted, "Call fit() first."
        contexts = self._encode_all(flow_vecs, batch_size)
        scaled = self.scaler.transform(contexts)

        preds = []
        center_ids = list(self._scaled_centers.keys())
        if not center_ids:
            return np.full(len(flow_vecs), self._default_label, dtype=np.int64)

        centers = np.stack([self._scaled_centers[k] for k in center_ids])

        for vec in scaled:
            dists = np.linalg.norm(centers - vec, axis=1)
            nearest = center_ids[int(np.argmin(dists))]
            preds.append(self.cluster_labels.get(nearest, self._default_label))

        return np.array(preds, dtype=np.int64)

    def _encode_all(
        self, flow_vecs: np.ndarray, batch_size: int
    ) -> np.ndarray:
        """Encode N flow vectors by duplicating as a window of size 1 for LSTM."""
        self.encoder.eval()
        all_ctx = []
        with torch.no_grad():
            for i in range(0, len(flow_vecs), batch_size):
                batch = flow_vecs[i: i + batch_size]
                t = torch.tensor(batch, dtype=torch.float32, device=self.device)
                # Expand to (B, 1, 12) → encoder handles single-step sequence
                t = t.unsqueeze(1)
                ctx = self.encoder(t).cpu().numpy()
                all_ctx.append(ctx)
        return np.concatenate(all_ctx, axis=0)
