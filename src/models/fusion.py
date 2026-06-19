"""
Cross-modal attention fusion module (Section III-C).

Architecture (exact from draft):
  Given E_IAM, E_flow, E_TI ∈ ℝ^d (from per-modality encoders):
    1. Construct input sequence X = [z_fuse; E_IAM; E_flow; E_TI] ∈ ℝ^{4×d}
       where z_fuse ∈ ℝ^d is a learnable fusion token (like [CLS]).
    2. Apply h-head self-attention (cross-modal attention) over X:
         Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
       with Q = X W_Q, K = X W_K, V = X W_V, W ∈ ℝ^{d × d_k}, d_k = d/h.
    3. Output is the z_fuse position vector → passed to classification head.
    4. Classification head: 2-layer MLP → 4-class logits.

Alternative fusion strategies (for ablation variants in Table III):
  - Arch-E: Concatenation of [E_IAM; E_flow; E_TI] → 2-layer MLP → logits
  - Arch-F: Average pooling over [E_IAM, E_flow, E_TI] → 2-layer MLP → logits

Modality-ablation variant (Section III-F / IV, "IAM-prioritized"):
  - IAMPriorityFusion: replaces the learnable z_fuse query token with E_IAM
    itself as the attention query over [E_IAM; E_flow; E_TI]. IAM is fixed
    as the priority modality rather than learning a separate fusion token.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Learnable z_fuse fusion (proposed model)
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalFusion(nn.Module):
    """
    8-head self-attention over [z_fuse; E_IAM; E_flow; E_TI].

    The output at the z_fuse position aggregates cross-modal context
    and is passed to the classification head.

    Parameters
    ----------
    d       : embedding dimension (default 256)
    n_heads : attention heads (default 8); d must be divisible by n_heads
    n_cls_layers : number of classification head layers (1 or 2)
    n_classes    : severity classes (4)
    dropout      : dropout in attention
    """

    def __init__(
        self,
        d: int = 256,
        n_heads: int = 8,
        n_cls_layers: int = 2,
        n_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d % n_heads == 0, f"d={d} must be divisible by n_heads={n_heads}"
        self.d = d

        # Learnable fusion token (z_fuse).
        self.z_fuse = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.z_fuse, std=0.02)

        # Modality-type embeddings for the 4 sequence positions
        # [z_fuse; E_IAM; E_flow; E_TI]. Without these the self-attention is
        # permutation-equivariant and cannot tell which token is which modality,
        # which crippled the fused representation; they let the attention
        # *select* the anomalous modality.
        self.type_emb = nn.Parameter(torch.zeros(1, 4, d))
        nn.init.normal_(self.type_emb, std=0.02)

        # Single self-attention block (8-head by default)
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d)

        # Classification head (1 or 2 layers)
        self.cls_head = _build_cls_head(d, n_cls_layers, n_classes, dropout)

    def forward(
        self,
        e_iam: torch.Tensor,   # (B, d)
        e_flow: torch.Tensor,  # (B, d)
        e_ti: torch.Tensor,    # (B, d)
    ) -> torch.Tensor:
        """Returns logits (B, n_classes)."""
        B = e_iam.shape[0]

        # Build sequence (B, 4, d) and add modality-type embeddings.
        z = self.z_fuse.expand(B, 1, self.d)
        x = torch.cat(
            [z, e_iam.unsqueeze(1), e_flow.unsqueeze(1), e_ti.unsqueeze(1)], dim=1
        )
        x = x + self.type_emb

        # Self-attention (cross-modal) + residual.
        out, _ = self.attn(x, x, x)
        out = self.norm(out + x)

        # Extract z_fuse output → (B, d).
        return self.cls_head(out[:, 0, :])


# ─────────────────────────────────────────────────────────────────────────────
# Concatenation + MLP fusion (Arch-E ablation)
# ─────────────────────────────────────────────────────────────────────────────

class ConcatFusion(nn.Module):
    """
    Arch-E: Concatenates [E_IAM; E_flow; E_TI] → 2-layer MLP → logits.
    Input dimension to MLP: 3d.
    """

    def __init__(
        self,
        d: int = 256,
        n_cls_layers: int = 2,
        n_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Project 3d → d first, then classification head
        self.proj = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cls_head = _build_cls_head(d, n_cls_layers, n_classes, dropout)

    def forward(
        self,
        e_iam: torch.Tensor,
        e_flow: torch.Tensor,
        e_ti: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([e_iam, e_flow, e_ti], dim=-1)  # (B, 3d)
        fused = self.proj(x)                           # (B, d)
        return self.cls_head(fused)


# ─────────────────────────────────────────────────────────────────────────────
# Average pooling fusion (Arch-F ablation)
# ─────────────────────────────────────────────────────────────────────────────

class AvgPoolFusion(nn.Module):
    """
    Arch-F: Average pooling over modality embeddings → MLP → logits.
    """

    def __init__(
        self,
        d: int = 256,
        n_cls_layers: int = 2,
        n_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cls_head = _build_cls_head(d, n_cls_layers, n_classes, dropout)

    def forward(
        self,
        e_iam: torch.Tensor,
        e_flow: torch.Tensor,
        e_ti: torch.Tensor,
    ) -> torch.Tensor:
        fused = (e_iam + e_flow + e_ti) / 3.0  # (B, d)
        return self.cls_head(fused)


# ─────────────────────────────────────────────────────────────────────────────
# IAM-prioritized fusion (modality ablation variant #7)
# ─────────────────────────────────────────────────────────────────────────────

class IAMPriorityFusion(nn.Module):
    """
    Modality-ablation variant: replaces the learnable z_fuse token with the
    IAM embedding E_IAM as a fixed attention query over
    [E_IAM; E_flow; E_TI]. Unlike CrossModalFusion, no separate fusion
    token is learned — IAM is hard-wired as the priority modality whose
    representation is updated by attending to the flow and TI embeddings.
    """

    def __init__(
        self,
        d: int = 256,
        n_heads: int = 8,
        n_cls_layers: int = 2,
        n_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d % n_heads == 0, f"d={d} must be divisible by n_heads={n_heads}"
        self.d = d

        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d)
        self.cls_head = _build_cls_head(d, n_cls_layers, n_classes, dropout)

    def forward(
        self,
        e_iam: torch.Tensor,   # (B, d)
        e_flow: torch.Tensor,  # (B, d)
        e_ti: torch.Tensor,    # (B, d)
    ) -> torch.Tensor:
        """Returns logits (B, n_classes)."""
        # Sequence over which IAM attends: [E_IAM; E_flow; E_TI]
        kv = torch.stack([e_iam, e_flow, e_ti], dim=1)   # (B, 3, d)
        q = e_iam.unsqueeze(1)                            # (B, 1, d) — IAM as fixed query

        out, _ = self.attn(q, kv, kv)                     # (B, 1, d)
        out = self.norm(out.squeeze(1) + e_iam)           # residual onto IAM embedding

        return self.cls_head(out)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_cls_head(
    d: int, n_layers: int, n_classes: int, dropout: float
) -> nn.Sequential:
    """Build a 1- or 2-layer classification head."""
    if n_layers == 1:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_classes))
    # 2-layer
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(d, d // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(d // 2, n_classes),
    )
