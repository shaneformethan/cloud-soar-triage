"""
Per-modality encoders (Section III-C).

Three encoders, each projecting their input to a d-dimensional embedding:

  IAMEncoder   : Transformer over event sequences with [CLS] token + positional
                 encoding → d-dim context vector (following LogBERT [b12]).
  FlowEncoder  : Feed-forward network with batch normalization,
                 mapping 12-dim flow feature vector → d-dim.
  TIEncoder    : Bottleneck MLP with batch normalization and ReLU,
                 projecting 16-dim TI vector → hidden_dim → d-dim.

Default d = 256, h (heads) = 8, IAM layers = 4. All architectural
hyperparameters are settable so the training script can sweep candidates
from Table I without rewriting this file.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# IAM Encoder  (Transformer + CLS token + positional encoding)
# ─────────────────────────────────────────────────────────────────────────────

class IAMEncoder(nn.Module):
    """
    Transformer encoder for IAM event sequences.

    Input:
        token_ids  : (B, T, E)  int32 token feature matrix from iam_loader
        padding_mask: (B, T)    bool, True where positions are padding

    Output:
        (B, d)  — [CLS] token representation after n_layers of self-attention.

    Architecture (Section III-C):
      1. Integer feature lookup → d-dim via per-feature embedding tables
         (action_type, resource_type, principal_role, src_ip_region, baseline_flag)
         → summed to single token embedding
      2. Prepend learnable [CLS] token
      3. Add sinusoidal positional encoding
      4. n_layers of Transformer encoder (post-LN, h heads, 4d FFN width)
      5. Return [CLS] position output
    """

    IAM_VOCAB_SIZES = [256, 10, 6, 5, 2]  # per feature column vocab sizes

    def __init__(
        self,
        d: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        max_seq_len: int = 64,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.d = d
        self.use_pe = use_positional_encoding

        # Per-feature embedding tables (feature values are integer indices)
        self.feature_embeds = nn.ModuleList([
            nn.Embedding(vocab_size + 1, d, padding_idx=0)
            for vocab_size in self.IAM_VOCAB_SIZES
        ])
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls_token, std=0.02)

        # Sinusoidal positional encoding (max_seq_len + 1 for CLS)
        if use_positional_encoding:
            self.register_buffer(
                "pos_enc",
                self._sinusoidal_pe(max_seq_len + 1, d),
            )
        else:
            self.pos_enc = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=4 * d,
            dropout=dropout,
            batch_first=True,
            norm_first=False,   # post-LN (original Transformer convention)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        token_ids: torch.Tensor,   # (B, T, 5) int
        padding_mask: torch.Tensor | None = None,  # (B, T) bool
    ) -> torch.Tensor:
        B, T, _ = token_ids.shape

        # Sum per-feature embeddings → (B, T, d)
        x = sum(
            self.feature_embeds[i](token_ids[:, :, i].clamp(0))
            for i in range(token_ids.shape[2])
        )

        # Prepend [CLS] token → (B, T+1, d)
        cls = self.cls_token.expand(B, 1, self.d)
        x = torch.cat([cls, x], dim=1)

        # Positional encoding
        if self.use_pe and self.pos_enc is not None:
            x = x + self.pos_enc[:, : T + 1, :]

        # Extend padding mask to include CLS position (never masked)
        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        else:
            src_key_padding_mask = None

        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        out = self.norm(out)
        return out[:, 0, :]  # [CLS] position → (B, d)

    @staticmethod
    def _sinusoidal_pe(max_len: int, d: int) -> torch.Tensor:
        pe = torch.zeros(1, max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float) * (-math.log(10000.0) / d)
        )
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div[: d // 2])
        return pe


# ─────────────────────────────────────────────────────────────────────────────
# Flow Encoder  (FFN + Batch Norm)
# ─────────────────────────────────────────────────────────────────────────────

class FlowEncoder(nn.Module):
    """
    Feed-forward encoder for 12-dim network flow feature vectors.

    Section III-C: "The network flow encoder is a feed-forward network with
    batch normalization, mapping the 12-dimensional flow feature vector to
    the same d-dimensional space."

    Architecture: Linear(12→d) → BN → ReLU → [Linear(d→d) → BN → ReLU] × (n_layers-1)
    """

    def __init__(self, d: int = 256, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 12
        for i in range(n_layers):
            out_dim = d
            layers += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, flow_vec: torch.Tensor) -> torch.Tensor:
        """flow_vec: (B, 12) → (B, d)"""
        return self.net(flow_vec)


# ─────────────────────────────────────────────────────────────────────────────
# TI Encoder  (Bottleneck MLP + Batch Norm)
# ─────────────────────────────────────────────────────────────────────────────

class TIEncoder(nn.Module):
    """
    Bottleneck MLP encoder for 16-dim threat intelligence feature vectors.

    Section III-C: "The threat intelligence encoder is a bottleneck MLP with
    batch normalization and ReLU activations, projecting the 16-dimensional
    TI vector through an intermediate hidden layer to d dimensions."

    Architecture: Linear(16→hidden) → BN → ReLU → Linear(hidden→d) → BN → ReLU
    n_layers controls how many intermediate hidden layers to use.
    """

    def __init__(
        self,
        d: int = 256,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 16
        dims = [hidden_dim] * max(n_layers - 1, 1) + [d]
        for out_dim in dims:
            layers += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, ti_vec: torch.Tensor) -> torch.Tensor:
        """ti_vec: (B, 16) → (B, d)"""
        return self.net(ti_vec)
