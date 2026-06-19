"""
Architectural ablation variants (Table III in draft, Section III-E).

All variants use the same FlowEncoder and TIEncoder as the proposed model.
Only the IAM encoder and/or fusion strategy changes.

Variants:
  Arch-A : 2-layer BiLSTM IAM encoder   + z_fuse fusion
  Arch-B : 2-layer BiGRU  IAM encoder   + z_fuse fusion
  Arch-C : 1D-CNN (3 layers, kernel 3)  + z_fuse fusion
  Arch-D : Transformer (no positional encoding) + z_fuse fusion
  Arch-E : 4-layer Transformer + Concatenation + 2-layer MLP
  Arch-F : 4-layer Transformer + Average pooling

Modality ablation variants share the same model class; unused modality
embeddings are zeroed out via forward_with_mask() in MultiModalTriageModel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders import FlowEncoder, TIEncoder
from src.models.fusion import CrossModalFusion, ConcatFusion, AvgPoolFusion
from src.models.model import ModelConfig, MultiModalTriageModel, build_padding_mask


# ─────────────────────────────────────────────────────────────────────────────
# Alternative IAM encoders
# ─────────────────────────────────────────────────────────────────────────────

class BiLSTMIAMEncoder(nn.Module):
    """
    Arch-A: Bidirectional LSTM over IAM event sequences.
    Output: concatenation of final forward/backward hidden states → d-dim.
    """
    IAM_VOCAB_SIZES = [256, 10, 6, 5, 2]

    def __init__(self, d: int = 256, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.feature_embeds = nn.ModuleList([
            nn.Embedding(vs + 1, d // 2, padding_idx=0)
            for vs in self.IAM_VOCAB_SIZES
        ])
        hidden = d // 2  # BiLSTM: hidden × 2 directions = d
        self.lstm = nn.LSTM(
            input_size=d // 2,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(
        self,
        token_ids: torch.Tensor,         # (B, T, 5)
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Sum per-feature embeddings → (B, T, d//2)
        x = sum(
            self.feature_embeds[i](token_ids[:, :, i].clamp(0))
            for i in range(token_ids.shape[2])
        )
        out, (h_n, _) = self.lstm(x)
        # Concatenate final forward and backward hidden states → (B, d)
        h_fwd = h_n[-2]  # last layer forward
        h_bwd = h_n[-1]  # last layer backward
        return torch.cat([h_fwd, h_bwd], dim=-1)  # (B, d)


class BiGRUIAMEncoder(nn.Module):
    """Arch-B: Bidirectional GRU over IAM event sequences."""
    IAM_VOCAB_SIZES = [256, 10, 6, 5, 2]

    def __init__(self, d: int = 256, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.feature_embeds = nn.ModuleList([
            nn.Embedding(vs + 1, d // 2, padding_idx=0)
            for vs in self.IAM_VOCAB_SIZES
        ])
        hidden = d // 2
        self.gru = nn.GRU(
            input_size=d // 2,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = sum(
            self.feature_embeds[i](token_ids[:, :, i].clamp(0))
            for i in range(token_ids.shape[2])
        )
        _, h_n = self.gru(x)
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        return torch.cat([h_fwd, h_bwd], dim=-1)


class CNNIAMEncoder(nn.Module):
    """
    Arch-C: 1D-CNN with 3 layers, kernel size 3, over IAM event sequences.
    Global max-pool over time → d-dim vector.
    """
    IAM_VOCAB_SIZES = [256, 10, 6, 5, 2]

    def __init__(self, d: int = 256, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.feature_embeds = nn.ModuleList([
            nn.Embedding(vs + 1, d, padding_idx=0)
            for vs in self.IAM_VOCAB_SIZES
        ])
        layers = []
        in_ch = d
        for i in range(n_layers):
            layers += [
                nn.Conv1d(in_ch, d, kernel_size=3, padding=1),
                nn.BatchNorm1d(d),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = d
        self.cnn = nn.Sequential(*layers)

    def forward(
        self,
        token_ids: torch.Tensor,   # (B, T, 5)
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = sum(
            self.feature_embeds[i](token_ids[:, :, i].clamp(0))
            for i in range(token_ids.shape[2])
        )                              # (B, T, d)
        x = x.permute(0, 2, 1)        # (B, d, T) for Conv1d
        x = self.cnn(x)                # (B, d, T)
        return x.max(dim=-1).values    # global max pool → (B, d)


# ─────────────────────────────────────────────────────────────────────────────
# Variant model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class AblationModel(nn.Module):
    """
    Generic wrapper for ablation variants.

    Selects IAM encoder and fusion strategy based on variant name.
    Flow and TI encoders are always the same as the proposed model.

    Parameters
    ----------
    variant : str
        One of: "arch_a", "arch_b", "arch_c", "arch_d", "arch_e", "arch_f"
    d, dropout, n_classes : shared hyperparameters
    """

    VARIANTS = {
        "arch_a": ("bilstm",  "zfuse"),
        "arch_b": ("bigru",   "zfuse"),
        "arch_c": ("cnn",     "zfuse"),
        "arch_d": ("nope",    "zfuse"),    # no positional encoding
        "arch_e": ("transformer", "concat"),
        "arch_f": ("transformer", "avgpool"),
    }

    def __init__(
        self,
        variant: str,
        d: int = 256,
        dropout: float = 0.1,
        n_classes: int = 4,
        n_heads: int = 8,
        cls_n_layers: int = 2,
        iam_n_layers_bilstm_gru: int = 2,
        iam_n_layers_transformer: int = 4,
    ):
        super().__init__()
        assert variant in self.VARIANTS, f"Unknown variant: {variant!r}"
        iam_arch, fusion_strat = self.VARIANTS[variant]

        # IAM encoder
        if iam_arch == "bilstm":
            self.iam_enc = BiLSTMIAMEncoder(d=d, n_layers=iam_n_layers_bilstm_gru, dropout=dropout)
        elif iam_arch == "bigru":
            self.iam_enc = BiGRUIAMEncoder(d=d, n_layers=iam_n_layers_bilstm_gru, dropout=dropout)
        elif iam_arch == "cnn":
            self.iam_enc = CNNIAMEncoder(d=d, dropout=dropout)
        elif iam_arch == "nope":
            # Transformer without positional encoding
            from src.models.encoders import IAMEncoder
            self.iam_enc = IAMEncoder(
                d=d, n_heads=n_heads, n_layers=iam_n_layers_transformer,
                dropout=dropout, use_positional_encoding=False
            )
        else:  # "transformer"
            from src.models.encoders import IAMEncoder
            self.iam_enc = IAMEncoder(
                d=d, n_heads=n_heads, n_layers=iam_n_layers_transformer,
                dropout=dropout, use_positional_encoding=True
            )

        self.flow_enc = FlowEncoder(d=d, n_layers=2, dropout=dropout)
        self.ti_enc   = TIEncoder(d=d, n_layers=2, dropout=dropout)

        if fusion_strat == "zfuse":
            self.fusion = CrossModalFusion(d=d, n_heads=n_heads, n_cls_layers=cls_n_layers,
                                           n_classes=n_classes, dropout=dropout)
        elif fusion_strat == "concat":
            self.fusion = ConcatFusion(d=d, n_cls_layers=cls_n_layers,
                                       n_classes=n_classes, dropout=dropout)
        elif fusion_strat == "avgpool":
            self.fusion = AvgPoolFusion(d=d, n_cls_layers=cls_n_layers,
                                        n_classes=n_classes, dropout=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        iam_tokens: torch.Tensor,
        iam_lengths: torch.Tensor,
        flow_vec: torch.Tensor,
        ti_vec: torch.Tensor,
    ) -> torch.Tensor:
        padding_mask = build_padding_mask(iam_lengths, iam_tokens.shape[1])
        e_iam  = self.iam_enc(iam_tokens, padding_mask)
        e_flow = self.flow_enc(flow_vec)
        e_ti   = self.ti_enc(ti_vec)
        return self.fusion(e_iam, e_flow, e_ti)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
