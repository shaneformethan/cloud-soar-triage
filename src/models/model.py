"""
Full multi-modal triage model (Section III).

Wires together:
  IAMEncoder + FlowEncoder + TIEncoder → CrossModalFusion → logits

Also provides:
  - forward_with_mask()  for modality ablation (zero out inactive modalities)
  - predict_proba()      for confidence-score output (post-calibration)
  - build_padding_mask() helper for variable-length IAM sequences
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders import IAMEncoder, FlowEncoder, TIEncoder
from src.models.fusion import CrossModalFusion, ConcatFusion, AvgPoolFusion, IAMPriorityFusion


# ─────────────────────────────────────────────────────────────────────────────
# Model configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    # Shared
    d: int = 256
    dropout: float = 0.1
    n_classes: int = 4

    # IAM encoder
    iam_n_layers: int = 4
    iam_n_heads: int = 8
    iam_max_seq_len: int = 64
    iam_use_positional_encoding: bool = True

    # Flow encoder
    flow_n_layers: int = 2

    # TI encoder
    ti_n_layers: int = 2
    ti_hidden_dim: int = 64

    # Fusion
    fusion_n_heads: int = 8
    fusion_strategy: str = "zfuse"   # "zfuse" | "concat" | "avgpool" | "iam_priority"
    cls_n_layers: int = 2


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class MultiModalTriageModel(nn.Module):
    """
    Proposed multi-modal cloud security incident triage model.

    Inputs (per batch):
        iam_tokens   : (B, 64, 5) int32 — IAM event sequence (zero-padded)
        iam_lengths  : (B,)       int   — actual sequence lengths (1–64)
        flow_vec     : (B, 12)    float32
        ti_vec       : (B, 16)    float32

    Output:
        logits : (B, 4) — pre-softmax severity class scores
    """

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg

        self.iam_enc = IAMEncoder(
            d=cfg.d,
            n_heads=cfg.iam_n_heads,
            n_layers=cfg.iam_n_layers,
            max_seq_len=cfg.iam_max_seq_len,
            dropout=cfg.dropout,
            use_positional_encoding=cfg.iam_use_positional_encoding,
        )
        self.flow_enc = FlowEncoder(
            d=cfg.d,
            n_layers=cfg.flow_n_layers,
            dropout=cfg.dropout,
        )
        self.ti_enc = TIEncoder(
            d=cfg.d,
            hidden_dim=cfg.ti_hidden_dim,
            n_layers=cfg.ti_n_layers,
            dropout=cfg.dropout,
        )

        if cfg.fusion_strategy == "zfuse":
            self.fusion = CrossModalFusion(
                d=cfg.d,
                n_heads=cfg.fusion_n_heads,
                n_cls_layers=cfg.cls_n_layers,
                n_classes=cfg.n_classes,
                dropout=cfg.dropout,
            )
        elif cfg.fusion_strategy == "concat":
            self.fusion = ConcatFusion(
                d=cfg.d,
                n_cls_layers=cfg.cls_n_layers,
                n_classes=cfg.n_classes,
                dropout=cfg.dropout,
            )
        elif cfg.fusion_strategy == "avgpool":
            self.fusion = AvgPoolFusion(
                d=cfg.d,
                n_cls_layers=cfg.cls_n_layers,
                n_classes=cfg.n_classes,
                dropout=cfg.dropout,
            )
        elif cfg.fusion_strategy == "iam_priority":
            self.fusion = IAMPriorityFusion(
                d=cfg.d,
                n_heads=cfg.fusion_n_heads,
                n_cls_layers=cfg.cls_n_layers,
                n_classes=cfg.n_classes,
                dropout=cfg.dropout,
            )
        else:
            raise ValueError(f"Unknown fusion_strategy: {cfg.fusion_strategy!r}")

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization (Section III-C)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        iam_tokens: torch.Tensor,       # (B, 64, 5) int
        iam_lengths: torch.Tensor,      # (B,)       int
        flow_vec: torch.Tensor,         # (B, 12)    float
        ti_vec: torch.Tensor,           # (B, 16)    float
    ) -> torch.Tensor:
        padding_mask = build_padding_mask(iam_lengths, iam_tokens.shape[1])
        e_iam  = self.iam_enc(iam_tokens, padding_mask)   # (B, d)
        e_flow = self.flow_enc(flow_vec)                   # (B, d)
        e_ti   = self.ti_enc(ti_vec)                       # (B, d)
        return self.fusion(e_iam, e_flow, e_ti)            # (B, 4)

    def forward_with_mask(
        self,
        iam_tokens: torch.Tensor,
        iam_lengths: torch.Tensor,
        flow_vec: torch.Tensor,
        ti_vec: torch.Tensor,
        active_modalities: set[str],   # subset of {"iam", "flow", "ti"}
    ) -> torch.Tensor:
        """
        For modality ablation: zero out embeddings of inactive modalities.
        The inactive encoder still runs (for computational simplicity) but
        its output is replaced with zeros before fusion.
        """
        padding_mask = build_padding_mask(iam_lengths, iam_tokens.shape[1])
        e_iam  = self.iam_enc(iam_tokens, padding_mask)
        e_flow = self.flow_enc(flow_vec)
        e_ti   = self.ti_enc(ti_vec)

        if "iam" not in active_modalities:
            e_iam = torch.zeros_like(e_iam)
        if "flow" not in active_modalities:
            e_flow = torch.zeros_like(e_flow)
        if "ti" not in active_modalities:
            e_ti = torch.zeros_like(e_ti)

        return self.fusion(e_iam, e_flow, e_ti)

    def predict_proba(
        self,
        iam_tokens: torch.Tensor,
        iam_lengths: torch.Tensor,
        flow_vec: torch.Tensor,
        ti_vec: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Return calibrated softmax probabilities (B, 4).
        temperature=1.0 → uncalibrated; use TemperatureScaler for fitted T.
        """
        logits = self.forward(iam_tokens, iam_lengths, flow_vec, ti_vec)
        return F.softmax(logits / temperature, dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Modality-ablation wrapper (Section III-F: 6 non-trivial modality subsets)
# ─────────────────────────────────────────────────────────────────────────────

MODALITY_SUBSETS: dict[str, set[str]] = {
    "iam_only":   {"iam"},
    "flow_only":  {"flow"},
    "ti_only":    {"ti"},
    "iam_flow":   {"iam", "flow"},
    "iam_ti":     {"iam", "ti"},
    "flow_ti":    {"flow", "ti"},
}


class MaskedModalityModel(nn.Module):
    """
    Thin wrapper around MultiModalTriageModel that fixes which modalities
    are active for modality-ablation experiments (Section III-F). Wraps a
    full MultiModalTriageModel and always calls forward_with_mask() with a
    fixed active_modalities set.
    """

    def __init__(self, active_modalities: set[str], cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.model = MultiModalTriageModel(cfg)
        self.active_modalities = set(active_modalities)
        self.cfg = cfg

    def forward(
        self,
        iam_tokens: torch.Tensor,
        iam_lengths: torch.Tensor,
        flow_vec: torch.Tensor,
        ti_vec: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.forward_with_mask(
            iam_tokens, iam_lengths, flow_vec, ti_vec, self.active_modalities
        )

    def predict_proba(
        self,
        iam_tokens: torch.Tensor,
        iam_lengths: torch.Tensor,
        flow_vec: torch.Tensor,
        ti_vec: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        logits = self.forward(iam_tokens, iam_lengths, flow_vec, ti_vec)
        return F.softmax(logits / temperature, dim=-1)

    def count_parameters(self) -> int:
        return self.model.count_parameters()


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Build a (B, T) boolean padding mask from sequence lengths.

    Returns True at positions >= length (i.e. padding positions, to be
    ignored by the Transformer's src_key_padding_mask).
    """
    device = lengths.device
    idx = torch.arange(max_len, device=device).unsqueeze(0)   # (1, T)
    return idx >= lengths.unsqueeze(1)                          # (B, T)


def load_model(path: str, device: str = "cpu") -> tuple[nn.Module, dict]:
    """
    Load a model checkpoint saved by train.py's _save_checkpoint().

    Reconstructs the correct model class (MultiModalTriageModel, or
    MaskedModalityModel if the checkpoint stores "active_modalities") from
    the stored ModelConfig, loads the state dict, moves it to `device`, and
    sets eval() mode.

    Returns
    -------
    (model, extra) where `extra` is the rest of the checkpoint dict
    (epoch, val_f1, train_cfg, active_modalities, ...).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = checkpoint.get("config") or {}
    cfg = ModelConfig(**cfg_dict) if cfg_dict else ModelConfig()

    if "active_modalities" in checkpoint:
        model: nn.Module = MaskedModalityModel(set(checkpoint["active_modalities"]), cfg)
    else:
        model = MultiModalTriageModel(cfg)

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    extra = {k: v for k, v in checkpoint.items() if k not in ("model_state", "config")}
    return model, extra
