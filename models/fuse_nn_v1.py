import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Dict, Optional, Tuple

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()
        self.bn  = nn.LayerNorm(out_features)
        self.dp  = nn.Dropout(dropout)
    def forward(self, x):
        x = self.act(self.lin(x))
        x = self.bn(x)
        x = self.dp(x)
        return x

class LowRankEncoder(nn.Module):
    """
    Low Rank：C -> r -> H
    Share Weight for Per-Channel Compression
    """
    def __init__(self, num_classes, rank=512, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(num_classes, rank, bias=True)
        self.fc2 = nn.Linear(rank, hidden_dim, bias=True)
        self.act = nn.ReLU()
        self.dp  = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C)
        x = self.dp(self.act(self.fc1(x)))
        x = self.dp(self.act(self.fc2(x)))
        return x  # (B, H)

class ChannelGatedFusion(nn.Module):
    """
    Gated Fusion for 3 Channels (B,3,H), output (B, H)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, h3):
        # h3: (B,3,H)
        w = self.gate(h3).squeeze(-1)          # (B,3)
        w = F.softmax(w, dim=1)                # (B,3)
        fused = (h3 * w.unsqueeze(-1)).sum(1)  # (B,H)
        return fused, w

class FusionProjectThenFuse(nn.Module):
    """
    Input: (B,3,C)
    First per-channel project to H, then fuse, then decode back to C
    Output: (B,C) logits
    """
    def __init__(self, num_classes, rank=512, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.enc = LowRankEncoder(num_classes, rank=rank, hidden_dim=hidden_dim, dropout=dropout)
        self.fuse = ChannelGatedFusion(hidden_dim)

        self.trunk = nn.Sequential(
            MLPBlock(hidden_dim, hidden_dim, dropout=dropout),
            Residual(MLPBlock(hidden_dim, hidden_dim, dropout=dropout)),
        )
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B,3,C)
        h_list = []
        for k in range(3):
            h_list.append(self.enc(x[:, k, :]))    # (B,H)
        h3 = torch.stack(h_list, dim=1)            # (B,3,H)

        fused, w = self.fuse(h3)                   # (B,H), (B,3)
        z = self.trunk(fused)                      # (B,H)
        logits = self.out(z)                       # (B,C)
        return logits, w


class PreNorm(nn.Module):
    """
    Pre-LN wrapper: x -> LN -> fn -> residual
    """
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    Simple MLP block for Transformer.
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# class SelfAttentionBlock(nn.Module):
#     """
#     Self-attention + FFN with Pre-LN.
#     """
#     def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
#         super().__init__()
#         self.attn = PreNorm(dim, nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True))
#         self.ffn = PreNorm(dim, FeedForward(dim, ffn_dim, dropout=dropout))

#     def forward(self, x):
#         # MultiheadAttention returns (out, attn_weights)
#         attn_out, attn_w = self.attn(x, x=x, x2=x, need_weights=True) \
#             if False else self._attn_forward(x)  # keep linter calm
#         x = x + attn_out
#         x = x + self.ffn(x)
#         return x, attn_w

#     def _attn_forward(self, x):
#         out, w = self.attn.fn(self.attn.norm(x), self.attn.norm(x), self.attn.norm(x), need_weights=True)
#         return out, w


# class CrossAttentionBlock(nn.Module):
#     """
#     Cross-attention: queries attend to context tokens (keys/values).
#     Pre-LN on both query and context.
#     """
#     def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
#         super().__init__()
#         self.q_norm = nn.LayerNorm(dim)
#         self.kv_norm = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

#     def forward(self, q, kv, need_weights: bool = True):
#         qn = self.q_norm(q)
#         kvn = self.kv_norm(kv)
#         out, w = self.attn(qn, kvn, kvn, need_weights=need_weights)
#         return out, w


class FusionTransformerLatent(nn.Module):
    """
    Fusion model for union-space predictions.

    Input:
      x: (B, 3, C) where 3 channels correspond to [zhao, esm2_v3, profunsom]
         Values are typically probabilities in [0,1] (recommended), or calibrated scores.

    Pipeline (practical version of your idea):
      (B,3,C) -> Linear(C->H) -> (B,3,H)
      + channel embeddings
      latent tokens (K) -> CrossAttn(latent <- channel_tokens) -> (B,K,H)
      -> (optional) self-attn blocks on latent tokens
      -> mean pool over K -> (B,H)
      -> Linear(H->C) -> logits (B,C)

    Returns:
      logits: (B,C)
      aux: dict with attention weights for analysis
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 512,
        latent_tokens: int = 8,
        num_heads: int = 8,
        depth: int = 2,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        share_channel_proj: bool = True,
        init_output_bias: float = -4.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.latent_tokens = latent_tokens
        self.num_heads = num_heads
        self.depth = depth
        self.ffn_dim = ffn_dim or (hidden_dim * 4)
        self.dropout = dropout

        # Project GO-dim scores to hidden dim.
        # If share_channel_proj=True, all 3 channels share the same projection weights.
        if share_channel_proj:
            self.in_proj = nn.Linear(num_classes, hidden_dim, bias=True)
        else:
            self.in_proj = nn.ModuleList([nn.Linear(num_classes, hidden_dim, bias=True) for _ in range(3)])

        self.in_ln = nn.LayerNorm(hidden_dim)
        self.in_drop = nn.Dropout(dropout)

        # Channel embedding to indicate "which base model"
        self.channel_embed = nn.Parameter(torch.zeros(1, 3, hidden_dim))
        nn.init.trunc_normal_(self.channel_embed, std=0.02)

        # Learnable latent tokens (K,H)
        self.latent = nn.Parameter(torch.zeros(1, latent_tokens, hidden_dim))
        nn.init.trunc_normal_(self.latent, std=0.02)

        # Cross-attention: latent queries attend to 3 channel tokens
        self.cross_attn = CrossAttentionBlock(hidden_dim, num_heads, dropout=dropout)
        self.cross_drop = nn.Dropout(dropout)

        # Self-attention blocks on latent tokens
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(hidden_dim, num_heads, self.ffn_dim, dropout=dropout) for _ in range(depth)]
        )

        # Pool & output head
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes, bias=True)

        # Conservative bias init to stabilize multi-label imbalance (small initial positive rate)
        if init_output_bias is not None:
            nn.init.constant_(self.out.bias, float(init_output_bias))

        # Optional: small weight init
        nn.init.trunc_normal_(self.out.weight, std=0.02)

    def _project_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,C) -> (B,3,H)
        """
        if isinstance(self.in_proj, nn.ModuleList):
            # Per-channel projection
            xs = []
            for i in range(3):
                xs.append(self.in_proj[i](x[:, i, :]))
            h = torch.stack(xs, dim=1)
        else:
            h = self.in_proj(x)
        h = self.in_ln(h)
        h = self.in_drop(h)
        return h

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
          x: (B,3,C)

        Returns:
          logits: (B,C)
          aux: dict
        """
        b, ch, c = x.shape
        if ch != 3:
            raise ValueError(f"Expected channel dimension=3, got {ch}")
        if c != self.num_classes:
            raise ValueError(f"Expected num_classes={self.num_classes}, got C={c}")

        # (B,3,H)
        tokens = self._project_input(x)
        tokens = tokens + self.channel_embed  # channel identity

        # Latent tokens: (B,K,H)
        latent = self.latent.expand(b, -1, -1)

        # Cross-attention: latent <- tokens
        lat_out, cross_w = self.cross_attn(latent, tokens, need_weights=True)
        latent = latent + self.cross_drop(lat_out)

        # Latent self-attention
        self_w_list = []
        for blk in self.blocks:
            latent, w = blk(latent)
            self_w_list.append(w)

        # Pool: mean over K
        z = latent.mean(dim=1)  # (B,H)
        z = self.out_ln(z)

        logits = self.out(z)  # (B,C)

        aux = {
            # cross_w shape usually (B, K, 3) when averaged over heads by MHA implementation
            "cross_attn_weights": cross_w,
            # self-attn weights list (depth entries); may be None depending on PyTorch settings
            "self_attn_weights": self_w_list,
            "pooled": z,
        }
        return logits, aux

def safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert probabilities to logits safely: log(p/(1-p)).
    If input is not probabilities, you can disable this in forward via a flag.
    """
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention: queries attend to context tokens (keys/values).
    Pre-LN on both query and context.
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, q, kv, need_weights: bool = True):
        qn = self.q_norm(q)
        kvn = self.kv_norm(kv)
        out, w = self.attn(qn, kvn, kvn, need_weights=need_weights)
        return out, w


class SelfAttentionBlock(nn.Module):
    """
    Self-attention + FFN with Pre-LN.
    """
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        xn = self.norm1(x)
        attn_out, attn_w = self.attn(xn, xn, xn, need_weights=True)
        x = x + self.drop1(attn_out)

        x = x + self.ffn(self.norm2(x))
        return x, attn_w


class FusionLinearBaseTransformerResidual(nn.Module):
    """
    Fusion model:
      - Per-channel calibration in logit space
      - Linear base fusion (fixed or learnable weights)
      - Transformer produces residual delta logits
      - Final logits = base + res_scale * delta

    Input:
      x: (B, 3, C) scores; recommended probabilities in [0,1].
    Output:
      logits: (B, C)
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 512,
        latent_tokens: int = 8,
        num_heads: int = 8,
        depth: int = 2,
        dropout: float = 0.1,
        use_logit_input: bool = True,
        # base weights
        base_weights: Tuple[float, float, float] = (0.35, 0.30, 0.35),
        learn_base_weights: bool = False,
        # residual control
        res_scale: float = 0.5,
        init_output_bias: float = -4.0,
        use_term_gate: bool = True,
        gate_scale_init: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.latent_tokens = latent_tokens
        self.num_heads = num_heads
        self.depth = depth
        self.dropout = dropout
        self.use_logit_input = use_logit_input

        # ---- per-channel calibration: logit -> a * logit + b ----
        # shape (1,3,1) broadcast to (B,3,C)
        self.calib_a = nn.Parameter(torch.ones(1, 3, 1))
        self.calib_b = nn.Parameter(torch.zeros(1, 3, 1))

        # ---- base fusion weights ----
        self.learn_base_weights = learn_base_weights
        if learn_base_weights:
            # softmax-parameterized weights, initialized near provided base_weights
            w = torch.tensor(base_weights, dtype=torch.float32)
            w = (w / w.sum()).clamp(1e-6, 1.0)
            # inverse softmax approx: log(w)
            self.base_logits = nn.Parameter(torch.log(w).view(1, 3))
        else:
            w = torch.tensor(base_weights, dtype=torch.float32)
            w = (w / w.sum()).view(1, 3, 1)  # (1,3,1)
            self.register_buffer("base_w", w)

        # ---- per-term channel gating (3 x C) ----
        self.use_term_gate = bool(use_term_gate)

        # gate_scale controls how much term gate can deviate (>=0)
        self.gate_scale = nn.Parameter(torch.tensor(float(gate_scale_init)))

        # Initialize term gate logits to match provided base_weights for every term
        w0 = torch.tensor(base_weights, dtype=torch.float32)
        w0 = (w0 / w0.sum()).clamp(1e-6, 1.0)
        gate_init = torch.log(w0).view(1, 3, 1).repeat(1, 1, num_classes)  # (1,3,C)
        self.term_gate_logits = nn.Parameter(gate_init)


        # ---- transformer residual branch ----
        # Project each channel (C -> H) to form 3 tokens
        self.in_proj = nn.Linear(num_classes, hidden_dim, bias=True)
        self.in_ln = nn.LayerNorm(hidden_dim)
        self.in_drop = nn.Dropout(dropout)

        # Channel embeddings help the transformer differentiate sources
        self.channel_embed = nn.Parameter(torch.zeros(1, 3, hidden_dim))
        nn.init.trunc_normal_(self.channel_embed, std=0.02)

        # Latent tokens (K,H)
        self.latent = nn.Parameter(torch.zeros(1, latent_tokens, hidden_dim))
        nn.init.trunc_normal_(self.latent, std=0.02)

        self.cross_attn = CrossAttentionBlock(hidden_dim, num_heads, dropout=dropout)
        self.cross_drop = nn.Dropout(dropout)

        ffn_dim = hidden_dim * 4
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout=dropout) for _ in range(depth)]
        )

        self.pool_ln = nn.LayerNorm(hidden_dim)

        # Residual head: H -> C (delta logits)
        self.delta_head = nn.Linear(hidden_dim, num_classes, bias=True)
        nn.init.trunc_normal_(self.delta_head.weight, std=0.02)
        nn.init.zeros_(self.delta_head.bias)

        # Residual scaling (fixed scalar)
        self.res_scale = float(res_scale)

        # Optional conservative global bias (helps imbalance)
        # Add to base logits (not to delta). If you already have bias in base models, you can disable.
        self.global_bias = nn.Parameter(torch.full((1, num_classes), float(init_output_bias)))

    def _get_base_weights(self) -> torch.Tensor:
        if self.learn_base_weights:
            w = F.softmax(self.base_logits, dim=-1)  # (1,3)
            return w.view(1, 3, 1)                   # (1,3,1)
        return self.base_w  # (1,3,1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: (B,3,C)
        """
        b, ch, c = x.shape
        if ch != 3:
            raise ValueError(f"Expected x shape (B,3,C), got channel={ch}")
        if c != self.num_classes:
            raise ValueError(f"Expected C={self.num_classes}, got {c}")

        # ---- convert to logits if input is probability-like ----
        if self.use_logit_input:
            x_logit = safe_logit(x)          # (B,3,C)
        else:
            x_logit = x

        # ---- per-channel calibration ----
        x_cal = self.calib_a * x_logit + self.calib_b  # (B,3,C)

        # ---- base fusion ----
        w_global = self._get_base_weights()  # (1,3,1)

        if self.use_term_gate:
            # (1,3,C), softmax over channel dim=1
            w_term = F.softmax(self.term_gate_logits * self.gate_scale, dim=1)  # (1,3,C)
            w = w_term
        else:
            w = w_global.repeat(1, 1, self.num_classes)  # (1,3,C)

        base = (w * x_cal).sum(dim=1)      # (B,C)
        base = base + self.global_bias     # (B,C)

        # ---- transformer residual branch ----
        # Tokens from calibrated logits (use calibrated space for consistency)
        tokens = self.in_proj(x_cal)                   # (B,3,H)
        tokens = self.in_ln(tokens)
        tokens = self.in_drop(tokens)
        tokens = tokens + self.channel_embed

        latent = self.latent.expand(b, -1, -1)         # (B,K,H)
        lat_out, cross_w = self.cross_attn(latent, tokens, need_weights=True)
        latent = latent + self.cross_drop(lat_out)

        self_w = []
        for blk in self.blocks:
            latent, w_self = blk(latent)
            self_w.append(w_self)

        z = latent.mean(dim=1)                         # (B,H)
        z = self.pool_ln(z)

        delta = self.delta_head(z)                     # (B,C)

        # Final logits = base + scaled residual
        logits = base + (self.res_scale * delta)

        aux = {
            "base_logits": base,
            "delta_logits": delta,
            "base_weights_global": w_global.squeeze(-1).detach(),                 # (1,3)
            "term_gate_scale": self.gate_scale.detach(),                          # ()
            "term_gate_mean": w.mean(dim=-1).squeeze(0).detach(),                 # (3,) mean over terms
            "cross_attn_weights": cross_w,
            "self_attn_weights": self_w,
        }

        return logits, aux