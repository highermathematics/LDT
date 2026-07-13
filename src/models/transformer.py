"""LDT 去噪网络：论文 adaLN Transformer（与论文图 1b 一致）。

使用 pre-norm adaLN: scale × LN(x) + shift，无 gate 残差。
adaLN bias 初始化 γ≈1, β≈0 保证训练初期接近恒等映射。
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import DiffusionStepEmbedding, PositionalEmbedding


# ---------------------------------------------------------------------------
# DiT 风格 adaLN 调制
# ---------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN 调制: x * scale + shift（论文图 1b）。

    Args:
        x: [B, seq, d_model]（已做 LayerNorm）。
        shift, scale: [B, d_model]。
    """
    return x * scale.unsqueeze(1) + shift.unsqueeze(1)


class AdaLNModulation(nn.Module):
    """论文 adaLN：从扩散步嵌入回归 shift, scale。

    初始化: shift=0, scale≈1，初始时 adaLN 退化为 LN → 恒等映射。
    """

    def __init__(self, d_model: int, d_temb: int):
        super().__init__()
        self.proj = nn.Linear(d_temb, 2 * d_model)
        # 零权重 + bias: scale≈1, shift=0
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias[:d_model])    # scale ≈ 1
        nn.init.zeros_(self.proj.bias[d_model:])   # shift = 0

    def forward(self, t_emb: torch.Tensor) -> tuple:
        scale, shift = self.proj(t_emb).chunk(2, dim=-1)
        return shift, scale


# ---------------------------------------------------------------------------
# DiT 风格 Transformer 块
# ---------------------------------------------------------------------------

class DiTEncoderLayer(nn.Module):
    """论文编码器块：pre-norm + adaLN（无 gate 残差）。

    x = x + Attn( modulate(LN(x), shift, scale) )
    x = x + FFN( modulate(LN(x), shift', scale') )
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_temb: Optional[int] = None,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        # 自注意力部分
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.adaln1 = AdaLNModulation(d_model, d_temb)

        # FFN 部分
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.adaln2 = AdaLNModulation(d_model, d_temb)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Self-attention sublayer
        shift, scale = self.adaln1(t_emb)
        x_norm = self.norm1(x)
        x_mod = modulate(x_norm, shift, scale)
        attn_out, _ = self.self_attn(x_mod, x_mod, x_mod)
        x = x + self.dropout1(attn_out)

        # FFN sublayer
        shift, scale = self.adaln2(t_emb)
        x_norm = self.norm2(x)
        x_mod = modulate(x_norm, shift, scale)
        ff_out = self.linear2(F.gelu(self.linear1(x_mod)))
        x = x + self.dropout2(ff_out)

        return x


class DiTDecoderLayer(nn.Module):
    """论文解码器块：自注意力 + 交叉注意力 + FFN，全部 pre-norm + adaLN（无 gate）。

    x = x + SelfAttn( modulate(LN(x), shift_s, scale_s) )
    x = x + CrossAttn( modulate(LN(x), shift_c, scale_c), memory )
    x = x + FFN( modulate(LN(x), shift_f, scale_f) )
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_temb: Optional[int] = None,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        # 自注意力
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.adaln1 = AdaLNModulation(d_model, d_temb)

        # 交叉注意力
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        self.adaln2 = AdaLNModulation(d_model, d_temb)

        # FFN
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.adaln3 = AdaLNModulation(d_model, d_temb)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        # Self-attention
        shift, scale = self.adaln1(t_emb)
        x_norm = self.norm1(x)
        x_mod = modulate(x_norm, shift, scale)
        attn_out, _ = self.self_attn(x_mod, x_mod, x_mod)
        x = x + self.dropout1(attn_out)

        # Cross-attention
        shift, scale = self.adaln2(t_emb)
        x_norm = self.norm2(x)
        x_mod = modulate(x_norm, shift, scale)
        cross_out, _ = self.cross_attn(x_mod, memory, memory)
        x = x + self.dropout2(cross_out)

        # FFN
        shift, scale = self.adaln3(t_emb)
        x_norm = self.norm3(x)
        x_mod = modulate(x_norm, shift, scale)
        ff_out = self.linear2(F.gelu(self.linear1(x_mod)))
        x = x + self.dropout3(ff_out)

        return x


# ---------------------------------------------------------------------------
# 去噪 Transformer（x̂_θ 网络）
# ---------------------------------------------------------------------------

class DenoisingTransformer(nn.Module):
    """LDT 去噪网络 x̂_θ — DiT 风格 adaLN Transformer。

    与论文图 1b 完全一致：
    1. 输入投影（2 层 MLP）
    2. 位置 + 时间嵌入
    3. DiT adaLN Transformer 编码器-解码器
    4. 输出投影 → ẑ₀

    Args:
        d_data: 时间序列特征维度 d。
        d_latent: 潜在维度 m。
        d_model: Transformer 嵌入维度。
        n_heads: 注意力头数。
        n_layers: 编码器/解码器层数。
        history_len: 历史窗口长度 T。
        pred_len: 预测长度 t。
        max_diffusion_steps: 最大扩散步 K。
        dropout: Dropout 概率。
    """

    def __init__(
        self,
        d_data: int,
        d_latent: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        history_len: int = 96,
        pred_len: int = 24,
        max_diffusion_steps: int = 300,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.d_model = d_model
        self.history_len = history_len
        self.pred_len = pred_len

        # 输入投影
        self.history_proj = nn.Sequential(
            nn.Linear(d_data, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # z_k 嵌入投影（2 层 MLP）
        self.z_proj = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # 自条件 GLU 门控：value ⊙ σ(gate) + 后投影
        self.self_cond_val = nn.Linear(d_latent, d_model)
        self.self_cond_gate = nn.Linear(d_latent, d_model)
        self.self_cond_post = nn.Linear(d_model, d_model)

        # 位置嵌入
        self.pos_emb = PositionalEmbedding(
            max_len=history_len + pred_len, d_model=d_model
        )

        # 时间嵌入（论文：single MLP layer）
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
        )

        # 扩散步嵌入 → adaLN 条件
        self.diff_step_emb = DiffusionStepEmbedding(
            d_latent=d_latent, max_steps=max_diffusion_steps
        )
        d_temb = self.diff_step_emb.n  # n = 4m

        # DiT adaLN Transformer
        self.encoder_layers = nn.ModuleList([
            DiTEncoderLayer(d_model, n_heads, d_temb, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DiTDecoderLayer(d_model, n_heads, d_temb, dropout=dropout)
            for _ in range(n_layers)
        ])

        # 输出投影（最后一层零初始化：确保初始输出为零）
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_latent),
        )
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        z_k: torch.Tensor,
        history: torch.Tensor,
        z_self_cond: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """从加噪 z_k 预测干净潜在变量 ẑ₀。"""
        B, T, _ = history.shape
        device = history.device
        t = z_k.shape[1]

        # 1. 投影
        h_hist = self.history_proj(history)                      # [B, T, d_model]

        # z_k 嵌入 + 自条件 GLU 门控
        z_emb = self.z_proj(z_k)                                 # [B, t, d_model]
        sc_val = self.self_cond_val(z_self_cond)                 # [B, t, d_model]
        sc_gate = torch.sigmoid(self.self_cond_gate(z_self_cond))  # [B, t, d_model]
        sc_gated = sc_val * sc_gate                              # GLU: value ⊙ σ(gate)
        sc_out = self.self_cond_post(sc_gated)                   # [B, t, d_model]
        h_latent = z_emb + sc_out                                # [B, t, d_model]

        # 2. 位置 + 时间嵌入
        pos_all = self.pos_emb(T + t)                             # [1, T+t, d_model]
        h_hist = h_hist + pos_all[:, :T, :]
        h_latent = h_latent + pos_all[:, T:, :]

        time_scale = max(T + t - 1, 1)
        time_idx_hist = torch.arange(T, device=device).float().view(1, T, 1) / time_scale
        time_idx_latent = torch.arange(T, T + t, device=device).float().view(1, t, 1) / time_scale
        h_hist = h_hist + self.time_mlp(time_idx_hist)
        h_latent = h_latent + self.time_mlp(time_idx_latent)

        # 3. 扩散步嵌入
        t_emb = self.diff_step_emb(k)                             # [B, 4m]

        # 4. 编码器
        h_enc = h_hist
        for layer in self.encoder_layers:
            h_enc = layer(h_enc, t_emb)

        # 5. 解码器（交叉注意力到编码器输出）
        h_dec = h_latent
        for layer in self.decoder_layers:
            h_dec = layer(h_dec, h_enc, t_emb)

        # 6. 输出投影
        z_0_pred = self.output_proj(h_dec)                        # [B, t, m]
        return z_0_pred
