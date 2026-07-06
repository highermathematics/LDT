"""LDT 去噪网络的 adaLN Transformer。

实现非自回归去噪骨干网络（图 1b），包括：
- 由扩散步条件控制的自适应层归一化（adaLN）
- Transformer 编码器-解码器架构
- 自条件输入（将 z_k 与之前的 ẑ₀ 估计拼接）
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import DiffusionStepEmbedding, PositionalEmbedding


# ---------------------------------------------------------------------------
# adaLN 工具
# ---------------------------------------------------------------------------

class AdaLayerNorm(nn.Module):
    """自适应层归一化：γ, β 由扩散步嵌入回归得到。

    γ, β = MLP(t_emb)，其中 t_emb 是扩散步嵌入。
    adaLN(γ, β, Y) = γ ⊙ LayerNorm(Y) + β

    Args:
        d_model: 特征维度。
        d_temb: 扩散步嵌入维度（n = 4m）。
    """

    def __init__(self, d_model: int, d_temb: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_temb, 2 * d_model)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        """应用 adaLN。

        Args:
            x: 输入 [B, seq_len, d_model]。
            t_emb: 扩散步嵌入 [B, d_temb]。

        Returns:
            归一化并调制后的张量 [B, seq_len, d_model]。
        """
        # LayerNorm
        x_norm = self.norm(x)  # [B, seq_len, d_model]

        # 从扩散嵌入获取缩放和偏移
        gamma_beta = self.proj(t_emb)  # [B, 2*d_model]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # 各 [B, d_model]

        # 广播并应用仿射变换
        gamma = gamma.unsqueeze(1)  # [B, 1, d_model]
        beta = beta.unsqueeze(1)    # [B, 1, d_model]

        return gamma * x_norm + beta


# ---------------------------------------------------------------------------
# adaLN Transformer 层
# ---------------------------------------------------------------------------

class AdaLNTransformerLayer(nn.Module):
    """单个 adaLN Transformer 块。

    用由扩散步条件控制的 adaLN 替代标准 LayerNorm。
    结构：自注意力 → adaLN → 前馈网络 → adaLN。

    Args:
        d_model: 特征维度。
        n_heads: 注意力头数。
        d_temb: 扩散步嵌入维度。
        dim_feedforward: 前馈网络隐藏维度。
        dropout: Dropout 概率。
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
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        # adaLN 模块
        self.adaln1 = AdaLayerNorm(d_model, d_temb)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        # 第二个 adaLN
        self.adaln2 = AdaLayerNorm(d_model, d_temb)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入 [B, seq_len, d_model]。
            t_emb: 扩散步嵌入 [B, d_temb]。
            attn_mask: 可选的注意力掩码。

        Returns:
            输出 [B, seq_len, d_model]。
        """
        # 自注意力 + 残差
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout1(attn_out)
        x = self.adaln1(x, t_emb)

        # 前馈网络 + 残差
        ff_out = self.linear2(F.gelu(self.linear1(x)))
        x = x + self.dropout2(ff_out)
        x = self.adaln2(x, t_emb)

        return x


# ---------------------------------------------------------------------------
# 去噪 Transformer（x̂_θ 网络）
# ---------------------------------------------------------------------------

class DenoisingTransformer(nn.Module):
    """LDT 去噪网络 x̂_θ。

    非自回归 adaLN Transformer，从加噪输入 z_k、历史条件 c
    和自条件 ẑ₀_prev 预测干净的潜在变量 ẑ₀。

    架构（图 1b）：
    1. 对 X̂ 和 Ẑ 的输入投影块（2 层 MLP）
    2. 位置 + 时间嵌入
    3. adaLN Transformer（编码器-解码器）
    4. 输出投影 → ẑ₀

    Args:
        d_data: 时间序列特征维度 d。
        d_latent: 潜在维度 m。
        d_model: Transformer 嵌入维度。
        n_heads: 注意力头数。
        n_layers: adaLN Transformer 层数（编码器+解码器）。
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
        self.d_data = d_data
        self.d_latent = d_latent
        self.d_model = d_model
        self.history_len = history_len
        self.pred_len = pred_len

        # 输入投影：每个输入流使用 2 层 MLP
        # 历史条件: d → d_model
        self.history_proj = nn.Sequential(
            nn.Linear(d_data, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # 加噪潜在变量 z_k: 2m → d_model（因为与自条件拼接）
        self.latent_proj = nn.Sequential(
            nn.Linear(2 * d_latent, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # 位置嵌入
        self.pos_emb = PositionalEmbedding(
            max_len=history_len + pred_len, d_model=d_model
        )

        # 时间嵌入（可学习的时序索引）
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # 扩散步嵌入（n = 4m）
        self.diff_step_emb = DiffusionStepEmbedding(
            d_latent=d_latent, max_steps=max_diffusion_steps
        )
        d_temb = self.diff_step_emb.n  # 4 * d_latent

        # adaLN Transformer 编码器（处理历史条件）
        self.encoder_layers = nn.ModuleList([
            AdaLNTransformerLayer(d_model, n_heads, d_temb, dropout=dropout)
            for _ in range(n_layers)
        ])

        # 交叉注意力解码器：从潜在位置关注编码器输出
        self.decoder_layers = nn.ModuleList([
            AdaLNTransformerDecoderLayer(
                d_model, n_heads, d_temb, dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # 输出投影: d_model → d_latent
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_latent),
        )

    def forward(
        self,
        z_k: torch.Tensor,
        history: torch.Tensor,
        z_self_cond: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """从加噪 z_k 预测干净潜在变量 ẑ₀。

        Args:
            z_k: 加噪潜在变量 [B, t, m]。
            history: 历史窗口 [B, T, d]（无条件时为零）。
            z_self_cond: 之前的 ẑ₀ 估计 [B, t, m]（无自条件时为零）。
            k: 扩散步索引 [B]。

        Returns:
            预测的干净潜在变量 ẑ₀ [B, t, m]。
        """
        B, T, _ = history.shape
        B2, t, _ = z_k.shape

        # 1. 投影历史: [B, T, d] → [B, T, d_model]
        h_hist = self.history_proj(history)  # [B, T, d_model]

        # 2. 拼接 z_k 与自条件并投影
        z_input = torch.cat([z_k, z_self_cond], dim=-1)  # [B, t, 2m]
        h_latent = self.latent_proj(z_input)              # [B, t, d_model]

        # 3. 添加位置嵌入
        pos_all = self.pos_emb(T + t)          # [1, T+t, d_model]
        h_hist = h_hist + pos_all[:, :T, :]    # [B, T, d_model]
        h_latent = h_latent + pos_all[:, T:, :]  # [B, t, d_model]

        # 4. 添加时间嵌入（基于索引）
        time_idx_hist = torch.arange(T, device=history.device).float().view(1, T, 1).expand(B, -1, -1)
        time_idx_latent = torch.arange(T, T + t, device=z_k.device).float().view(1, t, 1).expand(B, -1, -1)
        h_hist = h_hist + self.time_mlp(time_idx_hist)
        h_latent = h_latent + self.time_mlp(time_idx_latent)

        # 5. 获取用于 adaLN 的扩散步嵌入
        t_emb = self.diff_step_emb(k)  # [B, n]

        # 6. 编码器：处理历史
        h_enc = h_hist
        for layer in self.encoder_layers:
            h_enc = layer(h_enc, t_emb)

        # 7. 解码器：通过交叉注意力处理潜在变量
        h_dec = h_latent
        for layer in self.decoder_layers:
            h_dec = layer(h_dec, h_enc, t_emb)

        # 8. 输出投影: [B, t, d_model] → [B, t, m]
        z_0_pred = self.output_proj(h_dec)

        return z_0_pred


class AdaLNTransformerDecoderLayer(nn.Module):
    """单个 adaLN Transformer 解码器层，含交叉注意力。

    结构：自注意力 → adaLN → 交叉注意力 → adaLN → 前馈网络 → adaLN。

    Args:
        d_model: 特征维度。
        n_heads: 注意力头数。
        d_temb: 扩散步嵌入维度。
        dim_feedforward: 前馈网络隐藏维度。
        dropout: Dropout 概率。
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
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.adaln1 = AdaLayerNorm(d_model, d_temb)

        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        self.adaln2 = AdaLayerNorm(d_model, d_temb)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.adaln3 = AdaLayerNorm(d_model, d_temb)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            x: 解码器输入 [B, t, d_model]。
            memory: 编码器输出 [B, T, d_model]。
            t_emb: 扩散步嵌入 [B, d_temb]。

        Returns:
            输出 [B, t, d_model]。
        """
        # 自注意力
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_out)
        x = self.adaln1(x, t_emb)

        # 交叉注意力
        cross_out, _ = self.cross_attn(x, memory, memory)
        x = x + self.dropout2(cross_out)
        x = self.adaln2(x, t_emb)

        # 前馈网络
        ff_out = self.linear2(F.gelu(self.linear1(x)))
        x = x + self.dropout3(ff_out)
        x = self.adaln3(x, t_emb)

        return x
