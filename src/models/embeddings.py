"""LDT 模型的嵌入层。

提供：
- PositionalEmbedding: 正弦位置编码（Vaswani et al.）
- TimeEmbedding: 基于 MLP 的时间依赖嵌入
- DiffusionStepEmbedding: 用于 adaLN 条件的正弦扩散步嵌入
"""

import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """正弦位置嵌入（Vaswani et al. 2017）。

    Args:
        max_len: 最大序列长度。
        d_model: 嵌入维度。
    """

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, length: int) -> torch.Tensor:
        """获取指定序列长度的位置嵌入。

        Args:
            length: 序列长度。

        Returns:
            嵌入张量 [1, length, d_model]。
        """
        return self.pe[:, :length]


class TimeEmbedding(nn.Module):
    """通过 MLP 学习的时刻嵌入。

    将标量时间索引（如小时）映射为 d_model 向量。

    Args:
        d_model: 输出嵌入维度。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """嵌入时间值。

        Args:
            t: 时间索引 [B, seq_len, 1] 或 [B, seq_len]。

        Returns:
            时间嵌入 [B, seq_len, d_model]。
        """
        if t.dim() == 2:
            t = t.unsqueeze(-1)  # [B, seq_len, 1]
        return self.mlp(t.float())


class DiffusionStepEmbedding(nn.Module):
    """用于 adaLN 条件的正弦扩散步嵌入。

    将离散扩散步 k 映射为大小为 n=4m 的连续嵌入，
    再通过 MLP 投影到所需的缩放/偏移维度。

    Args:
        d_latent: 潜在维度 m。
        max_steps: 最大扩散步 K。
    """

    def __init__(self, d_latent: int, max_steps: int = 300):
        super().__init__()
        self.d_latent = d_latent
        self.max_steps = max_steps

        # 正弦嵌入维度 n = 4m
        n = 4 * d_latent
        self.n = n

        # 预计算正弦基
        pe = torch.zeros(max_steps, n)
        position = torch.arange(0, max_steps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n, 2, dtype=torch.float)
            * (-math.log(10000.0) / n)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        # 投影到 adaLN 缩放/偏移: n → （用于 γ 和 β）
        self.mlp = nn.Sequential(
            nn.Linear(n, n),
            nn.GELU(),
            nn.Linear(n, n),
        )

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """获取扩散步嵌入。

        Args:
            k: 扩散步索引 [B] 或 [B, 1]（0 起始）。

        Returns:
            步嵌入 [B, n]，其中 n = 4m。
        """
        if k.dim() == 0:
            k = k.unsqueeze(0)
        if k.dim() == 2:
            k = k.squeeze(-1)
        k = k.long().clamp(0, self.max_steps - 1)
        emb = self.pe[k]  # [B, n]
        return self.mlp(emb)
