"""第一阶段：用于时间序列压缩的对称统计感知 VAE。

实现 LDT 论文第 4.1 节中描述的自编码器：
- 编码器：3 层 Transformer 编码器，下采样 d → m
- 解码器：3 层 Transformer 编码器，上采样 m → d
- 判别器：1 层 Transformer 编码器，输出标量
- VN 归一化用于非平稳数据的稳定训练
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Transformer 构建块
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """单个 Transformer 编码器块：多头自注意力 + 前馈网络。

    Args:
        d_model: 嵌入维度。
        n_heads: 注意力头数。
        dim_feedforward: 前馈网络隐藏维度（默认: 4 × d_model）。
        dropout: Dropout 概率。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量 [B, seq_len, d_model]。

        Returns:
            输出张量 [B, seq_len, d_model]。
        """
        # 自注意力 + 残差连接
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))

        # 前馈网络 + 残差连接
        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff_out))

        return x


# ---------------------------------------------------------------------------
# 编码器
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """VAE 编码器：将归一化目标压缩为潜在表示。

    架构：输入投影 → 3 层 Transformer 编码器 → μ/σ 投影。

    Args:
        d_input: 输入特征维度 d（时间序列特征数）。
        d_latent: 潜在维度 m（m << d）。
        d_model: Transformer 嵌入维度。
        n_heads: 注意力头数。
        n_layers: Transformer 层数。
        dropout: Dropout 概率。
    """

    def __init__(
        self,
        d_input: int,
        d_latent: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_latent = d_latent
        self.d_model = d_model

        # 下采样投影: d → d_model
        self.input_proj = nn.Linear(d_input, d_model)

        # Transformer 编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # 输出层: d_model → d_latent（分别输出 μ 和 log_var）
        self.fc_mu = nn.Linear(d_model, d_latent)
        self.fc_logvar = nn.Linear(d_model, d_latent)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将输入编码为潜在分布参数。

        Args:
            x: 输入张量 [B, t, d]（归一化后的目标）。

        Returns:
            (z_mu [B, t, m], z_logvar [B, t, m]) 元组。
        """
        # 投影: [B, t, d] → [B, t, d_model]
        h = self.input_proj(x)

        # Transformer 层
        for layer in self.layers:
            h = layer(h)

        # 输出 μ 和 log σ²
        z_mu = self.fc_mu(h)          # [B, t, m]
        z_logvar = self.fc_logvar(h)  # [B, t, m]

        return z_mu, z_logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """确定性编码：仅返回均值。

        Args:
            x: 输入 [B, t, d]。

        Returns:
            潜在编码 z [B, t, m]。
        """
        z_mu, _ = self.forward(x)
        return z_mu


# ---------------------------------------------------------------------------
# 解码器
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """VAE 解码器：从潜在表示重建时间序列。

    架构：输入投影 → 3 层 Transformer 编码器 → 输出投影。

    Args:
        d_output: 输出特征维度 d。
        d_latent: 潜在维度 m。
        d_model: Transformer 嵌入维度。
        n_heads: 注意力头数。
        n_layers: Transformer 层数。
        dropout: Dropout 概率。
    """

    def __init__(
        self,
        d_output: int,
        d_latent: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_output = d_output
        self.d_latent = d_latent
        self.d_model = d_model

        # 上采样投影: m → d_model
        self.input_proj = nn.Linear(d_latent, d_model)

        # Transformer 编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # 输出投影: d_model → d
        self.output_proj = nn.Linear(d_model, d_output)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """将潜在编码解码回时间域。

        Args:
            z: 潜在张量 [B, t, m]。

        Returns:
            重建的时间序列 [B, t, d]。
        """
        # 投影: [B, t, m] → [B, t, d_model]
        h = self.input_proj(z)

        # Transformer 层
        for layer in self.layers:
            h = layer(h)

        # 输出投影: [B, t, d_model] → [B, t, d]
        y = self.output_proj(h)

        return y


# ---------------------------------------------------------------------------
# 判别器
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """基于时间块的 WGAN 判别器（Critic），用于 VAE 对抗训练。

    架构：输入投影 → 1 层 Transformer 编码器 → 标量输出头。
    WGAN-GP 模式下输出无 sigmoid，直接输出实数分数。

    Args:
        d_input: 输入特征维度 d。
        d_model: Transformer 嵌入维度。
        n_heads: 注意力头数。
        dropout: Dropout 概率。
    """

    def __init__(
        self,
        d_input: int,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model

        self.input_proj = nn.Linear(d_input, d_model)
        self.encoder = TransformerEncoderBlock(
            d_model, n_heads, dropout=dropout
        )
        # 沿时间维度池化后映射为标量（Critic 输出实数分数，无 sigmoid）
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输出 Wasserstein 临界分数（实数）。

        Args:
            x: 时间序列 [B, t, d]。

        Returns:
            临界分数 [B, 1]。
        """
        h = self.input_proj(x)     # [B, t, d_model]
        h = self.encoder(h)        # [B, t, d_model]
        h = h.mean(dim=1)          # [B, d_model] — 时间维度池化
        return self.classifier(h)  # [B, 1]


# ---------------------------------------------------------------------------
# VAE（完整第一阶段模型）
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    """用于时间序列压缩的对称统计感知 VAE。

    组合编码器、解码器和判别器，加入 KL 正则化。
    按 LDT 论文第 4.1 节所述进行对抗训练。

    Args:
        d_data: 时间序列特征维度 d。
        d_latent: 潜在维度 m。
        d_model: Transformer 嵌入维度。
        n_heads: 注意力头数（编码器/解码器/判别器共享）。
        n_layers: 编码器/解码器的 Transformer 层数。
        kl_weight: KL 散度正则化权重。
        dropout: Dropout 概率。
    """

    def __init__(
        self,
        d_data: int,
        d_latent: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        kl_weight: float = 1e-8,
        lambda_adv: float = 0.01,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_data = d_data
        self.d_latent = d_latent
        self.kl_weight = kl_weight
        self.lambda_adv = lambda_adv

        self.encoder = Encoder(
            d_input=d_data, d_latent=d_latent, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        )
        self.decoder = Decoder(
            d_output=d_data, d_latent=d_latent, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        )
        self.discriminator = Discriminator(
            d_input=d_data, d_model=d_model, n_heads=n_heads,
            dropout=dropout,
        )

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """重参数化技巧：z = μ + σ ⊙ ε。

        Args:
            mu: 均值 [B, t, m]。
            logvar: 对数方差 [B, t, m]。

        Returns:
            采样得到的潜在变量 z [B, t, m]。
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self, x: torch.Tensor, stochastic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """将输入编码为潜在编码。

        Args:
            x: 输入时间序列 [B, t, d]。
            stochastic: True 时通过重参数化采样，False 时仅返回均值。

        Returns:
            (z [B, t, m], mu [B, t, m], logvar [B, t, m]) 元组。
        """
        mu, logvar = self.encoder(x)
        if stochastic:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """将潜在编码解码回时间域。

        Args:
            z: 潜在张量 [B, t, m]。

        Returns:
            重建的时间序列 [B, t, d]。
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, stochastic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """完整前向传播：编码 → 解码。

        Args:
            x: 输入时间序列 [B, t, d]。
            stochastic: True 时使用重参数化。

        Returns:
            (y_recon [B, t, d], z [B, t, m], mu, logvar) 元组。
        """
        z, mu, logvar = self.encode(x, stochastic=stochastic)
        y_recon = self.decode(z)
        return y_recon, z, mu, logvar

    def kl_loss(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """与标准正态先验的 KL 散度。

        L_reg = 0.5 × Σ (μ² + σ² - 1 - log(σ²))

        Args:
            mu: 均值 [B, t, m]。
            logvar: 对数方差 [B, t, m]。

        Returns:
            标量 KL 损失（所有维度取平均）。
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl * self.kl_weight

    def generator_loss(
        self,
        y_real: torch.Tensor,
        y_recon: torch.Tensor,
        z_mu: torch.Tensor,
        z_logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算生成器（编码器 + 解码器）损失。

        WGAN-GP: 对抗损失 = -mean(D(fake))，即最大化 Critic 对生成样本的评分。

        Args:
            y_real: 真实目标 [B, t, d]。
            y_recon: 重建目标 [B, t, d]。
            z_mu: 潜在均值 [B, t, m]。
            z_logvar: 潜在对数方差 [B, t, m]。

        Returns:
            (total_loss, rec_loss, kl_loss_val, adv_loss) 元组。
        """
        # 重建损失（MSE）
        rec_loss = F.mse_loss(y_recon, y_real)

        # WGAN 对抗损失：最大化 Critic 对生成样本的评分
        disc_fake = self.discriminator(y_recon)
        adv_loss = -disc_fake.mean()

        # KL 正则化
        kl = self.kl_loss(z_mu, z_logvar)

        # 生成器总损失：重建 + λ·对抗 + KL
        weighted_adv = adv_loss * self.lambda_adv
        total = rec_loss + weighted_adv + kl

        return total, rec_loss, kl, adv_loss

    def gradient_penalty(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """WGAN-GP 梯度惩罚项。

        GP = E[(||∇_x̂ D(x̂)||_2 - 1)²], 其中 x̂ 是 real 和 fake 的随机插值。

        Args:
            real: 真实样本 [B, t, d]。
            fake: 生成样本 [B, t, d]（需保留梯度）。

        Returns:
            标量梯度惩罚值。
        """
        batch_size = real.size(0)
        # 每个样本独立的 ε ~ U(0, 1)
        epsilon = torch.rand(batch_size, 1, 1, device=real.device)
        # 随机插值: x̂ = ε·real + (1-ε)·fake
        interpolated = epsilon * real + (1 - epsilon) * fake
        interpolated.requires_grad_(True)

        # 计算 Critic 对插值样本的评分
        d_interpolated = self.discriminator(interpolated)

        # 计算梯度 ∂D/∂x̂
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        # 计算每个样本的梯度范数
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)

        # 惩罚项: (||∇D||₂ - 1)²
        return ((gradient_norm - 1.0) ** 2).mean()

    def discriminator_loss(
        self,
        y_real: torch.Tensor,
        y_recon: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 WGAN-GP 判别器（Critic）损失。

        WGAN loss = mean(D(fake)) - mean(D(real)) + λ · gradient_penalty

        Args:
            y_real: 真实值 [B, t, d]。
            y_recon: 重建值 [B, t, d]（需要保留梯度进行 GP 计算）。

        Returns:
            (total_loss, wasserstein_distance) 元组。
        """
        disc_real = self.discriminator(y_real)
        disc_fake = self.discriminator(y_recon)

        # Wasserstein 距离: D(fake) - D(real)
        w_dist = disc_fake.mean() - disc_real.mean()

        # 梯度惩罚（权重 λ=10，遵循 WGAN-GP 论文建议）
        gp = self.gradient_penalty(y_real, y_recon)

        # 总损失
        total = w_dist + 10.0 * gp

        return total, w_dist
