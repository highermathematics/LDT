"""第二阶段：潜在扩散 Transformer（LDT）。

在潜在空间中实现扩散过程：
- 平方根（sqrt）噪声调度
- x₀ 预测目标（而非 ε 预测）
- 自条件机制
- 无分类器引导
- DDIM 采样
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import DenoisingTransformer


# ---------------------------------------------------------------------------
# 噪声调度
# ---------------------------------------------------------------------------

class SqrtNoiseSchedule:
    """扩散模型的平方根噪声调度。

    定义：
        α̅_k = Π_{i=1}^k (1 - β_i)
        β_i 遵循从 β_1 到 β_T 的 sqrt 调度

    Args:
        K: 总扩散步数。
        beta_1: 起始噪声水平（默认: 1e-4）。
        beta_T: 终止噪声水平（默认: 0.1）。
    """

    def __init__(
        self,
        K: int,
        beta_1: float = 1e-4,
        beta_T: float = 0.1,
    ):
        self.K = K
        self.beta_1 = beta_1
        self.beta_T = beta_T

        # Sqrt 调度: β_i = (√β_1 + (i/(K-1)) × (√β_T - √β_1))²
        sqrt_b1 = math.sqrt(beta_1)
        sqrt_bT = math.sqrt(beta_T)
        betas = []
        for i in range(K):
            frac = i / max(K - 1, 1)
            beta = (sqrt_b1 + frac * (sqrt_bT - sqrt_b1)) ** 2
            betas.append(beta)
        betas = torch.tensor(betas, dtype=torch.float32)

        # 计算 alpha 及其累积乘积
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)  # α̅_k

        self.betas = betas
        self.alphas = alphas
        self.alpha_cumprod = alpha_cumprod            # [K]
        self.alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)

    def get_alpha_bar(self, k: torch.Tensor) -> torch.Tensor:
        """获取指定步索引的 α̅_k。

        Args:
            k: 步索引 [B]（1 起始）。

        Returns:
            α̅_k 值 [B]。
        """
        k_idx = (k - 1).long().clamp(0, self.K - 1)
        return self.alpha_cumprod[k_idx].to(k.device)

    def q_sample(
        self, z_0: torch.Tensor, k: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """前向扩散: z_k = √α̅_k · z_0 + √(1 - α̅_k) · ε。

        Args:
            z_0: 干净的潜在变量 [B, t, m]。
            k: 扩散步索引 [B]（1 起始）。
            noise: 高斯噪声 ε [B, t, m]。

        Returns:
            加噪后的潜在变量 z_k [B, t, m]。
        """
        alpha_bar = self.get_alpha_bar(k)  # [B]
        # 广播到 [B, 1, 1]
        alpha_bar = alpha_bar.view(-1, 1, 1)
        return torch.sqrt(alpha_bar) * z_0 + torch.sqrt(1.0 - alpha_bar) * noise

    def get_ddim_coeffs(
        self, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算步 k 的 DDIM 系数。

        返回 (μ̂_t, μ̂_x, σ_t) 使得:
        z_{k-1} = μ̂_t · z_k + μ̂_x · x̂_0 + σ_t · ε

        Args:
            k: 当前步 [B]（1 起始）。

        Returns:
            (mu_t [B], mu_x [B], sigma [B]) 元组，均可广播。
        """
        alpha_bar = self.get_alpha_bar(k)  # α̅_k
        # α̅_{k-1}: k=1 时 α̅_0 = 1.0
        prev_k = (k - 1).long()
        alpha_bar_prev = torch.where(
            prev_k >= 1,
            self.alpha_cumprod[(prev_k - 1).clamp(0, self.K - 1)].to(k.device),
            torch.ones_like(alpha_bar),
        )

        # DDIM 系数（确定性, σ=0）
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)

        # z_k 的系数
        mu_t = sqrt_alpha_bar_prev / sqrt_alpha_bar * torch.sqrt(1.0 - alpha_bar)
        # x̂_0 的系数
        mu_x = sqrt_alpha_bar_prev

        # 确定性 DDIM: σ = 0
        sigma = torch.zeros_like(alpha_bar)

        return mu_t, mu_x, sigma


# ---------------------------------------------------------------------------
# LDT 扩散模型
# ---------------------------------------------------------------------------

class LDiffusion(nn.Module):
    """潜在扩散 Transformer：第二阶段模型。

    训练去噪网络 x̂_θ 从加噪 z_k 预测干净潜在 z₀。
    支持自条件机制和无分类器引导。

    Args:
        d_data: 时间序列特征维度 d。
        d_latent: 潜在维度 m。
        d_model: Transformer 嵌入维度。
        n_heads: 去噪 Transformer 注意力头数。
        n_layers: Transformer 层数。
        history_len: 历史窗口长度 T。
        pred_len: 预测长度 t。
        diffusion_steps: 总扩散步数 K。
        beta_1: 起始噪声水平。
        beta_T: 终止噪声水平。
        p_uncond: 无条件训练概率（CFG）。
        self_cond_prob: 训练时使用自条件的概率。
        dropout: Transformer 中的 dropout。
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
        diffusion_steps: int = 100,
        beta_1: float = 1e-4,
        beta_T: float = 0.1,
        p_uncond: float = 0.1,
        self_cond_prob: float = 0.4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_data = d_data
        self.d_latent = d_latent
        self.history_len = history_len
        self.pred_len = pred_len
        self.diffusion_steps = diffusion_steps
        self.p_uncond = p_uncond
        self.self_cond_prob = self_cond_prob

        # 噪声调度
        self.noise_schedule = SqrtNoiseSchedule(
            K=diffusion_steps, beta_1=beta_1, beta_T=beta_T
        )

        # 去噪网络 x̂_θ
        self.denoiser = DenoisingTransformer(
            d_data=d_data,
            d_latent=d_latent,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            history_len=history_len,
            pred_len=pred_len,
            max_diffusion_steps=diffusion_steps,
            dropout=dropout,
        )

    def _scale_latent(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将潜在变量缩放到单位方差以稳定扩散训练。

        Ẑ = Z / σ̂，其中 σ̂² = (1/btm) Σ(z - μ̂)²

        Args:
            z: 潜在张量 [B, t, m]。

        Returns:
            (z_scaled [B, t, m], sigma_hat [B, 1, 1]) 元组。
        """
        b, t, m = z.shape
        mu = z.mean()
        var = ((z - mu) ** 2).mean()
        sigma_hat = torch.sqrt(var + 1e-8)
        z_scaled = z / sigma_hat
        return z_scaled, sigma_hat.detach()

    def _unscale_latent(
        self, z_scaled: torch.Tensor, sigma_hat: torch.Tensor
    ) -> torch.Tensor:
        """_scale_latent 的逆操作。

        Args:
            z_scaled: 缩放后的潜在变量 [B, t, m]。
            sigma_hat: 缩放因子 [B, 1, 1] 或标量。

        Returns:
            还原后的潜在变量 [B, t, m]。
        """
        return z_scaled * sigma_hat

    def training_step(
        self,
        z_0: torch.Tensor,
        history: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """按算法 1 执行单步训练。

        Args:
            z_0: 来自编码器的干净潜在变量 [B, t, m]。
            history: 历史窗口 [B, T, d]。

        Returns:
            (loss, metrics_dict) 元组。
        """
        B, t, m = z_0.shape
        device = z_0.device

        # 1. 采样扩散步 k ~ Uniform(1..K)
        k = torch.randint(1, self.diffusion_steps + 1, (B,), device=device)

        # 2. 采样噪声 ε ~ N(0, I)
        noise = torch.randn_like(z_0)

        # 3. 前向扩散: z_k = √α̅_k · z_0 + √(1-α̅_k) · ε
        z_k = self.noise_schedule.q_sample(z_0, k, noise)

        # 4. 无分类器引导：以概率 p_uncond 丢弃条件
        uncond_mask = torch.rand(B, device=device) < self.p_uncond
        history_cond = history.clone()
        history_cond[uncond_mask] = 0.0  # 将条件设为 ∅

        # 5. 自条件机制
        z_self_cond = torch.zeros_like(z_0)
        use_self_cond = torch.rand(1, device=device).item() < self.self_cond_prob

        if use_self_cond:
            with torch.no_grad():
                z_self_cond = self.denoiser(z_k, history_cond, z_self_cond, k)
                # 自条件估计停止梯度

        # 6. 预测 ẑ₀
        z_0_pred = self.denoiser(z_k, history_cond, z_self_cond, k)

        # 7. 损失: MSE(z_0, ẑ₀) — x₀ 预测
        loss = F.mse_loss(z_0_pred, z_0)

        metrics = {
            "loss": loss.item(),
            "k_mean": k.float().mean().item(),
        }

        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        guidance_strength: float = 3.0,
        num_steps: Optional[int] = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """按算法 2 进行 DDIM 采样。

        在反向扩散过程中实现自条件机制和无分类器引导。

        Args:
            history: 历史窗口 [B, T, d]。
            guidance_strength: 无分类器引导强度 w。
            num_steps: DDIM 采样步数（默认: 完整 K 步）。
            progress: 是否显示进度条。

        Returns:
            干净的潜在变量 z₀ [B, t, m]。
        """
        B, T, d = history.shape
        device = history.device
        t = self.pred_len
        m = self.d_latent
        K = self.diffusion_steps

        if num_steps is None:
            num_steps = K

        # 待采样的步索引（均匀间隔，降序）
        step_indices = torch.linspace(K, 1, num_steps, device=device).long()
        step_indices = torch.cat([step_indices, torch.tensor([0], device=device)])

        # 1. z_K ~ N(0, I)
        z = torch.randn(B, t, m, device=device)

        # 2. ẑ_0_prev = 0（自条件初始化）
        z_self_cond = torch.zeros(B, t, m, device=device)

        # 无条件模型的空条件
        empty_history = torch.zeros(B, T, d, device=device)

        # 迭代各步
        iterator = range(len(step_indices) - 1)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="DDIM 采样")

        for i in iterator:
            k_curr = step_indices[i]      # 当前步
            k_next = step_indices[i + 1]  # 下一步

            # 创建步张量
            k_tensor = torch.full((B,), k_curr, device=device, dtype=torch.long)

            # 7. 自条件预测
            z_self_cond = self.denoiser(z, history, z_self_cond, k_tensor)

            # 8. 条件预测
            z_0_cond = self.denoiser(z, history, z_self_cond, k_tensor)

            # 无条件预测
            z_0_uncond = self.denoiser(z, empty_history, z_self_cond, k_tensor)

            # 9. 无分类器引导
            z_0_guided = (1 + guidance_strength) * z_0_cond \
                - guidance_strength * z_0_uncond

            # 10. DDIM 步: z_{k-1} = √α̅_{k-1}·ẑ₀ + √(1-α̅_{k-1})·ε_pred
            alpha_bar_curr = self.noise_schedule.get_alpha_bar(k_tensor)

            # 从 z_k 和 ẑ₀ 计算 ε_pred
            sqrt_alpha = torch.sqrt(alpha_bar_curr).view(-1, 1, 1)
            sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bar_curr).view(-1, 1, 1)
            eps_pred = (z - sqrt_alpha * z_0_guided) / sqrt_one_minus_alpha

            # 获取 α̅_{k-1}
            if k_next > 0:
                k_next_tensor = torch.full((B,), k_next, device=device, dtype=torch.long)
                alpha_bar_next = self.noise_schedule.get_alpha_bar(k_next_tensor)
            else:
                alpha_bar_next = torch.ones(B, device=device)

            sqrt_alpha_next = torch.sqrt(alpha_bar_next).view(-1, 1, 1)
            sqrt_one_minus_alpha_next = torch.sqrt(1.0 - alpha_bar_next).view(-1, 1, 1)

            # z_{k-1} = √α̅_{k-1}·ẑ₀ + √(1-α̅_{k-1})·ε_pred
            z = sqrt_alpha_next * z_0_guided + sqrt_one_minus_alpha_next * eps_pred

            # 更新自条件
            z_self_cond = z_0_guided

        return z
