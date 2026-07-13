#!/usr/bin/env python
"""最小验证：用随机噪声目标 + 随机历史测试 LDT 能否学习。

如果模型能过拟合随机噪声 → 架构正确，问题在数据
如果模型不能过拟合随机噪声 → 架构有 bug
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", message="Using `json`-module")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from src.models.diffusion import LDiffusion


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    B, T, t, d, m = 16, 96, 24, 137, 32

    # 随机数据（固定 seed）
    torch.manual_seed(42)
    X = torch.randn(B, T, d, device=device)
    Y = torch.randn(B, t, d, device=device)
    z_0 = torch.randn(B, t, m, device=device)  # 模拟 VAE 编码后的潜变量

    # 创建模型
    ldt = LDiffusion(
        d_data=d, d_latent=m, d_model=128, n_heads=8, n_layers=3,
        history_len=T, pred_len=t, diffusion_steps=100,
        p_uncond=0.0, self_cond_prob=0.0,
    ).to(device)

    # 缩放潜变量（论文第4页）
    z_0_scaled, _ = ldt._scale_latent(z_0)
    print(f"z_0 均值: {z_0.mean().item():.4f}, "
          f"方差: {z_0.var().item():.4f}")
    print(f"z_0_scaled 均值: {z_0_scaled.mean().item():.4f}, "
          f"方差: {z_0_scaled.var().item():.4f}")
    print(f"零预测 baseline MSE: {(z_0_scaled ** 2).mean().item():.4f}")

    optimizer = torch.optim.AdamW(ldt.parameters(), lr=3e-3)
    K = ldt.diffusion_steps

    print("\n训练 500 步（随机噪声目标）...")
    best_loss = float("inf")

    for step in range(1, 501):
        optimizer.zero_grad()
        k = torch.randint(1, K + 1, (B,), device=device)
        noise = torch.randn_like(z_0_scaled)
        z_k = ldt.noise_schedule.q_sample(z_0_scaled, k, noise)
        z_pred = ldt.denoiser(z_k, X, torch.zeros_like(z_0_scaled), k)
        loss = F.mse_loss(z_pred, z_0_scaled)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ldt.parameters(), max_norm=5.0)
        optimizer.step()

        best_loss = min(best_loss, loss.item())

        if step % 50 == 0:
            grad_norm = sum(p.grad.norm().item() ** 2
                           for p in ldt.parameters()
                           if p.grad is not None) ** 0.5
            print(f"  步 {step:3d}: loss={loss.item():.4f}  "
                  f"best={best_loss:.4f}  grad={grad_norm:.2f}")

    if best_loss < 0.1:
        print(f"\n✅ 通过！模型可以学习随机噪声 (best_loss={best_loss:.4f})")
        print("   问题在数据 pipeline — 检查 VN 归一化 / VAE 编码 / 数据加载")
    elif best_loss < 0.5:
        print(f"\n⚠️  部分通过 (best_loss={best_loss:.4f})")
        print("   增加训练步数或模型容量")
    else:
        print(f"\n❌ 失败！模型无法学习随机噪声 (best_loss={best_loss:.4f})")
        print("   架构有 bug — 检查前向传播和梯度流")


if __name__ == "__main__":
    main()
