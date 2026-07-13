#!/usr/bin/env python
"""LDT 过拟合诊断脚本。

如果模型架构和训练逻辑正确，200 步内 loss 应该从 ~1.0 降到 <0.1。
否则说明存在底层 bug。
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="Using `json`-module")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from src.config import load_config
from src.data.dataset import create_dataloaders
from src.data.normalization import VarianceUpdateNorm
from src.models.autoencoder import Encoder
from src.models.diffusion import LDiffusion


def main():
    parser = argparse.ArgumentParser(description="LDT 过拟合诊断")
    parser.add_argument("--config", type=str, default="configs/solar.yaml")
    parser.add_argument("--stage1_ckpt", type=str,
                        default="checkpoints/solar_stage1/best_model.pt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device or config.training.device
                         if torch.cuda.is_available() else "cpu")

    # ── 加载冻结编码器 ────────────────────────────────────────
    ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=True)
    vae_cfg = ckpt["vae_config"]
    encoder = Encoder(
        d_input=vae_cfg["d_data"], d_latent=vae_cfg["d_latent"],
        d_model=vae_cfg["d_model"], n_heads=vae_cfg["n_heads"],
        n_layers=vae_cfg["n_layers"],
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # ── VN 层 ──────────────────────────────────────────────────
    vn = VarianceUpdateNorm(num_features=vae_cfg["d_data"]).to(device)
    if "vn" in ckpt:
        vn.load_state_dict(ckpt["vn"], strict=False)
    vn.eval()

    # ── 加载一个 batch ──────────────────────────────────────────
    train_loader, _, _, _ = create_dataloaders(
        name=config.dataset.name,
        prediction_length=config.dataset.prediction_length,
        lookback_window=config.dataset.lookback_window,
        batch_size=16,
        num_workers=0,
        force_dimension=vae_cfg["d_data"],
    )
    X, Y = next(iter(train_loader))
    X, Y = X.to(device), Y.to(device)
    B, T, d = X.shape
    _, t, _ = Y.shape

    # 更新 VN 统计量
    vn.reset_stats()
    vn.update_stats(torch.cat([X, Y], dim=1))
    E_hat, Var_hat = vn.get_stats()

    Y_norm = vn.normalize(Y, E_hat, Var_hat)
    with torch.no_grad():
        z_0 = encoder.encode(Y_norm)

    X_norm = vn.normalize(
        torch.cat([X, torch.zeros(B, t, d, device=device)], dim=1),
        E_hat, Var_hat,
    )[:, :T, :]

    # ── 创建 LDT ────────────────────────────────────────────────
    ldt = LDiffusion(
        d_data=d,
        d_latent=config.vae.latent_dim,
        d_model=config.diffusion.embed_dim,
        n_heads=config.diffusion.num_heads,
        n_layers=config.diffusion.num_layers,
        history_len=T,
        pred_len=t,
        diffusion_steps=config.diffusion.diffusion_steps,
        beta_1=config.diffusion.beta_1,
        beta_T=config.diffusion.beta_T,
        p_uncond=0.0,        # 过拟合测试：关闭 CFG 和自条件
        self_cond_prob=0.0,
    ).to(device)

    # ── 检查：不同输入下输出是否有变化 ──────────────────────────
    print("=" * 60)
    print("诊断 1: 模型输出对输入敏感度")
    print("=" * 60)

    with torch.no_grad():
        k_test = torch.ones(B, device=device, dtype=torch.long)
        noise1 = torch.randn(B, t, config.vae.latent_dim, device=device)
        z_k1 = ldt.noise_schedule.q_sample(z_0, k_test, noise1)
        noise2 = torch.randn(B, t, config.vae.latent_dim, device=device)
        z_k2 = ldt.noise_schedule.q_sample(z_0, k_test, noise2)

        out1 = ldt.denoiser(z_k1, X_norm, torch.zeros_like(z_0), k_test)
        out2 = ldt.denoiser(z_k2, X_norm, torch.zeros_like(z_0), k_test)
        diff = (out1 - out2).abs().mean().item()
        print(f"不同 z_k 的输出差异: {diff:.6f}  (应为 >0.01)")

        # 检查: 空条件 vs 有条件
        out_cond = ldt.denoiser(z_k1, X_norm, torch.zeros_like(z_0), k_test)
        out_uncond = ldt.denoiser(z_k1, torch.zeros_like(X_norm, device=device),
                                  torch.zeros_like(z_0), k_test)
        diff2 = (out_cond - out_uncond).abs().mean().item()
        print(f"条件 vs 非条件输出差异: {diff2:.6f}  (应为 >0.01)")

        # 检查: 不同 k 的输出
        k_early = torch.ones(B, device=device, dtype=torch.long)
        k_late = torch.full((B,), config.diffusion.diffusion_steps, device=device, dtype=torch.long)
        out_k1 = ldt.denoiser(z_k1, X_norm, torch.zeros_like(z_0), k_early)
        out_k2 = ldt.denoiser(z_k1, X_norm, torch.zeros_like(z_0), k_late)
        diff3 = (out_k1 - out_k2).abs().mean().item()
        print(f"不同扩散步 k 的输出差异: {diff3:.6f}  (应为 >0.01)")

    if diff < 0.001 and diff2 < 0.001 and diff3 < 0.001:
        print("\n⚠️  模型对输入完全不敏感！可能存在架构 bug。")
        print("   检查: adaLN 是否被正确调用，t_emb 是否正确传入")

    # ── 检查: sigma_hat 的量级 ──────────────────────────────────
    print("\n" + "=" * 60)
    print("诊断 2: 潜变量统计量")
    print("=" * 60)
    z_0_scaled, sigma_hat = ldt._scale_latent(z_0)
    print(f"z_0 均值: {z_0.mean().item():.4f}, 标准差: {z_0.std().item():.4f}")
    print(f"z_0_scaled 均值: {z_0_scaled.mean().item():.4f}, 标准差: {z_0_scaled.std().item():.4f}")
    print(f"sigma_hat: {sigma_hat.item():.4f}")
    print(f"零预测 MSE baseline: {(z_0_scaled ** 2).mean().item():.4f}")

    # ── 过拟合测试 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"诊断 3: 单 batch 过拟合测试 ({args.steps} 步)")
    print("=" * 60)

    optimizer = torch.optim.AdamW(ldt.parameters(), lr=3e-4)
    z_0_fixed = z_0_scaled.detach()
    X_fixed = X_norm.detach()  # VN 有可学习参数，断开其计算图
    best_loss = float("inf")

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()

        k = torch.randint(1, config.diffusion.diffusion_steps + 1, (B,), device=device)
        noise = torch.randn_like(z_0_fixed)
        z_k = ldt.noise_schedule.q_sample(z_0_fixed, k, noise)

        z_pred = ldt.denoiser(z_k, X_fixed, torch.zeros_like(z_0_fixed), k)
        loss = F.mse_loss(z_pred, z_0_fixed)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ldt.parameters(), max_norm=10.0)
        optimizer.step()

        best_loss = min(best_loss, loss.item())

        if step % 50 == 0 or step == 1:
            grad_norm = sum(p.grad.norm().item() ** 2
                           for p in ldt.parameters()
                           if p.grad is not None) ** 0.5
            print(f"  步 {step:3d}: loss={loss.item():.4f}  grad_norm={grad_norm:.2f}")

    final_loss = best_loss
    print(f"\n最终 best loss: {final_loss:.4f}")

    if final_loss < 0.05:
        print("✅ 过拟合成功 — DiT 架构完美，可进行全量训练")
    elif final_loss < 0.2:
        print("✅ 过拟合通过 — 架构正确，全量训练需更多epoch")
    elif final_loss < 0.5:
        print("⚠️  部分收敛 — 增加模型容量或训练步数")
    else:
        print("❌ 过拟合失败 — 存在底层 bug")


if __name__ == "__main__":
    main()
