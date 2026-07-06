"""第一阶段训练：对称统计感知 VAE。

使用生成器/判别器交替更新和 EMA 更新的 VN 统计量训练 VAE。
保存编码器和解码器供第二阶段使用。
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.data.normalization import VarianceUpdateNorm
from src.models.autoencoder import VAE


def train_vae_epoch(
    vae: VAE,
    vn: VarianceUpdateNorm,
    dataloader: DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    lookback: int,
    horizon: int,
    log_interval: int = 10,
) -> Dict[str, float]:
    """训练 VAE 一个 epoch，交替更新生成器和判别器。

    Args:
        vae: VAE 模型。
        vn: 方差更新归一化层。
        dataloader: 训练数据加载器。
        optimizer_g: 生成器（编码器+解码器）优化器。
        optimizer_d: 判别器优化器。
        device: 计算设备。
        lookback: 历史长度 T。
        horizon: 预测长度 t。
        log_interval: 日志记录间隔（批次数）。

    Returns:
        平均指标字典。
    """
    vae.train()
    metrics_sum = {"rec_loss": 0.0, "adv_loss": 0.0, "kl_loss": 0.0,
                   "disc_loss": 0.0, "gen_loss": 0.0}

    pbar = tqdm(dataloader, desc="VAE 训练")
    for batch_idx, (X, Y) in enumerate(pbar):
        X = X.to(device)  # [B, T, d]
        Y = Y.to(device)  # [B, t, d]

        B = X.shape[0]

        # 拼接用于 VN 更新
        W = torch.cat([X, Y], dim=1)  # [B, T+t, d]

        # 在完整窗口上更新 VN 统计量
        vn.update_stats(W.detach())

        # 使用更新后的统计量归一化目标
        E_hat, Var_hat = vn.get_stats()
        Y_norm = vn.normalize(Y, E_hat, Var_hat)

        # ---------------------------------------------------------------
        # 1. 训练生成器（编码器 + 解码器）
        # ---------------------------------------------------------------
        optimizer_g.zero_grad()

        y_recon, z, mu, logvar = vae(Y_norm, stochastic=True)
        gen_loss, rec_loss, kl_loss, adv_loss = vae.generator_loss(
            Y_norm, y_recon, mu, logvar
        )

        gen_loss.backward()
        optimizer_g.step()

        # ---------------------------------------------------------------
        # 2. 训练判别器
        # ---------------------------------------------------------------
        optimizer_d.zero_grad()

        with torch.no_grad():
            y_recon_detached, _, _, _ = vae(Y_norm, stochastic=True)

        disc_loss = vae.discriminator_loss(Y_norm, y_recon_detached)
        disc_loss.backward()
        optimizer_d.step()

        # 累计指标
        metrics_sum["rec_loss"] += rec_loss.item()
        metrics_sum["adv_loss"] += adv_loss.item()
        metrics_sum["kl_loss"] += kl_loss.item()
        metrics_sum["disc_loss"] += disc_loss.item()
        metrics_sum["gen_loss"] += gen_loss.item()

        if batch_idx % log_interval == 0:
            pbar.set_postfix(
                rec=f"{rec_loss.item():.4f}",
                adv=f"{adv_loss.item():.4f}",
                disc=f"{disc_loss.item():.4f}",
            )

    # 平均指标
    n = len(dataloader)
    return {k: v / n for k, v in metrics_sum.items()}


@torch.no_grad()
def validate_vae(
    vae: VAE,
    vn: VarianceUpdateNorm,
    dataloader: DataLoader,
    device: torch.device,
    lookback: int,
    horizon: int,
) -> Dict[str, float]:
    """在验证集上验证 VAE。

    Args:
        vae: VAE 模型。
        vn: VN 层（验证时统计量冻结）。
        dataloader: 验证数据加载器。
        device: 计算设备。
        lookback: 历史长度 T。
        horizon: 预测长度 t。

    Returns:
        平均验证指标字典。
    """
    vae.eval()

    total_rec = 0.0
    total_kl = 0.0

    for X, Y in dataloader:
        X = X.to(device)
        Y = Y.to(device)

        E_hat, Var_hat = vn.get_stats()
        Y_norm = vn.normalize(Y, E_hat, Var_hat)

        y_recon, z, mu, logvar = vae(Y_norm, stochastic=False)

        rec_loss = nn.functional.mse_loss(y_recon, Y_norm)
        kl_loss = vae.kl_loss(mu, logvar)

        total_rec += rec_loss.item()
        total_kl += kl_loss.item()

    n = len(dataloader)
    return {"val_rec": total_rec / n, "val_kl": total_kl / n}


def train_stage1(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> str:
    """运行第一阶段 VAE 训练。

    Args:
        config: 完整配置。
        train_loader: 训练数据加载器。
        val_loader: 验证数据加载器。

    Returns:
        保存的检查点目录路径。
    """
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    print(f"第一阶段: 在 {device} 上训练 VAE")

    d_data = config.dataset.dimension
    d_latent = config.vae.latent_dim
    d_model = config.vae.embed_dim
    lookback = config.dataset.lookback_window
    horizon = config.dataset.prediction_length

    # 创建模型
    vae = VAE(
        d_data=d_data,
        d_latent=d_latent,
        d_model=d_model,
        n_heads=config.vae.num_heads,
        n_layers=config.vae.num_layers,
        kl_weight=config.vae.kl_weight,
    ).to(device)

    # VN 归一化层
    vn = VarianceUpdateNorm(num_features=d_data).to(device)

    # 优化器
    optimizer_g = torch.optim.Adam(
        list(vae.encoder.parameters()) + list(vae.decoder.parameters()),
        lr=config.vae.lr,
    )
    optimizer_d = torch.optim.Adam(
        vae.discriminator.parameters(),
        lr=config.vae.lr,
    )

    # 检查点目录
    ckpt_dir = os.path.join(
        config.training.checkpoint_dir, f"{config.dataset.name}_stage1"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.vae.epochs + 1):
        # 每轮开始时重置 VN 统计量（重新构建 EMA）
        vn.reset_stats()

        # 预热：运行一次前向传播以初始化 VN 统计量
        for X, Y in train_loader:
            W = torch.cat([X, Y], dim=1).to(device)
            vn.update_stats(W)
            break

        # 训练
        train_metrics = train_vae_epoch(
            vae, vn, train_loader, optimizer_g, optimizer_d,
            device, lookback, horizon, config.training.log_interval,
        )

        # 验证
        val_metrics = validate_vae(vae, vn, val_loader, device, lookback, horizon)

        print(
            f"Epoch {epoch:3d} | "
            f"rec={train_metrics['rec_loss']:.4f} "
            f"adv={train_metrics['adv_loss']:.4f} "
            f"disc={train_metrics['disc_loss']:.4f} "
            f"kl={train_metrics['kl_loss']:.6f} | "
            f"val_rec={val_metrics['val_rec']:.4f}"
        )

        # 基于 val_rec 早停
        val_loss = val_metrics["val_rec"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存检查点
            torch.save(
                {
                    "encoder": vae.encoder.state_dict(),
                    "decoder": vae.decoder.state_dict(),
                    "vn": vn.state_dict(),
                    "vae_config": {
                        "d_data": d_data,
                        "d_latent": d_latent,
                        "d_model": d_model,
                        "n_heads": config.vae.num_heads,
                        "n_layers": config.vae.num_layers,
                    },
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                os.path.join(ckpt_dir, "best_model.pt"),
            )
            print(f"  -> 已保存最佳检查点 (val_rec={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.vae.early_stop_patience:
                print(f"在第 {epoch} 轮早停")
                break

    print(f"第一阶段完成。最佳 val_rec: {best_val_loss:.4f}")
    return ckpt_dir
