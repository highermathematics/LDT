"""第一阶段训练: 对称统计感知 VAE。

按照论文 Section 4.1: 对抗训练 VAE（生成器+判别器交替更新），
EMA 更新的 VN 统计量。损失: L_rec - L_adv + L_reg（判别器: log D(Y)）。
"""

import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.data.normalization import VarianceUpdateNorm
from src.models.autoencoder import VAE


@torch.no_grad()
def fit_vn_stats(
    vn: VarianceUpdateNorm,
    dataloader: DataLoader,
    device: torch.device,
) -> None:
    """Fit VN statistics once on the training set and keep them fixed."""
    vn.reset_stats()
    for X, Y in dataloader:
        W = torch.cat([X, Y], dim=1).to(device)
        vn.update_stats(W)


def train_vae_epoch(
    vae: VAE,
    vn: VarianceUpdateNorm,
    dataloader: DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int = 10,
) -> Dict[str, float]:
    """训练 VAE 一个 epoch，按论文对抗训练方式交替更新 G/D。"""
    vae.train()
    metrics_sum = {"rec_loss": 0.0, "adv_loss": 0.0, "kl_loss": 0.0,
                   "critic_loss": 0.0, "w_dist": 0.0}

    pbar = tqdm(dataloader, desc="VAE 训练")
    for batch_idx, (X, Y) in enumerate(pbar):
        X = X.to(device)
        Y = Y.to(device)

        E_hat, Var_hat = vn.get_stats()
        Y_norm = vn.normalize(Y, E_hat, Var_hat)

        # ============================================================
        # 1. 训练 Critic（判别器）: n_critic 次
        #    WGAN-GP loss = D(fake) - D(real) + λ · gradient_penalty
        # ============================================================
        n_critic = 1
        for _ in range(n_critic):
            optimizer_d.zero_grad()
            with torch.no_grad():
                y_recon_d, _, _, _ = vae(Y_norm, stochastic=True)
            disc_loss, w_dist = vae.discriminator_loss(
                Y_norm, y_recon_d
            )
            disc_loss.backward()
            optimizer_d.step()

        # ============================================================
        # 2. 训练生成器（编码器 + 解码器）
        #    WGAN loss = -mean(D(fake))
        # ============================================================
        optimizer_g.zero_grad()
        y_recon, z, mu, logvar = vae(Y_norm, stochastic=True)
        gen_loss, rec_loss, kl_loss, adv_loss = vae.generator_loss(
            Y_norm, y_recon, mu, logvar
        )
        gen_loss.backward()
        optimizer_g.step()

        metrics_sum["rec_loss"] += rec_loss.item()
        metrics_sum["adv_loss"] += adv_loss.item()
        metrics_sum["kl_loss"] += kl_loss.item()
        metrics_sum["critic_loss"] += disc_loss.item()
        metrics_sum["w_dist"] += w_dist.item()

        if batch_idx % log_interval == 0:
            pbar.set_postfix(
                adv=f"{adv_loss.item():+.4f}",
                critic=f"{disc_loss.item():.4f}",
                rec=f"{rec_loss.item():.4f}",
            )

    n = len(dataloader)
    return {k: v / n for k, v in metrics_sum.items()}


@torch.no_grad()
def validate_vae(
    vae: VAE,
    vn: VarianceUpdateNorm,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """验证 VAE。"""
    vae.eval()
    total_rec = 0.0
    for X, Y in dataloader:
        X, Y = X.to(device), Y.to(device)
        E_hat, Var_hat = vn.get_stats()
        Y_norm = vn.normalize(Y, E_hat, Var_hat)
        y_recon, z, mu, logvar = vae(Y_norm, stochastic=False)
        total_rec += F.mse_loss(y_recon, Y_norm).item()
    return {"val_rec": total_rec / len(dataloader)}


def train_stage1(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> str:
    """运行第一阶段 VAE 对抗训练（按论文）。"""
    device = torch.device(
        config.training.device if torch.cuda.is_available() else "cpu"
    )
    print(f"第一阶段: 在 {device} 上对抗训练 VAE")

    d_data = config.dataset.dimension
    d_latent = config.vae.latent_dim
    d_model = config.vae.embed_dim

    vae = VAE(
        d_data=d_data, d_latent=d_latent, d_model=d_model,
        n_heads=config.vae.num_heads, n_layers=config.vae.num_layers,
        kl_weight=config.vae.kl_weight,
        lambda_adv=config.vae.lambda_adv,
    ).to(device)

    vn = VarianceUpdateNorm(num_features=d_data).to(device)
    for param in vn.parameters():
        param.requires_grad = False

    optimizer_g = torch.optim.Adam(
        list(vae.encoder.parameters()) + list(vae.decoder.parameters()),
        lr=config.vae.lr,
    )
    optimizer_d = torch.optim.Adam(
        vae.discriminator.parameters(),
        lr=config.vae.lr,
    )

    ckpt_dir = os.path.join(
        config.training.checkpoint_dir, f"{config.dataset.name}_stage1"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    fit_vn_stats(vn, train_loader, device)

    for epoch in range(1, config.vae.epochs + 1):
        train_metrics = train_vae_epoch(
            vae, vn, train_loader, optimizer_g, optimizer_d,
            device, config.training.log_interval,
        )
        val_metrics = validate_vae(vae, vn, val_loader, device)

        print(
            f"Epoch {epoch:3d} | "
            f"rec={train_metrics['rec_loss']:.4f} "
            f"adv={train_metrics['adv_loss']:+.4f} "
            f"critic={train_metrics['critic_loss']:.4f} "
            f"W-dist={train_metrics['w_dist']:+.4f} "
            f"kl={train_metrics['kl_loss']:.6f} | "
            f"val_rec={val_metrics['val_rec']:.4f}"
        )

        val_loss = val_metrics["val_rec"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "encoder": vae.encoder.state_dict(),
                    "decoder": vae.decoder.state_dict(),
                    "vn": vn.state_dict(),
                    "vae_config": {
                        "d_data": d_data, "d_latent": d_latent,
                        "d_model": d_model, "n_heads": config.vae.num_heads,
                        "n_layers": config.vae.num_layers,
                    },
                    "dimension": d_data,
                    "epoch": epoch, "val_loss": val_loss,
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
