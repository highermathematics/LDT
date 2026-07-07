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
                   "disc_loss": 0.0, "gen_loss": 0.0, "r1": 0.0}

    pbar = tqdm(dataloader, desc="VAE 训练")
    for batch_idx, (X, Y) in enumerate(pbar):
        X = X.to(device)
        Y = Y.to(device)

        # 拼接 [X, Y] 更新 VN 统计量
        W = torch.cat([X, Y], dim=1)
        vn.update_stats(W.detach())

        E_hat, Var_hat = vn.get_stats()
        Y_norm = vn.normalize(Y, E_hat, Var_hat)

        # ============================================================
        # 1. 训练生成器 (编码器 + 解码器): min L_rec - L_adv + L_reg
        # ============================================================
        optimizer_g.zero_grad()
        y_recon, z, mu, logvar = vae(Y_norm, stochastic=True)
        gen_loss, rec_loss, kl_loss, adv_loss = vae.generator_loss(
            Y_norm, y_recon, mu, logvar
        )
        gen_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(vae.encoder.parameters()) + list(vae.decoder.parameters()),
            max_norm=5.0,
        )
        optimizer_g.step()

        # ============================================================
        # 2. 训练判别器: max log D(real) + log(1 - D(fake)) + R1
        # ============================================================
        optimizer_d.zero_grad()
        with torch.no_grad():
            y_recon_detached, _, _, _ = vae(Y_norm, stochastic=True)
        disc_loss, r1_penalty = vae.discriminator_loss(
            Y_norm.detach(), y_recon_detached,
            r1_gamma=1.0, label_smooth=0.1,
        )
        disc_loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.discriminator.parameters(), max_norm=5.0)
        optimizer_d.step()

        metrics_sum["rec_loss"] += rec_loss.item()
        metrics_sum["adv_loss"] += adv_loss.item()
        metrics_sum["kl_loss"] += kl_loss.item()
        metrics_sum["disc_loss"] += disc_loss.item()
        metrics_sum["gen_loss"] += gen_loss.item()
        metrics_sum["r1"] += r1_penalty.item()

        if batch_idx % log_interval == 0:
            pbar.set_postfix(
                rec=f"{rec_loss.item():.4f}",
                adv=f"{adv_loss.item():.4f}",
                disc=f"{disc_loss.item():.4f}",
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
    ).to(device)

    vn = VarianceUpdateNorm(num_features=d_data).to(device)

    optimizer_g = torch.optim.Adam(
        list(vae.encoder.parameters()) + list(vae.decoder.parameters()),
        lr=config.vae.lr,
    )
    optimizer_d = torch.optim.Adam(
        vae.discriminator.parameters(),
        lr=config.vae.lr * config.vae.disc_lr_mult,
    )

    ckpt_dir = os.path.join(
        config.training.checkpoint_dir, f"{config.dataset.name}_stage1"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.vae.epochs + 1):
        vn.reset_stats()
        for X, Y in train_loader:
            vn.update_stats(torch.cat([X, Y], dim=1).to(device))
            break

        train_metrics = train_vae_epoch(
            vae, vn, train_loader, optimizer_g, optimizer_d,
            device, config.training.log_interval,
        )
        val_metrics = validate_vae(vae, vn, val_loader, device)

        print(
            f"Epoch {epoch:3d} | "
            f"rec={train_metrics['rec_loss']:.4f} "
            f"adv={train_metrics['adv_loss']:.4f} "
            f"disc={train_metrics['disc_loss']:.4f} "
            f"r1={train_metrics['r1']:.4f} "
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
