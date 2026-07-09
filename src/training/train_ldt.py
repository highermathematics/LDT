"""第二阶段训练：潜在扩散 Transformer。

在潜在空间中训练去噪网络 x̂_θ，使用第一阶段冻结的编码器。
实现论文中的算法 1。
"""

import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.data.normalization import VarianceUpdateNorm
from src.models.autoencoder import Encoder
from src.models.diffusion import LDiffusion


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


@torch.no_grad()
def estimate_latent_sigma(
    encoder: Encoder,
    vn: VarianceUpdateNorm,
    dataloader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Estimate the global latent standard deviation used by LDT."""
    encoder.eval()
    total = 0
    sum_z = torch.zeros((), device=device)
    sum_z2 = torch.zeros((), device=device)
    E_hat, Var_hat = vn.get_stats()

    for _, Y in dataloader:
        Y = Y.to(device)
        Y_norm = vn.normalize(Y, E_hat, Var_hat)
        z = encoder.encode(Y_norm)
        total += z.numel()
        sum_z += z.sum()
        sum_z2 += (z * z).sum()

    mean = sum_z / max(total, 1)
    var = (sum_z2 / max(total, 1) - mean * mean).clamp(min=1e-8)
    return torch.sqrt(var)


def train_ldt_epoch(
    ldt: LDiffusion,
    encoder: Encoder,
    vn: VarianceUpdateNorm,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lookback: int,
    horizon: int,
    latent_sigma: torch.Tensor,
    log_interval: int = 10,
) -> Dict[str, float]:
    """训练 LDT 一个 epoch。

    Args:
        ldt: 潜在扩散模型。
        encoder: 冻结的 VAE 编码器。
        vn: VN 归一化层。
        dataloader: 训练数据加载器。
        optimizer: 去噪网络优化器。
        device: 计算设备。
        lookback: 历史长度 T。
        horizon: 预测长度 t。
        log_interval: 日志记录间隔。

    Returns:
        平均指标字典。
    """
    ldt.train()
    encoder.eval()

    total_loss = 0.0
    total_k = 0.0
    total_sigma = 0.0  # EMA of sigma_hat for inference unscaling

    pbar = tqdm(dataloader, desc="LDT 训练")
    for batch_idx, (X, Y) in enumerate(pbar):
        X = X.to(device)  # [B, T, d]
        Y = Y.to(device)  # [B, t, d]

        E_hat, Var_hat = vn.get_stats()

        # 归一化目标并编码到潜在空间（论文 Algorithm 1 第 4 行）
        Y_norm = vn.normalize(Y, E_hat, Var_hat)
        with torch.no_grad():
            z_0 = encoder.encode(Y_norm)  # [B, t, m]

        # 归一化历史窗口
        X_norm = vn.normalize(
            torch.cat([X, torch.zeros(X.shape[0], horizon, X.shape[2], device=device)], dim=1),
            E_hat, Var_hat,
        )[:, :lookback, :]  # [B, T, d]

        # 训练步
        optimizer.zero_grad()
        loss, metrics = ldt.training_step(z_0, X_norm, sigma_hat=latent_sigma)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_k += metrics["k_mean"]
        total_sigma += metrics["sigma_hat"]

        if batch_idx % log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(dataloader)
    return {"loss": total_loss / n, "k_mean": total_k / n, "sigma_hat": total_sigma / n}


@torch.no_grad()
def validate_ldt(
    ldt: LDiffusion,
    encoder: Encoder,
    vn: VarianceUpdateNorm,
    dataloader: DataLoader,
    device: torch.device,
    lookback: int,
    horizon: int,
    latent_sigma: torch.Tensor,
) -> Dict[str, float]:
    """在验证集上验证 LDT。

    与训练一致：在缩放潜在空间中计算去噪 MSE，
    并使用 self-conditioning 进行最优预测。

    Args:
        ldt: 潜在扩散模型。
        encoder: 冻结的 VAE 编码器。
        vn: VN 层。
        dataloader: 验证数据加载器。
        device: 计算设备。
        lookback: 历史长度 T。
        horizon: 预测长度 t。

    Returns:
        包含 val_loss 的字典。
    """
    ldt.eval()
    encoder.eval()

    total_loss = 0.0

    for X, Y in dataloader:
        X = X.to(device)
        Y = Y.to(device)

        E_hat, Var_hat = vn.get_stats()
        Y_norm = vn.normalize(Y, E_hat, Var_hat)
        z_0 = encoder.encode(Y_norm)

        X_norm = vn.normalize(
            torch.cat([X, torch.zeros(X.shape[0], horizon, X.shape[2], device=device)], dim=1),
            E_hat, Var_hat,
        )[:, :lookback, :]

        # 使用训练集全局潜变量尺度，保持训练/验证/推理一致
        z_0_scaled = z_0 / latent_sigma.clamp(min=1e-8)

        # 验证：始终使用自条件（论文 Algorithm 1 第 8-13 行）
        k = torch.randint(1, ldt.diffusion_steps + 1, (X.shape[0],), device=device)
        noise = torch.randn_like(z_0_scaled)
        z_k = ldt.noise_schedule.q_sample(z_0_scaled, k, noise)

        z_self_cond_init = torch.zeros_like(z_0_scaled)
        z_0_first = ldt.denoiser(z_k, X_norm, z_self_cond_init, k)
        z_0_pred = ldt.denoiser(z_k, X_norm, z_0_first, k)

        # 在缩放空间中计算 MSE（论文 Eq.6）
        total_loss += torch.nn.functional.mse_loss(z_0_pred, z_0_scaled).item()

    return {"val_loss": total_loss / len(dataloader)}


def train_stage2(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    stage1_ckpt_dir: str,
) -> str:
    """运行第二阶段 LDT 训练。

    Args:
        config: 完整配置。
        train_loader: 训练数据加载器。
        val_loader: 验证数据加载器。
        stage1_ckpt_dir: 第一阶段检查点目录路径。

    Returns:
        保存的检查点目录路径。
    """
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    print(f"第二阶段: 在 {device} 上训练 LDT")

    d_data = config.dataset.dimension
    d_latent = config.vae.latent_dim
    d_model = config.diffusion.embed_dim
    lookback = config.dataset.lookback_window
    horizon = config.dataset.prediction_length

    # 加载第一阶段检查点
    stage1_path = os.path.join(stage1_ckpt_dir, "best_model.pt")
    if not os.path.exists(stage1_path):
        raise FileNotFoundError(f"第一阶段检查点未找到: {stage1_path}")

    ckpt = torch.load(stage1_path, map_location=device)

    # 创建冻结的编码器
    vae_cfg = ckpt["vae_config"]
    d_latent = vae_cfg["d_latent"]
    encoder = Encoder(
        d_input=vae_cfg["d_data"],
        d_latent=vae_cfg["d_latent"],
        d_model=vae_cfg["d_model"],
        n_heads=vae_cfg["n_heads"],
        n_layers=vae_cfg["n_layers"],
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # VN 层
    vn = VarianceUpdateNorm(num_features=d_data).to(device)
    if "vn" in ckpt:
        vn.load_state_dict(ckpt["vn"], strict=False)
    vn.eval()
    for param in vn.parameters():
        param.requires_grad = False

    # 创建 LDT 模型
    ldt = LDiffusion(
        d_data=d_data,
        d_latent=d_latent,
        d_model=d_model,
        n_heads=config.diffusion.num_heads,
        n_layers=config.diffusion.num_layers,
        history_len=lookback,
        pred_len=horizon,
        diffusion_steps=config.diffusion.diffusion_steps,
        beta_1=config.diffusion.beta_1,
        beta_T=config.diffusion.beta_T,
        p_uncond=config.diffusion.p_uncond,
        self_cond_prob=config.diffusion.self_cond_prob,
    ).to(device)

    optimizer = torch.optim.Adam(ldt.parameters(), lr=config.diffusion.lr)

    # 检查点目录
    ckpt_dir = os.path.join(
        config.training.checkpoint_dir, f"{config.dataset.name}_stage2"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    # 在全量训练集上固定 VN 统计量与潜变量尺度
    fit_vn_stats(vn, train_loader, device)
    latent_sigma = estimate_latent_sigma(encoder, vn, train_loader, device)
    print(f"  潜变量全局 sigma_hat: {latent_sigma.item():.6f}")

    for epoch in range(1, config.diffusion.epochs + 1):
        # 训练
        train_metrics = train_ldt_epoch(
            ldt, encoder, vn, train_loader, optimizer,
            device, lookback, horizon, latent_sigma, config.training.log_interval,
        )

        # 验证
        val_metrics = validate_ldt(
            ldt, encoder, vn, val_loader, device, lookback, horizon, latent_sigma,
        )

        print(
            f"Epoch {epoch:3d} | "
            f"loss={train_metrics['loss']:.4f} "
            f"k_mean={train_metrics['k_mean']:.1f} | "
            f"val_loss={val_metrics['val_loss']:.4f}"
        )

        # 早停
        val_loss = val_metrics["val_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "ldt_state_dict": ldt.state_dict(),
                    "ldt_config": {
                        "d_data": d_data,
                        "d_latent": d_latent,
                        "d_model": d_model,
                        "n_heads": config.diffusion.num_heads,
                        "n_layers": config.diffusion.num_layers,
                        "history_len": lookback,
                        "pred_len": horizon,
                        "diffusion_steps": config.diffusion.diffusion_steps,
                        "beta_1": config.diffusion.beta_1,
                        "beta_T": config.diffusion.beta_T,
                        "p_uncond": config.diffusion.p_uncond,
                        "self_cond_prob": config.diffusion.self_cond_prob,
                    },
                    "sigma_hat": latent_sigma.detach().cpu().item(),  # 推理时反缩放用
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                os.path.join(ckpt_dir, "best_model.pt"),
            )
            print(f"  -> 已保存最佳检查点 (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.diffusion.early_stop_patience:
                print(f"在第 {epoch} 轮早停")
                break

    print(f"第二阶段完成。最佳 val_loss: {best_val_loss:.4f}")
    return ckpt_dir
