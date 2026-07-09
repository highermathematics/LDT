"""LDT 模型推理工具。

从训练好的 LDT 模型执行批量 DDIM 采样并解码回时间域。
使用训练时记录的 sigma_hat 对潜在变量反缩放（论文第4页）。
"""

from typing import Optional

import torch
from tqdm import tqdm

from src.data.normalization import VarianceUpdateNorm
from src.models.autoencoder import Decoder, Encoder
from src.models.diffusion import LDiffusion


class LDTInference:
    """完整 LDT 管线的推理封装。

    Args:
        ldt: 训练好的 LDT 扩散模型。
        encoder: 冻结的 VAE 编码器。
        decoder: 冻结的 VAE 解码器。
        vn: 方差更新归一化层。
        guidance_strength: 无分类器引导强度 w。
        num_samples: 概率预测的采样数。
        ddim_steps: DDIM 采样步数。
        sample_batch_size: 每批并行采样数（控制显存）。
        sigma_hat: 训练时记录的 EMA 潜在标准差，用于推理反缩放（论文第4页）。
    """

    def __init__(
        self,
        ldt: LDiffusion,
        encoder: Encoder,
        decoder: Decoder,
        vn: VarianceUpdateNorm,
        guidance_strength: float = 3.0,
        num_samples: int = 50,
        ddim_steps: Optional[int] = None,
        sample_batch_size: int = 8,
        sigma_hat: float = 1.0,
    ):
        self.ldt = ldt
        self.encoder = encoder
        self.decoder = decoder
        self.vn = vn
        self.guidance_strength = guidance_strength
        self.num_samples = num_samples
        self.ddim_steps = ddim_steps or 50
        self.sample_batch_size = sample_batch_size
        self.sigma_hat = sigma_hat

    @torch.no_grad()
    def predict(
        self,
        X_history: torch.Tensor,
        Y_target: Optional[torch.Tensor] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """批量并行 DDIM 采样 → 反缩放 → 解码。

        Args:
            X_history: 历史窗口 [B, T, d]。
            Y_target: 真实目标 [B, t, d]（仅用于 VN 统计量）。
            progress: 是否显示进度条。

        Returns:
            采样张量 [N, B, t, d]（原始数据空间）。
        """
        device = X_history.device
        B, T, d_data = X_history.shape
        t = self.ldt.pred_len
        N = self.num_samples
        K = self.sample_batch_size

        # 使用 VN 中保存的训练统计量，不随测试数据更新
        E_hat, Var_hat = self.vn.get_stats()
        X_norm = self.vn.normalize(
            torch.cat([X_history, torch.zeros(B, t, d_data, device=device)], dim=1),
            E_hat, Var_hat,
        )[:, :T, :]

        all_samples = []
        chunks = list(range(0, N, K))
        if progress and len(chunks) > 1:
            chunks = tqdm(chunks, desc=f"DDIM ({N}*{self.ddim_steps}步)")

        for start in chunks:
            n = min(K, N - start)
            X_batch = X_norm.repeat(n, 1, 1)  # [B*n, T, d]

            # DDIM 采样（缩放空间）
            z_scaled = self.ldt.sample(
                X_batch,
                guidance_strength=self.guidance_strength,
                num_steps=self.ddim_steps,
            )  # [B*n, t, m]

            # 反缩放: Z = Z_scaled × σ̂（论文第4页）
            z = z_scaled * self.sigma_hat

            # 解码 → 反归一化
            Y_pred = self.vn.denormalize(self.decoder(z), E_hat, Var_hat)  # [B*n, t, d]
            Y_pred = Y_pred.view(n, B, t, d_data)
            all_samples.append(Y_pred)

        return torch.cat(all_samples, dim=0)  # [N, B, t, d]


def load_model_from_checkpoints(
    stage1_path: str,
    stage2_path: str,
    device: torch.device,
    guidance_strength: float = 3.0,
    ddim_steps: Optional[int] = None,
) -> LDTInference:
    """从保存的检查点加载完整 LDT 推理管线。"""
    ckpt1 = torch.load(stage1_path, map_location=device)
    vae_cfg = ckpt1["vae_config"]

    encoder = Encoder(
        d_input=vae_cfg["d_data"], d_latent=vae_cfg["d_latent"],
        d_model=vae_cfg["d_model"], n_heads=vae_cfg["n_heads"],
        n_layers=vae_cfg["n_layers"],
    ).to(device)
    encoder.load_state_dict(ckpt1["encoder"])
    encoder.eval()

    decoder = Decoder(
        d_output=vae_cfg["d_data"], d_latent=vae_cfg["d_latent"],
        d_model=vae_cfg["d_model"], n_heads=vae_cfg["n_heads"],
        n_layers=vae_cfg["n_layers"],
    ).to(device)
    decoder.load_state_dict(ckpt1["decoder"])
    decoder.eval()

    vn = VarianceUpdateNorm(num_features=vae_cfg["d_data"]).to(device)
    if "vn" in ckpt1:
        vn.load_state_dict(ckpt1["vn"], strict=False)
    vn.eval()

    ckpt2 = torch.load(stage2_path, map_location=device)
    ldt_cfg = ckpt2["ldt_config"]
    sigma_hat = ckpt2.get("sigma_hat", 1.0)

    ldt = LDiffusion(
        d_data=ldt_cfg["d_data"], d_latent=ldt_cfg["d_latent"],
        d_model=ldt_cfg["d_model"], n_heads=ldt_cfg["n_heads"],
        n_layers=ldt_cfg["n_layers"], history_len=ldt_cfg["history_len"],
        pred_len=ldt_cfg["pred_len"], diffusion_steps=ldt_cfg["diffusion_steps"],
        beta_1=ldt_cfg["beta_1"], beta_T=ldt_cfg["beta_T"],
        p_uncond=ldt_cfg.get("p_uncond", 0.1),
        self_cond_prob=ldt_cfg.get("self_cond_prob", 0.5),
    ).to(device)
    ldt.load_state_dict(ckpt2["ldt_state_dict"])
    ldt.eval()

    return LDTInference(
        ldt=ldt, encoder=encoder, decoder=decoder, vn=vn,
        guidance_strength=guidance_strength, num_samples=50,
        ddim_steps=ddim_steps, sigma_hat=sigma_hat,
    )
