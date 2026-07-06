"""LDT 模型推理工具。

从训练好的 LDT 模型执行 DDIM 采样并将预测解码回时间域。
"""

from typing import List, Optional

import torch
from tqdm import tqdm

from src.data.normalization import VarianceUpdateNorm
from src.models.autoencoder import Decoder, Encoder
from src.models.diffusion import LDiffusion


class LDTInference:
    """完整 LDT 管线的推理封装。

    组合 VN 归一化、编码器、LDT 扩散模型和解码器，
    实现端到端的概率预测。

    Args:
        ldt: 训练好的 LDT 扩散模型。
        encoder: 冻结的 VAE 编码器。
        decoder: 冻结的 VAE 解码器。
        vn: 方差更新归一化层。
        guidance_strength: 无分类器引导强度 w。
        num_samples: 概率预测的采样数。
        ddim_steps: DDIM 采样步数（默认: 完整扩散步数）。
    """

    def __init__(
        self,
        ldt: LDiffusion,
        encoder: Encoder,
        decoder: Decoder,
        vn: VarianceUpdateNorm,
        guidance_strength: float = 3.0,
        num_samples: int = 100,
        ddim_steps: Optional[int] = None,
    ):
        self.ldt = ldt
        self.encoder = encoder
        self.decoder = decoder
        self.vn = vn
        self.guidance_strength = guidance_strength
        self.num_samples = num_samples
        self.ddim_steps = ddim_steps or ldt.diffusion_steps

    @torch.no_grad()
    def predict(
        self,
        X_history: torch.Tensor,
        Y_target: Optional[torch.Tensor] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """生成概率预测。

        流程：
        1. VN 归一化历史数据
        2. 编码历史（可选：编码目标用于评估）
        3. DDIM 采样 N 条潜在轨迹
        4. 将每条轨迹解码回时间域

        Args:
            X_history: 历史窗口 [B, T, d]。
            Y_target: 真实目标 [B, t, d]（仅用于 VN 统计量）。
            progress: 是否显示进度条。

        Returns:
            采样张量 [N, B, t, d]。
        """
        device = X_history.device
        B, T, d_data = X_history.shape
        t = self.ldt.pred_len

        # 如提供目标，则更新 VN 统计量
        if Y_target is not None:
            W = torch.cat([X_history, Y_target], dim=1)
            self.vn.update_stats(W)

        E_hat, Var_hat = self.vn.get_stats()

        # 归一化历史数据
        X_norm = self.vn.normalize(
            torch.cat([X_history, torch.zeros(B, t, d_data, device=device)], dim=1),
            E_hat, Var_hat,
        )[:, :T, :]  # [B, T, d]

        samples_list = []
        iterator = range(self.num_samples)
        if progress:
            iterator = tqdm(iterator, desc="生成采样")

        for _ in iterator:
            # 在潜在空间中 DDIM 采样
            z_0 = self.ldt.sample(
                X_norm,
                guidance_strength=self.guidance_strength,
                num_steps=self.ddim_steps,
            )  # [B, t, m]

            # 解码回时间域
            Y_norm_pred = self.decoder(z_0)  # [B, t, d]

            # 反归一化
            Y_pred = self.vn.denormalize(Y_norm_pred, E_hat, Var_hat)  # [B, t, d]

            samples_list.append(Y_pred.unsqueeze(0))  # [1, B, t, d]

        samples = torch.cat(samples_list, dim=0)  # [N, B, t, d]
        return samples


def load_model_from_checkpoints(
    stage1_path: str,
    stage2_path: str,
    device: torch.device,
    guidance_strength: float = 3.0,
    ddim_steps: Optional[int] = None,
) -> LDTInference:
    """从保存的检查点加载完整 LDT 推理管线。

    Args:
        stage1_path: 第一阶段检查点文件路径。
        stage2_path: 第二阶段检查点文件路径。
        device: 计算设备。
        guidance_strength: CFG 引导强度 w。
        ddim_steps: 采样的 DDIM 步数。

    Returns:
        可用于预测的 LDTInference 实例。
    """
    # 加载第一阶段
    ckpt1 = torch.load(stage1_path, map_location=device)
    vae_cfg = ckpt1["vae_config"]

    encoder = Encoder(
        d_input=vae_cfg["d_data"],
        d_latent=vae_cfg["d_latent"],
        d_model=vae_cfg["d_model"],
        n_heads=vae_cfg["n_heads"],
        n_layers=vae_cfg["n_layers"],
    ).to(device)
    encoder.load_state_dict(ckpt1["encoder"])
    encoder.eval()

    decoder = Decoder(
        d_output=vae_cfg["d_data"],
        d_latent=vae_cfg["d_latent"],
        d_model=vae_cfg["d_model"],
        n_heads=vae_cfg["n_heads"],
        n_layers=vae_cfg["n_layers"],
    ).to(device)
    decoder.load_state_dict(ckpt1["decoder"])
    decoder.eval()

    vn = VarianceUpdateNorm(num_features=vae_cfg["d_data"]).to(device)
    if "vn" in ckpt1:
        vn.load_state_dict(ckpt1["vn"])
    vn.eval()

    # 加载第二阶段
    ckpt2 = torch.load(stage2_path, map_location=device)
    ldt_cfg = ckpt2["ldt_config"]

    ldt = LDiffusion(
        d_data=ldt_cfg["d_data"],
        d_latent=ldt_cfg["d_latent"],
        d_model=ldt_cfg["d_model"],
        n_heads=ldt_cfg["n_heads"],
        n_layers=ldt_cfg["n_layers"],
        history_len=ldt_cfg["history_len"],
        pred_len=ldt_cfg["pred_len"],
        diffusion_steps=ldt_cfg["diffusion_steps"],
        beta_1=ldt_cfg["beta_1"],
        beta_T=ldt_cfg["beta_T"],
    ).to(device)
    ldt.load_state_dict(ckpt2["ldt_state_dict"])
    ldt.eval()

    return LDTInference(
        ldt=ldt,
        encoder=encoder,
        decoder=decoder,
        vn=vn,
        guidance_strength=guidance_strength,
        num_samples=100,
        ddim_steps=ddim_steps,
    )
