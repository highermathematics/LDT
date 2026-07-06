"""概率时间序列预测的评估指标。

实现 LDT 论文（AAAI 2024）中使用的 CRPS-sum（连续排序概率分数，
在时间步上求和）和 MSE。
"""

from typing import Optional

import torch


def crps_empirical(
    samples: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """计算单个时间步和维度的经验 CRPS。

    CRPS(F, y) = ∫ (F(z) - 1{z ≥ y})² dz

    使用基于样本的估计量：
        CRPS = (1/N) Σ|x_i - y| - (1/(2N²)) Σ|x_i - x_j|

    Args:
        samples: 预测样本 [N, B, t, d]（N 个采样）。
        target: 真实值 [B, t, d]。

    Returns:
        每个 (batch, timestep, dimension) 的 CRPS [B, t, d]。
    """
    N = samples.shape[0]

    # 平均绝对误差项: (1/N) Σ|x_i - y|
    mae = torch.abs(samples - target.unsqueeze(0)).mean(dim=0)  # [B, t, d]

    # 成对绝对差项: (1/(2N²)) Σ|x_i - x_j|
    # 使用向量化成对展开高效计算
    if N > 1:
        samples_expanded = samples.unsqueeze(1)    # [N, 1, B, t, d]
        samples_expanded_2 = samples.unsqueeze(0)  # [1, N, B, t, d]
        pairwise_diff = torch.abs(samples_expanded - samples_expanded_2)  # [N, N, B, t, d]
        pairwise_mean = pairwise_diff.sum(dim=(0, 1)) / (2 * N * N)       # [B, t, d]
    else:
        pairwise_mean = torch.zeros_like(mae)

    crps = mae - pairwise_mean  # [B, t, d]
    return crps


def crps_sum(
    samples: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """计算在所有时间步上求和的 CRPS-sum（沿时间维求和）。

    CRPS-sum = Σ_t CRPS(F_t, y_t)

    与 LDT 论文表 1 中使用的指标相同。

    Args:
        samples: 预测样本 [N, B, t, d]。
        target: 真实值 [B, t, d]。

    Returns:
        每个批次项的 CRPS-sum [B]。
    """
    crps = crps_empirical(samples, target)  # [B, t, d]
    return crps.sum(dim=(1, 2))  # [B] — 沿时间和维度求和


def mse_median(
    samples: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """使用中位数预测作为确定性预报计算 MSE。

    Args:
        samples: 预测样本 [N, B, t, d]。
        target: 真实值 [B, t, d]。

    Returns:
        每个批次项的 MSE [B]。
    """
    median_pred = samples.median(dim=0).values  # [B, t, d]
    se = (median_pred - target) ** 2            # [B, t, d]
    return se.mean(dim=(1, 2))                   # [B] — 沿时间和维度取均值


def compute_all_metrics(
    samples: torch.Tensor, target: torch.Tensor
) -> dict:
    """同时计算 CRPS-sum 和 MSE 指标。

    Args:
        samples: 预测样本 [N, B, t, d]。
        target: 真实值 [B, t, d]。

    Returns:
        包含 'crps_sum'（均值±标准差）和 'mse'（均值±标准差）的字典。
    """
    cs = crps_sum(samples, target)
    mse = mse_median(samples, target)

    return {
        "crps_sum_mean": cs.mean().item(),
        "crps_sum_std": cs.std().item(),
        "mse_mean": mse.mean().item(),
        "mse_std": mse.std().item(),
    }
