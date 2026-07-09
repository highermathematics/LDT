"""概率时间序列预测的评估指标。"""

import torch


def crps_empirical(
    samples: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """计算经验 CRPS，基于样本估计量。

        CRPS = (1/N) Σ|x_i - y| - (1/(2N²)) Σ|x_i - x_j|

    Args:
        samples: 预测样本 [N, B, t, d] 或 [N, B, t]。
        target: 真实值 [B, t, d] 或 [B, t]。

    Returns:
        每个位置的 CRPS，形状与 target 一致。
    """
    N = samples.shape[0]
    mae = torch.abs(samples - target.unsqueeze(0)).mean(dim=0)
    if N > 1:
        s1 = samples.unsqueeze(1)   # [N, 1, B, t, d]
        s2 = samples.unsqueeze(0)   # [1, N, B, t, d]
        pairwise_mean = (s1 - s2).abs().sum(dim=(0, 1)) / (2 * N * N)
    else:
        pairwise_mean = torch.zeros_like(mae)
    return mae - pairwise_mean


def crps_sum(
    samples: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """计算归一化 CRPS-sum。

    CRPS-sum 先把所有变量维度求和，再在聚合后的序列上计算 CRPS。
    这是 TimeGrad/LDT 多变量评估里 "sum" 指标的常用口径。

    Args:
        samples: [N, B, t, d]
        target:  [B, t, d]

    Returns:
        归一化 CRPS-sum [B]。
    """
    sample_sum = samples.sum(dim=-1)           # [N, B, t]
    target_sum = target.sum(dim=-1)            # [B, t]
    crps = crps_empirical(sample_sum, target_sum)
    target_abs_sum = target_sum.abs().sum(dim=1).clamp(min=1e-8)
    return crps.sum(dim=1) / target_abs_sum


def crps_mean_dimension(
    samples: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """逐维归一化 CRPS，用作诊断指标。"""
    crps = crps_empirical(samples, target)                    # [B, t, d]
    crps_t = crps.sum(dim=1)                                  # [B, d]
    target_abs_sum = target.abs().sum(dim=1).clamp(min=1e-8)  # [B, d]
    return (crps_t / target_abs_sum).mean(dim=1)              # [B]


def mse_median(
    samples: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """使用中位数预测计算 MSE。

    Args:
        samples: [N, B, t, d]
        target:  [B, t, d]

    Returns:
        每个批次项的 MSE [B]（所有时间和维度取均值）。
    """
    median_pred = samples.median(dim=0).values  # [B, t, d]
    se = (median_pred - target) ** 2            # [B, t, d]
    return se.mean(dim=(1, 2))                   # [B]


def compute_all_metrics(
    samples: torch.Tensor, target: torch.Tensor
) -> dict:
    """计算 CRPS-sum 和 MSE。

    Args:
        samples: [N, B, t, d]
        target:  [B, t, d]

    Returns:
        {'crps_sum_mean', 'crps_sum_std', 'mse_mean', 'mse_std'}
    """
    cs = crps_sum(samples, target)
    cs_dim = crps_mean_dimension(samples, target)
    mse = mse_median(samples, target)
    return {
        "crps_sum_mean": cs.mean().item(),
        "crps_sum_std": cs.std(unbiased=False).item(),
        "crps_dim_mean": cs_dim.mean().item(),
        "crps_dim_std": cs_dim.std(unbiased=False).item(),
        "mse_mean": mse.mean().item(),
        "mse_std": mse.std(unbiased=False).item(),
        "num_series": target.shape[0],
    }
