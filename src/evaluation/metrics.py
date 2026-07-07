"""概率时间序列预测的评估指标。

实现 LDT 论文（AAAI 2024）中使用的 CRPS-sum 和 MSE。
与 TimeGrad/CSDI 保持一致的计算方式：按维度归一化后取平均。
"""

import torch


def crps_empirical(
    samples: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """计算经验 CRPS，基于样本估计量。

        CRPS = (1/N) Σ|x_i - y| - (1/(2N²)) Σ|x_i - x_j|

    Args:
        samples: 预测样本 [N, B, t, d]。
        target: 真实值 [B, t, d]。

    Returns:
        每个 (batch, timestep, dim) 的 CRPS [B, t, d]。
    """
    N = samples.shape[0]
    mae = torch.abs(samples - target.unsqueeze(0)).mean(dim=0)  # [B, t, d]
    if N > 1:
        s1 = samples.unsqueeze(1)   # [N, 1, B, t, d]
        s2 = samples.unsqueeze(0)   # [1, N, B, t, d]
        pairwise_mean = (s1 - s2).abs().sum(dim=(0, 1)) / (2 * N * N)
    else:
        pairwise_mean = torch.zeros_like(mae)
    return mae - pairwise_mean  # [B, t, d]


def crps_sum(
    samples: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """计算归一化 CRPS-sum，与 TimeGrad/LDT 论文一致。

    计算方式：
        1. 逐维计算 CRPS ∊ R^{B×t×d}
        2. 沿时间求和 → [B, d]
        3. 逐维除以 Σ_t |target| 做归一化 → [B, d]
        4. 对维度取平均 → [B]

    Args:
        samples: [N, B, t, d]
        target:  [B, t, d]

    Returns:
        归一化 CRPS-sum [B]。
    """
    crps = crps_empirical(samples, target)                    # [B, t, d]
    crps_t = crps.sum(dim=1)                                   # [B, d] — 沿时间求和
    target_abs_sum = target.abs().sum(dim=1).clamp(min=1e-8)  # [B, d] — 归一化因子
    crps_norm = crps_t / target_abs_sum                        # [B, d] — 逐维归一化
    return crps_norm.mean(dim=1)                                # [B] — 维度取平均


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
    mse = mse_median(samples, target)
    return {
        "crps_sum_mean": cs.mean().item(),
        "crps_sum_std": cs.std().item(),
        "mse_mean": mse.mean().item(),
        "mse_std": mse.std().item(),
    }
