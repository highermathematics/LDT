"""自适应方差更新归一化（VN）层。

实现 LDT 论文第 4.1 节中描述的统计感知归一化。
使用 EMA 更新的全局统计量，在非平稳多元时间序列上实现
稳定的自编码器训练。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class VarianceUpdateNorm(nn.Module):
    """自适应方差更新归一化层。

    维护逐维度均值和方差的 EMA 更新统计量。
    使用从拼接的 [history, target] 窗口推导出的统计量来归一化目标。

    Args:
        num_features: 时间序列特征数 d。
        eps: 数值稳定性小常数。
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # 可学习的仿射参数 γ_d, β_d ∈ R^d
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # 运行中的 EMA 统计量，初始化为 None
        self.register_buffer("E_hat", None)    # [d]
        self.register_buffer("Var_hat", None)  # [d]
        self.register_buffer("n_batches", torch.zeros(1, dtype=torch.long))

    def compute_instance_stats(
        self, W: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """沿时间维度计算逐实例均值和方差。

        Args:
            W: 拼接窗口 [B, τ, d]，其中 τ = T + t。

        Returns:
            (E[W] ∈ R^d, Var[W] ∈ R^d) 元组，在 batch 上取平均。
        """
        tau = W.shape[1]                                # τ = T + t
        E_i = W.mean(dim=1)                             # [B, d] — 时间维度求均值
        Var_i = ((W - E_i.unsqueeze(1)) ** 2).mean(dim=1)  # [B, d]

        # 在 batch 上取平均
        E = E_i.mean(dim=0)    # [d]
        Var = Var_i.mean(dim=0)  # [d]

        return E, Var

    def update_stats(self, W: torch.Tensor) -> None:
        """使用 EMA（1/n 规则）更新运行统计量。

        Ê_{n+1} = (1/n) × (E_{n+1} + Ê_n × (n-1))
        V̂ar_{n+1} = (1/n) × (Var_{n+1} + V̂ar_n × (n-1))

        Args:
            W: 拼接窗口 [B, τ, d]。
        """
        E_new, Var_new = self.compute_instance_stats(W)

        n = self.n_batches.item() + 1

        if self.E_hat is None:
            # 第一个 batch：直接初始化
            self.E_hat = E_new.detach().clone()
            self.Var_hat = Var_new.detach().clone()
        else:
            # 以 1/n 权重进行 EMA 更新
            self.E_hat = (1.0 / n) * (E_new.detach() + self.E_hat * (n - 1))
            self.Var_hat = (1.0 / n) * (Var_new.detach() + self.Var_hat * (n - 1))

        self.n_batches.fill_(n)

    def normalize(
        self, Y: torch.Tensor, E_hat: Optional[torch.Tensor] = None,
        Var_hat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """使用全局统计量归一化目标 Y。

        Ŷ = γ_d × (Y - Ê[W]) / √(V̂ar[W] + ε) + β_d

        Args:
            Y: 目标时间序列 [B, t, d]。
            E_hat: 预计算的均值统计量 [d]，为 None 时使用 self.E_hat。
            Var_hat: 预计算的方差统计量 [d]，为 None 时使用 self.Var_hat。

        Returns:
            归一化后的目标 Ŷ [B, t, d]。
        """
        if E_hat is None:
            E_hat = self.E_hat
        if Var_hat is None:
            Var_hat = self.Var_hat

        if E_hat is None or Var_hat is None:
            raise RuntimeError("统计量尚未初始化，请先调用 update_stats()。")

        # 广播: [d] → [1, 1, d]
        E = E_hat.view(1, 1, -1)
        V = Var_hat.view(1, 1, -1)

        Y_norm = self.gamma.view(1, 1, -1) * (Y - E) / torch.sqrt(V + self.eps) \
            + self.beta.view(1, 1, -1)

        return Y_norm

    def denormalize(
        self, Y_norm: torch.Tensor, E_hat: Optional[torch.Tensor] = None,
        Var_hat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """normalize 的逆操作：恢复原始尺度。

        Y = (Ŷ - β_d) / γ_d × √(V̂ar[W] + ε) + Ê[W]

        Args:
            Y_norm: 归一化张量 [B, t, d]。
            E_hat, Var_hat: 要使用的统计量。

        Returns:
            反归一化张量 [B, t, d]。
        """
        if E_hat is None:
            E_hat = self.E_hat
        if Var_hat is None:
            Var_hat = self.Var_hat

        if E_hat is None or Var_hat is None:
            raise RuntimeError("统计量尚未初始化，请先调用 update_stats()。")

        E = E_hat.view(1, 1, -1)
        V = Var_hat.view(1, 1, -1)

        Y = (Y_norm - self.beta.view(1, 1, -1)) / self.gamma.view(1, 1, -1) \
            * torch.sqrt(V + self.eps) + E

        return Y

    def get_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取当前运行统计量。

        Returns:
            (E_hat [d], Var_hat [d]) 元组。
        """
        if self.E_hat is None or self.Var_hat is None:
            raise RuntimeError("统计量尚未初始化。")
        return self.E_hat, self.Var_hat

    def reset_stats(self) -> None:
        """重置运行统计量。"""
        self.E_hat = None
        self.Var_hat = None
        self.n_batches.fill_(0)
