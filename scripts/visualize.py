#!/usr/bin/env python
"""LDT 预测可视化工具。

生成论文中描述的三种图表：
1. 不确定性估计: 8 条采样轨迹（Solar/Taxi）
2. 确定性预测: 中位数 vs 真实值（Electricity/Traffic）
3. 消融对比: LDT vs LDT-g vs LDT-c
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.dataset import create_dataloaders
from src.evaluation.inference import load_model_from_checkpoints


def plot_uncertainty(
    samples: torch.Tensor,
    target: torch.Tensor,
    history: torch.Tensor,
    save_path: str,
    num_trajectories: int = 8,
    dim_idx: int = 0,
    batch_idx: int = 0,
    title: str = "不确定性估计",
):
    """绘制不确定性估计：多条采样轨迹 vs 真实值。

    Args:
        samples: 预测采样 [N, B, t, d]。
        target: 真实值 [B, t, d]。
        history: 历史窗口 [B, T, d]。
        save_path: 图片保存路径。
        num_trajectories: 绘制的采样轨迹数。
        dim_idx: 绘制的维度索引。
        batch_idx: 绘制的批次项索引。
        title: 图表标题。
    """
    samples_np = samples[:num_trajectories, batch_idx, :, dim_idx].cpu().numpy()  # [N, t]
    target_np = target[batch_idx, :, dim_idx].cpu().numpy()  # [t]
    history_np = history[batch_idx, :, dim_idx].cpu().numpy()  # [T]

    T = len(history_np)
    t = len(target_np)

    fig, ax = plt.subplots(figsize=(12, 5))

    # 绘制历史数据
    ax.plot(range(T), history_np, color="blue", linewidth=1.5, label="历史数据")

    # 绘制采样轨迹
    for i in range(num_trajectories):
        ax.plot(
            range(T, T + t), samples_np[i],
            color="gray", linewidth=0.5, alpha=0.6,
            label="采样" if i == 0 else None,
        )

    # 绘制真实值
    ax.plot(
        range(T, T + t), target_np,
        color="red", linewidth=1.5, label="真实值",
    )

    # 中位数预测
    median_np = samples[:, batch_idx, :, dim_idx].median(dim=0).values.cpu().numpy()
    ax.plot(
        range(T, T + t), median_np,
        color="green", linewidth=1.5, linestyle="--", label="中位数",
    )

    ax.axvline(x=T - 0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("时间步")
    ax.set_ylabel("值")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"不确定性图已保存到: {save_path}")


def plot_deterministic(
    samples: torch.Tensor,
    target: torch.Tensor,
    history: torch.Tensor,
    save_path: str,
    dim_idx: int = 0,
    batch_idx: int = 0,
    title: str = "确定性预测",
):
    """绘制确定性预测：中位数 vs 真实值。

    Args:
        samples: 预测采样 [N, B, t, d]。
        target: 真实值 [B, t, d]。
        history: 历史窗口 [B, T, d]。
        save_path: 图片保存路径。
        dim_idx: 绘制的维度索引。
        batch_idx: 绘制的批次项索引。
        title: 图表标题。
    """
    median_np = samples[:, batch_idx, :, dim_idx].median(dim=0).values.cpu().numpy()
    # 10% 和 90% 分位数
    lower_np = samples[:, batch_idx, :, dim_idx].quantile(0.1, dim=0).cpu().numpy()
    upper_np = samples[:, batch_idx, :, dim_idx].quantile(0.9, dim=0).cpu().numpy()
    target_np = target[batch_idx, :, dim_idx].cpu().numpy()
    history_np = history[batch_idx, :, dim_idx].cpu().numpy()

    T = len(history_np)
    t = len(target_np)

    fig, ax = plt.subplots(figsize=(12, 5))

    # 绘制历史数据
    ax.plot(range(T), history_np, color="blue", linewidth=1.5, label="历史数据")

    # 预测区间
    ax.fill_between(
        range(T, T + t), lower_np, upper_np,
        color="gray", alpha=0.3, label="80% 置信区间",
    )

    # 中位数预测
    ax.plot(
        range(T, T + t), median_np,
        color="green", linewidth=1.5, label="中位数预测",
    )

    # 真实值
    ax.plot(
        range(T, T + t), target_np,
        color="red", linewidth=1.5, label="真实值",
    )

    ax.axvline(x=T - 0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("时间步")
    ax.set_ylabel("值")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"确定性预测图已保存到: {save_path}")


def plot_ablation_comparison(
    ldt_samples: torch.Tensor,
    ldt_g_samples: torch.Tensor,
    ldt_c_samples: torch.Tensor,
    target: torch.Tensor,
    history: torch.Tensor,
    save_path: str,
    dim_idx: int = 0,
    batch_idx: int = 0,
    title: str = "消融实验",
):
    """绘制消融对比：LDT vs LDT-g（无自条件）vs LDT-c（无引导）。

    Args:
        ldt_samples: 完整 LDT 预测 [N, B, t, d]。
        ldt_g_samples: LDT-g 预测 [N, B, t, d]。
        ldt_c_samples: LDT-c 预测 [N, B, t, d]。
        target: 真实值 [B, t, d]。
        history: 历史窗口 [B, T, d]。
        save_path: 图片保存路径。
        dim_idx: 绘制的维度索引。
        batch_idx: 绘制的批次项索引。
        title: 图表标题。
    """
    ldt_med = ldt_samples[:, batch_idx, :, dim_idx].median(dim=0).values.cpu().numpy()
    ldt_g_med = ldt_g_samples[:, batch_idx, :, dim_idx].median(dim=0).values.cpu().numpy()
    ldt_c_med = ldt_c_samples[:, batch_idx, :, dim_idx].median(dim=0).values.cpu().numpy()
    target_np = target[batch_idx, :, dim_idx].cpu().numpy()
    history_np = history[batch_idx, :, dim_idx].cpu().numpy()

    T = len(history_np)
    t = len(target_np)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    variants = [
        (axes[0], ldt_med, "LDT（完整模型）", "green"),
        (axes[1], ldt_g_med, "LDT-g（无自条件）", "orange"),
        (axes[2], ldt_c_med, "LDT-c（无引导）", "purple"),
    ]

    for ax, pred, label, color in variants:
        ax.plot(range(T), history_np, color="blue", linewidth=1, alpha=0.7, label="历史")
        ax.plot(range(T, T + t), pred, color=color, linewidth=1.5, label=label)
        ax.plot(range(T, T + t), target_np, color="red", linewidth=1, label="目标", alpha=0.7)
        ax.axvline(x=T - 0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"消融对比图已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="可视化 LDT 预测结果。")
    parser.add_argument("--config", type=str, required=True,
                        help="数据集配置 YAML 路径。")
    parser.add_argument("--stage1_ckpt", type=str, required=True,
                        help="第一阶段检查点路径。")
    parser.add_argument("--stage2_ckpt", type=str, required=True,
                        help="第二阶段检查点路径。")
    parser.add_argument("--output_dir", type=str, default="plots",
                        help="图表输出目录。")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["uncertainty", "deterministic", "ablation", "all"],
                        help="可视化模式。")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device or config.training.device
                         if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print("正在加载模型...")
    inference = load_model_from_checkpoints(
        stage1_path=args.stage1_ckpt,
        stage2_path=args.stage2_ckpt,
        device=device,
        guidance_strength=config.diffusion.guidance_strength,
    )
    inference.num_samples = 8 if args.mode == "uncertainty" else 100

    # 获取一批测试数据
    _, _, test_loader, _ = create_dataloaders(
        name=config.dataset.name,
        prediction_length=config.dataset.prediction_length,
        lookback_window=config.dataset.lookback_window,
        batch_size=config.training.batch_size,
        num_workers=0,  # 可视化使用单进程
    )

    X, Y = next(iter(test_loader))
    X = X.to(device)
    Y = Y.to(device)

    print(f"正在生成预测（N={inference.num_samples}）...")
    samples = inference.predict(X, Y, progress=True)

    name = config.dataset.name

    if args.mode in ("uncertainty", "all"):
        plot_uncertainty(
            samples, Y, X,
            save_path=os.path.join(args.output_dir, f"{name}_uncertainty.png"),
            num_trajectories=8,
            title=f"{name.capitalize()} - 不确定性估计（LDT）",
        )

    if args.mode in ("deterministic", "all"):
        inference.num_samples = 100
        samples_100 = inference.predict(X, Y, progress=False)
        plot_deterministic(
            samples_100, Y, X,
            save_path=os.path.join(args.output_dir, f"{name}_deterministic.png"),
            title=f"{name.capitalize()} - 确定性预测（LDT）",
        )

    print("可视化完成！")


if __name__ == "__main__":
    main()
