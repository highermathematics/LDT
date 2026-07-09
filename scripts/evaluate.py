#!/usr/bin/env python
"""LDT 模型评估脚本。

在测试集上计算 CRPS-sum 和 MSE 指标。

用法:
    python scripts/evaluate.py \
        --config configs/solar.yaml \
        --stage1_ckpt checkpoints/solar_stage1/best_model.pt \
        --stage2_ckpt checkpoints/solar_stage2/best_model.pt
"""

import argparse
import os
import sys
import warnings

import torch

# 屏蔽 GluonTS JSON 模块警告
warnings.filterwarnings("ignore", message="Using `json`-module")

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.dataset import create_dataloaders
from src.evaluation.inference import LDTInference, load_model_from_checkpoints
from src.evaluation.metrics import compute_all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="在测试集上评估 LDT 模型。"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="数据集配置 YAML 路径。",
    )
    parser.add_argument(
        "--stage1_ckpt", type=str, required=True,
        help="第一阶段检查点路径（.pt 文件）。",
    )
    parser.add_argument(
        "--stage2_ckpt", type=str, required=True,
        help="第二阶段检查点路径（.pt 文件）。",
    )
    parser.add_argument(
        "--num_samples", type=int, default=100,
        help="经验 CRPS 的采样数（默认: 100）。",
    )
    parser.add_argument(
        "--guidance_strength", type=float, default=None,
        help="CFG 引导强度（默认: 来自配置）。",
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=None,
        help="DDIM 采样步数（默认: 来自配置）。",
    )
    parser.add_argument(
        "--device", type=str, default=None,
    )
    parser.add_argument(
        "--max_batches", type=int, default=None,
        help="最大评估批次数（用于快速检查）。",
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    device = torch.device(args.device or config.training.device
                         if torch.cuda.is_available() else "cpu")

    w = args.guidance_strength or config.diffusion.guidance_strength
    ddim = args.ddim_steps or config.diffusion.ddim_steps

    print(f"正在加载 {config.dataset.name} 的模型...")
    print(f"  第一阶段: {args.stage1_ckpt}")
    print(f"  第二阶段: {args.stage2_ckpt}")
    print(f"  引导强度: {w}")
    print(f"  设备: {device}")

    # 加载推理管线
    inference = load_model_from_checkpoints(
        stage1_path=args.stage1_ckpt,
        stage2_path=args.stage2_ckpt,
        device=device,
        guidance_strength=w,
        ddim_steps=ddim,
    )
    inference.num_samples = args.num_samples

    # 加载测试数据
    print("\n正在加载测试数据...")
    _, _, test_loader, dimension = create_dataloaders(
        name=config.dataset.name,
        prediction_length=config.dataset.prediction_length,
        lookback_window=config.dataset.lookback_window,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        force_dimension=inference.decoder.d_output,
    )
    print(f"  测试批次数: {len(test_loader)}")
    print(f"  维度: {dimension}")

    # 评估
    print(f"\n正在评估（N={args.num_samples} 个采样，共 {len(test_loader)} 批次）...")
    all_metrics = []
    total = min(args.max_batches, len(test_loader)) if args.max_batches else len(test_loader)

    for batch_idx, (X, Y) in enumerate(test_loader):
        X = X.to(device)
        Y = Y.to(device)

        # 生成采样（不使用测试集 Y 更新 VN 统计量）
        samples = inference.predict(X, progress=False)

        # 计算指标
        metrics = compute_all_metrics(samples, Y)
        all_metrics.append(metrics)

        # 每 10 批打印一次进度
        if (batch_idx + 1) % 10 == 0:
            print(f"  进度: {batch_idx + 1}/{len(test_loader)}")

        if args.max_batches and batch_idx + 1 >= args.max_batches:
            break

    # 汇总结果
    weights = torch.tensor([m["num_series"] for m in all_metrics], dtype=torch.float64)
    crps_vals = torch.tensor([m["crps_sum_mean"] for m in all_metrics], dtype=torch.float64)
    mse_vals = torch.tensor([m["mse_mean"] for m in all_metrics], dtype=torch.float64)
    crps_dim_vals = torch.tensor([m["crps_dim_mean"] for m in all_metrics], dtype=torch.float64)

    crps_mean = (crps_vals * weights).sum() / weights.sum()
    mse_mean = (mse_vals * weights).sum() / weights.sum()
    crps_dim_mean = (crps_dim_vals * weights).sum() / weights.sum()
    crps_std = torch.sqrt(((crps_vals - crps_mean) ** 2 * weights).sum() / weights.sum())
    mse_std = torch.sqrt(((mse_vals - mse_mean) ** 2 * weights).sum() / weights.sum())

    print("\n" + "=" * 50)
    print("最终结果")
    print("=" * 50)
    print(f"CRPS-sum: {crps_mean:.4f} ± {crps_std:.4f}")
    print(f"CRPS-dim: {crps_dim_mean:.4f}  (诊断用逐维平均)")
    print(f"MSE:      {mse_mean:.6e} ± {mse_std:.6e}")

    # 打印论文表 1 参考值
    paper_results = {
        "solar": (0.253, 7.7e2),
        "electricity": (0.021, 1.6e5),
        "traffic": (0.040, 4.1e-4),
        "taxi": (0.125, 2.2e0),
        "wikipedia": (0.061, 2.92e7),
    }
    name = config.dataset.name
    if name in paper_results:
        ref_crps, ref_mse = paper_results[name]
        print(f"\n论文参考值（表 1）: CRPS-sum={ref_crps}, MSE={ref_mse}")


if __name__ == "__main__":
    main()
