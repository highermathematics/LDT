#!/usr/bin/env python
"""LDT 模型主训练入口。

支持第一阶段（VAE）和第二阶段（LDT）训练。

用法:
    python scripts/train.py --config configs/solar.yaml --stage all
    python scripts/train.py --config configs/solar.yaml --stage 1
    python scripts/train.py --config configs/solar.yaml --stage 2
"""

import argparse
import os
import random
import sys
import warnings

import numpy as np
import torch

# 屏蔽 GluonTS JSON 模块警告
warnings.filterwarnings("ignore", message="Using `json`-module")

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.dataset import create_dataloaders
from src.training.train_vae import train_stage1
from src.training.train_ldt import train_stage2
from src.utils.logger import setup_logger


def set_seed(seed: int) -> None:
    """设置随机种子以确保可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="训练 LDT 模型进行概率时间序列预测。"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="数据集配置 YAML 路径（如 configs/solar.yaml）。",
    )
    parser.add_argument(
        "--stage", type=str, default="all", choices=["1", "2", "all"],
        help="训练阶段: 1 (VAE), 2 (LDT), 或 all (两者)。",
    )
    parser.add_argument(
        "--stage1_ckpt", type=str, default=None,
        help="第一阶段检查点目录路径（用于第二阶段训练）。"
             "默认: checkpoints/{dataset}_stage1/。",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="设备覆盖 (cuda/cpu)。",
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    if args.device:
        config.training.device = args.device

    # 设置随机种子
    set_seed(config.training.seed)

    # 检查设备
    if config.training.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU。")
        config.training.device = "cpu"

    # 启动日志记录（终端内容同步写入文件）
    dataset_name = config.dataset.name
    log_path = setup_logger(prefix=f"train_{dataset_name}_stage{args.stage}")
    print(f"日志文件: {log_path}\n")
    print(f"数据集: {dataset_name}")
    print(f"  维度: {config.dataset.dimension}")
    print(f"  预测长度: {config.dataset.prediction_length}")
    print(f"  历史窗口: {config.dataset.lookback_window}")
    print(f"  设备: {config.training.device}")

    # 第一阶段
    if args.stage in ("1", "all"):
        print("\n" + "=" * 60)
        print("第一阶段: 训练 VAE 自编码器")
        print("=" * 60)

        train_loader, val_loader, test_loader, dimension = create_dataloaders(
            name=dataset_name,
            prediction_length=config.dataset.prediction_length,
            lookback_window=config.dataset.lookback_window,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            train_split_ratio=config.dataset.train_split_ratio,
        )

        # 用实际维度覆盖配置
        config.dataset.dimension = dimension
        print(f"  实际维度: {dimension}")
        print(f"  训练批次数: {len(train_loader)}")
        print(f"  验证批次数: {len(val_loader)}")

        stage1_dir = train_stage1(config, train_loader, val_loader)
        print(f"第一阶段检查点已保存到: {stage1_dir}")
    else:
        stage1_dir = args.stage1_ckpt or os.path.join(
            config.training.checkpoint_dir, f"{dataset_name}_stage1"
        )

    # 第二阶段
    if args.stage in ("2", "all"):
        print("\n" + "=" * 60)
        print("第二阶段: 训练 LDT 扩散模型")
        print("=" * 60)

        # 从 Stage I 检查点读取维度，确保与 VAE 编码器一致
        import torch as _torch
        _ckpt = _torch.load(
            os.path.join(stage1_dir, "best_model.pt"),
            map_location="cpu", weights_only=True,
        )
        saved_dim = _ckpt.get("dimension", _ckpt["vae_config"]["d_data"])
        config.dataset.dimension = saved_dim

        # 重新创建 DataLoader，强制使用与 Stage I 一致的维度
        train_loader, val_loader, test_loader, dimension = create_dataloaders(
            name=dataset_name,
            prediction_length=config.dataset.prediction_length,
            lookback_window=config.dataset.lookback_window,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            force_dimension=saved_dim,
            train_split_ratio=config.dataset.train_split_ratio,
        )
        print(f"  使用 Stage I 维度: {dimension} (强制 d={saved_dim})")

        stage2_dir = train_stage2(config, train_loader, val_loader, stage1_dir)
        print(f"第二阶段检查点已保存到: {stage2_dir}")

    print("\n训练完成！")


if __name__ == "__main__":
    main()
