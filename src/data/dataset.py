"""基于 GluonTS 的数据集加载与预处理。

从多元时间序列数据集中创建滑动窗口（历史, 目标）对，
遵循 TimeGrad/CSDI 的预测长度和训练/验证/测试集划分惯例。

数据集存储在项目根目录的 datasets/ 文件夹下，首次运行自动下载。
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from gluonts.dataset.repository import get_dataset as gluonts_get_dataset
from torch.utils.data import DataLoader, Dataset, TensorDataset

# 项目根目录下的数据集存放路径
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "datasets"


def _get_gluonts_dataset(name: str):
    """按名称加载 GluonTS 内置数据集（存放在项目 datasets/ 目录下）。

    Args:
        name: 数据集名称（solar, electricity, traffic, taxi, wiki）。

    Returns:
        包含 train/test 划分的 GluonTS 数据集。
    """
    # 将常用名称映射到 GluonTS 注册名称
    name_map = {
        "solar": "solar_nips",
        "electricity": "electricity_nips",
        "traffic": "traffic_nips",
        "taxi": "taxi_30min",
        "wikipedia": "wiki-rolling_nips",
    }
    gluonts_name = name_map.get(name, name)

    # 确保 datasets 目录存在
    os.makedirs(str(DATA_DIR), exist_ok=True)

    # 指定下载路径为项目内的 datasets/ 目录
    return gluonts_get_dataset(
        gluonts_name,
        path=DATA_DIR,
        regenerate=False,
    )


def load_multivariate_data(
    name: str,
    prediction_length: int,
    lookback_window: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """加载并准备多元时间序列数据。

    GluonTS 中数据以多条单变量序列形式存储，本函数将其合并为
    一条多元序列 [总时间步, 特征维度 d]，然后创建滑动窗口。

    Args:
        name: 数据集名称（solar, electricity, traffic, taxi, wikipedia）。
        prediction_length: 预测长度 t。
        lookback_window: 历史长度 T，默认为 4 × prediction_length。

    Returns:
        (train_data, val_data, test_data, dimension) 元组，
        每个划分是形状为 [N, T+t, d] 的 numpy 数组。
    """
    if lookback_window is None:
        lookback_window = 4 * prediction_length

    total_len = lookback_window + prediction_length

    # 从 GluonTS 加载
    gluonts_data = _get_gluonts_dataset(name)

    def _stack_multivariate(entries, num_series: Optional[int] = None) -> np.ndarray:
        """将多条单变量 GluonTS 序列合并为一条多元序列。

        GluonTS 中每条 entry["target"] 形状为 [1, timesteps]，
        将所有序列对齐到相同长度后堆叠为 [timesteps, d]。

        Args:
            entries: GluonTS 数据集条目列表。
            num_series: 使用的序列数量（None 表示全部使用）。
        """
        arrays = []
        for entry in entries:
            vals = entry["target"]
            if vals.ndim == 2:
                vals = vals.squeeze(0)  # [1, T] → [T]
            arrays.append(vals.astype(np.float32))

        # 只取前 num_series 条（确保 train/test 维度一致）
        if num_series is not None:
            arrays = arrays[:num_series]

        # 对齐到最短长度
        min_len = min(len(a) for a in arrays)
        trimmed = [a[:min_len] for a in arrays]

        # 堆叠为多元序列: [d, min_len] → [min_len, d]
        stacked = np.stack(trimmed, axis=1)  # [min_len, d]
        return stacked

    # 确保 train/test 使用相同数量的序列（取两者中的最小值）
    n_train = len(gluonts_data.train)
    n_test = len(gluonts_data.test)
    n_common = min(n_train, n_test)
    if n_train != n_test:
        print(f"  [!] train 有 {n_train} 条序列, test 有 {n_test} 条序列, 截取前 {n_common} 条")

    train_mv = _stack_multivariate(gluonts_data.train, num_series=n_common)
    test_mv = _stack_multivariate(gluonts_data.test, num_series=n_common)

    dimension = train_mv.shape[1]
    print(f"  合并后多元序列: train={train_mv.shape}, test={test_mv.shape}, 维度 d={dimension}")

    def _create_windows(data: np.ndarray) -> np.ndarray:
        """在多元序列上创建滑动窗口 [N, T+t, d]。"""
        if data.shape[0] < total_len:
            raise ValueError(
                f"序列太短（{data.shape[0]}），不满足 total_len={total_len}。"
            )
        N = data.shape[0] - total_len + 1
        windows = np.zeros((N, total_len, data.shape[1]), dtype=np.float32)
        for i in range(N):
            windows[i] = data[i: i + total_len]
        return windows

    all_train = _create_windows(train_mv)
    all_test = _create_windows(test_mv)

    # 将训练集划分为 train/val（90%/10%）
    np.random.seed(42)
    n_train = len(all_train)
    indices = np.random.permutation(n_train)
    split = int(0.9 * n_train)
    train_idx, val_idx = indices[:split], indices[split:]

    train_data = all_train[train_idx]
    val_data = all_train[val_idx]
    test_data = all_test

    return train_data, val_data, test_data, dimension


class TimeSeriesWindowDataset(Dataset):
    """PyTorch 数据集，将预加载的窗口切分为 (X, Y) 对。

    每个样本为 (history, target) 元组：
        history ∈ R^{T×d}, target ∈ R^{t×d}
    """

    def __init__(self, data: np.ndarray, lookback: int, horizon: int):
        """
        Args:
            data: 形状为 [N, T+t, d] 的数组，包含完整窗口。
            lookback: T，历史时间步数。
            horizon: t，预测时间步数。
        """
        self.lookback = lookback
        self.horizon = horizon
        self.data = torch.from_numpy(data)  # [N, T+t, d]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.data[idx]              # [T+t, d]
        X = window[: self.lookback]          # [T, d]
        Y = window[self.lookback:]           # [t, d]
        return X, Y


def create_dataloaders(
    name: str,
    prediction_length: int,
    lookback_window: Optional[int] = None,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """为指定数据集创建训练/验证/测试 DataLoader。

    Args:
        name: 数据集名称。
        prediction_length: 预测长度 t。
        lookback_window: 历史长度 T。
        batch_size: 批次大小。
        num_workers: DataLoader 工作进程数。

    Returns:
        (train_loader, val_loader, test_loader, dimension) 元组。
    """
    if lookback_window is None:
        lookback_window = 4 * prediction_length

    train_data, val_data, test_data, dimension = load_multivariate_data(
        name, prediction_length, lookback_window
    )

    train_ds = TimeSeriesWindowDataset(train_data, lookback_window, prediction_length)
    val_ds = TimeSeriesWindowDataset(val_data, lookback_window, prediction_length)
    test_ds = TimeSeriesWindowDataset(test_data, lookback_window, prediction_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, dimension
