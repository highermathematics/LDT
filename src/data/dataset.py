"""基于 GluonTS 的数据集加载与预处理。

从多元时间序列数据集中创建滑动窗口（历史, 目标）对，
遵循 TimeGrad/CSDI 的预测长度和训练/验证/测试集划分惯例。

数据集存储在项目根目录的 datasets/ 文件夹下，首次运行自动下载。
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        path=str(DATA_DIR),
        regenerate=False,
    )


def _rolling_window(
    data: np.ndarray, lookback: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """从多元时间序列创建滑动窗口对。

    Args:
        data: 完整时间序列，形状为 [total_length, d]。
        lookback: 历史时间步数 T。
        horizon: 待预测的未来时间步数 t。

    Returns:
        (X, Y) 元组，X ∈ [N, T, d], Y ∈ [N, t, d]。
    """
    total_len = lookback + horizon
    if data.shape[0] < total_len:
        raise ValueError(
            f"时间序列太短（{data.shape[0]}），"
            f"无法满足 lookback={lookback} + horizon={horizon}"
        )

    N = data.shape[0] - total_len + 1
    X = np.zeros((N, lookback, data.shape[1]), dtype=np.float32)
    Y = np.zeros((N, horizon, data.shape[1]), dtype=np.float32)

    for i in range(N):
        X[i] = data[i: i + lookback]
        Y[i] = data[i + lookback: i + total_len]

    return X, Y


def load_multivariate_data(
    name: str,
    prediction_length: int,
    lookback_window: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """加载并准备多元时间序列数据。

    加载 GluonTS 数据集并创建用于训练、验证和测试的滑动窗口划分。

    Args:
        name: 数据集名称（solar, electricity, traffic, taxi, wikipedia）。
        prediction_length: 预测长度 t。
        lookback_window: 历史长度 T，默认为 4 × prediction_length。

    Returns:
        (train_data, val_data, test_data, dimension) 元组，
        每个划分是形状为 [N, total_len, d] 的 numpy 数组，
        total_len = lookback_window + prediction_length。
    """
    if lookback_window is None:
        lookback_window = 4 * prediction_length

    total_len = lookback_window + prediction_length

    # 从 GluonTS 加载
    gluonts_data = _get_gluonts_dataset(name)

    # 提取时间序列数组
    def extract_series(entries) -> List[np.ndarray]:
        """从 GluonTS 条目中提取目标值。"""
        series_list = []
        for entry in entries:
            vals = entry["target"].T  # [d, total_timesteps] → [total_timesteps, d]
            if vals.ndim == 1:
                vals = vals[:, np.newaxis]  # 单变量: [T, 1]
            series_list.append(vals.astype(np.float32))
        return series_list

    train_series = extract_series(gluonts_data.train)
    test_series = extract_series(gluonts_data.test)

    # 确定数据维度
    dimension = train_series[0].shape[1]

    def create_windows(series_list: List[np.ndarray]) -> np.ndarray:
        """对每条序列应用滑动窗口并堆叠。"""
        windows = []
        for s in series_list:
            if s.shape[0] >= total_len:
                X, Y = _rolling_window(s, lookback_window, prediction_length)
                windows.append(np.concatenate([X, Y], axis=1))  # [N, T+t, d]
        if not windows:
            raise ValueError(
                f"无法创建窗口。序列太短，不满足 total_len={total_len}。"
            )
        return np.concatenate(windows, axis=0)

    # 创建窗口
    all_train = create_windows(train_series)
    all_test = create_windows(test_series)

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
