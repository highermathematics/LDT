#!/usr/bin/env python
"""ESA Anomaly Dataset 预处理脚本。

将每个 Mission 下所有 channel zip 解压、读取 pickle、
重采样为小时级、对齐为统一多元矩阵，保存为 .npy 文件。

用法:
    python scripts/preprocess_esa.py --mission 1
    python scripts/preprocess_esa.py --mission all
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "ESA Anomaly Dataset"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "esa_processed"


def extract_channel_zip(zip_path: Path, dest_dir: Path) -> Path:
    """使用 PowerShell Expand-Archive 解压 channel zip（兼容增强 deflate）。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_name = zip_path.stem  # e.g. "channel_1"
    extracted_file = dest_dir / zip_name

    if extracted_file.exists():
        return extracted_file

    subprocess.run(
        [
            "powershell", "-Command",
            f"Expand-Archive -Path '{zip_path}' -DestinationPath '{dest_dir}' -Force"
        ],
        capture_output=True,
    )
    return extracted_file


def load_channel_pickle(file_path: Path) -> pd.DataFrame:
    """读取 pickled channel DataFrame。"""
    return pd.read_pickle(str(file_path))


def resample_channel(df: pd.DataFrame, freq: str = "1h") -> pd.Series:
    """将 channel DataFrame 重采样为指定频率。

    Args:
        df: 单列 DatetimeIndex DataFrame。
        freq: 目标频率（默认 '1h'）。

    Returns:
        重采样后的 Series，缺失区间填 NaN。
    """
    col_name = df.columns[0]
    # resample + mean: 15min→1h 取均值；18s→1h 同样取均值
    resampled = df[col_name].resample(freq).mean()
    return resampled


def build_multivariate_matrix(
    mission_dir: Path,
    temp_dir: Path,
    freq: str = "1h",
) -> np.ndarray:
    """将一个 Mission 的所有通道构建为多元矩阵。

    Args:
        mission_dir: 解压后的 Mission 目录（含 channels/ 子文件夹和 channels.csv）。
        temp_dir: 临时解压 channel zip 的目录。
        freq: 目标重采样频率。

    Returns:
        (data_matrix, channel_names) 元组。
        data_matrix: [timesteps, d] float32 数组。
        channel_names: 通道名称列表（长度 d）。
    """
    channels_dir = mission_dir / "channels"
    channel_zips = sorted(
        channels_dir.glob("channel_*.zip"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    print(f"  找到 {len(channel_zips)} 个通道 zip")

    resampled_series: Dict[str, pd.Series] = {}

    skipped_categorical = 0

    for i, zip_path in enumerate(channel_zips):
        ch_name = zip_path.stem
        print(f"  [{i+1}/{len(channel_zips)}] {ch_name}...", end=" ", flush=True)

        # 解压
        extracted = extract_channel_zip(zip_path, temp_dir)
        # 读取
        df = load_channel_pickle(extracted)

        # 跳过类别型通道（字符串值无法做 mean 重采样）
        if df.dtypes.iloc[0] == object:
            print(f"跳过 (类别型, {len(df):,} 行)")
            skipped_categorical += 1
            continue

        # 重采样
        sr = resample_channel(df, freq)
        resampled_series[ch_name] = sr
        print(f"{len(df):,}→{len(sr):,} 点, freq={freq}")

    if skipped_categorical > 0:
        print(f"  跳过 {skipped_categorical} 个类别型通道")

    # 构建公共时间索引
    common_index = None
    for sr in resampled_series.values():
        if common_index is None:
            common_index = sr.index
        else:
            common_index = common_index.union(sr.index)
    print(f"  公共时间索引: {len(common_index)} 个时间步")

    # 对齐所有通道
    d = len(resampled_series)
    T = len(common_index)
    matrix = np.full((T, d), np.nan, dtype=np.float32)
    channel_names = []

    for j, (ch_name, sr) in enumerate(resampled_series.items()):
        aligned = sr.reindex(common_index)
        matrix[:, j] = aligned.values.astype(np.float32)
        channel_names.append(ch_name)

    # 统计缺失情况
    nan_count = np.isnan(matrix).sum()
    nan_pct = 100 * nan_count / (T * d)
    print(f"  矩阵形状: {matrix.shape}, NaN 比例: {nan_pct:.2f}%")

    return matrix, channel_names


def fill_missing(matrix: np.ndarray) -> np.ndarray:
    """填充缺失值：先前向填充（ffill），剩余用列均值填充。"""
    df = pd.DataFrame(matrix)
    df = df.ffill().bfill()
    # 仍有 NaN 的用列均值填充
    col_means = df.mean()
    df = df.fillna(col_means)
    return df.values.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="ESA 数据集预处理")
    parser.add_argument(
        "--mission", type=str, default="all",
        help="Mission 编号: 1, 2, 3, 或 all",
    )
    parser.add_argument(
        "--freq", type=str, default="1h",
        help="重采样频率（默认: 1h）",
    )
    args = parser.parse_args()

    missions = []
    if args.mission == "all":
        missions = [1, 2, 3]
    else:
        missions = [int(args.mission)]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for m_num in missions:
        mission_name = f"ESA-Mission{m_num}"
        mission_dir = DATASET_DIR / mission_name
        if not mission_dir.exists():
            print(f"[!] {mission_dir} 不存在，跳过")
            continue

        print(f"\n{'='*60}")
        print(f"处理 {mission_name}")
        print(f"{'='*60}")

        temp_dir = OUTPUT_DIR / f"_temp_m{m_num}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        matrix, channel_names = build_multivariate_matrix(
            mission_dir, temp_dir, args.freq
        )

        # 填充缺失值
        print(f"  填充缺失值...")
        matrix_filled = fill_missing(matrix)
        remaining_nan = np.isnan(matrix_filled).sum()
        print(f"  填充后 NaN 数: {remaining_nan}")

        # 保存
        out_file = OUTPUT_DIR / f"esa_mission{m_num}.npy"
        np.save(str(out_file), matrix_filled)
        print(f"  已保存: {out_file}  (shape={matrix_filled.shape})")

        # 保存通道名
        names_file = OUTPUT_DIR / f"esa_mission{m_num}_channels.txt"
        with open(names_file, "w") as f:
            f.write("\n".join(channel_names))
        print(f"  已保存通道名: {names_file}")

        print(f"  完成: {mission_name}")


if __name__ == "__main__":
    main()
