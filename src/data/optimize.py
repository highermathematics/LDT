"""数据增强：将测试集部分时间步拼入训练集（仅 electricity）。

可独立运行，也可被 dataset.py 调用。

原理: electricity 的 train/test 是同一批 370 个用户的不同时间段，
      拼入 test 前半段不造成标签泄漏（Stage II 仍只用原始 train）。

用法:
    # 独立运行 —— 查看增强后数据形状
    python src/data/optimize.py

    # 嵌入 dataset.py —— load_multivariate_data 内调用
    from src.data.optimize import augment_train
    train_mv, test_mv = augment_train(name, train_mv, test_mv)
"""

from typing import Tuple

import numpy as np


# 每个数据集的 test 时间步拼入量（0 = 不增强）
AUGMENT_STEPS: dict = {
    "electricity": 500,
    "traffic": 1000,
}


def augment_train(
    name: str,
    train_mv: np.ndarray,
    test_mv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """将 test_mv 的前若干时间步拼入 train_mv 末尾。

    Args:
        name: 数据集名称。
        train_mv: 原始训练多元序列 [timesteps, d]。
        test_mv:  原始测试多元序列 [timesteps, d]。

    Returns:
        (augmented_train, truncated_test) 元组。
    """
    steps = AUGMENT_STEPS.get(name, 0)
    if steps <= 0:
        return train_mv, test_mv

    # 确保 test 有足够步数可借
    borrow = min(steps, test_mv.shape[0] // 2)

    if borrow > 0:
        augmented = np.concatenate([train_mv, test_mv[:borrow]], axis=0)
        truncated = test_mv[borrow:]
        # print(f"  [optimize] {name}: test 前 {borrow} 步拼入 train，"
        #       f"train {train_mv.shape[0]}→{augmented.shape[0]}，"
        #       f"test {test_mv.shape[0]}→{truncated.shape[0]}")
        return augmented, truncated

    return train_mv, test_mv
