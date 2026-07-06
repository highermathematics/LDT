"""LDT 模型配置管理。

加载 YAML 配置文件并与默认值合并，生成类型化的训练和评估配置。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DatasetConfig:
    """数据集相关配置。"""

    name: str = "solar"
    prediction_length: int = 24       # 预测长度 t
    lookback_window: int = 96         # 历史窗口长度 (4 × prediction_length)
    dimension: int = 137              # d: 时间序列特征数


@dataclass
class VAECConfig:
    """第一阶段 VAE 配置。"""

    latent_dim: int = 32              # m: 潜在维度
    embed_dim: int = 128              # Transformer 嵌入维度
    num_layers: int = 3               # 编码器/解码器 Transformer 层数
    num_heads: int = 4                # 注意力头数
    kl_weight: float = 1e-8           # KL 散度正则化权重
    lr: float = 1e-3                  # 学习率
    epochs: int = 100                 # 最大训练轮数
    early_stop_patience: int = 10     # 早停耐心值


@dataclass
class DiffusionConfig:
    """第二阶段 LDT 扩散模型配置。"""

    diffusion_steps: int = 100        # K: 总扩散步数
    beta_1: float = 1e-4              # 起始噪声水平
    beta_T: float = 0.1               # 终止噪声水平
    noise_schedule: str = "sqrt"      # 噪声调度: sqrt 或 linear
    embed_dim: int = 128              # d_model: 去噪 Transformer 嵌入维度
    num_layers: int = 3               # Transformer 编码器/解码器层数
    num_heads: int = 8                # 注意力头数
    p_uncond: float = 0.1             # 无条件训练概率（CFG）
    self_cond_prob: float = 0.4       # 自条件训练概率
    guidance_strength: float = 3.0    # w: 无分类器引导强度
    lr: float = 1e-3                  # 学习率
    epochs: int = 200                 # 最大训练轮数
    early_stop_patience: int = 15     # 早停耐心值
    ddim_steps: Optional[int] = None  # DDIM 采样步数，None 表示与 diffusion_steps 相同


@dataclass
class TrainingConfig:
    """通用训练配置。"""

    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10


@dataclass
class Config:
    """总配置，组合所有子配置。"""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    vae: VAECConfig = field(default_factory=VAECConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归将 override 合并到 base 字典中。"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str) -> Config:
    """加载 YAML 配置文件并与默认值合并。

    Args:
        config_path: YAML 配置文件路径（如 'configs/solar.yaml'）。

    Returns:
        合并后的 Config 对象。
    """
    # 加载默认配置
    default_path = Path(config_path).parent.parent / "configs" / "default.yaml"
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            defaults = yaml.safe_load(f) or {}
    else:
        defaults = {}

    # 加载数据集专属配置
    with open(config_path, "r", encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}

    # 合并
    merged = _merge_dicts(defaults, overrides)

    # 构建类型化配置
    return Config(
        dataset=DatasetConfig(**merged.get("dataset", {})),
        vae=VAECConfig(**merged.get("vae", {})),
        diffusion=DiffusionConfig(**merged.get("diffusion", {})),
        training=TrainingConfig(**merged.get("training", {})),
    )
