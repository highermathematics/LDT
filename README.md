# LDT: 潜在扩散 Transformer

基于潜在扩散 Transformer 的概率时间序列预测。

本仓库复现了 AAAI 2024 论文中的 LDT 模型：

> **Latent Diffusion Transformer for Probabilistic Time Series Forecasting**
> Shibo Feng, Chunyan Miao, Zhong Zhang, Peilin Zhao
> *AAAI 2024*

## 概述

LDT 是一个两阶段、非自回归的扩散模型，用于高维多元时间序列预测：

1. **第一阶段**：对称统计感知 VAE 将高维时间序列压缩到低维潜在空间。
2. **第二阶段**：带有自条件机制和无分类器引导的潜在扩散 Transformer（LDT）
   生成未来潜在表示，再由解码器还原到时间域。

## 安装

```bash
pip install -r requirements.txt
```

**依赖：** Python 3.8+、PyTorch、GluonTS、NumPy、Matplotlib、PyYAML、tqdm。

## 项目结构

```
LDT/
├── configs/              # 每个数据集一个 YAML 配置
│   ├── default.yaml      # 默认配置
│   ├── solar.yaml
│   ├── electricity.yaml
│   ├── traffic.yaml
│   ├── taxi.yaml
│   └── wikipedia.yaml
├── src/
│   ├── config.py          # 配置管理
│   ├── data/
│   │   ├── dataset.py     # GluonTS 数据加载
│   │   └── normalization.py  # 方差更新归一化（VN）
│   ├── models/
│   │   ├── autoencoder.py # 第一阶段：VAE（编码器、解码器、判别器）
│   │   ├── diffusion.py   # 第二阶段：LDT 扩散 + DDIM 采样
│   │   ├── transformer.py # adaLN Transformer 骨干网络
│   │   └── embeddings.py  # 位置/时间/扩散步嵌入
│   ├── training/
│   │   ├── train_vae.py   # 第一阶段训练循环
│   │   └── train_ldt.py   # 第二阶段训练循环
│   └── evaluation/
│       ├── metrics.py     # CRPS-sum、MSE 指标
│       └── inference.py   # DDIM 推理管线
├── scripts/
│   ├── train.py           # 单数据集训练
│   ├── evaluate.py        # 单数据集评估
│   ├── visualize.py       # 预测可视化
│   └── run_all.py         # 一键训练+评估全部 5 个数据集
└── checkpoints/           # 模型权重保存目录
```

## 使用方法

### 训练

同时训练两个阶段：
```bash
python scripts/train.py --config configs/solar.yaml --stage all
```

仅训练第一阶段（VAE）：
```bash
python scripts/train.py --config configs/solar.yaml --stage 1
```

仅训练第二阶段（LDT，需要第一阶段的检查点）：
```bash
python scripts/train.py --config configs/solar.yaml --stage 2
```

### 一键运行全部数据集

自动训练 + 评估全部 5 个数据集，最后打印汇总对比表：
```bash
python scripts/run_all.py
```

可选参数：
```bash
python scripts/run_all.py --datasets solar,taxi    # 只跑指定数据集
python scripts/run_all.py --skip_train             # 仅评估已有检查点
python scripts/run_all.py --device cuda            # 指定设备
```

### 评估

```bash
python scripts/evaluate.py \
    --config configs/solar.yaml \
    --stage1_ckpt checkpoints/solar_stage1/best_model.pt \
    --stage2_ckpt checkpoints/solar_stage2/best_model.pt \
    --num_samples 100
```

### 可视化

```bash
python scripts/visualize.py \
    --config configs/solar.yaml \
    --stage1_ckpt checkpoints/solar_stage1/best_model.pt \
    --stage2_ckpt checkpoints/solar_stage2/best_model.pt \
    --mode all \
    --output_dir plots/
```

## 数据集

| 数据集 | 维度 | 预测长度 | 领域 |
|--------|------|---------|------|
| Solar | 137 | 24 | 能源 |
| Electricity | 370 | 24 | 电力 |
| Traffic | 963 | 24 | 交通 |
| Taxi | 1214 | 24 | 交通 |
| Wikipedia | 2000 | 24 | 网页流量 |

数据集通过 GluonTS 自动加载。

## 关键技术细节

- **x₀ 预测**：直接预测干净潜在表示（而非噪声 ε），在非自回归时间序列中表现更优。
- **自条件机制**：训练时 40% 的步数使用上一轮的预测作为额外输入（停止梯度）。
- **无分类器引导**：以 10% 的无条件丢弃率训练；采样时对条件和无条件得分进行线性组合。
- **adaLN**：由扩散步嵌入控制的自适应 LayerNorm，替代 Transformer 骨干中的标准 LayerNorm。
- **VN 归一化**：使用 EMA 统计量的自适应方差更新归一化，提升非平稳数据的训练稳定性。

## 参考文献

```bibtex
@inproceedings{feng2024latent,
  title={Latent Diffusion Transformer for Probabilistic Time Series Forecasting},
  author={Feng, Shibo and Miao, Chunyan and Zhang, Zhong and Zhao, Peilin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```
