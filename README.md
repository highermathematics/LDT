# LDT: Latent Diffusion Transformer

基于潜在扩散 Transformer 的概率时间序列预测。

本仓库从零复现了 AAAI 2024 论文：

> **Latent Diffusion Transformer for Probabilistic Time Series Forecasting**
> Shibo Feng, Chunyan Miao, Zhong Zhang, Peilin Zhao — *AAAI 2024*

---

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [数据集](#数据集)
- [训练](#训练)
- [评估](#评估)
- [评估指标说明](#评估指标说明)
- [可视化](#可视化)
- [一键运行](#一键运行)
- [ESA 卫星数据](#esa-卫星数据)
- [诊断验证](#诊断验证)
- [项目结构](#项目结构)
- [配置系统](#配置系统)
- [模型架构](#模型架构)
- [日志](#日志)
- [参考文献](#参考文献)

---

## 安装

```bash
pip install -r requirements.txt
```

**依赖：** Python 3.8+, PyTorch ≥ 1.13, GluonTS ≥ 0.13, NumPy, Pandas, Matplotlib, PyYAML, tqdm.

> Windows 下数据加载器自动使用单进程模式，无需额外配置。

---

## 快速开始

以 Solar 数据集为例，两阶段训练 + 评估：

```bash
# 训练（VAE + LDT 两阶段）
python scripts/train.py --config configs/solar.yaml --stage all

# 评估
python scripts/evaluate.py \
    --config configs/solar.yaml \
    --stage1_ckpt checkpoints/solar_stage1/best_model.pt \
    --stage2_ckpt checkpoints/solar_stage2/best_model.pt \
    --num_samples 100
```

---

## 数据集

### Benchmark（GluonTS 自动下载）

| 数据集 | 维度 d | 预测长度 | 回顾窗口 | 领域 |
|--------|--------|---------|---------|------|
| Solar | 137 | 24 | 96 | 能源 |
| Electricity | 370 | 24 | 96 | 电力 |
| Traffic | 963 | 24 | 96 | 交通 |
| Taxi | 1214 | 24 | 96 | 交通 |
| Wikipedia | 2000 | 24 | 96 | 网页流量 |

数据划分与 TimeGrad (2021) / CSDI (2021) 一致。Electricity 和 Traffic 训练集较短，会自动借入测试集前缀增强。

### ESA 卫星数据集（需手动准备）

| 数据集 | 通道数 | 时间范围 | 采样 |
|--------|--------|---------|------|
| ESA-M1 | 76 | 2000–2013 | 15min → 1h |
| ESA-M2 | 90 | 2000–2003 | 18s → 5min |

原始数据需先放入 `datasets/ESA Anomaly Dataset/`，然后运行预处理。详见 [ESA 卫星数据](#esa-卫星数据) 一节。

---

## 训练

```bash
# 两阶段全跑
python scripts/train.py --config configs/solar.yaml --stage all

# 仅第一阶段 VAE
python scripts/train.py --config configs/solar.yaml --stage 1

# 仅第二阶段 LDT（自动读取第一阶段检查点并冻结编码器）
python scripts/train.py --config configs/solar.yaml --stage 2

# 手动指定第一阶段检查点
python scripts/train.py --config configs/solar.yaml --stage 2 \
    --stage1_ckpt checkpoints/solar_stage1
```

训练日志自动保存至 `logs/` 目录（详见 [日志](#日志)）。

---

## 评估

```bash
python scripts/evaluate.py \
    --config configs/solar.yaml \
    --stage1_ckpt checkpoints/solar_stage1/best_model.pt \
    --stage2_ckpt checkpoints/solar_stage2/best_model.pt \
    --num_samples 100
```

可选参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--num_samples` | 100 | CRPS 采样数 |
| `--guidance_strength` | 配置文件值 | CFG 引导强度 |
| `--ddim_steps` | 配置文件值 | DDIM 采样步数 |
| `--max_batches` | 全部 | 限制评估批次数（快速抽查） |
| `--device` | 配置文件值 | cuda / cpu |

---

## 评估指标说明

评估脚本输出三个指标：

```
==================================================
最终结果
==================================================
CRPS-sum: 0.2530 ± 0.0150    ← 核心指标
CRPS-dim: 0.3100              ← 诊断指标
MSE:      7.700000e+02         ← 确定性精度
```

### CRPS-sum （Continuous Ranked Probability Score，越小越好）

**论文 Table 1 所用的指标。** 先将所有通道的值按时间步求和为一维序列，再计算 CRPS。衡量预测分布与真实分布之间的整体差距：

```
CRPS = (1/N) Σ|采样_i - 真实值| - (1/2N²) Σ|采样_i - 采样_j|
```

除以真实值的绝对值之和做归一化，消除量纲差异。值越小说明概率预测越准。

### CRPS-dim （诊断用，越大说明通道间差异大）

**逐通道分别计算 CRPS 然后取平均。** 不先求和，而是每个通道独立算。如果这个值明显大于 CRPS-sum，说明部分通道预测方差较大或模型在某些通道上表现较弱——这在多通道遥测数据中很常见。

### MSE （确定性预测的均方误差，越小越好）

100 个采样取**中位数**作为确定性点预测，直接与真实值计算 MSE。衡量"如果只给一个数"时的准确度。

---

## 可视化

```bash
python scripts/visualize.py \
    --config configs/solar.yaml \
    --stage1_ckpt checkpoints/solar_stage1/best_model.pt \
    --stage2_ckpt checkpoints/solar_stage2/best_model.pt \
    --mode all --output_dir plots/
```

三种模式：

| `--mode` | 内容 |
|-----------|------|
| `uncertainty` | 8 条采样轨迹叠加，展示预测不确定性 |
| `deterministic` | 采样中位数 + 80% 置信区间 vs 真实值 |
| `ablation` | LDT / LDT-g / LDT-c 三变体同图对比 |
| `all` | 以上全部 |

---

## 一键运行

```bash
# 全部 5 个 benchmark 训练 + 评估
python scripts/run_all.py

# 只跑指定数据集
python scripts/run_all.py --datasets solar,taxi

# 只跑 ESA
python scripts/run_all.py --datasets esa_m1,esa_m2

# 仅评估已有检查点 / 仅训练
python scripts/run_all.py --skip_train
python scripts/run_all.py --skip_eval
```

运行结束打印与论文 Table 1 的对比表。

---

## ESA 卫星数据

### 1. 下载原始数据

将 [ESA Anomaly Dataset](https://pan.baidu.com/s/1BT3A7V216xpGhLQcKL29hg?pwd=u7ib) 下载解压到：

```
datasets/ESA Anomaly Dataset/
├── ESA-Mission1/
│   └── channels/
│       ├── channel_1.zip
│       ├── channel_2.zip
│       └── ...
└── ESA-Mission2/
    └── channels/
        ├── channel_1.zip
        └── ...
```

### 2. 预处理

```bash
# M1 和 M2 分别预处理（15min/18s → 1h 重采样，通道对齐，NaN 填充）
python scripts/preprocess_esa.py --mission 1
python scripts/preprocess_esa.py --mission 2

# 或一起处理
python scripts/preprocess_esa.py --mission all
```

生成文件：`datasets/esa_processed/esa_mission1.npy`、`esa_mission2.npy`。

### 3. 训练 + 评估

```bash
# ESA-M1（76 通道）
python scripts/train.py --config configs/esa_m1.yaml --stage all
python scripts/evaluate.py --config configs/esa_m1.yaml \
    --stage1_ckpt checkpoints/esa_m1_stage1/best_model.pt \
    --stage2_ckpt checkpoints/esa_m1_stage2/best_model.pt

# ESA-M2（90 通道）
python scripts/train.py --config configs/esa_m2.yaml --stage all
python scripts/evaluate.py --config configs/esa_m2.yaml \
    --stage1_ckpt checkpoints/esa_m2_stage1/best_model.pt \
    --stage2_ckpt checkpoints/esa_m2_stage2/best_model.pt
```

ESA 配置使用 `guidance_strength=3.0`、`epochs=300`，训练时间比 benchmark 数据集更长。

---

## 诊断验证

正式训练前，用诊断脚本确认架构无 bug：

```bash
# 随机噪声过拟合 —— 隔离纯架构问题（loss 应快速 → 0）
python scripts/debug_synthetic.py --config configs/solar.yaml

# 真实数据单 batch 过拟合 —— 验证完整管线（loss 应在 200-500 步内从 ~1 降至 <0.05）
python scripts/debug_overfit.py --config configs/solar.yaml
```

---

## 项目结构

```
LDT/
├── configs/                    # YAML 配置（逐数据集）
│   ├── default.yaml            #   默认基类
│   ├── solar.yaml              #   Solar (d=137)
│   ├── electricity.yaml        #   Electricity (d=370)
│   ├── traffic.yaml            #   Traffic (d=963)
│   ├── taxi.yaml               #   Taxi (d=1214)
│   ├── wikipedia.yaml          #   Wikipedia (d=2000)
│   ├── esa_m1.yaml             #   ESA M1 (d=76)
│   └── esa_m2.yaml             #   ESA M2 (d=90)
├── src/
│   ├── config.py               # 配置管理（dataclass + YAML 合并）
│   ├── data/
│   │   ├── dataset.py          # 数据加载（GluonTS + 滑动窗口 + ESA .npy）
│   │   ├── normalization.py    # VN 方差更新归一化
│   │   └── optimize.py         # 训练集增强（借入测试集前缀）
│   ├── models/
│   │   ├── autoencoder.py      # Stage I: VAE（Encoder / Decoder / WGAN-GP Discriminator）
│   │   ├── diffusion.py        # Stage II: LDT 扩散模型 + DDIM 采样
│   │   ├── transformer.py      # adaLN DiT Transformer（自条件 + 交叉注意力）
│   │   └── embeddings.py       # 位置 / 时间 / 扩散步嵌入
│   ├── training/
│   │   ├── train_vae.py        # Stage I 训练循环
│   │   └── train_ldt.py        # Stage II 训练循环
│   ├── evaluation/
│   │   ├── metrics.py          # CRPS-sum / CRPS-dim / MSE
│   │   └── inference.py        # DDIM 推理管线 + 检查点重建
│   └── utils/
│       └── logger.py           # 训练日志记录（Tee 双写）
├── scripts/
│   ├── train.py                # 单数据集训练入口
│   ├── evaluate.py             # 单数据集评估入口
│   ├── visualize.py            # 可视化（不确定性 / 确定性 / 消融）
│   ├── run_all.py              # 一键训练+评估 + 汇总表
│   ├── preprocess_esa.py       # ESA 卫星数据预处理
│   ├── debug_synthetic.py      # 诊断：随机噪声过拟合
│   └── debug_overfit.py        # 诊断：单 batch 过拟合
├── datasets/                   # 数据集存放位置
├── logs/                       # 训练日志（自动生成）
└── checkpoints/                # 模型存放位置
```

---

## 配置系统

采用分层 YAML：`default.yaml` 提供基础默认值，每个数据集的 YAML 只写差异部分，加载时自动合并。

```yaml
# 示例：ESA M1 配置覆写了 VAE 潜在维度和扩散嵌入维度
dataset:
  name: esa_m1
  prediction_length: 24
  lookback_window: 96
vae:
  latent_dim: 6       # m ≈ √(76/2)
  embed_dim: 64
  epochs: 300
diffusion:
  embed_dim: 128
  guidance_strength: 3.0
  epochs: 300
```

**关键约束：**
- 潜在维度 `m` 满足 `d/m ≥ 2m`（下采样因子至少是潜在维度的 2 倍）
- 回顾窗口 = 4 × 预测长度

---

## 模型架构

### 概览

```
原始时间序列 (B, T+t, d)
       │
       ▼
    VN 归一化 ────── 统计量 (EMA, 冻结)
       │
       ▼
  ┌─────────────────────────────┐
  │        Stage I: VAE         │
  │  Encoder (3层 Transformer)  │──► Z ~ N(μ, σ²)   (B, t, m)
  │  Decoder (3层 Transformer)  │──► Ŷ               (B, t, d)
  │  Discriminator (WGAN-GP)    │──► 标量评分
  │  Loss: MSE + KL + WGAN-GP   │
  └─────────────────────────────┘
       │ (冻结 Encoder)
       ▼
  ┌─────────────────────────────────────────┐
  │           Stage II: LDT                 │
  │  Z₀ → Zₖ (前向扩散, sqrt schedule)      │
  │  Ẑθ (去噪 Transformer, adaLN)           │
  │  ←── 自条件 (GLU gate, 40% prob)        │
  │  ←── CFG 引导 (p_uncond=0.1, w=3.0)     │
  │  Loss: MSE(x₀-prediction)               │
  │  Sampling: DDIM (确定性反向)             │
  └─────────────────────────────────────────┘
       │
       ▼
    Decoder → VN 逆归一化 → 预测样本 (N, B, t, d)
```

### 核心设计

| 设计 | 说明 |
|------|------|
| **x₀ 预测** | 直接预测干净潜在表示，非噪声 ε，更适合非自回归时序 |
| **adaLN** | 扩散步嵌入控制 LayerNorm 的 shift/scale，零初始化实现恒等启动 |
| **潜在缩放** | Ẑ = Z / σ̂，σ̂ 在全体训练集估计，稳定扩散训练 |
| **自条件** | 训练时 40% 概率用上一轮预测作额外输入（stop-grad），推理时迭代更新 |
| **CFG** | 训练时 10% 概率丢弃条件，采样时 (1+w)·cond - w·uncond |
| **VN 归一化** | EMA 更新的逐维统计量，训练开始时拟合一次并冻结 |

### 检查点格式

**Stage I** (`best_model.pt`)：
```
encoder, decoder, vn, vae_config, dimension, epoch, val_loss
```

**Stage II** (`best_model.pt`)：
```
ldt_state_dict, ldt_config, sigma_hat, epoch, val_loss
```

---

## 日志

训练和评估脚本启动时自动在 `logs/` 目录创建带时间戳的日志文件。终端里 tqdm 进度条正常原地刷新，日志文件只保留完整行输出（epoch 摘要、checkpoint 保存等），干净可读。

```bash
python scripts/train.py --config configs/solar.yaml --stage all
# → logs/train_solar_stageall_2026-07-14_14-35-52.log

python scripts/evaluate.py --config configs/solar.yaml ...
# → logs/eval_solar_2026-07-14_15-00-00.log

python scripts/run_all.py
# → logs/run_all_all_2026-07-14_16-00-00.log
```

---

## 参考文献

```bibtex
@inproceedings{feng2024latent,
  title={Latent Diffusion Transformer for Probabilistic Time Series Forecasting},
  author={Feng, Shibo and Miao, Chunyan and Zhang, Zhong and Zhao, Peilin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

**相关基础工作：**
- DDPM — Ho et al., NeurIPS 2020
- DDIM — Song et al., ICLR 2021
- LDM — Rombach et al., CVPR 2022
- DiT — Peebles & Xie, ICCV 2023
- TimeGrad — Rasul et al., ICML 2021
- CSDI — Tashiro et al., NeurIPS 2021
