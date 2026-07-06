# LDT (Latent Diffusion Transformer) 复现提示词

## 论文信息
- **标题**: Latent Diffusion Transformer for Probabilistic Time Series Forecasting
- **会议**: AAAI 2024
- **链接**: 项目目录下的 `4-AAAI2024-LDT.pdf`
- **参考实现**: PyTorch + GluonTS

## 任务目标
在 `D:\Code\cpp\LDT` 下从零搭建 LDT 模型的完整复现代码，只复刻 LDT 这一个模型（不包含基线模型），要求代码结构清晰、可运行、可复现论文中 Table 1 的 LDT 行结果。

---

## 一、项目目录结构

```
D:\Code\cpp\LDT\
├── PROMPT.md                    # 本文件
├── README.md                    # 项目说明
├── requirements.txt             # 依赖
├── configs/
│   ├── default.yaml             # 默认配置
│   ├── solar.yaml
│   ├── electricity.yaml
│   ├── traffic.yaml
│   ├── taxi.yaml
│   └── wikipedia.yaml
├── src/
│   ├── __init__.py
│   ├── config.py                # 配置管理
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # 数据集加载与预处理
│   │   └── normalization.py     # Instance Norm / VN 归一化
│   ├── models/
│   │   ├── __init__.py
│   │   ├── autoencoder.py       # Stage I: VAE (Encoder + Decoder + Discriminator)
│   │   ├── diffusion.py         # Stage II: LDT 扩散模型 (前向+反向+采样)
│   │   ├── transformer.py       # adaLN Transformer 层
│   │   └── embeddings.py        # 各类 Embedding (位置/时间/扩散步)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_vae.py         # Stage I 训练
│   │   └── train_ldt.py         # Stage II 训练
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py           # CRPS-sum, MSE
│       └── inference.py         # DDIM 采样 + 解码
├── scripts/
│   ├── train.py                 # 主训练入口
│   ├── evaluate.py              # 评估入口
│   └── visualize.py             # 可视化
└── checkpoints/                 # 模型保存目录
```

---

## 二、模型架构详解

### 2.1 整体流程 (两阶段)

**Stage I**: 训练 VAE，将高维时间序列压缩到低维潜在空间
**Stage II**: 在潜在空间训练扩散 Transformer，从历史窗口生成未来潜在表示，再用解码器还原

### 2.2 Stage I: 对称统计感知 VAE

#### 输入输出
- 输入: 拼接后的 [X_history, Y_target]，其中 X ∈ R^{B×T×d}, Y ∈ R^{B×t×d}
- 输出: 重建的 Ỹ ∈ R^{B×t×d}
- 潜在变量: Z ∈ R^{B×t×m}, m << d

#### 自适应方差更新归一化层 (VN)
对每个样本 i 拼接 W_i = [X_i, Y_i] ∈ R^{(T+t)×d}：

```
E[W_i] = (1/τ) Σ_{j=1}^{τ} W_i^j          # τ = T+t
Var[W_i] = (1/τ) Σ_{j=1}^{τ} (W_i^j - E[W_i])²
```

在 batch 间使用 EMA 更新全局统计量：
```
Ê_{n+1} = (1/n) × (E_{n+1} + Ê_n × (n-1))
V̂ar_{n+1} = (1/n) × (Var_{n+1} + V̂ar_n × (n-1))
```

归一化 target：
```
Ŷ = γ_d × (Y - Ê[W]) / √(V̂ar[W] + ε) + β_d
```
其中 γ_d, β_d ∈ R^d 是可学习参数。

#### 编码器 E
- 3 层 Transformer Encoder
- 每层: 4 个注意力头
- 下采样: 将 d 维映射到 m 维 (f = d/m, f ≥ 2m)
- 输出: Z (mean) 和 log_var (用于 KL 正则化)

#### 解码器 D
- 3 层 Transformer Encoder (对称结构)
- 4 个注意力头
- 上采样: 将 m 维映射回 d 维

#### 判别器 Disc
- 1 层 Transformer Encoder, 4 个注意力头
- 输出标量: 真/假

#### 损失函数
```
L = min_E,D max_Disc [
    L_rec(Y, D(E(Ŷ)))           # MSE 重建损失
    - L_adv(D(E(Ŷ)))            # 对抗损失 (生成器部分)
    + log Disc(Y)               # 判别器真样本
    + L_reg(Y; E, D)            # KL 散度正则化, 权重 1e-8
]
```

正则化项使潜在空间 Z 零中心且小方差。

### 2.3 Stage II: 潜在扩散 Transformer (LDT)

#### 扩散前向过程 (固定)
```
z_0 = E(Ŷ)                                    # 编码后的潜在表示
z_k = √(α̅_k) · z_0 + √(1 - α̅_k) · ε         # ε ~ N(0, I)
α̅_k = Π_{i=1}^k (1 - β_i)
```
- 噪声调度: sqrt schedule, β_1 = 1e-4, β_T = 0.1
- 扩散步数 K ∈ {50, 100, 200, 300}

#### 去噪网络 x̂_θ (核心, 非自回归)

**输入处理**:
1. 从 VN 获取 Ê[W] 和 V̂ar[W]
2. 归一化历史窗口: X̂ = (X - Ê[W]) / √(V̂ar[W] + ε)
3. 缩放潜在变量: Ẑ = Z / σ̂，其中 σ̂² = (1/btm) Σ(z - μ̂)²
4. 通过 2 层 MLP 投影块: X̂ → X̂_emb ∈ R^{T×d_model}, Ẑ → Ẑ_emb ∈ R^{t×d_model}

**网络结构 (Fig.1b)**:
- 位置嵌入 p_emb (正弦)
- 时间嵌入 s_emb (MLP)
- 扩散步嵌入 t_emb (正弦, n=4m 维) → 控制 adaLN
- 3 层 adaLN Transformer Encoder + Decoder
- 8 个注意力头
- 预测目标: ẑ_0 (干净潜在表示), 非噪声 ε

**adaLN (自适应层归一化)**:
```
γ, β = MLP(t_emb)               # 从扩散步嵌入回归
adaLN(γ, β, Y) = γ ⊙ Y + β      # 逐特征仿射变换
```
替换标准 Transformer 的 LayerNorm。

**自条件机制 (Self-Conditioning)**:
- 训练时:
  - 60% 概率: ẑ_0 = 0 (无自条件)
  - 40% 概率: 先算 ẑ_0_pred = x̂(z_k, 0, c, k), stop_gradient(ẑ_0_pred)
- 推理时:
  - ẑ_0 初始化为 0
  - 每步用上一轮预测的 ẑ_0 更新

**无分类器引导 (Classifier-Free Guidance)**:
- 以概率 p_uncond 将条件 c 设为 ∅
- 采样时线性组合:
  ```
  x̃ = (1 + w) · x̂(z_k, c, ẑ_0, k) - w · x̂(z_k, ∅, ẑ_0, k)
  ```
- w 为引导强度

#### 训练目标
```
L_LDM = E[ ||z_0 - x̂(z_k, c, ẑ_0, k)||²₂ ]
```
预测干净数据 z_0 而非噪声 ε。

#### 推理采样 (DDIM)
```
z_K ~ N(0, I)
ẑ_0_prev = 0

for k = K downto 1:
    # 自条件预测
    ẑ_0 = x̂(z_k, X̂, ẑ_0_prev, k)
    
    # 引导组合
    z_0_cond = x̂(z_k, X̂, ẑ_0, k)
    z_0_uncond = x̂(z_k, ∅, ẑ_0, k)
    z_0_guided = (1+w) · z_0_cond - w · z_0_uncond
    
    # DDIM 步
    ε_pred = (z_k - √(α̅_k) · z_0_guided) / √(1 - α̅_k)
    z_{k-1} = √(α̅_{k-1}) · z_0_guided + √(1 - α̅_{k-1}) · ε_pred
    
    ẑ_0_prev = z_0_guided

Y_pred = D(z_0)      # 解码回时间域
```

### 2.4 关键超参数汇总

| 参数 | 值 |
|------|-----|
| Lookback window | 4 × prediction_length |
| 潜在维度 m | [1/4, 1/8] × d (数据特征数) |
| 下采样因子 f | d/m, f ≥ 2m |
| VAE 嵌入维度 | 32/64/128/256 |
| VAE Encoder/Decoder 层数 | 3 |
| VAE 注意力头数 | 4 |
| VAE KL 权重 | 1e-8 |
| 扩散步数 K | 50/100/200/300 |
| 噪声调度 | sqrt schedule |
| β_1 | 1e-4 |
| β_T | 0.1 |
| 扩散模型层数 | 3 |
| 扩散模型注意力头数 | 8 |
| 扩散模型嵌入维度 | 32/64/128/256 |
| p_uncond | 需设 (如 0.1) |
| 自条件概率 | 40% (训练时) |
| 引导强度 w | 0~5 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| DDIM 采样步数 | 与 K 相同或更少 |

---

## 三、数据集

使用 GluonTS 加载：

| 数据集 | 维度 d | 预测长度 | 领域 |
|--------|--------|---------|------|
| Solar | 137 | 参考 TimeGrad 设置 | 能源 |
| Electricity | 370 | 同上 | 电力 |
| Traffic | 963 | 同上 | 交通 |
| Taxi | 1214 | 同上 | 交通 |
| Wikipedia | 2000 | 同上 | 网页流量 |

预测长度、训练/验证/测试分割应与 TimeGrad (Rasul et al. 2021) 和 CSDI (Tashiro et al. 2021) 保持一致。

---

## 四、评估指标

### CRPS-sum (连续排序概率分数)
```
CRPS(F, y) = ∫ (F(z) - 1{z ≥ y})² dz
```
- 对预测分布采样 N=100 个样本
- 使用经验 CDF 估计
- 对所有时间步求和: CRPS-sum = Σ_t CRPS(F_t, y_t)

### MSE
- 采样中位数作为确定性预测
- 计算与真实值的均方误差

---

## 五、训练流程

### Stage I 训练 (VAE)
1. 加载数据，构建 DataLoader (batch_size=64)
2. 初始化 VAE (E, D, Disc)
3. 交替训练生成器和判别器
4. 监控 L_rec, L_adv, L_reg
5. 早停，保存最佳 E 和 D

### Stage II 训练 (LDT)
1. 加载冻结的 E (不需要 D 和 Disc，仅推理时用 D)
2. 初始化去噪网络 x̂_θ
3. 按 Algorithm 1 训练循环
4. 监控 L_LDM
5. 保存最佳模型

---

## 六、消融实验

实现以下变体以验证各组件作用：

1. **LDT (完整模型)**: 自条件 + 分类器自由引导
2. **LDT-g**: 去除自条件 (ẑ_0 ≡ 0)
3. **LDT-c**: 去除引导 (w = 0)
4. **ε-prediction**: 切换去噪目标为噪声预测而非 x-prediction

---

## 七、可视化

1. **不确定性估计**: 对 Solar/Taxi，同一输入采样 8 条轨迹，绘图
2. **确定性预测**: 对 Electricity/Traffic，采样中位数 vs Ground Truth
3. **消融对比**: LDT-g vs LDT-c vs LDT 在同一图上对比

---

## 八、实现要求

1. 代码用 Python 3.8+，PyTorch，GluonTS
2. 所有模块写清楚的 docstring 和类型注解
3. 配置文件驱动，每个数据集一个 YAML
4. 支持 GPU 训练 (单卡即可)
5. Checkpoint 自动保存和恢复
6. 训练日志记录 loss 曲线
7. 每个模块可独立测试

---

## 九、参考实现资源

- 论文 PDF: `4-AAAI2024-LDT.pdf`
- 论文 Algorithm 1, 2 见正文第 6-7 页
- 扩散基础: DDPM (Ho et al. 2020), DDIM (Song et al. 2020)
- 潜在扩散: LDM (Rombach et al. 2022)
- 数据集: GluonTS 库
- 基线对比参考: TimeGrad, CSDI (两者有开源代码)
