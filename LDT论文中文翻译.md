# 潜在扩散Transformer用于概率性多元时间序列预测

## 第三十八届AAAI人工智能会议 (AAAI-24)

**作者：** Shibo Feng¹,²,³, Chunyan Miao¹,²,³, Zhong Zhang⁴, Peilin Zhao⁴*

¹ 新加坡南洋理工大学（NTU）计算机科学与工程学院
² NTU-UBC积极老龄化卓越联合研究中心（LILY），南洋理工大学
³ 微众银行-NTU金融科技联合研究院，南洋理工大学
⁴ 腾讯AI Lab，中国深圳

邮箱：{shibo001, ascymiao}@ntu.edu.sg, {todzhang, masonzhao}@tencent.com

*通讯作者

版权所有 © 2024，人工智能促进协会（www.aaai.org）。保留所有权利。

---

## 摘要

多元时间序列的概率预测是一项极具挑战性但又具有重要实践意义的任务。本研究提出将高维多元时间序列预测浓缩为潜在空间时间序列生成问题，以提高每个时间戳的表达能力，使预测更加可控。为解决现有工作难以扩展到高维多元时间序列的问题，我们提出了一个名为**潜在扩散Transformer（Latent Diffusion Transformer，简称LDT）**的潜在多元时间序列扩散框架，该框架由对称的统计感知自编码器和基于扩散的条件生成器组成，以实现这一思想。通过精心设计，时间序列自编码器能够通过考虑动态统计量，将多元时间戳模式压缩成简洁的潜在表示。然后，基于扩散的条件生成器能够在连续潜在空间上，通过一种以非自回归方式建模的新型自条件引导机制，高效地生成逼真的多元时间戳值。大量实验表明，我们的模型在许多流行的高维多元时间序列数据集上取得了最先进的性能。

---

## 1. 引言

时间序列数据的预测在金融（Sezer, Gudelek和Ozbayoglu 2020）、能源（Cao等 2020）、交通（Liu等 2016；Feng等 2023）和人体识别（Rao和Miao 2022；Rao等 2021）等多个领域都至关重要。多元预测在实际应用中普遍存在，在工业领域更为重要和流行。例如，电力公司分析来自众多客户的数十亿数据点来监测电力消耗，这反映了此任务的复杂性和重要性。

潜在扩散模型（Rombach等 2022）是一种简单有效的方法，能够在不降低质量的前提下，显著提高去噪扩散模型（Ho, Jain和Abbeel 2020）的训练和采样效率。这一类潜在生成模型近年来获得了显著的认可和成就，特别是在处理高维数据类型方面，如高分辨率图像（Ho, Jain和Abbeel 2020；Takagi和Nishimoto 2023）、自然语言（Li等 2022a；Yuan等 2022）和音频（Huang等 2022；Ruan等 2023）。

多元时间序列预测旨在准确预测未来趋势，但由于其复杂性和计算需求而面临挑战。使用深度自回归模型（Woo等 2022；Liu等 2022；Wu等 2020）来预测未来时间戳的常见方法受到数据高维度和模型结构的阻碍。这导致了两个主要问题：显著的算力资源需求限制了可扩展性，以及预测中误差的累积，特别是在高维序列中。因此，迫切需要一种能够高效且有效地预测未来趋势、降低计算负载并提高速度的创新性预测框架。

潜在空间生成是时间序列预测中一种高效的替代方案，它采用预训练的自编码器来减轻数据冗余，将生成从时间域转移到潜在域。主要挑战是分布偏移问题，因为均值、方差等统计属性随时间变化（Fan等 2023；Kim等 2021）。传统模型在使用历史时间戳作为自编码器输入时，存在数值不准确的问题。我们的新方法在预训练中动态更新统计参数，确保为每个时间戳提供高质量、准确的潜在表示。

现有的多元时间序列扩散模型面临两个主要问题。第一，自回归结构（Rasul等 2021）导致远距离预测性能差、误差累积和推理速度慢。第二，大多数模型在低维序列上表现出色（Tashiro等 2021；Alcaraz和Strodthoff 2022；Shen和Kwok 2023），但在高维度上表现不佳。为克服这些挑战，我们的方法强调使用非自回归、资源高效的去噪网络进行预测。我们引入了一种基于自条件的Transformer去噪结构，该结构能够在连续潜在空间中有效地对时间变量进行去噪，并结合协变量特征，类似于图像生成中的策略（Chen, Zhang和Hinton 2022；Yang等 2022；Ho和Salimans 2022）。与自回归模型相比，这个Transformer扩散模块显著降低了计算复杂度、资源使用，并提高了采样速度。

在本文中，我们为多元概率预测引入了一种新颖的两阶段、非自回归扩散架构。我们在多个真实世界数据集上的实验表明，该模型在高维多元时间序列预测方面超越了现有的最先进生成模型。我们工作的**主要贡献**如下：

- **引入LDT模型**——多元时间序列预测的新方法，利用潜在空间表示在高维场景下实现高精度预测。
- **开发了实用的LDT结构**，具有独特的自条件机制和非自回归Transformer，能实现约束的自条件预测。
- **进行了大量实验**，使用多个多元预测数据集，展示了LDT在多元时间序列概率预测方面相较于近期最先进预测方法的卓越性能。

---

## 2. 背景

### 扩散模型

扩散模型（Ho, Jain和Abbeel 2020）是一种概率生成模型，旨在通过迭代去噪正态分布变量来生成目标数据分布p(x)。扩散概率模型由固定的前向过程和可学习的反向过程组成，这是一个长度为T的马尔可夫链。

**前向过程。** 这是一个从数据分布到高斯分布的转移和固定扩散过程。给定数据样本 $x \in \mathbb{R}^d \sim p(x)$ 和一些潜在变量 $\{z_0, z_1, \ldots, z_T\}$，这些变量在数据分布和高斯分布之间随着扩散步数的增加进行插值。前向过程可以正式描述为由一系列方差 $\beta_t$ 参数化的马尔可夫链，其中 $\alpha_t := 1 - \beta_t$：

$$q(z_{1:T}|z_0) = \prod_{t=1}^{T} q(z_t|z_{t-1}) \quad \text{(1)}$$

其中 $q(z_t|z_{t-1}) \sim \mathcal{N}(\sqrt{1 - \beta_t}z_{t-1}, \beta_t I)$。

由于扩散过程的步数越多，添加的噪声越多，$q(z_t|x)$ 具有一个闭式解，可以用一般形式描述：

$$z_t \sim q(z_t|x) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t}x, (1 - \bar{\alpha}_t)I\right) \quad \text{(2)}$$

$$z_t = \sqrt{\bar{\alpha}_t}x + \sqrt{1 - \bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\bar{\alpha}_t = \prod_{i=1}^{t}(1 - \beta_i) \in (0, 1)$，$z_0 = x$，且 $z_T \sim \mathcal{N}(0, I)$。随着扩散步数的增加，潜在变量 $z_t$ 变得越来越嘈杂，直到 $z_T$ 近似为高斯变量，与起点 $x$ 无关。

**反向过程。** 可学习的反向过程由反向马尔可夫链定义：$p_\theta(z_{0:T}) = p(z_T)\prod_{t=t}^{T} p_\theta(z_{t-1}|z_t)$，其中 $p(z_T) = \mathcal{N}(0, I)$ 是已知的。$p_\theta(z_{t-1}|z_t)$ 可以近似地由以下方程推导：$q(z_{t-1}|z_t, x) = \mathcal{N}(\tilde{\mu}_t(z_t, x), \tilde{\beta}_t^2 I)$，其中 $\tilde{\mu}_t(z_t, x)$ 具有闭式解，$\tilde{\beta}_t$ 是超参数。为得到 $p_\theta(z_{t-1}|z_t)$，我们训练一个去噪网络 $\epsilon_\theta$ 来近似给定噪声潜在变量 $z_t$ 和时间步 $t$ 的 $x$：

$$\mathcal{L} = \mathbb{E}_{p(x), t \sim \mathcal{U}\{1,\ldots,T\}, z_t}\left[ \|\hat{x}_\theta(z_t, t) - x\|_2^2 \right] \quad \text{(3)}$$

其中 $z_t \sim q(z_t|x)$，$p_\theta(z_{t-1}|z_t)$ 是 $q(z_{t-1}|z_t, \hat{x}_\theta(z_t, t))$ 的近似，这使得我们能够从闭式中采样，并通过采样 $z_{t-1}$ 进行去噪，直到得到 $z_0 = x \sim p(x)$：

$$z_{t-1} \sim p_\theta(z_{t-1}|z_t) = q(z_{t-1}|z_t, \hat{x}_\theta(z_t, t)) \quad \text{(4)}$$

对于从训练好的扩散模型进行采样，我们利用Song等（2020）的推理分布，因此从 $q(z_{t-1}|z_t, x) = \mathcal{N}(\mu_q(z_t, x), \sigma_q(t)I)$ 推导采样：

$$q(z_{t-1}|z_t, x) \sim \mathcal{N}\left(z_{t-1}; \gamma_{t1}z_t + \gamma_{t2}x, \sigma_t I\right) \quad \text{(5)}$$

当设置 $\sigma_q(t) = 0$ 时得到确定性DDIM采样器，其中 $\gamma_{t1} = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}$，$\gamma_{t2} = \frac{\sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)}{1 - \bar{\alpha}_t}$，$\sigma_t = \frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}$。详细的训练和采样过程在方法部分展示。

---

## 3. 方法

我们对高维多元时间序列预测的方法包括一个两阶段过程：基于统计量的时间自编码器和潜在扩散Transformer（LDT）生成器。自编码器在训练过程中动态更新全局统计量，以实现准确的未来时间戳重建。LDT生成器然后利用自条件和引导机制生成潜在条件，并结合相关协变量。该方法高效地捕获了时间序列数据中的内在动力学和相关性，LDT的框架如图1所示。具体的算法细节见算法1和算法2。

### 3.1 对称时间序列压缩

为确保模型在生成高能量潜在嵌入方面的通用性和有效性，我们构建了一个简单而准确的自编码器结构。均值、方差等统计属性在时间序列中往往随时间变化，先前的工作"RevIN（Kim等 2021）、DIT-sh（Fan等 2023）"声称，不同输入序列之间的差异会显著降低模型性能。我们发现，在非平稳多元时间序列中，随机采样的不同批次样本会有高方差偏差，这会降低自编码器训练的稳定性和有效性。因此，我们提出了一种简单而有效的对称自编码器结构，带有自适应方差更新归一化层（VN）。

更具体地说，给定时间空间中的回溯窗口数据 $X \in \mathbb{R}^{\mathcal{T} \times d}$ 和目标 $Y \in \mathbb{R}^{\tau \times d}$，归一化层VN将目标 $Y$ 归一化为 $\hat{Y} = \text{VN}([X, Y])$。然后，编码器 $\mathcal{E}$ 将 $\hat{Y}$ 编码为潜在表示 $Z = \mathcal{E}(\hat{Y})$，解码器 $\mathcal{D}$ 从潜在表示重建目标时间序列，得到 $\tilde{Y} = \mathcal{D}(Z) = \mathcal{D}(\mathcal{E}(\hat{Y}))$，其中 $Z \in \mathbb{R}^{\tau \times m}$ （$m << d$）。重要的是，编码器按因子 $f = d/m$ 下采样时间序列，我们使用的因子 $f \geq 2^m$，其中 $m \in \mathbb{N}$。选择这样的 $f$ 大小是为了降低扩散模型中加噪高维多元时间序列的训练难度。我们的自编码器框架概览如图1所示。

如图所示，我们首先使用实例归一化（Ulyanov, Vedaldi和Lempitsky 2016）来计算每个输入 $W^i = [X^i, Y^i] \in \mathbb{R}^{\Lambda \times d}$ 的实例特定均值和标准差，其中 $\Lambda = \mathcal{T} + \tau$，定义为：

$$\mathbb{E}[W^i] = \frac{1}{\Lambda}\sum_{j=1}^{\Lambda} W_j^i, \quad \text{Var}[W^i] = \frac{1}{\Lambda}\sum_{j=1}^{\Lambda} (W_j^i - \mathbb{E}[W^i])^2$$

以及自适应更新的均值：$\hat{\mathbb{E}}_{n+1}[W^i] = \frac{1}{n}(\mathbb{E}_{n+1}[W^i] + \hat{\mathbb{E}}_n[W^i] \times (n - 1))$，其中 $n$ 是批次数量，方差的自适应更新函数 $\hat{\text{Var}}_{n+1}$ 与 $\hat{\mathbb{E}}_{n+1}$ 相同。我们通过这些更新的统计量对目标 $Y^i$ 进行归一化：$\hat{Y}^i = \gamma_d \frac{Y^i - \hat{\mathbb{E}}_{n+1}[W^i]}{\sqrt{\hat{\text{Var}}_{n+1}[W^i] + \epsilon}} + \delta_d$，其中 $\gamma_d, \delta_d \in \mathbb{R}^d$ 是可学习的仿射参数。

我们逐步更新用于正则化的实例方差和均值。一方面，目标序列中的非平稳信息可以被削弱，使自编码器更容易训练。另一方面，自编码器的生成结果使第二阶段扩散模型的训练更加稳定和准确。

具体来说，我们的自编码器结构是一个对称模型，具体模块由于篇幅限制在附录中展示。同时，为避免任意高方差的潜在空间，我们遵循"潜在扩散模型"中提出的正则化策略，对学习到的潜在变量施加KL惩罚（即将KL项的权重设为 $10^{-8}$），使其趋向标准正态分布，以更好地保留 $Y$ 的细节。我们以对抗方式训练所有自编码器模型，使得基于时间戳的判别器 $\Psi$ 被优化来区分原始目标时间序列和重建 $\mathcal{D}(\mathcal{E}(\hat{Y}))$。我们自编码器的完整目标训练损失函数 $\mathcal{L}$ 如下：

$$\mathcal{L} = \min_{\mathcal{E}, \mathcal{D}} \max_{\Psi}\left[\mathcal{L}_{rec}(Y, \mathcal{D}(\mathcal{E}(Y))) - \mathcal{L}_{adv}(\mathcal{D}(\mathcal{E}(Y))) + \log \Psi(Y) + \mathcal{L}_{reg}(Y; \mathcal{E}, \mathcal{D})\right]$$

其中 $\mathcal{L}_{reg}$ 是一个正则化损失项，用于将潜在变量 $Z$ 正则化为零中心并获得小方差。我们发现不同的时间序列总是具有大方差，这可能导致后续潜在扩散模型的训练极不稳定。详细的解释在实验中描述。

### 3.2 潜在扩散Transformer

**潜在表示的生成建模。** 与直接在高维多元时间序列的时间域中应用扩散模型相比，我们引入了由 $\mathcal{E}$ 和 $\mathcal{D}$ 组成的经过训练的时间压缩模型，将高效的低维时间序列表示传递给后续的去噪网络。

与以往依赖时间空间中自回归生成模型的工作（Min等 2022b；Yi等 2023）不同，我们利用基于注意力的Transformer模型（Min等 2022a；Xu等 2021）建立了一个非自回归的去噪网络结构，包括自适应归一化层（adaLN）（Park等 2019）、Transformer编码器-解码器块和自条件引导块。我们非自回归去噪网络的细节见图1(b)。潜在扩散模型中的训练目标如下：

$$\mathcal{L}_{LDM} := \mathbb{E}_{\mathcal{E}(x), x \sim p(x), \epsilon, t}\left[\|x - \hat{x}_\theta(z_t, c, t)\|_2^2\right] \quad \text{(6)}$$

其中我们模型的去噪主干 $\hat{x}_\theta$ 是一个自引导Transformer结构，$c$ 是回溯窗口数据和协变量等条件。由于前向过程是固定的，$z_t$ 可以从训练好的 $\mathcal{E}$ 高效获得，我们发现训练 $x$ 作为去噪目标相比 $\epsilon$ 能提高生成性能。最终，来自 $p(x)$ 的样本可以直接通过训练好的解码器 $\mathcal{D}$ 解码回时间空间。

### 3.3 自条件引导

首先，条件扩散模型可以简单描述为 $\hat{x}_\theta(z_t, c, t)$。为了以无分类器引导的方式训练潜在扩散模型，我们选择训练一个通过得分估计器 $\epsilon_\theta(z_t, t)$ 参数化的无条件去噪扩散模型 $p(z)$，以及条件模型 $\epsilon_\theta(z_t, c, t)$。我们使用单个神经网络来参数化这两个模型，对于无条件模型，我们在训练去噪网络时仅将回溯窗口条件视为缺失值，即 $\epsilon_\theta(z_t, t) = \epsilon_\theta(z_t, c = \varnothing, t)$。我们通过简单地以某个概率 $p_{uncond}$（设为超参数）将回溯窗口数据条件设置为 $\varnothing$ 来联合训练这两个模型。当条件潜在扩散模型训练好后，我们使用一个简单但有效的线性组合得分估计进行采样：

$$\hat{\epsilon}_\theta(z_t, c, t) = (1 + w)\epsilon_\theta(z_t, c, t) - w\epsilon_\theta(z_t, t) \quad \text{(7)}$$

其中 $w$ 是引导强度。当 $w = 0$ 时，方程变为标准条件扩散模型。当 $w > 0$ 时，去噪网络的更新梯度将更多地偏向第一项并偏离后一项。具体来说，如果我们可以获得精确的预测得分 $\epsilon_\theta(z_t, c, t)$ 和 $\epsilon_\theta(z_t, t)$，那么这个去噪网络的梯度将是 $\nabla_{z_t}\log p(c|z_t) = -\frac{1}{\sigma_t}[\epsilon_\theta(z_t, c, t) - \epsilon_\theta(z_t, t)]$。

此外，我们引入了一种自条件机制，可以看作是在迭代采样过程中对其自身先前生成的样本进行直接条件化。具体来说，我们的条件潜在扩散模型 $\hat{x}_\theta(z_t, c, t)$ 被替换为略有不同的去噪网络 $\hat{x}_\theta(z_t, \hat{z}_0, c, t)$，其中 $\hat{z}_0$ 是先前估计的并迭代更新。在我们的设置中，我们将 $z_t$ 与从采样链中去噪网络的较早预测获得的先前估计 $\hat{z}_0$ 连接起来。在训练阶段，我们以某个概率（例如60%）设置 $\hat{z}_0 = 0$，这退回到没有自条件化的建模。除此之外，我们首先预测 $\hat{z}_0 = \hat{x}_\theta(z_t, 0, c, t)$，然后用它进行自条件化。注意，我们不会通过估计的 $\hat{z}_0$ 进行反向传播。

### 3.4 潜在扩散Transformer网络

完整的去噪网络如图1(b)所示。为了以非自回归方式进行时间序列预测，我们需要涵盖如何处理时间序列输入（回溯窗口数据、目标）以及 $\hat{x}_\theta$ 的架构。

首先，我们描述如何处理时间序列数据作为去噪网络训练的输入。如第3.1节所定义，$\hat{\mathbb{E}}_t[W^i]$ 和 $\hat{\text{Var}}[W^i]$ 来自包含回溯窗口数据和预测目标的完整时间序列。我们首先使用 $\hat{X} = \frac{X_t^i - \hat{\mathbb{E}}[W^i]}{\sqrt{\hat{\text{Var}}[W^i] + \epsilon}} \in \mathbb{R}^{\mathcal{T} \times d}$ 对回溯窗口条件 $X \in \mathbb{R}^{\mathcal{T} \times d}$ 进行归一化，并使用 $\hat{Z} = \frac{\mathcal{E}(\hat{X})}{\sigma}$ 重新缩放潜在表示 $Z = \mathcal{E}(Y)$，其中 $\hat{\sigma}^2 = \frac{1}{btm}\sum_{b,t,m}(z_{b,t,m} - \hat{\mu})^2$，$\hat{\mu} = \frac{1}{btm}\sum_{b,t,m} Z_{b,t,m}$，来自每个训练批次的更新结果（$b$ 是批次大小，$t$ 是预测长度，$m$ 是隐藏大小），以获得去噪网络的输入 $\hat{Z} \in \mathbb{R}^{\tau \times m}$。

然后，我们通过由两个多层感知器层组成的输入投影块，获得 $\hat{X}$ 的嵌入 $\hat{X}_{emb} \in \mathbb{R}^{\mathcal{T} \times m}$ 和潜在表示 $\hat{Z}$ 的嵌入 $\hat{Z}_{emb} \in \mathbb{R}^{\tau \times m}$。

在我们的去噪网络中，我们引入时间嵌入 $s_{emb} = [s_{1:\tau}]$ 来学习时间依赖性，该嵌入通过单个MLP层获得，以及位置嵌入 $p_{emb} = [p_{1:\tau}]$（Vaswani等 2017中定义）。此外，扩散步嵌入 $t_{emb} \in \mathbb{R}^{n \times 1}$ （$n = 4m$）被编码为正弦位置嵌入，用于引导Transformer残差层中的自适应层归一化，替代标准层归一化，定义为：

$$\gamma_{i,c} = f_c(x), \quad \beta_{i,c} = h_c(x) \quad \text{(8)}$$

其中 $x$ 表示任意向量输入，$\gamma_{i,c}$ 和 $\beta_{i,c}$ 通过特征级仿射变换调制神经网络的激活 $Y_{i,c}$（下标表示第 $i$ 个输入的第 $c$ 个特征图）：

$$\text{adaLN}(\gamma_{i,c}, Y_{i,c}, \beta_{i,c}) = \gamma_{i,c}Y_{i,c} + \beta_{i,c} \quad \text{(9)}$$

$f_c$ 和 $h_c$ 可以是任意函数，如神经网络。在我们的实践中，更容易将 $f_c$ 和 $h_c$ 视为输出单个向量的单个函数（$\gamma \in \mathbb{R}^m, \beta \in \mathbb{R}^m$）。在我们的残差层中，我们通过扩散步嵌入 $t_{emb}$ 学习维度级的缩放和偏移参数 $\gamma_{i,c}$ 和 $\beta_{i,c}$。

**训练。** 我们LDT去噪网络的整体结构如图1所示，训练目标可参考方法部分的公式6。此外，具体的训练和推理过程见算法1和算法2。

**推理。** 对于反向过程中的每个时间步 $t$，学习到的由 $\theta$ 参数化的去噪分布 $p_\theta$ 基于前一个更嘈杂的样本 $z_t$ 生成样本 $z_{t-1}$。在反向去噪过程到达 $T = 0$ 后，我们将生成的 $z_0$ 的每个时间戳四舍五入到嵌入空间中其最近的值，并通过训练好的解码器 $\mathcal{D}$ 获得最终目标：

$$z_{t-1} = \hat{\gamma}z_t + \hat{\gamma}\hat{x}(z_t, t|X) + \sigma_t\epsilon \quad \text{(10)}$$

$$Y = \mathcal{D}(z_0) \quad \text{(11)}$$

其中 $\hat{\gamma} = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}$，$\hat{\gamma} = \frac{\sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)}{1 - \bar{\alpha}_t}$，$\sigma_t = \frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}$，$\epsilon \sim \mathcal{N}(0, 1)$。注意，在整个训练和采样的扩散过程中，我们应用 $x$ 而不是 $\epsilon$。在实验中，我们发现使用 $\epsilon$ 难以完成多元时间序列预测。

---

### 算法1：LDT训练

**输入：** 训练集中的样本 $x_0^{1:\mathcal{T}}$（历史）和 $x_0$（目标）；扩散步数 $K$；预训练自编码器中的编码器 $\mathcal{E}$。
**输出：** 训练好的去噪函数 $\hat{x}_\theta$。

1. 重复；
2. &emsp;$k \sim \text{Uniform}(\{1, 2, \ldots, K\})$；
3. &emsp;$\epsilon \sim \mathcal{N}(0, I)$；
4. &emsp;生成噪声潜在嵌入 $z_k = \sqrt{\bar{\alpha}_k}z_0 + \sqrt{1 - \bar{\alpha}_k}\epsilon$；
5. &emsp;生成潜在嵌入 $z_0 = \mathcal{E}(x_0^{1:\mathcal{T}}, x_0)$；
6. &emsp;使用正弦位置嵌入获取扩散步 $k$ 的嵌入 $p_{emb}$；
7. &emsp;以概率 $p_{uncond}$ 将 $x_0^{1:\mathcal{T}} \leftarrow \varnothing$；
8. &emsp;初始化自条件 $\hat{z}_0 = \text{zeros\_like}(z_k)$；
9. &emsp;如果 $\text{Uniform}(0, I) > 0.5$，则：
10. &emsp;&emsp;$\hat{z}_0^{pred} = \hat{x}_\theta(z_k, x_0^{1:\mathcal{T}}, \hat{z}_0, k)$；
11. &emsp;&emsp;$\hat{z}_0^{pred} = \text{Stop\_gradient}(\hat{z}_0^{pred})$；
12. &emsp;使用去噪网络通过 $\hat{x}_\theta(\hat{z}_0^{pred}, x_0^{1:\mathcal{T}}, z_k, t)$ 生成去噪样本 $z_0$；
13. &emsp;通过公式7获取 $z_0$，若 $x_0^{1:\mathcal{T}} = \varnothing$ 则 $c = \varnothing$；
14. &emsp;通过公式6计算损失 $\mathcal{L}_k(\theta)$；
15. &emsp;对 $\mathcal{L}_k(\theta)$ 执行梯度下降步；
16. 直到收敛。

---

### 算法2：LDT生成

**输入：** 训练好的去噪网络 $\hat{x}_\theta$，预训练自编码器中的解码器 $\mathcal{D}$，样本 $x_0^{1:\mathcal{T}}$（历史），引导强度 $w$。
**输出：** 生成的对应未来目标 $\hat{x}_0$。

1. $z_K \sim \mathcal{N}(0, I)$；
2. $\hat{z}_0 = \text{zeros\_like}(z_K)$；
3. $x_{1:\mathcal{T}} = \text{zeros\_like}(x_0^{1:\mathcal{T}})$；
4. 对于 $k$ 从 $K$ 到 1：
5. &emsp;若 $k > 1$，$\epsilon \sim \mathcal{N}(0, I)$，否则 $\epsilon = 0$；
6. &emsp;使用正弦位置嵌入获取扩散步 $k$ 的嵌入 $p_{emb}$；
7. &emsp;获取自条件 $\hat{z}_0 = \hat{x}_\theta(z_k, x_{1:\mathcal{T}}, \hat{z}_0, p_{emb})$；
8. &emsp;获取目标 $z_0 = \hat{x}_\theta(z_k, x_0^{1:\mathcal{T}}, \hat{z}_0, p_{emb})$；
9. &emsp;通过公式7使用 $x_{1:\mathcal{T}}$ 获取基于引导的目标 $z_0$；
10. &emsp;通过公式10估计 $z_{k-1}$；
11. 返回 $\hat{x}_0$。

---

## 4. 定量实验

### 数据集

我们在五个真实世界基准上广泛评估了提出的LDT，涵盖了主流的多元时间序列概率预测应用：

- **能源：** Solar（Lai等 2018）（137维）和 Electricity（370维）
- **交通：** Traffic（963维）和 Taxi（1214维）
- **维基百科：** Wikipedia（2000维）

实验中使用的数据集属性可参考先前的工作（Rasul等 2021；Tashiro等 2021），并在附录C中展示。

### 评估指标

对于概率估计，我们报告**跨时间序列总和的连续排序概率得分CRPS-sum**（Matheson和Winkler 1976；Jordan, Krüger和Lerch 2017）和**MSE**（均方误差）误差指标，分别测量整体联合分布模式拟合和联合分布中心趋势拟合。由于篇幅有限，指标的具体形式在附录B中展示。

### 基线模型

我们包含了多种基线方法。对于经典设置和竞争性多元时间序列基线概率模型：高斯过程模型GP（Roberts等 2013）、KVAE（Krishnan, Shalit和Sontag 2017）、Vec-LSTM-ind-scaling、GP-scaling和GP-Copula（Salinas等 2019）。对于时间序列扩散模型，包括TimeGrad（Rasul等 2021）、CSDI（Tashiro等 2021）、SSSD（Alcaraz和Strodthoff 2022）、D3VAE（Li等 2022b）作为竞争性自回归基线。此外，对于非自回归建模和基于流的架构，我们选择了TLAE（Nguyen和Quanz 2021）、HMGT（Ding等 2020）、LSTM-Real-NVP和LSTM-MAF（Rasul等 2020）。

### 实现细节

在第一阶段的自编码器结构中，编码器和解码器都使用了3层Transformer编码器层（4个注意力头），判别器中使用1层Transformer编码器层（4个注意力头机制）。最大回溯窗口数据为预测目标的4倍，与（Rasul等 2020）中的设置相同，嵌入维度 $m \in [1/4, 1/8]$ 数据特征，扩散步数 $T = [50, 100, 200, 300]$，平方根噪声调度（Li等 2022a）和二次方差调度 $\beta_1 = 10^{-4}$ 直到 $\beta_T = 0.1$。在我们的去噪网络结构中，使用3层Transformer结构，8个注意力头，嵌入维度 = [32, 64, 128, 256]。我们的方法依赖于ADAM优化器（Kingma和Ba 2014），初始学习率为 $1 \times 10^{-3}$，批次大小为64。所有实验重复五次以上，使用PyTorch（Paszke等 2019）和GluonTS（Alexandrov等 2020）实现。不同数据集对应的具体实验超参数见附录C。

---

## 5. 主要结果

### 真实世界数据集结果

我们使用CRPS、CRPS-sum和MSE比较了LDT与上述基线在测试时间预测上的表现。多元设置下的概率预测结果如表1所示。

**表1：基线模型和我们的模型LDT在测试集上的CRPS-sum（C-S）和MSE比较（越低越好），"-"表示运行失败（数值问题），(*)表示实验方差。VLIS、LSRP分别是Vec-LSTM-ind-scaling和LSTM-Real-NVP的缩写。CSDI中的"-"表示内存不足（OOM）。**

| 方法 | SOLAR C-S | SOLAR MSE | ELECTRICITY C-S | ELECTRICITY MSE | TRAFFIC C-S | TRAFFIC MSE | TAXI C-S | TAXI MSE | WIKIPEDIA C-S | WIKIPEDIA MSE |
|------|-----------|-----------|-----------------|-----------------|-------------|-------------|----------|----------|---------------|---------------|
| GP | 0.828(.010) | 9.3e2 | 0.947(.016) | 2.1e5 | 2.198(.774) | 6.3e-4 | 0.425(.199) | - | 0.93(.003) | 7.3e7 |
| KVAE | 0.340(.025) | 1.1e3 | 0.051(.019) | 1.8e5 | 0.100(.005) | 5.2e-4 | - | - | 0.095(.012) | - |
| VLIS | 0.391(.017) | 9.8e2 | 0.025(.001) | 2.4e5 | 0.087(.041) | 6.9e-4 | 0.506(.005) | - | - | - |
| GP-scaling | 0.368(.012) | 9.1e2 | 0.022(.000) | 2.5e5 | 0.079(.000) | 6.9e-4 | 0.183(.395) | 7.3e | 0.133(.002) | 5.5e7 |
| GP-Copula | 0.337(.024) | 9.8e2 | 0.024(.001) | 1.8e5 | 0.078(.002) | 4.9e-4 | 0.208(.183) | 2.7e | 1.483(1.034) | 7.2e7 |
| LSRP | 0.331(.020) | 9.4e2 | 0.024(.001) | 2.1e5 | 0.078(.001) | 4.4e-4 | 0.175(.001) | 3.1e | 0.086(.004) | 4.0e7 |
| LSTM-MAF | 0.315(.032) | 8.3e2 | 0.023(.000) | 2.7e5 | 0.069(.002) | 5.0e-4 | 0.161(.002) | 2.6e | 0.078(.001) | 4.7e7 |
| HMGT | 0.327(.013) | 9.9e2 | 0.022(.003) | 2.1e5 | 0.052(.002) | 4.6e-4 | 0.158(.042) | 2.4e | 0.067(.002) | 3.8e7 |
| TLAE | 0.124(.014) | 9.4e2 | 0.040(.001) | 2.4e5 | 0.069(.005) | 4.4e-4 | 0.130(.010) | 2.4e | 0.074(.011) | 3.0e7 |
| TimeGrad | 0.317(.020) | 5.4e2 | 0.025(.001) | 2.3e5 | 0.050(.006) | 4.5e-4 | 0.137(.013) | 2.6e | 0.241(.012) | 3.8e7 |
| CSDI | 0.298(.004) | 9.2e2 | 0.029(.002) | 2.4e5 | 0.053(.009) | 4.5e-4 | 0.133(.006) | 2.4e | 0.064(.003) | 3.1e7 |
| SSSD | 0.275(.004) | 7.7e2 | 0.026(.001) | 2.4e5 | 0.047(.002) | 4.5e-4 | 0.130(.011) | - | 0.069(.004) | 3.2e7 |
| D3VAE | 0.332(.002) | - | 0.030(.000) | - | 0.049(.001) | - | 0.125(.007) | - | 0.065(.001) | 2.99e7 |
| **LDT** | **0.253(.002)** | **7.7e2** | **0.021(.001)** | **1.6e5** | **0.040(.000)** | **4.1e-4** | **0.125(.007)** | **2.2e** | **0.061(.002)** | **2.92e7** |

与其他生成模型相比，我们观察到LDT在几乎所有基准上取得了（据我们所知）最先进的CRPS-sum。值得注意的是，我们的模型在Electricity上显示了显著的CRPS-sum降低**16%**（0.025 → 0.021），在Traffic上降低**14.8%**（0.047 → 0.040），在Taxi上降低**4%**（0.130 → 0.125）。此外，在MSE指标方面，在上述三个数据集中我们分别获得了**22%**（$2.1 \times 10^5 \to 1.6 \times 10^5$）、**8%**（$4.5 \times 10^{-4} \to 4.1 \times 10^{-4}$）和**4%**（$2.2 \to 2.3$，注：原文数据有限）的提升。

### 不确定性估计

不确定性可以通过在预测时估计结果序列的噪声来评估。我们发现我们的模型在两类数据集中显示出明显的不确定性估计。如图2所示，对于Solar数据集，尽管数据具有很强的周期性，但数据集中数值幅度的周期性变化和周期长度方面存在很大差异，这会导致两个相似的回溯窗口数据产生两个不同的预测目标。此外，在具有高随机性的Taxi数据集中，我们发现当遇到极端值时，我们模型中估计的不确定性会迅速增长。

### 确定性估计

除了上述不确定性估计方法外，我们的工作还揭示了当应用于具有有限极端变化的数据集（如Electricity和Traffic）时，我们的模型表现出确定性估计结果。如图3所示，我们的一次性LDT可以更准确地预测相对平稳的高维时间序列。我们发现模型中引导强度的变化会影响确定性预测的性能，这在附录D中也有展示。我们观察到，在具有确定性预测的数据集如Electricity和Traffic中，更大的引导 $w$ 会产生更好的结果，而在Solar和Taxi中，较低的引导 $w$ 效果更好。这表明我们的模型可以通过调整引导强度来适应不同的预测场景，实现不同类型数据集的确定性和不确定性预测。

---

## 6. 消融研究

在本节中，我们研究了我们结构中提出的各组件的有效性。引入了表1中的三个代表性多元数据集：Solar、Electricity和Traffic，这些数据集是非平稳且高维的。

### 自条件引导机制

在本实验中，我们研究了第3.3节中描述的自条件引导机制的有效性。我们考虑了三种不同的设置来验证我们模块的有效性，其中**LDT-g**是不使用自条件来训练去噪网络，**LDT-c**是不使用引导，**LDT**是我们在本文中提出的完整模型。表4显示了在相同引导强度 $w = 3.0$ 下的MSE和CRPS-sum两个指标的结果。

**表4：消融研究中模型和我们的模型LDT在测试集上的CRPS-sum（C-S）和MSE比较（越低越好）。**

| 方法 | Solar C-S | Solar MSE | Electricity C-S | Electricity MSE | Traffic C-S | Traffic MSE |
|------|-----------|-----------|-----------------|-----------------|-------------|-------------|
| LDT-g | 0.301(.001) | 8.9e2 | 0.024(.000) | 2.1e5 | 0.050(.003) | 4.3e-4 |
| LDT-c | 0.264(.004) | 8.0e2 | 0.023(.003) | 1.8e5 | 0.047(.004) | 4.5e-4 |
| **LDT** | **0.253(.002)** | **7.7e2** | **0.021(.001)** | **1.6e5** | **0.040(.000)** | **4.1e-4** |

为了验证不同部分的作用，我们可视化了Electricity数据集中更复杂样本生成的结果。如图4所示，LDT-g可以捕获预测目标的详细变化模式，但在数值的准确性上存在偏差。LDT-c可以有效地学习预测未来目标数值变化的区间，但细节不够精细。而我们提出的LDT有效地结合了这两个因素的优点：引导因子学习预测目标的详细模式，自条件因子学习预测目标的数值。

### 预测 $x$ vs. 预测 $\epsilon$

在本实验中，我们讨论了工作中不同的去噪策略。我们在五个数据集上比较了两种不同的训练策略，表2显示了我们的比较结果。

**表2：$\epsilon$ 策略和 $x$ 去噪策略在测试集上的CRPS-sum（C-S）和MSE比较（越低越好）。**

| 策略 | Solar C-S | Solar MSE | Electricity C-S | Electricity MSE | Traffic C-S | Traffic MSE | Taxi C-S | Taxi MSE | Wikipedia C-S | Wikipedia MSE |
|------|-----------|-----------|-----------------|-----------------|-------------|-------------|----------|----------|---------------|---------------|
| $\epsilon$ | 0.528(.006) | 1.4e3 | 0.044(.007) | 3.0e5 | 0.074(.012) | 6.4e-4 | 0.218(.012) | 3.2e | 0.079(.010) | 4.1e7 |
| **$x$** | **0.253(.002)** | **7.7e2** | **0.021(.001)** | **1.6e5** | **0.040(.000)** | **4.1e-4** | **0.125(.007)** | **2.2e** | **0.061(.002)** | **2.92e7** |

我们发现去噪过程在 $\epsilon$ 策略中表现出极差的性能，但我们发现像TimeGrad这样的基于自回归扩散的方法可以使用 $\epsilon$ 作为去噪目标。我们认为这一结果有两个原因：（1）在非自回归条件下，目标被设为噪声，这使得模型忽略了时间戳之间的相关性；（2）时间序列通常包含高度非线性的噪声，这很容易与扩散过程生成的噪声混淆。

### 推理效率

在本实验中，我们比较了提出的LDT与其他时间序列扩散模型基线TimeGrad、SSSD、CSDI和D3VAE的推理效率。表3显示了在Solar和Electricity多元数据集上两种不同预测长度（24, 48）下的推理时间。

**表3：在Solar和Electricity多元数据集上两种不同预测长度H=(24, 48)的推理时间（秒）。**

| 方法 | Solar H=24 | Solar H=48 | Electricity H=24 | Electricity H=48 |
|------|------------|------------|------------------|------------------|
| TimeGrad | 104.51(.73) | 203.16(.90) | 302.61(.35) | 615.02(.06) |
| SSSD | 80.36(.28) | 132.39(.71) | 176.23(.82) | 295.75(.54) |
| CSDI | 92.23(.53) | 147.52(.17) | 203.52(.74) | 314.23(.76) |
| D3VAE | 87.53(.29) | 153.43(.11) | 198.61(.19) | 304.76(.82) |
| **LDT** | **13.72(.25)** | **14.03(.37)** | **22.13(.14)** | **25.29(.18)** |

在生成效率方面，我们的一次性潜在结构LDT表现优异。LDT比TimeGrad快约**8-24倍**，比CSDI快约**7-12倍**，展示了非自回归潜在扩散方法的显著效率优势。

---

## 7. 结论

在本研究中，我们引入了一种利用潜在空间表示的多元概率时间序列预测方法。我们的方法结合了自条件引导机制，将自条件偏置与基于条件的引导相结合，以增强潜在扩散模型中的去噪过程。此外，我们为高维多元时间序列预测开发了一种一次性、非自回归的潜在扩散Transformer（LDT）。我们在五个标准时间序列基准上对LDT模型的评估设定了新的基准，超越了现有的生成方法。消融研究验证了我们模型中每个组件的贡献。我们的目标是进一步完善去噪结构，以建模高维多元时间序列。

---

## 致谢

本研究部分得到了新加坡总理办公室国家研究基金会（National Research Foundation）的支持，包括其AI Singapore计划（AISG Award No: AISGGC-2019-003）和NRF研究者计划（NRFI Award No. NRF-NRFI05-2019-0002）。本文表达的任何意见、发现结论或建议均为作者的观点，不代表新加坡国家研究基金会的观点。此外，作者衷心感谢评审人的建议和编辑的鼓励。

---

## 参考文献

1. Alcaraz, J. M. L.; 和 Strodthoff, N. 2022. 基于扩散的时间序列插补和预测的结构化状态空间模型. *arXiv preprint arXiv:2208.09399*.
2. Alexandrov, A.; Benidis, K.; Bohlke-Schneider, M.; Flunkert, V.; Gasthaus, J.; Januschowski, T.; Maddix, D. C.; Rangapuram, S.; Salinas, D.; Schulz, J.; 等. 2020. Gluonts：Python中的概率和神经时间序列建模. *机器学习研究杂志*, 21(1): 4629–4634.
3. Cao, D.; Wang, Y.; Duan, J.; Zhang, C.; Zhu, X.; Huang, C.; Tong, Y.; Xu, B.; Bai, J.; Tong, J.; 等. 2020. 用于多元时间序列预测的频谱时空图神经网络. *神经信息处理系统进展*, 33: 17766–17778.
4. Chen, T.; Zhang, R.; 和 Hinton, G. 2022. 模拟比特：使用带自条件的扩散模型生成离散数据. *arXiv preprint arXiv:2208.04202*.
5. Ding, Q.; Wu, S.; Sun, H.; Guo, J.; 和 Guo, J. 2020. 用于股票走势预测的分层多尺度高斯Transformer. *IJCAI*, 4640–4646.
6. Fan, W.; Wang, P.; Wang, D.; Wang, D.; Zhou, Y.; 和 Fu, Y. 2023. Dish-TS：缓解时间序列预测中分布偏移的通用范式. *AAAI人工智能会议论文集*, 37卷, 7522–7529.
7. Feng, S.; Miao, C.; Xu, K.; Wu, J.; Wu, P.; Zhang, Y.; 和 Zhao, P. 2023. 用于概率时间序列预测的多尺度注意力流. *IEEE知识与数据工程汇刊*.
8. Ho, J.; Jain, A.; 和 Abbeel, P. 2020. 去噪扩散概率模型. *神经信息处理系统进展*, 33: 6840–6851.
9. Ho, J.; 和 Salimans, T. 2022. 无分类器扩散引导. *arXiv preprint arXiv:2207.12598*.
10. Huang, R.; Lam, M. W.; Wang, J.; Su, D.; Yu, D.; Ren, Y.; 和 Zhao, Z. 2022. Fastdiff：用于高质量语音合成的快速条件扩散模型. *arXiv preprint arXiv:2204.09934*.
11. Jordan, A.; Krüger, F.; 和 Lerch, S. 2017. 使用scoringRules评估概率预测. *arXiv preprint arXiv:1709.04743*.
12. Kim, T.; Kim, J.; Tae, Y.; Park, C.; Choi, J.-H.; 和 Choo, J. 2021. 可逆实例归一化：针对分布偏移的准确时间序列预测. *国际学习表征会议*.
13. Kingma, D. P.; 和 Ba, J. 2014. Adam：一种随机优化方法. *arXiv preprint arXiv:1412.6980*.
14. Krishnan, R.; Shalit, U.; 和 Sontag, D. 2017. 非线性状态空间模型的结构化推理网络. *AAAI人工智能会议论文集*, 31卷.
15. Lai, G.; Chang, W.-C.; Yang, Y.; 和 Liu, H. 2018. 使用深度神经网络建模长短期时间模式. *第41届国际ACM SIGIR信息检索研究与发展会议*, 95–104.
16. Li, X.; Thickstun, J.; Gulrajani, I.; Liang, P. S.; 和 Hashimoto, T. B. 2022a. Diffusion-lm改进了可控文本生成. *神经信息处理系统进展*, 35: 4328–4343.
17. Li, Y.; Lu, X.; Wang, Y.; 和 Dou, D. 2022b. 使用扩散、去噪和解耦的生成式时间序列预测. *神经信息处理系统进展*, 35: 23009–23022.
18. Liu, C.; Hoi, S. C.; Zhao, P.; 和 Sun, J. 2016. 用于时间序列预测的在线ARIMA算法. *AAAI人工智能会议论文集*, 30卷.
19. Liu, Y.; Wu, H.; Wang, J.; 和 Long, M. 2022. 非平稳Transformer：探索时间序列预测中的平稳性. *神经信息处理系统进展*, 35: 9881–9893.
20. Matheson, J. E.; 和 Winkler, R. L. 1976. 连续概率分布的评分规则. *管理科学*, 22(10): 1087–1096.
21. Min, E.; Chen, R.; Bian, Y.; Xu, T.; Zhao, K.; Huang, W.; Zhao, P.; Huang, J.; Ananiadou, S.; 和 Rong, Y. 2022a. 图Transformer：架构视角概述. *arXiv preprint arXiv:2202.08455*.
22. Min, E.; Rong, Y.; Xu, T.; Bian, Y.; Luo, D.; Lin, K.; Huang, J.; Ananiadou, S.; 和 Zhao, P. 2022b. 通过图掩码Transformer基于邻居交互的点击率预测. *第45届国际ACM SIGIR信息检索研究与发展会议论文集*, 353–362.
23. Nguyen, N.; 和 Quanz, B. 2021. 时间潜在自编码器：一种概率多元时间序列预测方法. *AAAI人工智能会议论文集*, 35卷, 9117–9125.
24. Park, T.; Liu, M.-Y.; Wang, T.-C.; 和 Zhu, J.-Y. 2019. 使用空间自适应归一化的语义图像合成. *IEEE/CVF计算机视觉与模式识别会议论文集*, 2337–2346.
25. Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.; Chanan, G.; Killeen, T.; Lin, Z.; Gimelshein, N.; Antiga, L.; 等. 2019. Pytorch：命令式风格的高性能深度学习库. *神经信息处理系统进展*, 32.
26. Rao, H.; Hu, X.; Cheng, J.; 和 Hu, B. 2021. SM-SGE：用于行人重识别的自监督多尺度骨架图编码框架. *第29届ACM国际多媒体会议论文集*, 1812–1820.
27. Rao, H.; 和 Miao, C. 2022. SimMC：用于无监督行人重识别的简单掩码对比骨架表征学习. *arXiv preprint arXiv:2204.09826*.
28. Rasul, K.; Seward, C.; Schuster, I.; 和 Vollgraf, R. 2021. 用于多元概率时间序列预测的自回归去噪扩散模型. *国际机器学习会议*, 8857–8868. PMLR.
29. Rasul, K.; Sheikh, A.-S.; Schuster, I.; Bergmann, U.; 和 Vollgraf, R. 2020. 通过条件归一化流的多元概率时间序列预测. *arXiv preprint arXiv:2002.06103*.
30. Roberts, S.; Osborne, M.; Ebden, M.; Reece, S.; Gibson, N.; 和 Aigrain, S. 2013. 时间序列建模的高斯过程. *皇家学会哲学汇刊A：数学、物理与工程科学*, 371(1984): 20110550.
31. Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; 和 Ommer, B. 2022. 使用潜在扩散模型的高分辨率图像合成. *IEEE/CVF计算机视觉与模式识别会议论文集*, 10684–10695.
32. Ruan, L.; Ma, Y.; Yang, H.; He, H.; Liu, B.; Fu, J.; Yuan, N. J.; Jin, Q.; 和 Guo, B. 2023. Mm-diffusion：学习用于联合音频和视频生成的多模态扩散模型. *IEEE/CVF计算机视觉与模式识别会议论文集*, 10219–10228.
33. Salinas, D.; Bohlke-Schneider, M.; Callot, L.; Medico, R.; 和 Gasthaus, J. 2019. 使用低秩高斯Copula过程的高维多元预测. *神经信息处理系统进展*, 32.
34. Sezer, O. B.; Gudelek, M. U.; 和 Ozbayoglu, A. M. 2020. 使用深度学习的金融时间序列预测：2005–2019系统文献综述. *应用软计算*, 90: 106181.
35. Shen, L.; 和 Kwok, J. 2023. 用于时间序列预测的非自回归条件扩散模型. *arXiv preprint arXiv:2306.05043*.
36. Takagi, Y.; 和 Nishimoto, S. 2023. 使用潜在扩散模型从人脑活动进行高分辨率图像重建. *IEEE/CVF计算机视觉与模式识别会议论文集*, 14453–14463.
37. Tashiro, Y.; Song, J.; Song, Y.; 和 Ermon, S. 2021. CSDI：用于概率时间序列插补的条件得分扩散模型. *神经信息处理系统进展*, 34: 24804–24816.
38. Ulyanov, D.; Vedaldi, A.; 和 Lempitsky, V. 2016. 实例归一化：快速风格化的缺失成分. *arXiv preprint arXiv:1607.08022*.
39. Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, L.; 和 Polosukhin, I. 2017. 注意力就是你所需要的一切. *神经信息处理系统进展*, 30.
40. Woo, G.; Liu, C.; Sahoo, D.; Kumar, A.; 和 Hoi, S. 2022. Etsformer：用于时间序列预测的指数平滑Transformer. *arXiv preprint arXiv:2202.01381*.
41. Wu, S.; Xiao, X.; Ding, Q.; Zhao, P.; Wei, Y.; 和 Huang, J. 2020. 用于时间序列预测的对抗稀疏Transformer. *神经信息处理系统进展*, 33: 17105–17115.
42. Xu, K.; Zhang, Y.; Ye, D.; Zhao, P.; 和 Tan, M. 2021. 用于投资组合策略学习的关系感知Transformer. *第二十九届国际人工智能联合会议论文集*, 4647–4653.
43. Yang, L.; Zhang, Z.; Song, Y.; Hong, S.; Xu, R.; Zhao, Y.; Shao, Y.; Zhang, W.; Cui, B.; 和 Yang, M.-H. 2022. 扩散模型：方法和应用综述. *arXiv preprint arXiv:2209.00796*.
44. Yi, Y.; Wan, X.; Bian, Y.; Ou-Yang, L.; 和 Zhao, P. 2023. ETDock：用于蛋白质-配体对接的新型等变Transformer. *arXiv preprint arXiv:2310.08061*.
45. Yuan, H.; Yuan, Z.; Tan, C.; Huang, F.; 和 Huang, S. 2022. Seqdiffuseq：使用编码器-解码器Transformer的文本扩散. *arXiv preprint arXiv:2212.10325*.

---

*翻译说明：本文翻译自AAAI 2024会议论文《Latent Diffusion Transformer for Probabilistic Time Series Forecasting》。翻译力求准确、专业，保留了原文的学术风格和技术术语的精确性。表格数据保持原始数值不变。公式保持原始数学符号。*
