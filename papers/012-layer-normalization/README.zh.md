# 012 — Layer Normalization（层归一化）

**作者：** Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton  
**发表时间：** 2016年（arXiv: 1607.06450）  
**会议/期刊：** arXiv 预印本（被广泛采用；于 NIPS 2016 研讨会展示）  
**类别：** 深度学习 / 训练技术  
**链接：** https://arxiv.org/abs/1607.06450

---

## 背景

批归一化（Batch Normalization）通过跨**批次维度**计算统计量来稳定训练。然而，这一设计在批次较小或批次大小不固定的场景下会带来严重问题：

- **循环神经网络（RNN）：** 序列长度不一，跨时间步应用 BN 十分别扭，且每个时间步的行为不同。
- **在线推理/单设备推理：** 批次大小为 1 时，BN 统计量毫无意义。
- **生成模型与强化学习：** 单样本生成或环境交互打破了 BN 的基本假设。

层归一化（Layer Normalization）从根本上解决了这一问题：它将归一化方向从批次维度转移到**特征维度**。每个样本独立计算自己的均值和方差，完全不依赖同一批次中的其他样本。因此，LN 在训练和推理阶段的行为完全一致，对批次大小没有任何要求。

如今，LN 已成为 Transformer、GPT、BERT、LLaMA 等几乎所有大型语言模型的标准归一化方式。

---

## 核心思想

### 1. 在特征维度归一化，而非批次维度

对单个样本，设隐藏层有 `H` 个神经元，输入向量为 `x ∈ ℝᴴ`：

```
μ  = (1/H) Σⱼ xⱼ                      （特征均值）
σ² = (1/H) Σⱼ (xⱼ - μ)²               （特征方差）
x̂ⱼ = (xⱼ - μ) / √(σ² + ε)            （归一化）
yⱼ = γⱼ · x̂ⱼ + βⱼ                    （缩放与平移）
```

与批归一化的关键对比：

| | 批归一化 (BN) | 层归一化 (LN) |
|---|---|---|
| 归一化方向 | 批次 (N) | 特征 (H) |
| 统计量依赖 | 同批其他样本 | 当前样本自身 |
| 训练/推理行为一致 | 否 | 是 |
| 批次大小=1 时有效 | 否 | 是 |
| 适用于 RNN | 不自然 | 自然 |

### 2. 无需维护滑动统计量

由于每个样本独立计算统计量，不需要维护 `running_mean` 和 `running_var`，也不需要区分训练模式和推理模式。实现更简单，部署更方便。

### 3. 可学习的仿射变换

与 BN 相同，LN 为每个特征维度保留可学习的缩放参数 `γ`（初始为 1）和偏移参数 `β`（初始为 0），让网络能够恢复任意分布，保持表达能力。

### 4. Pre-LN 与 Post-LN

原始 Transformer 使用"Post-LN"（归一化在残差加法之后）：

```
x → 子层 → + 残差 → LayerNorm → 输出
```

现代 LLM（如 GPT-2、LLaMA）多使用"Pre-LN"（归一化在子层之前）：

```
x → LayerNorm → 子层 → + 残差 → 输出
```

Pre-LN 的梯度流更稳定，训练初期不易爆炸，所需的预热步数更少。

---

## 实现要点

### 前向传播（单样本）

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta
```

无论批次大小为多少，这一函数都能正确工作。

### 在 RNN 中的应用

LN 最初正是为 RNN 设计的。在每个时间步独立应用：

```
h_t = tanh(LayerNorm(W_h @ h_{t-1} + W_x @ x_t + b))
```

这与 BN 在 RNN 中需要每个时间步分别维护统计量形成鲜明对比。

---

## 历史影响

- 论文发表后，LN 迅速成为 RNN 领域的标准归一化方法。
- Transformer 架构采用 LN，使其在 NLP 领域一统天下。
- 所有主流 LLM（GPT、BERT、T5、LLaMA 等）均使用 LN 或其变体。
- **RMSNorm**（LLaMA 使用）是 LN 的简化版本：去掉均值中心化，仅用均方根归一化，计算量减少约 30%，效果相当。

---

## 关键结论

1. **LN 对每个样本独立归一化**，彻底消除批次大小的限制，训练和推理行为完全一致。

2. **Transformer 和 LLM 的核心组件**：理解 LN 是理解现代大模型的基础。

3. **Pre-LN 更稳定**：现代实践中 Pre-LN 优于 Post-LN，训练曲线更平滑。

4. **实现极为简洁**：无需滑动统计量，无需区分训练/推理模式，约 5 行代码搞定。

5. **RMSNorm 是流行的进一步简化**：LLaMA、Mistral 等模型采用，去掉均值中心化步骤，在速度和精度之间取得更好平衡。

---

## 参考文献

- Ba et al., "Layer Normalization", arXiv 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019
- Xiong et al., "On Layer Normalization in the Transformer Architecture", ICML 2020（Pre-LN 分析）
