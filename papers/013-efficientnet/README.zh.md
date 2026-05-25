# 013 — EfficientNet：重新思考卷积神经网络的模型缩放

**作者：** Mingxing Tan, Quoc V. Le  
**发表时间：** 2019年（arXiv: 1905.11946）  
**会议/期刊：** ICML 2019  
**类别：** 深度学习 / 架构设计  
**链接：** https://arxiv.org/abs/1905.11946

---

## 背景

提升 CNN 性能的方式，历来不外乎三条路：加深（更多层）、加宽（更多通道）、提高分辨率（更大输入图像）。过去的做法是各自为政——堆更多层，或者加更大图像，直到显存耗尽为止。

Tan & Le 提出了一个根本性的问题：**这三个维度是否存在相互依赖，能否以一种统一的原则同时缩放它们？**

他们的答案是**复合缩放（Compound Scaling）**：用一个标量系数 φ（phi）将深度、宽度和分辨率联动缩放。基于此，他们设计了 **EfficientNet** 系列模型——在 2019 年以最少的参数量和计算量，达到了当时最高的 ImageNet 精度。

---

## 核心思想

### 1. 为什么三个维度会相互影响

- 单纯加深：梯度消失，收益递减
- 单纯加宽：特征浅、感受野小
- 单纯提高分辨率：图像细节多了，但网络容量不够，无法充分利用

关键洞察：**高分辨率图像需要更深的网络（更大感受野）和更宽的网络（更大容量）**，三者必须协同增长。

### 2. 复合缩放公式

给定用户定义的复合系数 **φ ≥ 0**，按如下规则缩放：

```
深度：      d = α^φ
宽度：      w = β^φ
分辨率：    r = γ^φ
```

约束条件：

```
α · β² · γ² ≈ 2
α ≥ 1,  β ≥ 1,  γ ≥ 1
```

该约束保证每将 φ 增加 1，总 FLOPs 约翻倍（FLOPs ∝ d · w² · r² ≈ 2^φ）。

最优系数通过对基础模型（B0）做一次小规模网格搜索确定，论文找到 **α=1.2, β=1.1, γ=1.15**（满足 1.2 × 1.1² × 1.15² ≈ 2.0）。

### 3. EfficientNet-B0：基础架构

作者首先用 NAS（神经架构搜索）寻找精度/FLOPs 最优的基础网络 **EfficientNet-B0**，包含：

- **MBConv 模块**（来自 MobileNetV2 的移动反转瓶颈卷积）
- **SE（Squeeze-and-Excitation）模块**：通道注意力机制
- **Swish 激活函数**（`x · σ(x)`），替代 ReLU
- 轻量参数：5.3M 参数，约 0.39B FLOPs（输入 224×224）

### 4. B1 到 B7：模型族

用递增的 φ 对 B0 进行复合缩放：

| 模型 | φ | 分辨率 | 参数量 | Top-1 精度 |
|------|---|--------|--------|------------|
| B0 | 0 | 224 | 5.3M | 77.1% |
| B1 | 1 | 240 | 7.8M | 79.1% |
| B3 | 3 | 300 | 12M  | 81.6% |
| B5 | 5 | 456 | 30M  | 83.6% |
| B7 | 7 | 600 | 66M  | 84.3% |

EfficientNet-B7 在 2019 年以 **84.3% ImageNet top-1 精度**创下新高，同时比当时其他最优模型**小 8.4 倍、快 6.1 倍**。

### 5. MBConv：核心模块

```
输入 (C_in 通道)
  → 1×1 卷积（升维至 C_in × 扩展比，通常 ×6）
  → 深度可分离卷积 k×k（空间混合，每通道独立滤波器）
  → SE 模块（全局平均池化 → FC → FC → 通道缩放）
  → 1×1 卷积（降维至 C_out 通道）
  → 随机深度（Stochastic Depth，训练时随机跳过）
  → 残差连接（若输入输出形状一致）
```

深度可分离卷积将 FLOPs 降低约 k² 倍：不再对 C_in 个通道做联合卷积，而是每个通道独立做 k×k 卷积，再用 1×1 卷积融合通道信息。

### 6. Squeeze-and-Excitation（通道注意力）

```python
z = global_avg_pool(x)               # 压缩至 (C,)
s = sigmoid(W₂ · relu(W₁ · z))      # 两层 FC，输出通道权重
y = s * x                            # 逐通道缩放
```

SE 模块用极少的参数，让网络学会动态放大重要通道、抑制冗余通道。

---

## 实现要点

### Swish 激活函数

```python
def swish(x):
    return x * sigmoid(x)
```

Swish（又称 SiLU）在负值区域有非零梯度，对深层网络比 ReLU 更有利，现被 EfficientNet、GPT-Neo 等广泛使用。

### 随机深度（Stochastic Depth）

训练时以概率 `p_drop` 随机跳过整个残差分支，相当于对深度的 Dropout，有效正则化超深网络：

```python
def stochastic_depth(x, residual, drop_prob, training):
    if not training or drop_prob == 0.0:
        return x + residual
    keep_prob = 1.0 - drop_prob
    mask = (random.uniform(0, 1) < keep_prob)
    return x + residual * mask / keep_prob
```

### 复合缩放的实际计算

给定 B0 的基础层数 $d_0$、宽度 $w_0$、分辨率 $r_0$，B_n 对应：

```python
phi = n  # n in {0,1,2,3,4,5,6,7}
alpha, beta, gamma = 1.2, 1.1, 1.15

depth      = round(d_0 * alpha**phi)
width_mult = beta**phi          # 应用于每层通道数
resolution = round(r_0 * gamma**phi)
```

---

## 影响与延伸

- **EfficientNetV2**（2021）：引入 Fused-MBConv（合并扩展卷积与深度卷积）+渐进式训练，进一步提升速度与精度
- **EfficientDet**：将 EfficientNet 用作目标检测 backbone，同时对 FPN 做复合缩放
- **CoAtNet**（2021）：将 EfficientNet 的缩放思路与 Transformer 结合
- 复合缩放范式成为此后高效视觉架构设计的基本原则，影响深远

---

## 关键结论

1. **模型缩放有三个维度，且相互依赖**：深度、宽度、分辨率需要协同增长，单独缩放会迅速遇到收益递减的边界。

2. **复合系数 φ 极具实用价值**：只需搜索一次基础模型的最优 (α, β, γ)，后续通过调整 φ 即可生成整个模型族。

3. **基础架构的质量决定缩放上限**：EfficientNet-B0 的精心设计是整个系列成功的根基——缩放只会放大架构本身的优劣。

4. **MBConv 是高效视觉架构的黄金模块**：深度可分离卷积 + SE 通道注意力 + 残差连接，几乎成为 2019 年后所有高效 CNN 的标准基础单元。

5. **Swish/SiLU 是比 ReLU 更好的激活函数**（在深层网络中）：已被现代 LLM 和视觉模型广泛采用。

---

## 参考文献

- Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018
- Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
- Tan & Le, "EfficientNetV2: Smaller Models and Faster Training", ICML 2021
- Ramachandran et al., "Searching for Activation Functions" (Swish), arXiv 2017
