# 007 — 深度残差学习（ResNet）

**论文：** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun，2015 | **发表于：** CVPR 2016  
**链接：** https://arxiv.org/abs/1512.03385

---

## 背景

2012 年 AlexNet 用 8 层卷积网络在 ImageNet 上大胜，之后研究者的直觉是：层越深，网络越强。于是 VGGNet 堆到 19 层，GoogLeNet 堆到 22 层，效果确实在提升。但当人们试图继续叠层时，遇到了一个令人困惑的现象——**退化问题（degradation problem）**：56 层的网络在 CIFAR-10 上的**训练误差**竟然高于 20 层网络。注意，这不是过拟合，而是模型根本无法在训练集上收敛。

按道理，更深的网络应该至少能"复现"浅层网络的结果——只需把多出来的层设成恒等映射即可。但优化器（SGD）做不到这一点。原因在于：让一堆带非线性激活的卷积层学出精确的恒等映射，梯度回传时会严重衰减（梯度消失），优化器很难找到这条路。

He 等人的洞察极其简洁：**改变学习目标**。不要让网络直接学期望的映射 H(x)，而是让它学残差 F(x) = H(x) − x，然后通过 shortcut 连接把输入 x 直接加回来：

```
输出 = F(x) + x
```

如果恒等映射是最优解，那么网络只需让 F(x) → 0，这比学出精确的恒等映射容易得多。这个改动几乎没有额外参数，却从根本上改善了优化景观。ResNet-152 在 ImageNet 2015 比赛中以 3.57% 的 Top-5 错误率夺冠，将当时的最佳成绩减半。

---

## 核心思想

### 残差块

基本结构如下：

```
y = F(x, {Wᵢ}) + x
```

- **x**：块的输入
- **F(x, {Wᵢ})**：残差函数，通常是 2~3 个卷积层 + BatchNorm + ReLU 的组合
- **+ x**：shortcut 连接（恒等映射，零参数开销）

加法之后再接一个 ReLU 作为输出激活。

从梯度角度看，损失对 x 的梯度变为：

```
∂L/∂x = ∂L/∂y · (∂F/∂x + I)
```

多了一个恒等项 I，梯度可以直接"跳过"任意多层回传，不再依赖每一层的 Jacobian 乘积，从根本上缓解了梯度消失。

### 维度不匹配时的投影 Shortcut

当特征图尺寸或通道数改变时（例如 stride=2 的下采样），输入 x 和 F(x) 维度不同，不能直接相加。解决方法是用 1×1 卷积做投影：

```
y = F(x) + Wₛ · x
```

其中 Wₛ 是 1×1 卷积，仅在维度变化的块中使用，其余块仍用纯恒等 shortcut。

### Bottleneck 设计（50 层以上）

ResNet-50/101/152 用了 Bottleneck 块来降低计算量：

```
输入（C 通道）
→ 1×1 conv → C/4 通道   （降维）
→ 3×3 conv → C/4 通道   （特征提取）
→ 1×1 conv → C 通道     （升维）
→ + shortcut
```

3×3 卷积在降维后的小张量上操作，计算量大幅减少，同时表达能力不降反升（因为深度更大）。

### 整体架构

ResNet 的结构非常规整，分成四个 Stage，每个 Stage 内特征图尺寸固定：

| 阶段 | 分辨率 | 通道数 | 块数（ResNet-50） |
|------|--------|--------|-----------------|
| conv1 + pool | 56×56 | 64 | — |
| Stage 1 | 56×56 | 256 | 3 |
| Stage 2 | 28×28 | 512 | 4 |
| Stage 3 | 14×14 | 1024 | 6 |
| Stage 4 | 7×7 | 2048 | 3 |

最后接全局平均池化（GAP）+ 全连接层（1000 类），无需大量参数的全连接堆叠。

---

## 为什么残差有效？一些直觉

Veit 等人（2016）提出了一个有趣的**集成视角**：一个有 n 个残差块的 ResNet，实际上隐式地包含了 2ⁿ 条不同深度的路径（因为每个块都可以走 shortcut 或走残差分支）。训练时，浅路径的梯度更大，主导了权重更新；推理时，所有路径协同贡献。这类似于一个深度自适应的模型集成，使网络对单层的去除非常鲁棒——这与普通深网截然不同。

---

## 实现说明

### 我们实现了

- 完整的 BasicBlock 和 Bottleneck 块（数学完全一致）
- 维度不匹配时的投影 shortcut（1×1 卷积）
- BatchNorm + ReLU 的标准摆放方式
- ResNet-18、ResNet-34（BasicBlock）和 ResNet-50（Bottleneck）
- 面向 CIFAR-10 的轻量版 ResNet-20（去掉初始大卷积和池化）
- 梯度范数对比演示（直观展示残差连接对梯度流的改善）

### 我们简化了

- 没有完整训练循环（ImageNet 训练成本太高）
- 没有多 GPU 或分布式训练
- 演示中使用随机输入，不涉及真实数据集加载

### 关键代码片段

**BasicBlock 的前向传播：**
```python
def forward(self, x):
    identity = x
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    if self.shortcut is not None:
        identity = self.shortcut(x)
    out = self.relu(out + identity)  # 残差相加，再激活
    return out
```

**Bottleneck 的前向传播：**
```python
def forward(self, x):
    identity = x
    out = self.relu(self.bn1(self.conv1(x)))   # 1×1 降维
    out = self.relu(self.bn2(self.conv2(out))) # 3×3 卷积
    out = self.bn3(self.conv3(out))            # 1×1 升维
    if self.shortcut is not None:
        identity = self.shortcut(x)
    out = self.relu(out + identity)
    return out
```

---

## 运行方法

```bash
pip install torch torchvision
python implementation.py
```

期望输出：
```
ResNet Architecture Demo
========================================
ResNet-18  | Params:   11,689,512 | Output: torch.Size([2, 1000])
ResNet-34  | Params:   21,797,672 | Output: torch.Size([2, 1000])
ResNet-50  | Params:   25,557,032 | Output: torch.Size([2, 1000])

CIFAR-10 ResNet-20 Demo
========================================
ResNet-20 (CIFAR) | Params: 269,722 | Output: torch.Size([4, 10])

Residual Connection Benefit Demo
========================================
Depth  5 | Plain loss: 2.3301 | ResNet loss: 2.6308
Depth 10 | Plain loss: 2.3301 | ResNet loss: 2.6976
Depth 20 | Plain loss: 2.3301 | ResNet loss: 3.1554

Gradient norm comparison (depth=20):
  Plain  network: gradient norm = 10994.7100
  ResNet network: gradient norm = 233.3394
```

---

## 核心收获

ResNet 的影响远超图像分类本身。残差连接（或其变体）已成为现代深度学习的基础设施，出现在 Transformer（注意力层后的残差）、GPT 系列、扩散模型的 U-Net 骨干等几乎所有现代架构中。它的核心贡献不是某个复杂的算法，而是一个极其简单的想法：**把学习目标从"映射本身"改为"映射与输入的差"**，从而让优化器在面对恒等映射时有了更容易走的路。这个思路的优雅之处在于，它以零额外参数的代价解决了一个根本性的优化难题。
