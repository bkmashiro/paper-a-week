# 005 — Dropout：一种简单防止神经网络过拟合的方法

[English](README.md)

**论文：** Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov — JMLR 2014  
**arXiv：** https://arxiv.org/abs/1207.0580  
**发表于：** Journal of Machine Learning Research，Vol. 15，2014

---

## 背景

在 Dropout 出现之前，神经网络对付过拟合的主要手段是：

- **L2/L1 正则化** — 在损失函数里加权重惩罚项
- **早停（Early Stopping）** — 验证集 loss 不再下降就停训练
- **集成（Ensemble）** — 训练多个模型取平均，效果好但代价高

这些方法都没有从根本上解决一个更深的问题：**协同适应（co-adaptation）**。

大型网络里的神经元倾向于"抱团"：神经元 A 只有在 B、C、D 同时激活时才有用。它从不学会独立提取有意义的特征——因为它根本不需要。一旦测试时 B 和 C 的激活稍微弱一点，A 就彻底失效了。

Dropout 的解法极其简单：训练时，每个神经元以概率 `p` 被随机"关掉"。被关掉的神经元既不参与前向传播，也不接收梯度。这样，每个神经元都必须学会"孤立地"提取有用特征，因为它永远不能指望某个特定的"队友"在场。

---

## 核心思想

### 1. Dropout 掩码（Mask）

训练的每一步，对每个隐藏单元，独立采样一个伯努利变量：

```
r_j ~ Bernoulli(1 - p)
```

thinned network（变薄的子网络）的输出变为 `y = f(W(r ⊙ x))`。

典型超参数：隐藏层 `p = 0.5`，输入层 `p = 0.2`。

### 2. 反转 Dropout（Inverted Dropout）

朴素 Dropout 需要在测试时对所有权重乘以 `(1-p)` 来校正期望值。现代框架普遍使用 **反转 Dropout**：在训练时就做缩放，测试时不用改动。

```python
mask = (random(shape) < keep_prob).astype(float)
mask /= keep_prob   # 训练时缩放，测试时直接用
output = activation * mask
```

这样推理阶段的计算图和普通前向传播完全相同。本实现使用反转 Dropout。

### 3. 几何直觉：隐式集成

有 `n` 个神经元的网络，理论上存在 `2^n` 个不同的子网络。Dropout 每一步都从中采样一个。测试时使用完整网络，近似于对所有 `2^n` 个子模型取几何平均。这是指数级的集成——代价却只有一个模型。

这也解释了 Dropout 为什么有效：集成多个不同的弱模型，往往比单个强模型更鲁棒。

### 4. 与权重正则化的联系

Dropout 会让依赖特定神经元组合的权重收到噪声梯度（因为"队友"随机消失）。网络被迫把信息分散到多条路径上，而不是集中在几个高度协调的神经元里。这种冗余表征本身就是一种隐式正则化。

### 5. 测试阶段行为

使用反转 Dropout，推理时不需要任何修改：
- 所有单元全部激活
- 不需要乘以任何缩放因子
- 前向传播和普通网络完全一样

这就是 PyTorch 中 `model.eval()` 只需关掉 `training` 标志的原因——Dropout 层自动退化为恒等映射。

---

## 实现说明

### 网络结构

两个 256 维隐藏层，ReLU 激活，每层后面接 Dropout。输出层用 4 类 Softmax + 交叉熵损失。

**文件：** `implementation.py`（约 180 行），`demo.py`（约 90 行）

### 关键代码

**反转 Dropout 掩码**：
```python
def dropout_mask(shape, p, rng):
    keep = 1.0 - p
    mask = (rng.random(shape) < keep).astype(np.float64)
    return mask / keep   # 训练时缩放
```

**前向传播带 Dropout**：
```python
h1 = relu(self.l1.forward(x))
m1 = dropout_mask(h1.shape, self.drop_p if self.training else 0.0, self.rng)
h1d = h1 * m1
```

**反向传播**（掩码必须和前向一致）：
```python
d_h2d = self.l3.backward(d_logits)
d_h2 = d_h2d * self._m2        # 重用前向的掩码
d_a2 = d_h2 * relu_grad(self._h2)
```

被丢弃的神经元不参与前向，也不接收梯度——关键在于反向传播复用完全相同的掩码。

### 简化内容

- 使用玩具 4 类高斯分类任务，而非 MNIST/CIFAR
- 优化器用普通 SGD，不引入 Adam 等变量
- 没有学习率调度
- 各层使用固定 `p`

---

## 运行方式

```bash
# 依赖：numpy（matplotlib 可选，用于画图）
pip install numpy

cd papers/005-dropout
python demo.py
```

预期输出：

```
Paper 005: Dropout — Comparison Demo
Dataset: 200 training samples, 500 validation, 50 features, 4 classes

No Dropout (baseline)  val_acc = 0.xxx
Dropout 0.5            val_acc = 0.xxx
Delta                  = +0.0xx
```

如果安装了 `matplotlib`，会保存 `dropout_comparison.png`。

---

## 关键收获

### 1. Dropout = 指数级集成，代价仅一个模型

训练一个 Dropout 网络，等价于隐式地对 `2^n` 个子网络做平均。这就是它能有效减少过拟合的原因——集成多样性是免费获得的。

### 2. 反转 Dropout 让推理零开销

训练时缩放（`mask /= keep_prob`），测试时前向传播不做任何修改。PyTorch、JAX、TensorFlow 都默认使用反转 Dropout。

### 3. Dropout 让更宽的网络变得可行

没有 Dropout 时，增大层宽往往导致更严重的过拟合。有了 Dropout，更宽的网络通常反而更好——额外的容量用于构建冗余表征，而不是记住噪声。

### 4. 反向传播必须复用前向的掩码

关掉一个神经元意味着它既不参与前向计算，也不参与梯度传播。如果前向和反向用的掩码不同，梯度就会流过那些对输出没有贡献的路径——这是错误的。

### 5. Dropout 塑造了现代深度学习的默认设置

Dropout 是 AlexNet（2012）成功的关键组件，此后成为深度学习的标配正则化手段。尽管 Transformer、BatchNorm 主导的网络有时对 Dropout 依赖更少，但随机正则化的思想依然深刻影响着今天的架构设计。
