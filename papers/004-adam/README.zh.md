# 004 — Adam：一种随机优化方法

[English](README.md)

**论文：** Kingma 和 Ba，2014 | **发表于：** ICLR 2015  
**链接：** https://arxiv.org/abs/1412.6980

---

## 背景

在 Adam 出现之前，训练神经网络时最常见的选择通常是：

- **SGD / Momentum**：实现简单、额外内存小，但对学习率非常敏感，特征尺度一不一致就很难调
- **AdaGrad / RMSProp**：能按参数自适应调整步长，但 AdaGrad 会衰减得越来越保守，而 RMSProp 在论文层面缺少一个统一、清晰的 bias correction 表达

Kingma 和 Ba 的工作是把两个思路合并到一起：

1. 维护梯度的一阶矩估计（first moment），也就是梯度本身的指数滑动平均
2. 维护梯度二阶矩估计（second moment），也就是梯度平方的指数滑动平均

这样，更新公式的分子有 momentum 的效果，分母有 RMS 归一化的效果。Adam 因此很快成为深度学习最常用的默认优化器之一。

---

## 核心思想

### 1. 一阶矩与二阶矩估计

给定第 `t` 步的梯度 `g_t`，Adam 维护：

```python
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
```

- `m_t`：平滑后的梯度方向，类似 momentum
- `v_t`：平滑后的梯度平方，用来估计每个参数的梯度尺度

### 2. Bias Correction

由于 `m` 和 `v` 都从 0 开始，训练最开始的若干步会明显偏小。Adam 用下面的修正消除这种偏差：

```python
m_hat_t = m_t / (1 - beta1^t)
v_hat_t = v_t / (1 - beta2^t)
```

如果没有 bias correction，优化器在最开始会因为统计量过小而行为失真。

### 3. 最终更新公式

```python
theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)
```

含义是：每个参数都有自己的有效学习率。梯度长期偏大的参数会被自动缩小步长；梯度较小或较稀疏的参数会得到相对更大的更新。

### 4. 为什么 Adam 常常更稳

Adam 特别适合以下场景：

- 不同参数的梯度尺度差异很大
- 目标函数噪声较大
- 你希望默认超参数就能稳定收敛，而不是先花很多时间调学习率

但要注意，"默认更好用"不等于"最终泛化一定最好"。在一些大规模训练任务里，精调过的 SGD 或 AdamW 可能优于原始 Adam。

---

## 实现解析

### 保留了什么

- 原始论文中的 Adam 更新：`m`、`v`、bias correction 全部保留
- 同一任务上的纯 SGD 基线，方便直接对比
- 纯 NumPy 的完整训练循环
- 打印若干关键 step 的 loss 和最终参数误差
- 如果安装了 `matplotlib`，可额外导出 loss 曲线图

### 简化了什么

- **只用线性回归玩具任务**：没有上深层网络
- **手写 MSE 梯度**：不依赖自动求导
- **全量 batch**：不做 mini-batch，便于看清优化器本身差异
- **不实现 AdamW**：这里只实现原始论文里的 vanilla Adam

### 核心代码解读

**Adam 更新**（`adam.py`）：

```python
self.m[name] = beta1 * self.m[name] + (1 - beta1) * grad
self.v[name] = beta2 * self.v[name] + (1 - beta2) * (grad * grad)
m_hat = self.m[name] / (1 - beta1 ** t)
v_hat = self.v[name] / (1 - beta2 ** t)
param -= lr * m_hat / (np.sqrt(v_hat) + eps)
```

**线性回归的手写梯度**：

```python
err = preds - y
loss = np.mean(err ** 2)
grads["W"] = (2 / n) * x.T @ err
grads["b"] = (2 / n) * np.sum(err, axis=0)
```

**对比任务设计**：一个特征量级是 `1e-2`，另一个是 `1e1`。这种条件数很差的问题，会让单一全局学习率的 SGD 很难同时照顾两个坐标，而 Adam 的逐参数自适应步长会更从容。

---

## 中文摘要

Adam 的直觉可以概括成一句话：**用一阶矩决定方向，用二阶矩决定步长。**

具体来说：

1. `m_t` 累积"最近梯度大概往哪边走"
2. `v_t` 累积"这个参数的梯度通常有多大"
3. 更新时用 `m_hat_t / sqrt(v_hat_t)`，相当于把方向和尺度分开处理

这让 Adam 在很多任务上都比普通 SGD 更省调参，尤其是在不同参数梯度尺度差异明显时。

---

## 运行方法

```bash
# 依赖：numpy
pip install numpy

# 可选：导出 loss 曲线图
pip install matplotlib

cd papers/004-adam
python demo.py
```

预期输出：

```text
Paper 004: Adam optimizer — NumPy training demo

Loss Comparison
step            Adam            SGD
1            ...
5            ...
20           ...

Final Metrics
Adam final loss: ...
SGD final loss:  ...
```

如果安装了 `matplotlib`，还会保存 `loss_comparison.png`。

---

## 关键收获

### 1. Adam = Momentum + 自适应学习率

分子里的 `m_hat` 负责平滑方向，分母里的 `sqrt(v_hat)` 负责按参数梯度尺度做归一化。这就是 Adam "默认好用" 的来源。

### 2. Bias Correction 不是可有可无

因为 `m`、`v` 初始都是 0，如果不做修正，前几步的统计量会被明显低估。尤其当 `beta2=0.999` 这样非常接近 1 时，修正项的作用更明显。

### 3. Adam 在病态问题上优势明显

本实现里故意构造了一个特征尺度相差 1000 倍的回归任务。SGD 必须在"一个方向太大"和"另一个方向太小"之间折中；Adam 则能逐参数调节步长，所以收敛更快。

### 4. 历史影响

Adam 几乎成了现代深度学习的标准优化器。从 Transformer 到扩散模型，再到大量预训练语言模型，Adam 家族一直都是训练主力。后来的 AdamW 则在保持 Adam 自适应核心的同时，把权重衰减处理得更合理。
