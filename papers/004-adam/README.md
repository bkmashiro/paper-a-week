# 004 — Adam: A Method for Stochastic Optimization

**Paper:** Kingma and Ba, 2014 | **Venue:** ICLR 2015  
**Link:** https://arxiv.org/abs/1412.6980

---

## Background

Before Adam, practitioners usually picked between two imperfect options:

- **SGD / momentum**: simple and memory-light, but sensitive to learning-rate tuning and feature scaling
- **AdaGrad / RMSProp**: adapt step sizes per parameter, but either decay too aggressively (AdaGrad) or lacked a clean bias-corrected formulation

Kingma and Ba combined two ideas into one optimizer:

1. Track the **first moment** of gradients: an exponential moving average of the gradient itself
2. Track the **second moment**: an exponential moving average of squared gradients

This yields an update rule that behaves like momentum in the numerator and RMS-style normalization in the denominator. Adam quickly became the default optimizer for deep learning because it works well out of the box across a wide range of models.

## Core Ideas

### 1. First and Second Moment Estimates

Given gradient `g_t` at step `t`, Adam maintains:

```python
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
```

- `m_t`: smoothed gradient, similar to momentum
- `v_t`: smoothed squared gradient, used to normalize step size

### 2. Bias Correction

At early steps, both moving averages are biased toward zero because they start from zero. Adam corrects this with:

```python
m_hat_t = m_t / (1 - beta1^t)
v_hat_t = v_t / (1 - beta2^t)
```

Without bias correction, the optimizer would take steps that are too small at the start of training.

### 3. Final Update Rule

```python
theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)
```

This means each parameter gets its own effective step size. Parameters with consistently large gradients get damped; parameters with small or sparse gradients get relatively larger updates.

### 4. Why Adam Helps

Adam is especially useful when:

- gradients have very different scales across parameters
- the objective is noisy
- you want a stable default optimizer without extensive tuning

Its weakness is that "works well by default" does not always mean "best final generalization." For some large-scale training setups, tuned SGD or AdamW can outperform vanilla Adam.

---

## 中文摘要

Adam 的核心思想是：同时对梯度的一阶矩和二阶矩做指数滑动平均，然后用这两个统计量来决定每个参数的更新方向和步长。

公式是：

```python
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)
```

其中 `m_hat_t` 和 `v_hat_t` 是 bias correction 后的一阶矩和二阶矩估计。

直觉上：

1. **`m_t` 像 momentum**：平滑掉梯度方向的抖动
2. **`v_t` 像自适应缩放器**：梯度大的参数步子小一点，梯度小的参数步子大一点
3. **bias correction**：解决前几步统计量严重偏小的问题

因此 Adam 通常比普通 SGD 更不怕特征尺度不一致，也更容易在默认超参数下快速下降。

---

## Implementation Notes

### What We Kept

- Exact Adam update with `m`, `v`, and bias correction
- Plain SGD baseline for apples-to-apples comparison
- Full NumPy training loop on the same regression task
- Printed loss checkpoints and final parameter error
- Optional plot export if `matplotlib` is installed

### What We Simplified

- **Tiny linear regression** instead of a deep network
- **Manual gradients** for MSE loss, no autograd
- **No mini-batching** — full-batch optimization keeps the signal clean
- **No AdamW / weight decay** — this is vanilla Adam from the original paper

### Key Code Points

**Adam step** (`adam.py`):
```python
self.m[name] = beta1 * self.m[name] + (1 - beta1) * grad
self.v[name] = beta2 * self.v[name] + (1 - beta2) * (grad * grad)
m_hat = self.m[name] / (1 - beta1 ** t)
v_hat = self.v[name] / (1 - beta2 ** t)
param -= lr * m_hat / (np.sqrt(v_hat) + eps)
```

**Toy regression gradients**:
```python
err = preds - y
loss = np.mean(err ** 2)
grads["W"] = (2 / n) * x.T @ err
grads["b"] = (2 / n) * np.sum(err, axis=0)
```

**Comparison setup**: one feature has scale `1e-2`, the other `1e1`, making the problem badly conditioned for one global SGD learning rate.

---

## Running

```bash
# Requirements: numpy
pip install numpy

# Optional for the loss plot
pip install matplotlib

cd papers/004-adam
python demo.py
```

Expected output:

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

If `matplotlib` is available, the demo also saves `loss_comparison.png`.

---

## Key Takeaways

### 1. Adam is momentum + adaptive scaling

The numerator (`m_hat`) gives a momentum-like direction; the denominator (`sqrt(v_hat)`) shrinks updates for parameters with consistently large gradients. This combination is why Adam feels robust in practice.

### 2. Bias correction is not cosmetic

Because `m_0 = 0` and `v_0 = 0`, early moving averages are biased low. The correction terms matter most in the first dozens of steps, especially when `beta2` is close to 1.

### 3. Adam shines on ill-conditioned problems

In this demo, one feature is tiny and one is large. A single SGD learning rate struggles to serve both coordinates. Adam adapts per parameter, so it gets near the optimum much faster.

### 4. Adam changed deep learning practice

Adam became the standard default optimizer for transformers, diffusion models, and many large neural networks. Later variants like AdamW keep the same adaptive core while fixing the interaction between Adam and weight decay.
