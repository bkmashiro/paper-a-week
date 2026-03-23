# 005 — Dropout: A Simple Way to Prevent Neural Networks from Overfitting

**Paper:** Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov — JMLR 2014  
**arXiv:** https://arxiv.org/abs/1207.0580  
**Venue:** Journal of Machine Learning Research, Vol. 15, 2014

---

## Background

Before dropout, the dominant approach to fighting overfitting in neural networks was either:

- **L2/L1 regularization** — penalize large weights globally
- **Early stopping** — stop when validation loss plateaus
- **Ensembles** — train multiple models and average predictions (accurate but expensive)

The deeper problem is **co-adaptation**: neurons in a large network learn to rely on each other. If neuron A always fires when a specific set of neurons B, C, D fire, it never develops a robust, standalone feature detector. When the network sees a new example where B and C happen to fire weakly, A is helpless.

Srivastava et al. proposed a simple fix: during training, randomly set each neuron's output to zero with probability `p`. This forces each neuron to be useful *on its own*, because it can never count on any particular co-conspirator being present.

---

## Core Ideas

### 1. The Dropout Mask

At each training step, for each hidden unit, draw a Bernoulli random variable:

```
r_j ~ Bernoulli(1 - p)
```

The thinned network for that step is `y = f(W(r ⊙ x))`. Each unit is present with probability `1-p` (kept) or absent with probability `p` (dropped).

A typical choice is `p = 0.5` for hidden layers and `p = 0.2` for the input layer.

### 2. Inverted Dropout (Scale at Train Time)

Naive dropout requires scaling outputs by `1/(1-p)` at test time to maintain expected activation magnitudes. The more common modern variant — **inverted dropout** — applies the scale at train time instead:

```python
mask = (random(shape) < keep_prob).astype(float)
mask /= keep_prob       # scale now so test is unchanged
output = activation * mask
```

This means at inference you use the full network with no changes. The implementation here uses inverted dropout.

### 3. Why It Works: A Geometric View

A network with `n` units has `2^n` possible thinned sub-networks. Dropout approximately samples one of these sub-networks on each training step. At test time, the full network (with weights scaled down) approximates an ensemble average over all `2^n` models — exponentially cheaper than explicit ensembling.

### 4. Connection to Weight Regularization

Dropout implicitly regularizes weights. Units that have learned to exploit specific co-activations will find their gradients noisy when those co-units are randomly silenced. The network learns to distribute signal across many redundant pathways instead of concentrating it.

Empirically, models trained with dropout can use significantly larger hidden layer sizes without overfitting, often achieving better final performance than smaller networks without dropout.

### 5. Test-Time Behavior

With inverted dropout, inference is identical to a normal forward pass — no scaling, no masks. Every unit is active and contributes its full weight. This approximates the geometric mean of all sub-network predictions.

---

## Implementation Notes

### Architecture

Two hidden layers of 256 units, ReLU activations, dropout after each hidden layer. Output is a 4-class softmax with cross-entropy loss.

**File:** `implementation.py` (~180 lines), `demo.py` (~90 lines)

### Key Code Paths

**Inverted dropout mask** (`implementation.py`):
```python
def dropout_mask(shape, p, rng):
    keep = 1.0 - p
    mask = (rng.random(shape) < keep).astype(np.float64)
    return mask / keep   # inverted: scale at train time
```

**Forward pass with dropout**:
```python
h1 = relu(self.l1.forward(x))
m1 = dropout_mask(h1.shape, self.drop_p if self.training else 0.0, self.rng)
h1d = h1 * m1
```

**Backward pass** (mask flows through):
```python
d_h2d = self.l3.backward(d_logits)
d_h2 = d_h2d * self._m2        # same mask used in forward
d_a2 = d_h2 * relu_grad(self._h2)
```

The key point: the same binary mask applied in the forward pass is re-used in the backward pass. Dropped units contribute zero gradient.

### What We Simplified

- **Toy 4-class Gaussian classification** instead of MNIST or CIFAR
- **Plain SGD** (no momentum) to keep the optimizer out of the story
- No learning-rate schedule
- `p` is fixed per-layer (no input-layer separate rate)

---

## Running

```bash
# Requirements: numpy (matplotlib optional for plots)
pip install numpy

cd papers/005-dropout
python demo.py
```

Expected output:

```
Paper 005: Dropout — Comparison Demo

Dataset: 200 training samples, 500 validation, 50 features, 4 classes
Architecture: 50 → 256 → 256 → 4  (ReLU + optional Dropout)

==================================================
  No Dropout (baseline)  (drop_p=0.0)
==================================================
  epoch  10 | loss ... | val_acc ...
  ...

==================================================
  With Dropout (p=0.5)  (drop_p=0.5)
==================================================
  epoch  10 | loss ... | val_acc ...
  ...

==================================================
  Summary
==================================================
  No Dropout  val_acc = 0.xxx
  Dropout 0.5 val_acc = 0.xxx
  Delta       = +0.0xx
```

If `matplotlib` is available, `dropout_comparison.png` is saved.

---

## Key Takeaways

### 1. Dropout is approximate ensemble averaging for free

Training one dropout network is equivalent to implicitly averaging over an exponential number of thinned sub-networks. This is why it reduces overfitting so effectively — you get ensemble diversity at the cost of a single model.

### 2. Inverted dropout means zero test-time overhead

By scaling at train time (`mask /= keep_prob`), inference is a plain forward pass with no modification. This is why virtually all modern frameworks (PyTorch, JAX, TensorFlow) default to inverted dropout.

### 3. Dropout enables wider networks

Without dropout, doubling the layer size usually means worse generalization. With dropout, wider is often strictly better — the extra capacity is used for redundant representations rather than memorized noise.

### 4. The mask must be the same in forward and backward

Dropping a unit means removing it from both the forward computation and the gradient flow. Reusing the identical mask in the backward pass is essential; otherwise you'd send gradient through pathways that contributed nothing to the output.

### 5. Dropout shaped modern deep learning defaults

Dropout was central to the success of AlexNet (2012) and became a near-universal component of deep learning for years. While newer architectures (transformers, batch-norm-heavy CNNs) sometimes rely less on it, the idea of stochastic regularization remains highly influential.
