# 012 — Layer Normalization

**Authors:** Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton  
**Published:** 2016 (arXiv: 1607.06450)  
**Venue:** arXiv preprint (widely adopted; presented at NIPS 2016 workshops)  
**Category:** Deep Learning / Training Techniques  
**Link:** https://arxiv.org/abs/1607.06450

---

## Background

Batch Normalization (BN) transformed deep learning by stabilizing training, but it carries a fundamental limitation: it computes normalization statistics **across the batch dimension**. This creates serious problems in settings where small or variable batch sizes are unavoidable:

- **Recurrent Neural Networks (RNNs):** sequences vary in length; applying BN across time steps is awkward and produces different behavior at each step.
- **Online / on-device inference:** batch size of 1 makes BN statistics meaningless and noisy.
- **Generative models and RL:** single-sample generation or environment rollouts break BN assumptions.
- **Distributed training:** synchronizing batch statistics across workers adds latency.

Layer Normalization (LN) eliminates the batch dependency entirely by normalizing **across the feature dimension** instead — each sample normalizes itself, independently of any other sample in the batch. This makes LN equally effective at train and test time, with batch size 1 or 10,000.

Layer Norm became the normalization of choice in Transformers (both encoder and decoder), all GPT/BERT variants, and virtually every large language model trained today.

---

## Core Ideas

### 1. Normalize Across Features, Not Batch

Given a hidden layer with `H` units and input vector `x ∈ ℝᴴ` for a single sample:

```
μ  = (1/H) Σⱼ xⱼ                      (mean over features)
σ² = (1/H) Σⱼ (xⱼ - μ)²               (variance over features)
x̂ⱼ = (xⱼ - μ) / √(σ² + ε)            (normalize)
yⱼ = γⱼ · x̂ⱼ + βⱼ                    (scale and shift)
```

The key contrast with Batch Norm:

| | Batch Norm | Layer Norm |
|---|---|---|
| Normalize over | Batch (N) | Features (H) |
| Statistics depend on | Other samples | Current sample only |
| Train ≠ Test behavior | Yes | No |
| Works with batch=1 | No | Yes |
| Works in RNNs | Awkward | Natural |

### 2. Per-Sample, Per-Layer Statistics

Because each sample computes its own `μ` and `σ²`, there are no running averages to maintain. The same formula applies at training and inference — no mode-switching required.

### 3. Learnable Affine Transform

Just like BN, LN keeps learnable gain `γ` (initialized to 1) and bias `β` (initialized to 0) parameters per feature. These let the network re-scale the normalized values to any distribution it needs.

### 4. Where LN Is Applied in Transformers

In the original Transformer ("post-LN"), normalization follows the residual addition:

```
x → Sublayer → + residual → LayerNorm → output
```

Modern LLMs mostly use "pre-LN" (GPT-2 style), where LN precedes the sublayer:

```
x → LayerNorm → Sublayer → + residual → output
```

Pre-LN training is more stable and requires less warmup — it avoids the gradient explosion risk at the start of training that post-LN can suffer.

---

## Implementation Details

### Forward Pass

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta
```

This single function handles the full forward pass for any batch size, including 1.

### Gradients

```
∂L/∂γ = Σ (∂L/∂y · x̂)
∂L/∂β = Σ ∂L/∂y
∂L/∂x = (γ/√(σ²+ε)) · [∂L/∂x̂ - mean(∂L/∂x̂) - x̂·mean(∂L/∂x̂ · x̂)]
```

The form is identical to BN gradients, but sums are over features rather than batch elements.

### RNN Application

In an RNN, LN is applied at every time step independently:

```
h_t = tanh(LayerNorm(W_h @ h_{t-1} + W_x @ x_t + b))
```

This was the original motivation: unlike BN, LN does not require time-step-specific statistics or separate inference handling.

---

## Results

The original paper showed:

- **RNNs:** LN-equipped LSTMs significantly outperformed BN-LSTM and vanilla LSTM on sequential tasks (Penn Treebank, text8, etc.)
- **Training stability:** Smoother loss curves with less tuning required
- **Generalization:** Slight regularization effect, similar to BN

The true vindication came later, when the Transformer architecture adopted LN as its normalization layer of choice, making LN one of the most widely deployed operations in modern deep learning.

---

## Key Takeaways

1. **LN normalizes within each sample, not across samples**: this eliminates all batch-size dependencies and makes behavior identical at train and test time.

2. **Essential for Transformers and LLMs**: every GPT, BERT, LLaMA, and similar model uses Layer Norm. Understanding it is non-negotiable for working with modern architectures.

3. **Pre-LN vs Post-LN matters**: pre-LN (normalize before the sublayer) offers better gradient flow and training stability; post-LN matches the original Transformer paper but is trickier to train.

4. **No running statistics to maintain**: simpler implementation than BN — no `running_mean`/`running_var`, no train/eval mode switching.

5. **RMSNorm is a popular simplification**: LLaMA and many modern LLMs use RMSNorm, which drops the mean subtraction step (`μ=0`) and only normalizes by RMS, cutting ~30% of computation with minimal quality loss.

---

## References

- Ba et al., "Layer Normalization", arXiv 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017 (Transformer post-LN)
- Press et al., "Improving Transformers with Pre-LN" — analysis of pre-LN vs post-LN stability
- Zhang & Sennrich, "Root Mean Square Layer Normalization (RMSNorm)", NeurIPS 2019
