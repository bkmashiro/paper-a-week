# 001 — Attention Is All You Need

**Paper:** Vaswani et al., 2017 | **Venue:** NeurIPS 2017  
**Link:** https://arxiv.org/abs/1706.03762

---

## Background

Before 2017, sequence-to-sequence models relied on recurrent architectures (RNNs, LSTMs, GRUs). These process tokens one at a time, making parallelization difficult and causing the gradient to "fade" over long sequences. Attention mechanisms existed as an add-on — a way for a decoder to look back at encoder states — but they weren't the primary computation.

Vaswani et al. made a radical proposal: **discard recurrence entirely** and build the entire model out of attention. The result was dramatically faster to train (full parallelism over sequence length), better at capturing long-range dependencies, and set the foundation for every major language model since — GPT, BERT, T5, and all their descendants.

The key contribution is the **Transformer architecture**: an encoder-decoder model where both encoder and decoder are stacks of identical layers, each containing a multi-head self-attention sublayer and a feedforward sublayer.

## Core Ideas

### Scaled Dot-Product Attention

Given queries Q, keys K, and values V (all matrices), attention is:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

- **Q · Kᵀ**: compute a similarity score between each query and all keys
- **/ √d_k**: scale down to avoid vanishing gradients in softmax (large d_k → large dot products → softmax saturation)
- **softmax**: convert scores to a probability distribution
- **· V**: weighted sum of values — the "answer" for each query

### Multi-Head Attention

Instead of one attention function, use **h** parallel attention "heads":

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
where head_i = Attention(Q W_Qi, K W_Ki, V W_Vi)
```

Each head learns to attend to different parts of the representation. d_k = d_model / h, so total compute stays constant.

### Feed-Forward Network

Each layer has a position-wise FFN applied identically to each token:

```
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
```

The inner dimension is typically 4× the model dimension.

### Residual Connections & Layer Norm

Each sublayer uses `LayerNorm(x + Sublayer(x))`. Residuals enable gradient flow through depth; layer norm stabilizes training.

---

## 中文摘要

Transformer 模型完全抛弃了 RNN/LSTM 的循环结构，仅用注意力机制（Attention）来建模序列关系。

核心公式是 Scaled Dot-Product Attention：用 Query 与所有 Key 做点积相似度打分，再 softmax 归一化后对 Value 加权求和。Multi-Head Attention 是将这个操作并行做 h 次（用不同的投影矩阵），让模型能同时关注不同位置/语义的信息。

Transformer 的优势：
1. **完全并行**：所有 token 同时计算，不像 RNN 那样串行
2. **长程依赖**：任意两个位置的 attention 路径长度为 O(1)，不存在梯度消失
3. **可解释性**：attention 权重可以可视化，看模型在"看"哪里

---

## Implementation Notes

### What We Kept

- Scaled dot-product attention (exact formula from paper)
- Multi-head attention with per-head projections
- Feed-forward layer with inner dimension 4× model dim
- Residual connections + layer normalization
- Full `TransformerBlock` = attention + FFN + norms

### What We Simplified

- **No positional encoding** — we focus on the attention mechanism itself; adding sinusoidal PE is straightforward but obscures the core
- **Encoder only** — no cross-attention decoder (would double the code)
- **numpy only** — no autograd; forward pass only, weights are random (shape demo)
- **No masking** — causal/padding masks omitted for clarity

### Key Code Points

**Scaled dot-product attention** (lines ~30–50):
```python
scores = Q @ K.T / math.sqrt(d_k)   # (seq, seq)
weights = softmax(scores)             # attention distribution
return weights @ V                    # weighted sum
```

**Multi-head** (lines ~55–90): split d_model into h heads, apply attention independently, concatenate:
```python
head_out = attention(Q @ Wq, K @ Wk, V @ Wv)   # (seq, d_k)
# after all heads: concat → (seq, d_model) → project
```

**TransformerBlock** (lines ~110–145): MHA + FFN each wrapped in residual + layernorm.

---

## Running

```bash
# Requirements: numpy only
pip install numpy

python implementation.py
# or run the demo directly:
python demo.py
```

Expected output:
```
TransformerBlock forward pass
Input shape:  (8, 32)   — seq_len=8, d_model=32
Output shape: (8, 32)   — same shape as input ✓

Stack of 4 blocks:
Input shape:  (8, 32)
Output shape: (8, 32)   ✓

Multi-head attention weights shape: (8, 8)
  — each token attends to all 8 positions ✓
```
