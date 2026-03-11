# 003 — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Paper:** Dao et al., 2022 | **Venue:** NeurIPS 2022  
**Link:** https://arxiv.org/abs/2205.14135

---

## Background

Standard attention is usually written as:

```python
softmax(Q @ K.T) @ V
```

This looks simple, but on modern GPUs the real bottleneck is often **memory traffic**, not raw FLOPs. For a sequence of length N, the score matrix `QK^T` is `N x N`, so vanilla attention materializes an **O(N^2)** intermediate in HBM (high-bandwidth memory). HBM is still far slower than on-chip SRAM, so a large fraction of runtime is spent moving the attention matrix back and forth.

Dao et al. argue that attention should be analyzed as an **IO problem**: how many reads/writes happen between HBM and fast SRAM. Their answer is FlashAttention, an exact attention algorithm that avoids storing the full attention matrix and instead computes attention in tiles that fit in SRAM.

The result is not an approximation. It is still exact softmax attention, but implemented in a way that is much more hardware-friendly.

## Core Ideas

### 1. The real problem is HBM traffic, not math

Vanilla attention performs:

1. Write `S = QK^T` to HBM
2. Read `S` back for softmax
3. Write probabilities `P` to HBM
4. Read `P` back to multiply by `V`

So even though the arithmetic is straightforward, the `N x N` matrix causes expensive memory movement. FlashAttention reduces this by never materializing `S` or `P` globally.

### 2. Tile Q/K/V so blocks stay in SRAM

Instead of computing all pairwise scores at once, FlashAttention splits:

- `Q` into row blocks
- `K, V` into column blocks

For each `(Q_i, K_j, V_j)` tile:

```python
S_ij = Q_i @ K_j.T
P_ij = softmax contribution for this tile
O_i += P_ij @ V_j
```

Because each tile is small, it can stay in fast on-chip SRAM. This dramatically reduces HBM reads/writes.

### 3. Online softmax trick

The hard part is softmax. Normally, to compute:

```python
softmax(s) = exp(s - max(s)) / sum(exp(s - max(s)))
```

you need the full row to know the row max and denominator. FlashAttention avoids this by keeping only two running statistics per query row:

- `m_i`: running max
- `l_i`: running sum of exponentials under that max

When a new score block arrives, update:

```python
m_new = max(m_old, m_block)
l_new = exp(m_old - m_new) * l_old + exp(m_block - m_new) * l_block
```

This lets us compute the exact final softmax in one tiled pass, without storing the whole `N x N` matrix.

### 4. Forward pass algorithm

At a high level (Algorithm 1 in the paper):

1. Initialize output block `O_i`, running max `m_i`, and running denominator `l_i`
2. Loop over all `K_j, V_j` blocks
3. Compute block scores `S_ij = Q_i K_j^T`
4. Compute tile-local max/sum
5. Update the running softmax statistics online
6. Update the partial output `O_i`
7. Normalize at the end with `l_i`

This preserves exact attention while using much less HBM memory.

### 5. Memory complexity drops from O(N^2) to O(N)

Vanilla attention needs to store the full score/probability matrices, which is **O(N^2)** memory.

FlashAttention only needs:

- the input/output tensors
- one query tile
- one key/value tile
- running vectors `m` and `l`

So the extra memory is **O(N)** rather than **O(N^2)**.

### 6. Speed in practice

The paper reports FlashAttention is **2-4x faster** than standard attention on an **A100 GPU** for sequence lengths **1024-4096**, while also reducing memory enough to enable longer-context training.

Important nuance: the speedup comes from the CUDA kernel and IO-aware schedule. A small NumPy reference implementation like ours is useful for understanding correctness, but it will not be faster than dense BLAS on CPU.

---

## 中文摘要

FlashAttention 的核心观点是：标准 Attention 的主要瓶颈并不是算力，而是 **HBM 显存读写**。

普通实现会先算出完整的 `QK^T` 分数矩阵，再做 softmax，再乘 `V`。问题在于这个分数矩阵大小是 `N x N`，序列一长就必须频繁在 HBM 和片上 SRAM 之间搬运大量数据。真正慢的是 **I/O**，不是公式本身。

FlashAttention 的做法是把 `Q / K / V` 切成小块（tile）：

1. `Q` 按行分块
2. `K / V` 按列分块
3. 每次只处理一个 `(Q_i, K_j, V_j)` 小块
4. 小块尽量留在 SRAM 中，不把整张 `N x N` attention matrix 写回 HBM

关键难点是 softmax 需要整行归一化。论文用了 **online softmax** 技巧：对每一行维护运行中的最大值 `m` 和分母和 `l`，每来一个新块就更新一次。这样即使看不到整行，也能得到与标准 softmax **完全一致** 的结果。

因此，FlashAttention：

1. **不近似**，结果仍然是 exact attention
2. **额外内存从 O(N^2) 降到 O(N)**
3. **GPU 上更快**，因为减少了昂贵的 HBM 访问

---

## Implementation Notes

### What We Kept

- Exact scaled dot-product attention
- Tiled forward pass over `Q`, `K`, `V`
- Online softmax with running `m` and `l`
- Final output identical to standard attention up to numerical tolerance

### What We Simplified

- **Forward pass only** — no backward kernel
- **Single-head, 2D tensors** — `Q, K, V` are `(N, d)`
- **NumPy reference** — pedagogical, not optimized CUDA
- **Equal block sizes** — one `block_size` for both query and key/value tiles

### Key Code Points

**Standard attention** (`flash_attention.py`):
```python
scores = (Q @ K.T) * scale
weights = softmax(scores)
return weights @ V
```

**FlashAttention tile update**:
```python
m_new = np.maximum(m_i, m_ij)
l_new = np.exp(m_i - m_new) * l_i + np.exp(m_ij - m_new) * l_ij
O_i = (
    (np.exp(m_i - m_new) * l_i)[:, None] * O_i
    + np.exp(m_ij - m_new)[:, None] * (P_ij @ V_j)
) / l_new[:, None]
```

This is the online softmax merge step: combine the previous partial result with the new tile without ever storing a full attention matrix.

---

## Running

```bash
# Requirements: numpy only
pip install numpy

cd papers/003-flashattention
python demo.py
```

Expected output:

```text
Paper 003: FlashAttention — tiled forward pass demo

seq=64    max error ~1e-15
seq=256   max error ~1e-15
seq=1024  max error ~1e-15

Memory comparison:
standard attention scores/probs: O(N^2)
flash running state + tiles:     O(N)
```
