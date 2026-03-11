# 003 — FlashAttention：IO 感知的快速精确注意力

[English](README.md)

**论文：** Dao et al., 2022 | **发表于：** NeurIPS 2022  
**链接：** https://arxiv.org/abs/2205.14135

---

## 背景

标准注意力的实现通常写成这样：

```python
softmax(Q @ K.T) @ V
```

看起来很简单。但在现代 GPU 上，**真正的瓶颈往往不是计算量（FLOPs），而是内存带宽（Memory Bandwidth）。**

要理解这一点，需要了解 GPU 的存储层级：

- **HBM（高带宽内存）**：也就是我们平常说的"显存"（VRAM），容量大（数十 GB），但访问速度相对慢。
- **SRAM（片上静态内存）**：在 GPU 芯片上，容量极小（每个 SM 只有几十 KB），但访问速度极快（比 HBM 快约 10-100 倍）。

标准注意力的问题在于：对于长度为 N 的序列，分数矩阵 `QKᵀ` 的大小是 `N × N`。以 seq=1024、d=64 为例，这个矩阵就是 1024×1024 = 100 万个浮点数，必须先写入 HBM，再从 HBM 读出来做 softmax，再写回去，再读出来乘 V……

Dao 等人的核心论点是：**应该把注意力视为一个 I/O 问题**——不是数学问题。他们发现，标准实现中大量时间浪费在搬运这个 N×N 矩阵上，而不是在真正的矩阵乘法计算上。

FlashAttention 的解决方案是：**永远不把完整的 N×N 注意力矩阵写到 HBM**，而是分块（tiling）在 SRAM 中计算，逐块更新输出。

重要的是：这不是近似算法。结果与标准注意力**完全一致**（在浮点精度范围内），只是换了一种更硬件友好的计算顺序。

---

## 核心思想

### 1. 真正的瓶颈：HBM 读写次数，不是 FLOPs

标准注意力的内存访问模式：

1. 计算 `S = QKᵀ`，将 `S`（N×N 矩阵）写入 HBM
2. 从 HBM 读取 `S`，计算 softmax，将概率矩阵 `P` 写入 HBM
3. 从 HBM 读取 `P`，乘以 `V`，将输出写入 HBM

每一步都需要从 HBM 读写一个 O(N²) 的矩阵。对于 N=4096，这是 16M 个浮点数，来回搬运数次。HBM 带宽是瓶颈，不是 CUDA 核心。

FlashAttention 通过分块计算，使得这个 N×N 矩阵**从不以完整形式出现**，HBM 读写量从 O(N²) 降到 O(N)（对于中间状态）。

### 2. 分块计算：让数据留在 SRAM

FlashAttention 的分块策略：

- 将 `Q` 按行分成若干块（每块 B_r 行）
- 将 `K` 和 `V` 按列/行分成若干块（每块 B_c 列/行）
- 对每个 `(Q_i, K_j, V_j)` 小块组合，在 SRAM 中计算局部注意力

```python
# 伪代码
for i in range(num_q_blocks):
    Q_i = Q[i*Br:(i+1)*Br, :]           # 从 HBM 读入，放入 SRAM
    for j in range(num_kv_blocks):
        K_j = K[j*Bc:(j+1)*Bc, :]       # 从 HBM 读入，放入 SRAM
        V_j = V[j*Bc:(j+1)*Bc, :]
        S_ij = Q_i @ K_j.T              # 在 SRAM 中计算，不写回 HBM
        # 用 online softmax 更新 O_i
    # 内层循环结束后，将 O_i 写回 HBM
```

关键：每个小块 `S_ij` 只在 SRAM 中存在，计算完就丢弃，永远不写入 HBM。只有最终输出 `O_i` 才写回 HBM——每个查询块只写一次。

### 3. Online Softmax：无需看完整行也能归一化

这是 FlashAttention 最精妙的数学技巧。

Softmax 的定义是：

```python
softmax(s_i) = exp(s_i - max(s)) / sum(exp(s_j - max(s)) for all j)
```

正常来说，计算 softmax 需要先知道整行的最大值 `max(s)` 和分母 `sum(exp(...))`，必须把整行一次性读入才能计算。但 FlashAttention 分块处理，每次只看一个 key 块。

解决方案：**维护运行状态，在线更新**。对每个查询行，只需跟踪两个标量：

- `m_i`：目前见过的所有 score 的最大值（running max）
- `l_i`：目前 softmax 分母的累计值（running sum of exp）

当处理新的 key 块，得到新的分数块 `S_ij` 时，更新规则为：

```python
m_new = np.maximum(m_i, m_ij)                               # 更新最大值
l_new = np.exp(m_i - m_new) * l_i + np.exp(m_ij - m_new) * l_ij  # 重新缩放旧的，加上新的
```

同时，部分输出也要相应地重新缩放：

```python
O_i = (
    (np.exp(m_i - m_new) * l_i)[:, None] * O_i      # 旧输出按新最大值重新缩放
    + np.exp(m_ij - m_new)[:, None] * (P_ij @ V_j)  # 加上新块的贡献
) / l_new[:, None]
```

当所有 key 块都处理完毕，`O_i` 就是准确的注意力输出——**与一次性计算完整 softmax 结果完全相同**，而从未存储过 N×N 矩阵。

### 4. 前向传播算法（高层概览）

对应论文中的 Algorithm 1：

1. 将 Q 分成 T_r 个行块；将 K, V 分成 T_c 个块
2. 为每个查询块初始化：`O_i = 0`，`m_i = -∞`，`l_i = 0`
3. 外层循环：遍历所有 K_j, V_j 块
4. 计算分数块：`S_ij = Q_i K_j^T / √d`
5. 计算局部 max 和 sum：`m_ij = rowmax(S_ij)`，`l_ij = rowsum(exp(S_ij - m_ij))`
6. 用 online softmax 公式更新 `m_i`、`l_i`、`O_i`
7. 最终用 `l_i` 归一化 `O_i`
8. 将 `O_i` 写回 HBM（每个查询块只写一次）

### 5. 内存复杂度：从 O(N²) 降到 O(N)

| | 标准注意力 | FlashAttention |
|--|-----------|----------------|
| 中间矩阵（S, P）| O(N²) | 不存储（在 SRAM 中即算即丢）|
| 运行状态（m, l）| 不需要 | O(N)（每行两个标量）|
| 总额外内存 | **O(N²)** | **O(N)** |

对于 N=4096，这意味着从 ~64MB（fp32）降到几乎忽略不计的运行状态。

### 6. 实际速度提升

论文报告：在 A100 GPU 上，对于序列长度 1024–4096，FlashAttention 比标准注意力快 **2–4 倍**，同时内存占用大幅降低，使得更长上下文的训练成为可能。

一个重要的注意事项：**速度提升来自 CUDA kernel 的 IO 感知调度**。像我们这样的 NumPy 参考实现只能验证数值正确性，在 CPU 上并不会更快——因为 CPU 的存储层级与 GPU 完全不同，NumPy 的 BLAS 实现已经相当高效。

---

## 实现解析

### 保留了什么

- 精确的缩放点积注意力
- 针对 Q / K / V 的分块前向传播
- 带运行状态 `m` 和 `l` 的 online softmax
- 最终输出与标准注意力在数值精度内完全一致

### 简化了什么

- **仅前向传播**：没有实现反向传播 kernel（论文中有，需要存储 softmax 归一化因子 `l` 用于反向）
- **单头、2D 张量**：Q, K, V 形状为 (N, d)，没有 batch 和 head 维度
- **NumPy 参考实现**：用于理解算法，非优化 CUDA
- **等大块大小**：查询块和键值块使用相同的 block_size

### 核心代码解读

**标准注意力**（`flash_attention.py`）：

```python
scores = (Q @ K.T) * scale
weights = softmax(scores)
return weights @ V
```

**FlashAttention 分块更新**：

```python
m_new = np.maximum(m_i, m_ij)
l_new = np.exp(m_i - m_new) * l_i + np.exp(m_ij - m_new) * l_ij
O_i = (
    (np.exp(m_i - m_new) * l_i)[:, None] * O_i
    + np.exp(m_ij - m_new)[:, None] * (P_ij @ V_j)
) / l_new[:, None]
```

这就是 online softmax 的合并步骤：将上一个块的部分输出与新块的贡献合并，同时正确地重新归一化——全程不存储完整的注意力矩阵。

---

## 运行方法

```bash
# 依赖：仅需 numpy
pip install numpy

cd papers/003-flashattention
python demo.py
```

预期输出：

```text
Paper 003: FlashAttention — tiled forward pass demo

seq=64    max error ~1e-15
seq=256   max error ~1e-15
seq=1024  max error ~1e-15

Memory comparison:
standard attention scores/probs: O(N^2)
flash running state + tiles:     O(N)
```

误差在 `1e-15` 量级，这是 IEEE 754 double 精度的数值误差，完全正常。结果与标准注意力**精确等价**。

---

## 关键收获

### 1. "算法复杂度"≠"实际速度"的教训

FlashAttention 的浮点运算量（FLOPs）与标准注意力相同，甚至略高（需要做更多的重新缩放计算）。但它在实际 GPU 上更快，因为 **I/O 才是真正的瓶颈**。

这是一个非常重要的系统思维转变：当你分析一个算法时，不仅要问"这需要多少次乘法加法"，还要问"这需要多少次内存读写"、"数据在哪一级缓存中"。

### 2. Online Softmax 的数学优雅性

Online softmax 不是 FlashAttention 的独创，但 FlashAttention 将其与 tiling 结合的方式非常巧妙。核心洞见是：softmax 的最终值依赖于全局 max 和 sum，但这两个统计量可以**增量地**更新，而不需要一次性看到所有数据。这与流算法（streaming algorithms）的思想一脉相承。

### 3. FlashAttention 的历史影响

FlashAttention 的出现直接改变了 LLM 的训练和推理格局：

- **更长的上下文**：内存省了 N² → N，seq_len 可以大幅提升。GPT-4（32K context）、Claude（100K context）等长上下文模型背后都有类似优化。
- **FlashAttention-2**（2023）：进一步优化并行度，在 A100 上接近理论带宽上限，速度是 FA-1 的约 2 倍。
- **FlashAttention-3**（2024）：针对 H100 GPU 的新特性（异步 Tensor Core + WGMMA）进行深度优化。
- **成为标配**：PyTorch 2.0 的 `F.scaled_dot_product_attention` 在 CUDA 后端默认调用 FlashAttention 风格的实现。

### 4. 反向传播的巧妙设计

虽然本周实现没有涉及，但值得一提：FlashAttention 的反向传播同样不存储 N×N 矩阵。技巧是存储 softmax 的归一化因子 `l_i`（每行一个标量），在反向传播时重新计算前向的注意力分数——用计算换内存。这在深度学习中称为"重计算"（recomputation / gradient checkpointing）思想。

### 5. IO 感知算法设计的启示

FlashAttention 代表了一类"IO 感知算法"（IO-aware algorithms）的设计理念，在 HPC（高性能计算）领域早有应用，但在深度学习中直到 FlashAttention 才被广泛重视。这种思维方式对于未来设计高效 GPU 算子至关重要：

1. 分析算法的内存访问模式
2. 识别哪些中间结果可以留在 SRAM 中
3. 重排计算顺序以减少 HBM 读写
4. 用轻量的在线统计替代全局聚合

这个框架可以应用于 Attention 之外的许多算子优化场景。
