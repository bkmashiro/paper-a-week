# 002 — LoRA: Low-Rank Adaptation of Large Language Models

**Paper:** Hu et al., 2021 | **Venue:** ICLR 2022
**Link:** https://arxiv.org/abs/2106.09685

---

## Background

Fine-tuning large language models like GPT-3 (175B parameters) requires updating every parameter — storing a full copy per task, consuming enormous memory, and making deployment impractical. If you want GPT-3 for 10 tasks, you need 10 copies of 175B parameters.

Hu et al. observed that the weight updates during fine-tuning have **low intrinsic rank** — most of the information in ΔW can be captured by a much smaller matrix. This led to LoRA: freeze the pretrained weights entirely, and inject small trainable low-rank matrices alongside them. The result is 10,000× fewer trainable parameters with no loss in task performance.

LoRA has become the dominant parameter-efficient fine-tuning method, powering everything from instruction-tuned LLMs to Stable Diffusion customization. Its simplicity (no architectural changes, no extra inference latency) makes it a rare case of a method that is both theoretically elegant and immediately practical.

## Core Ideas

### Low-Rank Decomposition of Weight Updates

For a pretrained weight matrix W ∈ ℝ^{d×d}, instead of learning a full ΔW, decompose it:

```
ΔW = BA    where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×d}, r << d
```

The forward pass becomes:

```
h = Wx + BAx × (α/r)
```

For a 1024×1024 weight matrix (1M params), with rank r=8:
- Full fine-tune: 1,048,576 params
- LoRA: 1024×8 + 8×1024 = **16,384 params** (1.6% of original)

### Initialization Strategy

- **B = 0**: ensures ΔW = BA = 0 at initialization, so the model starts as the exact pretrained model
- **A ~ Kaiming uniform**: standard random initialization provides diverse directions for A

This is critical — random initialization of both A and B would immediately corrupt the pretrained representations.

### Scaling Factor α/r

The output is scaled by α/r, where α is a hyperparameter. This decouples the learning rate from the rank: when you change r, you don't need to retune the learning rate. The paper typically sets α = 2r.

### Weight Merging

At inference time, merge LoRA back into the base weights:

```
W_merged = W + BA × (α/r)
```

Result: **zero additional latency** compared to the original model. You can even swap different LoRA adapters in and out without changing the inference code.

---

## 中文摘要

LoRA 的核心思想是：大模型微调时的权重变化矩阵 ΔW 具有**低秩特性**，不需要更新全部参数。

具体做法是冻结预训练权重 W，在旁边插入两个小矩阵 B ∈ ℝ^{d×r} 和 A ∈ ℝ^{r×d}（r 远小于 d），只训练这两个矩阵。前向传播为 h = Wx + BAx × (α/r)。

关键设计：
1. **B 初始化为零**：训练开始时 ΔW = BA = 0，完全保留预训练模型的行为
2. **α/r 缩放**：使得调整 rank 时不需要重新调学习率
3. **权重合并**：推理时将 BA 合并回 W，不增加任何延迟

效果：在 GPT-3 175B 上，LoRA 只需训练 ~4.7M 参数（占比 0.003%），在 GLUE 等任务上性能与全量微调持平。这使得普通用户也能在消费级 GPU 上微调大模型。

---

## Implementation Notes

### What We Kept

- Exact LoRA formulation: ΔW = BA with B=0 init, A Kaiming init
- Scaling by α/r
- Freeze base weights, only train A and B
- Weight merging for inference
- `apply_lora` utility to inject LoRA into arbitrary models

### What We Simplified

- **No dropout on LoRA path** — paper uses dropout on the LoRA branch, omitted for clarity
- **No per-layer rank tuning** — same rank for all target layers
- **Tiny model** — demo uses a small GPT (not actual pretrained weights)

### Key Code Points

**LoRALinear forward** (lora.py, lines ~60–65):
```python
base = self.linear(x)                                      # frozen W
lora = (x @ self.lora_A.T @ self.lora_B.T) * (alpha / r)  # low-rank path
return base + lora
```

**Weight merging** (lora.py, lines ~68–75):
```python
delta = (self.lora_B @ self.lora_A) * (self.alpha / self.rank)
self.linear.weight.add_(delta)  # merge into W, no extra latency
```

**apply_lora** (lora.py, lines ~85–100): walks the module tree, replaces target `nn.Linear` layers with `LoRALinear` wrappers.

---

## Running

```bash
# Requirements: torch
pip install torch

# Run the demo
cd papers/002-lora
python demo.py
```

Expected output:
```
╔══════════════════════════════════════════════════════════╗
║  Paper 002: LoRA — Low-Rank Adaptation demo             ║
╚══════════════════════════════════════════════════════════╝

─────────── Scaling Intuition: rank vs params ────────────
  Weight matrix: 1024×1024 = 1,048,576 params

    rank   LoRA params    % of full
  ──────  ────────────  ──────────
       1         2,048       0.20%
       4         8,192       0.78%
       8        16,384       1.56%
      64       131,072      12.50%

──────────── Parameter Comparison ─────────────
  Full fine-tune: ~3.6M params
  LoRA (rank=8, Q+V only): ~33K trainable (< 1%)

──────────── Training with LoRA ───────────────
  Step 1-5: loss decreases ✓

──────────── Weight Merging ───────────────────
  Merged 8 LoRA layers, output diff: ~0 ✓
```
