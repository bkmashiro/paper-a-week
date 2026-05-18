"""
012 — Layer Normalization: Demo
Runs all sections from implementation.py and prints readable output.
"""

from implementation import (
    LayerNorm, RMSNorm, FeedForwardBlock, LNLSTMCell,
    compare_bn_vs_ln, gradient_check,
)
import numpy as np


def separator(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


# -----------------------------------------------------------------------
# Demo 1 — LayerNorm is batch-size agnostic
# -----------------------------------------------------------------------
separator("Demo 1: LayerNorm works for any batch size (even 1)")

ln = LayerNorm(normalized_shape=6)
for batch in [1, 4, 32]:
    x = np.random.randn(batch, 6) * 5 + 3   # non-zero mean, non-unit var
    y = ln(x)
    print(f"  batch={batch:2d}  | input mean={x.mean():.2f}  "
          f"out mean={y.mean():.4f}  out std={y.std():.4f}")

print("\n  (Output mean ≈ 0, std ≈ 1 regardless of batch size)")


# -----------------------------------------------------------------------
# Demo 2 — RMSNorm vs LayerNorm output
# -----------------------------------------------------------------------
separator("Demo 2: RMSNorm — simplified normalization (no mean centering)")

np.random.seed(7)
x = np.random.randn(4, 8) * 3
ln_out  = LayerNorm(8)(x)
rms_out = RMSNorm(8)(x)

print(f"  LayerNorm — mean: {ln_out.mean():.4f}, std: {ln_out.std():.4f}")
print(f"  RMSNorm   — mean: {rms_out.mean():.4f}, std: {rms_out.std():.4f}")
print("  (RMSNorm does not center to zero — it only scales by RMS)")


# -----------------------------------------------------------------------
# Demo 3 — Pre-LN vs Post-LN gradient norms (stability)
# -----------------------------------------------------------------------
separator("Demo 3: Pre-LN vs Post-LN — gradient flow stability")

np.random.seed(42)
d_model, d_ff, batch, seq = 16, 64, 4, 10

# Stack 4 blocks and measure how much the residual signal grows/shrinks
for label, pre_ln in [("Post-LN", False), ("Pre-LN", True)]:
    x = np.random.randn(batch, seq, d_model)
    signal_norms = [np.linalg.norm(x)]
    for _ in range(4):
        block = FeedForwardBlock(d_model, d_ff, pre_ln=pre_ln)
        x = block(x)
        signal_norms.append(np.linalg.norm(x))
    trend = " → ".join(f"{v:.1f}" for v in signal_norms)
    print(f"  {label:8s}  signal norm through 4 layers: {trend}")

print("\n  Pre-LN keeps signal norms stable; Post-LN can drift.")


# -----------------------------------------------------------------------
# Demo 4 — LN-LSTM unrolled over a sequence
# -----------------------------------------------------------------------
separator("Demo 4: LN-LSTM — 5-step sequence processing")

np.random.seed(3)
cell = LNLSTMCell(input_size=4, hidden_size=8)
h, c = np.zeros((2, 8)), np.zeros((2, 8))
print(f"  {'Step':>4}  {'|h| mean':>10}  {'|c| mean':>10}")
for t in range(5):
    x_t = np.random.randn(2, 4)
    h, c = cell(x_t, h, c)
    print(f"  {t:>4}  {np.abs(h).mean():>10.4f}  {np.abs(c).mean():>10.4f}")
print("\n  Hidden and cell states stay well-behaved across time steps.")


# -----------------------------------------------------------------------
# Demo 5 — BN vs LN stability at small batch sizes
# -----------------------------------------------------------------------
separator("Demo 5: BatchNorm vs LayerNorm — variance at small batches")
compare_bn_vs_ln(batch_sizes=[1, 2, 4, 8, 16, 32])


# -----------------------------------------------------------------------
# Demo 6 — Gradient check
# -----------------------------------------------------------------------
separator("Demo 6: Gradient correctness check")
gradient_check()


print("\nAll demos complete.\n")
