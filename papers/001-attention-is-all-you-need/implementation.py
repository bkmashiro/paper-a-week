"""
Attention Is All You Need — Core Implementation
Vaswani et al., NeurIPS 2017 | https://arxiv.org/abs/1706.03762

Implements:
  - scaled_dot_product_attention
  - MultiHeadAttention
  - FeedForward
  - LayerNorm
  - TransformerBlock
  - TransformerEncoder (stack of blocks)

Pure numpy, forward-pass only. No positional encoding (encoder only demo).
Total: ~230 lines (including docstrings and comments)
"""

import numpy as np
import math


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along `axis`."""
    x = x - x.max(axis=axis, keepdims=True)   # subtract max for stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


# ─────────────────────────────────────────────────────────────
# Scaled Dot-Product Attention
# ─────────────────────────────────────────────────────────────

def scaled_dot_product_attention(
    Q: np.ndarray,   # (seq_q, d_k)
    K: np.ndarray,   # (seq_k, d_k)
    V: np.ndarray,   # (seq_k, d_v)
    mask: np.ndarray = None,   # (seq_q, seq_k) optional boolean mask
) -> tuple[np.ndarray, np.ndarray]:
    """
    Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

    Returns:
        output: (seq_q, d_v) — weighted sum of values
        weights: (seq_q, seq_k) — attention distribution (for visualization)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d_k)   # (seq_q, seq_k) similarity scores

    if mask is not None:
        # set masked positions to -inf so softmax gives ~0 weight
        scores = np.where(mask, scores, -1e9)

    weights = softmax(scores, axis=-1)   # (seq_q, seq_k) attention probs
    output = weights @ V                 # (seq_q, d_v) attended values
    return output, weights


# ─────────────────────────────────────────────────────────────
# Layer Normalization
# ─────────────────────────────────────────────────────────────

class LayerNorm:
    """
    LayerNorm(x) = γ * (x - μ) / (σ + ε) + β
    Normalizes across the last dimension (feature dim).
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones(d_model)   # learnable scale
        self.beta = np.zeros(d_model)   # learnable shift
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


# ─────────────────────────────────────────────────────────────
# Multi-Head Attention
# ─────────────────────────────────────────────────────────────

class MultiHeadAttention:
    """
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
    where head_i = Attention(Q W_Qi, K W_Ki, V W_Vi)

    d_k = d_v = d_model / h  — each head works in a subspace
    """
    def __init__(self, d_model: int, num_heads: int, seed: int = 42):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        rng = np.random.default_rng(seed)

        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads   # per-head dimension

        # Per-head projection matrices (stored stacked for convenience)
        # Shape: (h, d_model, d_k)
        scale = math.sqrt(2.0 / d_model)
        self.W_q = rng.normal(0, scale, (num_heads, d_model, self.d_k))
        self.W_k = rng.normal(0, scale, (num_heads, d_model, self.d_k))
        self.W_v = rng.normal(0, scale, (num_heads, d_model, self.d_k))
        # Output projection: (d_model, d_model)
        self.W_o = rng.normal(0, scale, (d_model, d_model))

        self.last_weights = None   # for inspection

    def __call__(
        self,
        Q: np.ndarray,   # (seq, d_model) — query
        K: np.ndarray,   # (seq, d_model) — key
        V: np.ndarray,   # (seq, d_model) — value
        mask: np.ndarray = None,
    ) -> np.ndarray:
        seq_len = Q.shape[0]
        head_outputs = []
        all_weights = []

        for i in range(self.h):
            # Project to per-head subspace: (seq, d_k)
            Qi = Q @ self.W_q[i]
            Ki = K @ self.W_k[i]
            Vi = V @ self.W_v[i]

            # Compute attention for this head
            head_out, weights = scaled_dot_product_attention(Qi, Ki, Vi, mask)
            head_outputs.append(head_out)    # (seq, d_k)
            all_weights.append(weights)

        # Concatenate all heads: (seq, d_model)
        concat = np.concatenate(head_outputs, axis=-1)

        # Final linear projection
        output = concat @ self.W_o   # (seq, d_model)

        # Store attention weights of last head for inspection
        self.last_weights = np.stack(all_weights)   # (h, seq, seq)
        return output


# ─────────────────────────────────────────────────────────────
# Feed-Forward Network
# ─────────────────────────────────────────────────────────────

class FeedForward:
    """
    FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2

    d_ff = 4 * d_model  — paper uses this ratio
    Applied independently to each position (position-wise).
    """
    def __init__(self, d_model: int, d_ff: int = None, seed: int = 0):
        rng = np.random.default_rng(seed)
        d_ff = d_ff or 4 * d_model
        scale = math.sqrt(2.0 / d_model)
        self.W1 = rng.normal(0, scale, (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.normal(0, scale, (d_ff, d_model))
        self.b2 = np.zeros(d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (seq, d_model)
        hidden = relu(x @ self.W1 + self.b1)   # (seq, d_ff)
        return hidden @ self.W2 + self.b2       # (seq, d_model)


# ─────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────

class TransformerBlock:
    """
    One Transformer encoder layer:
      1. Multi-Head Self-Attention + residual + LayerNorm
      2. Feed-Forward + residual + LayerNorm

    Pre-LN variant: LayerNorm BEFORE sublayer (more stable training).
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int = None, seed: int = 42):
        self.attn = MultiHeadAttention(d_model, num_heads, seed=seed)
        self.ffn = FeedForward(d_model, d_ff, seed=seed + 1)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def __call__(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        # Self-attention sublayer (Q = K = V = x)
        normed = self.norm1(x)
        attn_out = self.attn(normed, normed, normed, mask)
        x = x + attn_out   # residual connection

        # Feed-forward sublayer
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out    # residual connection

        return x


# ─────────────────────────────────────────────────────────────
# Transformer Encoder (stack of blocks)
# ─────────────────────────────────────────────────────────────

class TransformerEncoder:
    """
    Stack of N identical TransformerBlocks.
    Input/output shape: (seq_len, d_model)
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int = None):
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, seed=i * 100)
            for i in range(num_layers)
        ]

    def __call__(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        for block in self.blocks:
            x = block(x, mask)
        return x


# ─────────────────────────────────────────────────────────────
# Demo / self-test
# ─────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(0)

    # ── Single block test ──────────────────────────────────
    d_model, num_heads, seq_len = 32, 4, 8
    x = rng.normal(0, 1, (seq_len, d_model))

    block = TransformerBlock(d_model=d_model, num_heads=num_heads)
    out = block(x)

    print("=" * 50)
    print("TransformerBlock forward pass")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"
    print("  ✓ Shapes match")

    # Check attention weights
    weights = block.attn.last_weights
    print(f"\n  Attention weights shape: {weights.shape}  (num_heads, seq, seq)")
    # Each head's weights should sum to 1 over keys
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5), "Attention weights should sum to 1"
    print("  ✓ Attention weights sum to 1")

    # ── Encoder stack test ────────────────────────────────
    print("\n" + "=" * 50)
    print("TransformerEncoder (4 layers)")
    encoder = TransformerEncoder(num_layers=4, d_model=d_model, num_heads=num_heads)
    enc_out = encoder(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {enc_out.shape}")
    assert enc_out.shape == x.shape
    print("  ✓ Shapes match")

    # ── Bigger model test (paper config scaled down) ──────
    print("\n" + "=" * 50)
    print("Paper-like config (d_model=512, heads=8, layers=6, seq=20)")
    d_model_big = 512
    x_big = rng.normal(0, 1, (20, d_model_big))
    encoder_big = TransformerEncoder(
        num_layers=6, d_model=d_model_big, num_heads=8, d_ff=2048
    )
    out_big = encoder_big(x_big)
    print(f"  Input shape:  {x_big.shape}")
    print(f"  Output shape: {out_big.shape}")
    assert out_big.shape == x_big.shape
    print("  ✓ All assertions passed")

    # ── Parameter count ───────────────────────────────────
    def count_params(encoder):
        total = 0
        for block in encoder.blocks:
            attn = block.attn
            # Q, K, V projections: h * (d_model * d_k) * 3
            total += attn.W_q.size + attn.W_k.size + attn.W_v.size
            # Output projection
            total += attn.W_o.size
            # FFN
            total += block.ffn.W1.size + block.ffn.b1.size
            total += block.ffn.W2.size + block.ffn.b2.size
            # LayerNorms (gamma + beta)
            total += block.norm1.gamma.size * 2 + block.norm2.gamma.size * 2
        return total

    params = count_params(encoder_big)
    print(f"\n  Parameter count (encoder only): {params:,}")
    print(f"  (Paper's full model: ~65M params)")

    print("\n" + "=" * 50)
    print("All tests passed! ✓")


if __name__ == "__main__":
    main()
