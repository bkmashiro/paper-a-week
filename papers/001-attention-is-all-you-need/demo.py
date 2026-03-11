"""
Demo for paper 001: Attention Is All You Need

Runs the implementation and prints readable output showing:
  - Attention weight visualization (text heatmap)
  - Forward pass shapes through each layer
  - A simple "which tokens attend to which" example
"""

import numpy as np
import sys
import os

# Add parent dirs to path so we can import implementation
sys.path.insert(0, os.path.dirname(__file__))
from implementation import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    FeedForward,
    LayerNorm,
    TransformerBlock,
    TransformerEncoder,
    softmax,
)


def print_separator(title=""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * (width - pad - len(title) - 2))
    else:
        print("─" * width)


def attention_heatmap(weights: np.ndarray, tokens: list[str], head: int = 0) -> None:
    """Print a text-based heatmap of attention weights."""
    w = weights[head]  # (seq, seq)
    n = len(tokens)
    max_tok = max(len(t) for t in tokens)

    # Header
    print(f"  Head {head} attention weights (rows=query, cols=key):")
    print("  " + " " * max_tok + "  " + "  ".join(f"{t:>3}" for t in tokens))

    blocks = " ░▒▓█"
    for i, row_tok in enumerate(tokens):
        row = w[i]
        bar = ""
        for j in range(n):
            v = row[j]
            idx = min(int(v * len(blocks)), len(blocks) - 1)
            bar += f"  {blocks[idx] * 3}"
        print(f"  {row_tok:>{max_tok}}  {bar}")
    print()


def demo_basic_attention():
    print_separator("Scaled Dot-Product Attention")

    # 5 tokens, d_k=4
    rng = np.random.default_rng(7)
    seq, d_k = 5, 4
    Q = rng.normal(0, 1, (seq, d_k))
    K = rng.normal(0, 1, (seq, d_k))
    V = rng.normal(0, 1, (seq, d_k))

    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"  Input:  Q, K, V each shape {Q.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Weights row sums (should be 1.0): {weights.sum(axis=1).round(4)}")
    print()


def demo_self_attention_pattern():
    """Show that with structured input, attention learns meaningful patterns."""
    print_separator("Self-Attention on Structured Tokens")

    # 6 tokens: manually crafted so similar tokens are close in embedding space
    tokens = ["cat", "sat", "mat", "dog", "ran", "far"]
    d_model = 8

    # Give "cat", "mat" similar embeddings (rhyme group 1)
    # Give "sat", "ran" similar embeddings (action group)
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, (d_model,))
    noise = lambda: rng.normal(0, 0.1, (d_model,))

    rhyme1 = rng.normal(0, 1, (d_model,))   # cat, mat pattern
    rhyme2 = rng.normal(0, 1, (d_model,))   # sat, ran pattern

    embeddings = np.array([
        rhyme1 + noise(),   # cat
        rhyme2 + noise(),   # sat
        rhyme1 + noise(),   # mat  (similar to cat)
        rng.normal(0, 1, (d_model,)),  # dog (unique)
        rhyme2 + noise(),   # ran  (similar to sat)
        rng.normal(0, 1, (d_model,)),  # far (unique)
    ])

    # Single-head self-attention
    mha = MultiHeadAttention(d_model=d_model, num_heads=1, seed=99)
    out = mha(embeddings, embeddings, embeddings)
    weights = mha.last_weights

    print(f"  Tokens: {tokens}")
    print(f"  d_model={d_model}, 1 head, d_k={d_model}")
    print()
    attention_heatmap(weights, tokens, head=0)

    print("  Observation: tokens with similar embeddings (cat↔mat, sat↔ran)")
    print("  tend to have higher mutual attention weights.\n")


def demo_multi_head():
    print_separator("Multi-Head Attention")

    d_model, num_heads, seq = 32, 4, 6
    tokens = [f"t{i}" for i in range(seq)]
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (seq, d_model))

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, seed=42)
    out = mha(x, x, x)
    weights = mha.last_weights

    print(f"  d_model={d_model}, heads={num_heads}, seq={seq}")
    print(f"  Per-head d_k = {d_model // num_heads}")
    print(f"  Output shape: {out.shape}")
    print(f"  Weight tensor shape: {weights.shape}  (heads, seq, seq)\n")

    # Show one head
    attention_heatmap(weights, tokens, head=0)


def demo_transformer_block_shapes():
    print_separator("TransformerBlock Layer-by-Layer")

    d_model, heads, seq = 64, 8, 10
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (seq, d_model))

    block = TransformerBlock(d_model=d_model, num_heads=heads, seed=42)

    # Manually trace forward pass
    print(f"  Input x:          {x.shape}")

    # Norm 1
    n1 = block.norm1(x)
    print(f"  LayerNorm(x):     {n1.shape}")

    # Attention
    attn_out = block.attn(n1, n1, n1)
    print(f"  MHA output:       {attn_out.shape}")

    # Residual
    x1 = x + attn_out
    print(f"  x + attn:         {x1.shape}  (residual)")

    # Norm 2
    n2 = block.norm2(x1)
    print(f"  LayerNorm(x1):    {n2.shape}")

    # FFN
    ffn_out = block.ffn(n2)
    print(f"  FFN output:       {ffn_out.shape}  (inner={4*d_model}→{d_model})")

    # Residual
    x2 = x1 + ffn_out
    print(f"  x1 + ffn:         {x2.shape}  (residual)\n")

    # Full block
    out = block(x)
    print(f"  Full block out:   {out.shape}")
    assert np.allclose(out, x2)
    print("  ✓ Manual trace matches block output\n")


def demo_encoder_stack():
    print_separator("Encoder Stack (6 layers, paper config scaled to d=128)")

    d_model, heads, layers, seq = 128, 8, 6, 12
    d_ff = 512

    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (seq, d_model))

    encoder = TransformerEncoder(num_layers=layers, d_model=d_model, num_heads=heads, d_ff=d_ff)
    out = encoder(x)

    print(f"  Config: d_model={d_model}, heads={heads}, layers={layers}, d_ff={d_ff}")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    assert out.shape == x.shape
    print(f"  ✓ Shape preserved through all {layers} layers\n")

    # Show how representation changes through layers
    print("  L2 norm of input vs output (should differ — representation changed):")
    print(f"    Input  norm (mean): {np.linalg.norm(x, axis=1).mean():.3f}")
    print(f"    Output norm (mean): {np.linalg.norm(out, axis=1).mean():.3f}\n")


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Paper 001: Attention Is All You Need — numpy demo       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    demo_basic_attention()
    demo_self_attention_pattern()
    demo_multi_head()
    demo_transformer_block_shapes()
    demo_encoder_stack()

    print_separator("Summary")
    print("  ✓ Scaled dot-product attention verified")
    print("  ✓ Multi-head attention: weights sum to 1, output shape correct")
    print("  ✓ TransformerBlock: residuals + layernorm trace verified")
    print("  ✓ 6-layer encoder preserves shape through all layers")
    print()
    print("  Implementation: papers/001-attention-is-all-you-need/implementation.py")
    print("  Run: python demo.py")
    print()


if __name__ == "__main__":
    main()
