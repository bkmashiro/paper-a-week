"""
FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
Dao et al. (2022) — https://arxiv.org/abs/2205.14135

Reference implementation of the tiled forward pass in NumPy.
This is a pedagogical implementation — not CUDA, not fast, but correct.
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def qk_scores(Q: np.ndarray, K: np.ndarray, scale: float) -> np.ndarray:
    """Compute scaled QK^T without materializing extra transposes."""
    return np.einsum("id,jd->ij", Q, K) * scale


def pv_product(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Compute P @ V."""
    return np.einsum("ij,jd->id", P, V)


def standard_attention(Q, K, V, scale=None):
    """Standard O(N^2) attention for comparison."""
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    _, d = Q.shape
    if scale is None:
        scale = d ** -0.5

    scores = qk_scores(Q, K, scale)
    weights = softmax(scores, axis=-1)
    return pv_product(weights, V)


def flash_attention_forward(Q, K, V, block_size=64, scale=None):
    """
    Flash attention forward pass with tiling.

    Processes K/V in blocks to avoid materializing full N×N attention matrix.
    Uses the online softmax trick (numerically stable).

    Args:
        Q, K, V: (N, d) arrays
        block_size: tile size (Bc = Br = block_size)
    Returns:
        O: (N, d) output
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    N, d = Q.shape
    if K.shape != (N, d) or V.shape != (N, d):
        raise ValueError("Q, K, V must all have shape (N, d)")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if scale is None:
        scale = d ** -0.5

    O = np.zeros((N, d), dtype=np.float64)

    # Outer loop: tile over Q (row blocks)
    for i in range(0, N, block_size):
        Qi = Q[i:i + block_size]  # (Br, d)
        Br = Qi.shape[0]

        Oi = np.zeros((Br, d), dtype=np.float64)
        li = np.zeros(Br, dtype=np.float64)
        mi = np.full(Br, -np.inf, dtype=np.float64)

        # Inner loop: tile over K, V (column blocks)
        for j in range(0, N, block_size):
            Kj = K[j:j + block_size]  # (Bc, d)
            Vj = V[j:j + block_size]  # (Bc, d)

            Sij = qk_scores(Qi, Kj, scale)           # (Br, Bc)
            mij = np.max(Sij, axis=-1)               # (Br,)
            Pij = np.exp(Sij - mij[:, None])         # (Br, Bc)
            lij = np.sum(Pij, axis=-1)               # (Br,)

            mi_new = np.maximum(mi, mij)
            alpha = np.exp(mi - mi_new)
            beta = np.exp(mij - mi_new)
            li_new = alpha * li + beta * lij

            Oi = (
                (alpha * li)[:, None] * Oi
                + beta[:, None] * pv_product(Pij, Vj)
            ) / li_new[:, None]

            mi = mi_new
            li = li_new

        O[i:i + block_size] = Oi

    return O
