"""
012 — Layer Normalization
Ba et al., 2016 | https://arxiv.org/abs/1607.06450

Full implementation in <300 lines covering:
  - Core LayerNorm (forward + manual backward)
  - RMSNorm variant (used in LLaMA / modern LLMs)
  - LayerNorm in a Transformer feed-forward block (Pre-LN and Post-LN)
  - LayerNorm in an LSTM cell (the original motivation)
  - Comparison: BN vs LN behavior at batch=1

Requirements: numpy only (no torch)
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Core Layer Normalization
# ---------------------------------------------------------------------------

class LayerNorm:
    """
    Layer Normalization — normalizes over the last (feature) dimension.

    Forward:
        μ  = mean(x, axis=-1)
        σ² = var(x, axis=-1)
        x̂ = (x - μ) / sqrt(σ² + ε)
        y  = γ * x̂ + β

    Parameters γ (gain) and β (bias) are learned; both shape (H,).
    Statistics are per-sample — no batch dependency whatsoever.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.H = normalized_shape
        self.eps = eps
        # Learnable parameters
        self.gamma = np.ones(normalized_shape, dtype=np.float64)   # scale
        self.beta  = np.zeros(normalized_shape, dtype=np.float64)  # shift
        # Cache for backward pass
        self._cache = {}

    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (..., H) — any leading batch dimensions are supported.
        Returns y of the same shape.
        """
        mean = x.mean(axis=-1, keepdims=True)          # (..., 1)
        var  = x.var(axis=-1, keepdims=True)           # (..., 1)
        x_hat = (x - mean) / np.sqrt(var + self.eps)  # (..., H)
        y = self.gamma * x_hat + self.beta             # (..., H)

        self._cache = {"x_hat": x_hat, "var": var, "x": x}
        return y

    # ------------------------------------------------------------------
    def backward(self, dy: np.ndarray):
        """
        Computes gradients w.r.t. x, gamma, and beta.
        dy: same shape as the forward output (..., H).
        Returns dx of the same shape as x.
        """
        x_hat = self._cache["x_hat"]
        var   = self._cache["var"]
        H     = self.H

        # Gradients for learnable params (sum over all but last axis)
        dgamma = (dy * x_hat).sum(axis=tuple(range(dy.ndim - 1)))
        dbeta  = dy.sum(axis=tuple(range(dy.ndim - 1)))

        # Gradient w.r.t. x_hat
        dx_hat = dy * self.gamma                                  # (..., H)

        # Gradient w.r.t. x (derivation mirrors BN, over features)
        inv_std = 1.0 / np.sqrt(var + self.eps)                  # (..., 1)
        dx = inv_std / H * (
            H * dx_hat
            - dx_hat.sum(axis=-1, keepdims=True)
            - x_hat * (dx_hat * x_hat).sum(axis=-1, keepdims=True)
        )

        return dx, dgamma, dbeta

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"LayerNorm(H={self.H}, eps={self.eps})"


# ---------------------------------------------------------------------------
# 2. RMSNorm — the modern simplification (LLaMA, Mistral, Gemma …)
# ---------------------------------------------------------------------------

class RMSNorm:
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, NeurIPS 2019).

    Drops the mean-centering step; normalizes only by RMS:

        RMS(x) = sqrt( mean(x², axis=-1) + ε )
        x̂     = x / RMS(x)
        y      = γ * x̂

    ~30% fewer ops than full LayerNorm; empirically similar quality.
    No β (bias) parameter by convention.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        self.H = normalized_shape
        self.eps = eps
        self.gamma = np.ones(normalized_shape, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + self.eps)
        return self.gamma * (x / rms)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"RMSNorm(H={self.H}, eps={self.eps})"


# ---------------------------------------------------------------------------
# 3. Pre-LN Transformer Feed-Forward Block
# ---------------------------------------------------------------------------

def gelu(x):
    """GELU activation (approximate)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class FeedForwardBlock:
    """
    Transformer feed-forward block with Pre-LN (modern style):

        Post-LN (original):   FFN(x) → + residual → LayerNorm
        Pre-LN  (GPT-2):      LayerNorm(x) → FFN → + residual

    Pre-LN is more stable: gradients flow directly through the residual
    path without passing through LayerNorm, preventing the vanishing
    gradient issue at initialization.
    """

    def __init__(self, d_model: int, d_ff: int, pre_ln: bool = True):
        self.pre_ln = pre_ln
        self.ln = LayerNorm(d_model)
        # Weight matrices (simplified: no bias)
        scale = np.sqrt(2.0 / d_model)
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.W2 = np.random.randn(d_ff, d_model) * scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, seq, d_model)"""
        if self.pre_ln:
            # Pre-LN: normalize first, then FFN, then residual add
            h = self.ln(x)
            h = gelu(h @ self.W1) @ self.W2
            return x + h
        else:
            # Post-LN: FFN first, then residual add, then normalize
            h = gelu(x @ self.W1) @ self.W2
            return self.ln(x + h)

    def __call__(self, x):
        return self.forward(x)


# ---------------------------------------------------------------------------
# 4. Layer-Normalized LSTM Cell (original paper motivation)
# ---------------------------------------------------------------------------

class LNLSTMCell:
    """
    LSTM cell with Layer Normalization applied to the pre-gate activations.

    Without LN, RNNs suffer from exploding/vanishing activations across
    time steps. BN cannot be applied naturally (variable-length sequences,
    time-step-specific statistics). LN applies per-sample, per-time-step,
    making it a perfect fit.

    The LN is applied to the combined (h_{t-1}, x_t) → gates computation.
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.H = hidden_size
        # Input projection: maps [x_t; h_{t-1}] → 4H (four LSTM gates)
        combined = input_size + hidden_size
        scale = np.sqrt(1.0 / combined)
        self.W = np.random.randn(combined, 4 * hidden_size) * scale
        self.b = np.zeros(4 * hidden_size)
        # One LayerNorm per gate group (applied to the 4H pre-activations)
        self.ln_gates = LayerNorm(4 * hidden_size)
        # LayerNorm on cell state before output gate
        self.ln_cell  = LayerNorm(hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        """
        x_t   : (batch, input_size)
        h_prev: (batch, hidden_size)
        c_prev: (batch, hidden_size)
        Returns (h_t, c_t).
        """
        combined = np.concatenate([x_t, h_prev], axis=-1)  # (B, input+hidden)
        gates_raw = combined @ self.W + self.b              # (B, 4H)
        gates = self.ln_gates(gates_raw)                    # apply LN

        H = self.H
        i = _sigmoid(gates[:, :H])          # input gate
        f = _sigmoid(gates[:, H:2*H])       # forget gate
        g = np.tanh(gates[:, 2*H:3*H])     # cell gate
        o = _sigmoid(gates[:, 3*H:])        # output gate

        c_t = f * c_prev + i * g
        h_t = o * np.tanh(self.ln_cell(c_t))  # LN on cell before output gate
        return h_t, c_t

    def __call__(self, x_t, h_prev, c_prev):
        return self.forward(x_t, h_prev, c_prev)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ---------------------------------------------------------------------------
# 5. BatchNorm vs LayerNorm: behavior at small batch sizes
# ---------------------------------------------------------------------------

class SimpleBatchNorm:
    """Minimal 1D BatchNorm for comparison purposes (train mode only)."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = np.ones(num_features)
        self.beta  = np.zeros(num_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, H)"""
        mean = x.mean(axis=0, keepdims=True)   # (1, H) — over batch
        var  = x.var(axis=0, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

    def __call__(self, x):
        return self.forward(x)


def compare_bn_vs_ln(batch_sizes=(1, 2, 8, 32), H=16, seed=42):
    """
    Show how BN becomes unreliable at small batch sizes while LN is stable.
    We measure the variance of normalized outputs — ideally close to 1.0.
    """
    rng = np.random.default_rng(seed)
    print("=" * 60)
    print("BatchNorm vs LayerNorm — output variance stability")
    print("(ideal normalized output variance ≈ 1.0)")
    print("=" * 60)
    print(f"{'Batch Size':>12} | {'BN out var':>12} | {'LN out var':>12}")
    print("-" * 42)

    for N in batch_sizes:
        # Simulate activations with deliberately skewed distribution
        x = rng.exponential(scale=2.0, size=(N, H))

        bn = SimpleBatchNorm(H)
        ln = LayerNorm(H)

        bn_out = bn(x)
        ln_out = ln(x)

        # Variance of outputs (before gamma scaling — γ=1, β=0 initially)
        bn_var = bn_out.var()
        ln_var = ln_out.var()

        print(f"{N:>12} | {bn_var:>12.4f} | {ln_var:>12.4f}")

    print()


# ---------------------------------------------------------------------------
# 6. Gradient check
# ---------------------------------------------------------------------------

def numerical_gradient(f, x, eps=1e-5):
    """Compute numerical gradient of scalar f(x) w.r.t. x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = f(x).sum()
        x[idx] = old - eps
        fm = f(x).sum()
        grad[idx] = (fp - fm) / (2 * eps)
        x[idx] = old
        it.iternext()
    return grad


def gradient_check(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((3, 8))   # (batch=3, H=8)
    ln = LayerNorm(8)

    # Analytical backward
    y = ln.forward(x)
    dy = rng.standard_normal(y.shape)
    dx_analytic, dgamma, dbeta = ln.backward(dy)

    # Numerical gradient for dx
    dx_numeric = numerical_gradient(lambda z: ln.forward(z) * dy, x.copy())

    # Compare
    err = np.abs(dx_analytic - dx_numeric).max()
    print(f"Gradient check — max |analytic − numeric|: {err:.2e}", end="  ")
    print("✓ PASS" if err < 1e-6 else "✗ FAIL")


# ---------------------------------------------------------------------------
# 7. Self-test sequence
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)

    print("\n--- 1. LayerNorm basic forward ---")
    ln = LayerNorm(4)
    x = np.array([[1.0, 2.0, 3.0, 4.0],
                  [0.1, 0.2, 0.3, 0.4]])
    y = ln(x)
    print("Input:\n", x)
    print("Output:\n", y)
    print("Output mean (per row, should ≈ 0):", y.mean(axis=-1))
    print("Output std  (per row, should ≈ 1):", y.std(axis=-1))

    print("\n--- 2. RMSNorm ---")
    rms = RMSNorm(4)
    yr = rms(x)
    print("RMSNorm output:\n", yr)
    rms_vals = np.sqrt((yr ** 2).mean(axis=-1))
    print("RMS of output rows (should ≈ 1):", rms_vals)

    print("\n--- 3. Pre-LN vs Post-LN Feed-Forward ---")
    np.random.seed(1)
    block_pre  = FeedForwardBlock(d_model=8, d_ff=32, pre_ln=True)
    block_post = FeedForwardBlock(d_model=8, d_ff=32, pre_ln=False)
    xb = np.random.randn(2, 5, 8)   # (batch=2, seq=5, d_model=8)
    out_pre  = block_pre(xb)
    out_post = block_post(xb)
    print(f"Pre-LN  output shape: {out_pre.shape}, norm: {np.linalg.norm(out_pre):.4f}")
    print(f"Post-LN output shape: {out_post.shape}, norm: {np.linalg.norm(out_post):.4f}")

    print("\n--- 4. LN-LSTM Cell ---")
    np.random.seed(2)
    cell = LNLSTMCell(input_size=4, hidden_size=8)
    x_t   = np.random.randn(3, 4)    # batch=3
    h0    = np.zeros((3, 8))
    c0    = np.zeros((3, 8))
    h1, c1 = cell(x_t, h0, c0)
    print(f"h_t shape: {h1.shape}, mean abs: {np.abs(h1).mean():.4f}")
    print(f"c_t shape: {c1.shape}, mean abs: {np.abs(c1).mean():.4f}")

    print("\n--- 5. BN vs LN stability ---")
    compare_bn_vs_ln()

    print("--- 6. Gradient check ---")
    gradient_check()
    print()
