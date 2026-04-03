"""
Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
Ioffe & Szegedy, ICML 2015 — arXiv:1502.03167

This implementation covers:
  - BatchNorm1d (fully-connected layers): normalize over (N,) per feature
  - BatchNorm2d (convolutional layers): normalize over (N, H, W) per channel
  - Full forward pass with train/eval mode switching
  - Manual backward pass (no autograd) for educational clarity
  - Running mean/variance tracking for inference
"""

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------

def bn_forward_train(
    x: np.ndarray,        # (N, C) or (N, C, H, W)
    gamma: np.ndarray,    # (C,)
    beta: np.ndarray,     # (C,)
    eps: float = 1e-5,
) -> Tuple[np.ndarray, dict]:
    """
    BN forward pass (training mode).
    Returns normalized output and cache for backward pass.
    """
    original_shape = x.shape
    N = x.shape[0]

    # Reshape to (N, C, -1) for unified treatment of 1D and 2D inputs
    if x.ndim == 2:
        # (N, C) → (N, C, 1)
        x_flat = x.reshape(N, -1, 1)
    elif x.ndim == 4:
        # (N, C, H, W) → (N, C, H*W)
        N, C, H, W = x.shape
        x_flat = x.reshape(N, C, H * W)
    else:
        raise ValueError(f"Expected 2D or 4D input, got {x.ndim}D")

    # x_flat: (N, C, M) where M = 1 for FC, H*W for Conv
    # Compute statistics over (N, M) dimensions for each channel C
    # Result shapes: (1, C, 1)
    mu = x_flat.mean(axis=(0, 2), keepdims=True)       # batch mean per channel
    var = x_flat.var(axis=(0, 2), keepdims=True)        # batch variance per channel
    std = np.sqrt(var + eps)

    x_hat = (x_flat - mu) / std                         # normalized: (N, C, M)

    # Reshape gamma, beta for broadcasting: (1, C, 1)
    g = gamma.reshape(1, -1, 1)
    b = beta.reshape(1, -1, 1)
    out_flat = g * x_hat + b

    # Restore original shape
    out = out_flat.reshape(original_shape)

    cache = {
        "x_hat": x_hat,       # (N, C, M)
        "mu": mu,              # (1, C, 1)
        "var": var,            # (1, C, 1)
        "std": std,            # (1, C, 1)
        "gamma": gamma,        # (C,)
        "eps": eps,
        "original_shape": original_shape,
        "N_eff": x_flat.shape[0] * x_flat.shape[2],   # N * M (effective batch size)
    }
    return out, cache


def bn_backward(
    dout: np.ndarray,   # same shape as x
    cache: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    BN backward pass.
    Returns (dx, dgamma, dbeta).
    """
    x_hat = cache["x_hat"]     # (N, C, M)
    std = cache["std"]          # (1, C, 1)
    gamma = cache["gamma"]      # (C,)
    original_shape = cache["original_shape"]
    N_eff = cache["N_eff"]      # N * M

    N = dout.shape[0]
    if dout.ndim == 2:
        dout_flat = dout.reshape(N, -1, 1)
    elif dout.ndim == 4:
        N, C, H, W = dout.shape
        dout_flat = dout.reshape(N, C, H * W)
    else:
        raise ValueError(f"Expected 2D or 4D dout, got {dout.ndim}D")

    # Gradients for gamma and beta (sum over N and spatial dims)
    dgamma = (dout_flat * x_hat).sum(axis=(0, 2))    # (C,)
    dbeta = dout_flat.sum(axis=(0, 2))                # (C,)

    g = gamma.reshape(1, -1, 1)

    # Gradient w.r.t. x_hat
    dx_hat = dout_flat * g                             # (N, C, M)

    # Gradient w.r.t. x (through the normalization formula)
    # dx = (1/std) * [dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat)]
    dx_flat = (1.0 / std) * (
        dx_hat
        - dx_hat.mean(axis=(0, 2), keepdims=True)
        - x_hat * (dx_hat * x_hat).mean(axis=(0, 2), keepdims=True)
    )

    dx = dx_flat.reshape(original_shape)
    return dx, dgamma, dbeta


def bn_forward_eval(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,  # (C,)
    running_var: np.ndarray,   # (C,)
    eps: float = 1e-5,
) -> np.ndarray:
    """BN forward pass (evaluation/inference mode)."""
    original_shape = x.shape
    N = x.shape[0]

    if x.ndim == 2:
        x_flat = x.reshape(N, -1, 1)
    elif x.ndim == 4:
        N, C, H, W = x.shape
        x_flat = x.reshape(N, C, H * W)
    else:
        raise ValueError(f"Expected 2D or 4D input, got {x.ndim}D")

    mu = running_mean.reshape(1, -1, 1)
    std = np.sqrt(running_var.reshape(1, -1, 1) + eps)
    g = gamma.reshape(1, -1, 1)
    b = beta.reshape(1, -1, 1)

    x_hat = (x_flat - mu) / std
    out = g * x_hat + b
    return out.reshape(original_shape)


# ---------------------------------------------------------------------------
# BatchNorm Layer classes (numpy, educational)
# ---------------------------------------------------------------------------

class BatchNorm:
    """
    Generic Batch Normalization layer.
    Supports 1D (FC) and 2D (Conv) inputs via unified implementation.

    Parameters
    ----------
    num_features : int
        Number of channels C (FC: feature dim; Conv: channel dim).
    eps : float
        Numerical stability constant, default 1e-5.
    momentum : float
        Momentum for running statistics update, default 0.1 (PyTorch default).
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones(num_features, dtype=np.float64)   # scale
        self.beta = np.zeros(num_features, dtype=np.float64)   # shift

        # Running statistics (used during eval)
        self.running_mean = np.zeros(num_features, dtype=np.float64)
        self.running_var = np.ones(num_features, dtype=np.float64)

        # Cached for backward
        self._cache: Optional[dict] = None

        # Mode
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            out, cache = bn_forward_train(x, self.gamma, self.beta, self.eps)
            self._cache = cache

            # Update running statistics
            # mu and var from cache: (1, C, 1) → squeeze to (C,)
            batch_mean = cache["mu"].squeeze()
            batch_var = cache["var"].squeeze()
            self.running_mean = (
                (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var + self.momentum * batch_var
            )
            return out
        else:
            return bn_forward_eval(
                x, self.gamma, self.beta,
                self.running_mean, self.running_var, self.eps,
            )

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (dx, dgamma, dbeta)."""
        if self._cache is None:
            raise RuntimeError("backward() called before forward() in training mode")
        return bn_backward(dout, self._cache)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __repr__(self):
        return (
            f"BatchNorm(num_features={self.num_features}, "
            f"eps={self.eps}, momentum={self.momentum})"
        )


# Convenience aliases
class BatchNorm1d(BatchNorm):
    """BatchNorm for 2D inputs (N, C)."""
    pass


class BatchNorm2d(BatchNorm):
    """BatchNorm for 4D inputs (N, C, H, W)."""
    pass


# ---------------------------------------------------------------------------
# Simple MLP with BN for testing
# ---------------------------------------------------------------------------

class Linear:
    """Basic fully-connected layer."""

    def __init__(self, in_features: int, out_features: int):
        # He initialization (good for ReLU)
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray):
        x = self._cache
        dW = x.T @ dout
        db = dout.sum(axis=0)
        dx = dout @ self.W.T
        return dx, dW, db


def relu(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ReLU activation, returns (output, mask)."""
    mask = (x > 0).astype(x.dtype)
    return x * mask, mask


class BNMLPBlock:
    """
    One block: Linear → BN → ReLU
    Used to demonstrate BN in practice.
    """

    def __init__(self, in_features: int, out_features: int):
        self.linear = Linear(in_features, out_features)
        self.bn = BatchNorm1d(out_features)
        self._relu_mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = self.linear.forward(x)
        out = self.bn.forward(out)
        out, self._relu_mask = relu(out)
        return out

    def backward(self, dout: np.ndarray):
        dout = dout * self._relu_mask          # through ReLU
        dout, dgamma, dbeta = self.bn.backward(dout)
        dx, dW, db = self.linear.backward(dout)
        return dx, {"W": dW, "b": db, "gamma": dgamma, "beta": dbeta}

    def train(self):
        self.bn.train()

    def eval(self):
        self.bn.eval()


if __name__ == "__main__":
    # Quick sanity check
    np.random.seed(42)

    print("=== BatchNorm1d sanity check ===")
    bn = BatchNorm1d(num_features=4)
    x = np.random.randn(8, 4)

    # Training forward
    out_train = bn.forward(x)
    print(f"Input  mean per feature: {x.mean(axis=0).round(3)}")
    print(f"Output mean per feature: {out_train.mean(axis=0).round(3)}  (should be ~0)")
    print(f"Output std  per feature: {out_train.std(axis=0).round(3)}   (should be ~1)")

    # Switch to eval
    bn.eval()
    out_eval = bn.forward(x)
    print(f"\nEval output (using running stats): same shape {out_eval.shape} ✓")

    print("\n=== BatchNorm2d sanity check ===")
    bn2d = BatchNorm2d(num_features=3)
    x2d = np.random.randn(4, 3, 8, 8)  # batch=4, channels=3, 8x8 spatial
    out2d = bn2d.forward(x2d)
    print(f"Input shape: {x2d.shape} → Output shape: {out2d.shape} ✓")

    print("\n=== Running stats update check ===")
    bn3 = BatchNorm1d(num_features=2)
    for _ in range(100):
        x_test = np.random.randn(32, 2) * 3 + 5   # mean≈5, std≈3
        bn3.forward(x_test)
    print(f"True mean  ≈ [5, 5], running_mean ≈ {bn3.running_mean.round(2)}")
    print(f"True var   ≈ [9, 9], running_var  ≈ {bn3.running_var.round(2)}")
