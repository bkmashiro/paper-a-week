"""
Demo: Batch Normalization — The Training Stability Effect
=========================================================

This demo trains a 4-layer MLP on synthetic data with and without BN,
showing the key benefits: faster convergence and more stable gradient flow.

Run:
    python demo.py
"""

import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Add parent dir to path so we can import implementation
sys.path.insert(0, os.path.dirname(__file__))
from implementation import BatchNorm1d, Linear, relu


# ---------------------------------------------------------------------------
# Minimal MLP (with or without BN)
# ---------------------------------------------------------------------------

class MLP:
    """4-layer MLP, optionally with BatchNorm after each hidden layer."""

    def __init__(self, dims: list, use_bn: bool = True, seed: int = 42):
        np.random.seed(seed)
        self.use_bn = use_bn
        self.layers = []
        self.bns = []
        self.relu_masks = []

        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1]))
            if use_bn and i < len(dims) - 2:   # no BN on output layer
                self.bns.append(BatchNorm1d(dims[i + 1]))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.relu_masks = []
        bn_idx = 0
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            is_hidden = (i < len(self.layers) - 1)
            if is_hidden:
                if self.use_bn:
                    x = self.bns[bn_idx].forward(x)
                    bn_idx += 1
                x, mask = relu(x)
                self.relu_masks.append(mask)
        return x   # logits (no activation on output)

    def train(self):
        for bn in self.bns:
            bn.train()

    def eval(self):
        for bn in self.bns:
            bn.eval()


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> tuple:
    """Returns (loss, dlogits)."""
    N = logits.shape[0]
    probs = softmax(logits)
    log_probs = np.log(probs + 1e-12)
    loss = -log_probs[np.arange(N), labels].mean()
    dlogits = probs.copy()
    dlogits[np.arange(N), labels] -= 1
    dlogits /= N
    return loss, dlogits


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return (logits.argmax(axis=1) == labels).mean()


def sgd_step(params: list, grads: list, lr: float):
    for p, g in zip(params, grads):
        p -= lr * g


def clip_grad(g: np.ndarray, max_norm: float = 5.0) -> np.ndarray:
    norm = np.linalg.norm(g)
    if norm > max_norm:
        g = g * (max_norm / (norm + 1e-8))
    return g


# ---------------------------------------------------------------------------
# Data: two-spiral dataset (notoriously hard without normalization)
# ---------------------------------------------------------------------------

def make_spiral_data(n_per_class: int = 200, n_classes: int = 3, noise: float = 0.2):
    """Generate n_classes-way spiral classification problem."""
    np.random.seed(0)
    X, y = [], []
    for c in range(n_classes):
        t = np.linspace(0, 1, n_per_class)
        r = t
        theta = t * 4 * np.pi + (2 * np.pi * c / n_classes)
        X.append(np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1))
        y.append(np.full(n_per_class, c, dtype=int))
    X = np.vstack(X) + np.random.randn(n_per_class * n_classes, 2) * noise
    y = np.concatenate(y)
    perm = np.random.permutation(len(y))
    return X[perm].astype(np.float64), y[perm]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(use_bn: bool, n_epochs: int = 200, lr: float = 0.05, batch_size: int = 64):
    X, y = make_spiral_data()
    N = len(X)
    dims = [2, 64, 64, 64, 3]

    model = MLP(dims, use_bn=use_bn)
    model.train()

    losses = []
    accs = []

    for epoch in range(n_epochs):
        # Shuffle
        perm = np.random.permutation(N)
        X, y = X[perm], y[perm]

        epoch_loss = 0.0
        for start in range(0, N, batch_size):
            xb = X[start:start + batch_size]
            yb = y[start:start + batch_size]

            # Forward
            logits = model.forward(xb)
            loss, dlogits = cross_entropy_loss(logits, yb)
            epoch_loss += loss

            # Backward (simplified: only update last layer and BNs via SGD)
            # For a proper MLP backprop, you'd unroll all layers.
            # Here we use a simplified update for demo clarity.
            grads_W = []
            grads_b = []

            # Backprop through output layer
            dx = dlogits
            out_layer = model.layers[-1]
            dW = out_layer._cache.T @ dx
            db = dx.sum(axis=0)
            dx = dx @ out_layer.W.T
            grads_W.append(dW)
            grads_b.append(db)

            # Gradient clipping + SGD update for output layer
            dW, db = clip_grad(dW), clip_grad(db)
            out_layer.W -= lr * dW
            out_layer.b -= lr * db

            # For hidden layers: update BN params + linear via simplified gradient step
            # (full unrolled backprop omitted for brevity — see implementation.py)
            bn_idx = len(model.bns) - 1
            for i in range(len(model.layers) - 2, -1, -1):
                # ReLU backward
                mask_idx = i if i < len(model.relu_masks) else -1
                if mask_idx < len(model.relu_masks):
                    dx = dx * model.relu_masks[mask_idx]

                # BN backward
                if use_bn and bn_idx >= 0:
                    dx, dgamma, dbeta = model.bns[bn_idx].backward(dx)
                    model.bns[bn_idx].gamma -= lr * dgamma
                    model.bns[bn_idx].beta -= lr * dbeta
                    bn_idx -= 1

                # Linear backward
                layer = model.layers[i]
                dW = layer._cache.T @ dx
                db = dx.sum(axis=0)
                dx = dx @ layer.W.T
                dW, db = clip_grad(dW), clip_grad(db)
                layer.W -= lr * dW
                layer.b -= lr * db

        # Eval
        model.eval()
        logits_full = model.forward(X)
        model.train()
        acc = accuracy(logits_full, y)
        avg_loss = epoch_loss / (N // batch_size)
        losses.append(avg_loss)
        accs.append(acc)

        if epoch % 40 == 0 or epoch == n_epochs - 1:
            label = "BN " if use_bn else "No-BN"
            print(f"  [{label}] Epoch {epoch:3d} | loss={avg_loss:.4f} | acc={acc:.2%}")

    return losses, accs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Batch Normalization Demo")
    print("Task: 3-class spiral classification")
    print("=" * 60)

    print("\n>>> Training WITHOUT Batch Normalization:")
    losses_no_bn, accs_no_bn = train(use_bn=False, n_epochs=200, lr=0.05)

    print("\n>>> Training WITH Batch Normalization:")
    losses_bn, accs_bn = train(use_bn=True, n_epochs=200, lr=0.05)

    print("\n" + "=" * 60)
    print("Final Results (epoch 200):")
    print(f"  No BN  — loss: {losses_no_bn[-1]:.4f}  accuracy: {accs_no_bn[-1]:.2%}")
    print(f"  With BN — loss: {losses_bn[-1]:.4f}  accuracy: {accs_bn[-1]:.2%}")
    print("=" * 60)

    # Show convergence speed: how many epochs to reach 70% accuracy?
    def epochs_to_acc(accs, target=0.70):
        for i, a in enumerate(accs):
            if a >= target:
                return i
        return None

    e_no_bn = epochs_to_acc(accs_no_bn)
    e_bn = epochs_to_acc(accs_bn)
    print(f"\nEpochs to reach 70% accuracy:")
    print(f"  No BN:   {e_no_bn if e_no_bn else 'never'}")
    print(f"  With BN: {e_bn if e_bn else 'never'}")
    if e_no_bn and e_bn:
        print(f"  Speedup: ~{e_no_bn / e_bn:.1f}x")

    print("\n>>> Running stats verification (BatchNorm1d):")
    from implementation import BatchNorm1d as BN
    bn = BN(num_features=3)
    # Feed 500 batches from a distribution with mean=[1,2,3], std=[1,2,3]
    np.random.seed(7)
    for _ in range(500):
        xb = np.random.randn(32, 3) * np.array([1, 2, 3]) + np.array([1, 2, 3])
        bn.forward(xb)
    bn.eval()
    print(f"  True mean  = [1, 2, 3],  running_mean ≈ {bn.running_mean.round(2)}")
    print(f"  True var   = [1, 4, 9],  running_var  ≈ {bn.running_var.round(2)}")

    print("\n>>> Train vs. Eval mode behavior:")
    bn2 = BN(num_features=2)
    x_single = np.array([[5.0, -3.0]])  # single sample
    bn2.running_mean = np.array([5.0, -3.0])
    bn2.running_var = np.array([1.0, 1.0])
    bn2.eval()
    out_eval = bn2.forward(x_single)
    print(f"  Input: {x_single[0]}, running_mean={bn2.running_mean}")
    print(f"  Eval output (should be ~[0, 0]): {out_eval[0].round(4)}")


if __name__ == "__main__":
    main()
