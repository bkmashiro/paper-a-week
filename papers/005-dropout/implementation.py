"""
Dropout: A Simple Way to Prevent Neural Networks from Overfitting
Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov — JMLR 2014
arXiv: 1207.0580 | https://arxiv.org/abs/1207.0580

Pure NumPy reference implementation of a dropout-regularized MLP.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(x.dtype)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Dropout mask
# ---------------------------------------------------------------------------

def dropout_mask(shape: tuple, p: float, rng: np.random.Generator) -> np.ndarray:
    """
    Bernoulli mask scaled by 1/(1-p) so that expected value is preserved.
    This is the "inverted dropout" trick — no scaling needed at inference.

    Args:
        shape: shape of the activation tensor
        p:     probability of dropping a unit (0 = no dropout, 0.5 = half dropped)
        rng:   NumPy random generator

    Returns:
        mask: float array with 0 or 1/(1-p) entries
    """
    if p == 0.0:
        return np.ones(shape)
    keep = 1.0 - p
    mask = (rng.random(shape) < keep).astype(np.float64)
    return mask / keep          # scale so E[output] = input


# ---------------------------------------------------------------------------
# Dense layer (weights + bias)
# ---------------------------------------------------------------------------

class Linear:
    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / in_dim)   # He init (good with ReLU)
        self.W = rng.normal(0.0, scale, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        self.dW = self._x.T @ d_out
        self.db = d_out.sum(axis=0)
        return d_out @ self.W.T


# ---------------------------------------------------------------------------
# Two-hidden-layer MLP with dropout
# ---------------------------------------------------------------------------

class DropoutMLP:
    """
    Architecture: Input → Linear → ReLU → Dropout
                        → Linear → ReLU → Dropout
                        → Linear → Softmax
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        drop_p: float = 0.5,
        seed: int = 0,
    ):
        self.rng = np.random.default_rng(seed)
        self.drop_p = drop_p
        self.training = True

        self.l1 = Linear(in_dim, hidden, self.rng)
        self.l2 = Linear(hidden, hidden, self.rng)
        self.l3 = Linear(hidden, out_dim, self.rng)

        # Cache for backward
        self._h1: np.ndarray | None = None
        self._h2: np.ndarray | None = None
        self._m1: np.ndarray | None = None
        self._m2: np.ndarray | None = None
        self._probs: np.ndarray | None = None

    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Hidden 1
        a1 = self.l1.forward(x)
        h1 = relu(a1)
        m1 = dropout_mask(h1.shape, self.drop_p if self.training else 0.0, self.rng)
        h1d = h1 * m1

        # Hidden 2
        a2 = self.l2.forward(h1d)
        h2 = relu(a2)
        m2 = dropout_mask(h2.shape, self.drop_p if self.training else 0.0, self.rng)
        h2d = h2 * m2

        # Output
        logits = self.l3.forward(h2d)
        probs = softmax(logits)

        # Cache
        self._h1, self._h2 = h1, h2
        self._m1, self._m2 = m1, m2
        self._probs = probs
        return probs

    # ------------------------------------------------------------------
    def backward(self, y_one_hot: np.ndarray) -> None:
        """Cross-entropy + softmax combined gradient."""
        n = y_one_hot.shape[0]
        d_logits = (self._probs - y_one_hot) / n       # (N, out_dim)

        d_h2d = self.l3.backward(d_logits)

        # Dropout 2
        d_h2 = d_h2d * self._m2
        d_a2 = d_h2 * relu_grad(self._h2)

        d_h1d = self.l2.backward(d_a2)

        # Dropout 1
        d_h1 = d_h1d * self._m1
        d_a1 = d_h1 * relu_grad(self._h1)

        self.l1.backward(d_a1)

    # ------------------------------------------------------------------
    def sgd_step(self, lr: float) -> None:
        for layer in (self.l1, self.l2, self.l3):
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

    # ------------------------------------------------------------------
    def cross_entropy(self, probs: np.ndarray, y: np.ndarray) -> float:
        """
        Args:
            probs: (N, C) softmax probabilities
            y:     (N,)  integer labels
        """
        n = y.shape[0]
        return -float(np.log(probs[np.arange(n), y] + 1e-12).mean())

    # ------------------------------------------------------------------
    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        was = self.training
        self.training = False
        probs = self.forward(x)
        self.training = was
        return float((probs.argmax(axis=1) == y).mean())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: DropoutMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 40,
    lr: float = 0.05,
    batch_size: int = 64,
    report_every: int = 5,
) -> dict:
    rng = np.random.default_rng(0)
    n = x_train.shape[0]
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.training = True
        idx = rng.permutation(n)
        epoch_loss = 0.0
        batches = 0

        for start in range(0, n, batch_size):
            b = idx[start : start + batch_size]
            xb, yb = x_train[b], y_train[b]
            yb_oh = np.eye(model.l3.W.shape[1])[yb]

            probs = model.forward(xb)
            loss = model.cross_entropy(probs, yb)
            model.backward(yb_oh)
            model.sgd_step(lr)

            epoch_loss += loss
            batches += 1

        avg_loss = epoch_loss / batches
        val_acc = model.accuracy(x_val, y_val)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        if epoch % report_every == 0:
            print(f"  epoch {epoch:3d} | loss {avg_loss:.4f} | val_acc {val_acc:.3f}")

    return history
