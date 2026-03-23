"""
Adam: A Method for Stochastic Optimization
Kingma and Ba (2014) — https://arxiv.org/abs/1412.6980

Pure NumPy reference implementation of SGD and Adam on a tiny linear model.
"""

import numpy as np


class SGD:
    """Plain SGD with optional momentum."""

    def __init__(self, params: dict[str, np.ndarray], lr: float = 1e-2, momentum: float = 0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = {name: np.zeros_like(value) for name, value in params.items()}

    def step(self, grads: dict[str, np.ndarray]) -> None:
        for name, grad in grads.items():
            self.velocity[name] = self.momentum * self.velocity[name] - self.lr * grad
            self.params[name] += self.velocity[name]


class Adam:
    """
    Adam with bias correction.

    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: dict[str, np.ndarray],
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = {name: np.zeros_like(value) for name, value in params.items()}
        self.v = {name: np.zeros_like(value) for name, value in params.items()}
        self.t = 0

    def step(self, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for name, grad in grads.items():
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad * grad)

            m_hat = self.m[name] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1.0 - self.beta2 ** self.t)
            self.params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class LinearRegression:
    """y = XW + b with manual gradients."""

    def __init__(self, in_features: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.params = {
            "W": rng.normal(0.0, 0.1, size=(in_features, 1)),
            "b": np.zeros((1,), dtype=np.float64),
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.params["W"] + self.params["b"]

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        preds = self.forward(x)
        err = preds - y
        loss = float(np.mean(err ** 2))
        scale = 2.0 / x.shape[0]
        grads = {
            "W": scale * x.T @ err,
            "b": scale * np.sum(err, axis=0),
        }
        return loss, grads


def make_toy_regression(n_samples: int = 256, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an ill-conditioned regression problem where feature scales differ a lot.
    Adam handles this more gracefully than vanilla SGD with one global step size.
    """
    rng = np.random.default_rng(seed)
    x_small = rng.normal(0.0, 1e-2, size=(n_samples, 1))
    x_large = rng.normal(0.0, 10.0, size=(n_samples, 1))
    x = np.concatenate([x_small, x_large], axis=1)
    true_w = np.array([[300.0], [-2.0]])
    noise = rng.normal(0.0, 0.1, size=(n_samples, 1))
    y = x @ true_w + 0.5 + noise
    return x.astype(np.float64), y.astype(np.float64), true_w


def train(
    optimizer_name: str,
    steps: int = 300,
    seed: int = 0,
) -> dict[str, object]:
    x, y, true_w = make_toy_regression(seed=42)
    model = LinearRegression(in_features=x.shape[1], seed=seed)

    if optimizer_name.lower() == "adam":
        optimizer = Adam(model.params, lr=0.2)
    elif optimizer_name.lower() == "sgd":
        optimizer = SGD(model.params, lr=3e-4)
    else:
        raise ValueError("optimizer_name must be 'adam' or 'sgd'")

    history = []
    for step in range(1, steps + 1):
        loss, grads = model.loss_and_grads(x, y)
        optimizer.step(grads)
        history.append(loss)

    final_loss, _ = model.loss_and_grads(x, y)
    param_error = float(np.linalg.norm(model.params["W"] - true_w))
    return {
        "losses": np.array(history, dtype=np.float64),
        "final_loss": final_loss,
        "param_error": param_error,
        "W": model.params["W"].copy(),
        "b": model.params["b"].copy(),
    }
