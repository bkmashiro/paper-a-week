"""
GAN: Generative Adversarial Networks
Goodfellow et al., NeurIPS 2014
https://arxiv.org/abs/1406.2661

Pure Python implementation (stdlib only: math, random).
Demonstrates the minimax game on a 1D Gaussian mixture target.
Generator and Discriminator are 2-hidden-layer MLPs trained with BCE.

Run:
    python implementation.py   # or python3 implementation.py
"""

import math
import random


# ---------------------------------------------------------------------------
# Math primitives (no numpy — works on any Python 3.8+ install)
# ---------------------------------------------------------------------------

def tanh(v):
    return [math.tanh(x) for x in v]


def sigmoid(v):
    """Numerically stable sigmoid."""
    out = []
    for x in v:
        x = max(-30.0, min(30.0, x))
        out.append(1.0 / (1.0 + math.exp(-x)))
    return out


def mat_vec(W, x):
    """Matrix-vector multiply: W (m×n) @ x (n,) → (m,)."""
    return [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]


def vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def clip(x, lo=1e-7, hi=1 - 1e-7):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Neural network primitives
# ---------------------------------------------------------------------------

class Linear:
    """
    Single linear layer y = W @ x + b.
    Stores forward activations for backprop. Accumulates gradients.
    """

    def __init__(self, n_in: int, n_out: int):
        # He initialization for tanh
        scale = math.sqrt(2.0 / n_in)
        self.W = [[random.gauss(0, scale) for _ in range(n_in)] for _ in range(n_out)]
        self.b = [0.0] * n_out
        self._zero_grad()
        self._x = None  # cached input

    def _zero_grad(self):
        n_in = len(self.W[0]) if self.W else 0
        n_out = len(self.W)
        self.dW = [[0.0] * n_in for _ in range(n_out)]
        self.db = [0.0] * n_out

    def forward(self, x):
        self._x = x
        return vec_add(mat_vec(self.W, x), self.b)

    def backward(self, grad_out):
        """Accumulate gradients; return grad w.r.t. input."""
        x = self._x
        n_in, n_out = len(x), len(grad_out)
        for i in range(n_out):
            self.db[i] += grad_out[i]
            for j in range(n_in):
                self.dW[i][j] += grad_out[i] * x[j]
        # gradient w.r.t. input
        return [sum(self.W[i][j] * grad_out[i] for i in range(n_out)) for j in range(n_in)]

    def step(self, lr: float):
        for i in range(len(self.W)):
            self.b[i] -= lr * self.db[i]
            for j in range(len(self.W[i])):
                self.W[i][j] -= lr * self.dW[i][j]
        self._zero_grad()


class MLP:
    """
    Multi-layer perceptron: Linear → Tanh → Linear → Tanh → Linear → [sigmoid|linear].
    dims: list of layer widths, e.g. [2, 32, 32, 1].
    """

    def __init__(self, dims: list, output: str = "linear"):
        self.layers = [Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.output = output   # 'sigmoid' for D, 'linear' for G
        self._hiddens = []     # (pre_act, post_act) for each hidden layer
        self._out = None

    def forward(self, x: list) -> list:
        self._hiddens = []
        h = x
        for layer in self.layers[:-1]:
            z = layer.forward(h)
            h = tanh(z)
            self._hiddens.append((z, h))
        z_final = self.layers[-1].forward(h)
        if self.output == "sigmoid":
            self._out = sigmoid(z_final)
        else:
            self._out = z_final
        return self._out

    def backward(self, grad_out: list) -> list:
        """Backprop through the network; accumulates gradients in each Linear layer."""
        # Gradient through output activation
        if self.output == "sigmoid":
            p = self._out
            grad = [grad_out[i] * p[i] * (1.0 - p[i]) for i in range(len(p))]
        else:
            grad = list(grad_out)
        # Backprop through output Linear
        grad = self.layers[-1].backward(grad)
        # Backprop through hidden Tanh → Linear pairs (reverse order)
        for i in range(len(self.layers) - 2, -1, -1):
            _, h = self._hiddens[i]
            grad = [grad[j] * (1.0 - h[j] ** 2) for j in range(len(h))]
            grad = self.layers[i].backward(grad)
        return grad

    def step(self, lr: float):
        for layer in self.layers:
            layer.step(lr)

    def zero_grad(self):
        for layer in self.layers:
            layer._zero_grad()


# ---------------------------------------------------------------------------
# Data distribution
# ---------------------------------------------------------------------------

def sample_real(n: int) -> list:
    """
    Target distribution: equal mixture of N(-2, 0.3²) and N(2, 0.3²).
    Tests whether G can discover multi-modality or collapses to the mean (0).
    """
    samples = []
    for _ in range(n):
        if random.random() < 0.5:
            samples.append([random.gauss(-2.0, 0.3)])
        else:
            samples.append([random.gauss(2.0, 0.3)])
    return samples


def sample_noise(n: int, dim: int) -> list:
    """z ~ N(0, I), shape (n, dim)."""
    return [[random.gauss(0, 1) for _ in range(dim)] for _ in range(n)]


# ---------------------------------------------------------------------------
# Loss functions (single-sample, averaged over minibatch in training loop)
# ---------------------------------------------------------------------------

def bce_loss(p: float, y: float) -> float:
    p = clip(p)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def bce_grad_presigmoid(p: float, y: float) -> float:
    """
    dL/dz where L = BCE(sigmoid(z), y).
    Combined sigmoid + BCE gradient = p - y  (clean closed form).
    """
    return p - y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    epochs: int = 3000,
    batch_size: int = 64,
    lr: float = 3e-3,
    latent_dim: int = 4,
    log_every: int = 300,
) -> tuple:
    """
    Alternating gradient descent: 1 D update then 1 G update per step.
    Both G and D are 3-layer MLPs (2 hidden layers of 32 units each).
    """
    random.seed(42)

    G = MLP([latent_dim, 32, 32, 1], output="linear")
    D = MLP([1, 32, 32, 1], output="sigmoid")

    print("Training GAN on Gaussian mixture: N(-2, 0.3) and N(2, 0.3)")
    print(f"{'Epoch':>6}  {'D_loss':>8}  {'G_loss':>8}  {'G_mean':>8}  {'G_std':>8}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):

        # ── Update Discriminator ─────────────────────────────────────────
        reals = sample_real(batch_size)
        zs    = sample_noise(batch_size, latent_dim)

        D.zero_grad()
        d_loss = 0.0
        for x in reals:
            p = D.forward(x)[0]
            d_loss += bce_loss(p, 1.0)
            # dL/dz_out = p - y = p - 1  (real sample, y=1)
            D.backward([bce_grad_presigmoid(p, 1.0) / batch_size])

        for z in zs:
            fake = G.forward(z)          # G output, treated as D input
            p = D.forward(fake)[0]
            d_loss += bce_loss(p, 0.0)
            # dL/dz_out = p - y = p - 0 = p  (fake sample, y=0)
            D.backward([bce_grad_presigmoid(p, 0.0) / batch_size])

        D.step(lr)

        # ── Update Generator ─────────────────────────────────────────────
        # Non-saturating loss: maximize E[log D(G(z))]
        zs = sample_noise(batch_size, latent_dim)

        G.zero_grad()
        D.zero_grad()     # will NOT call D.step() — gradients flow through D to G only
        g_loss = 0.0
        for z in zs:
            fake = G.forward(z)
            p = D.forward(fake)[0]
            g_loss += bce_loss(p, 1.0)   # G wants D to output 1 on its fakes
            # dL/dz_D_out = p - 1; backprop through D gives grad w.r.t. fake
            grad_fake = D.backward([bce_grad_presigmoid(p, 1.0) / batch_size])
            G.backward(grad_fake)         # continue backprop through G

        G.step(lr)
        # (D.zero_grad already called above; gradients discarded, D not stepped)

        if epoch % log_every == 0:
            n_eval = 1000
            gen_samples = [G.forward(z)[0] for z in sample_noise(n_eval, latent_dim)]
            g_mean = sum(gen_samples) / n_eval
            g_var  = sum((s - g_mean) ** 2 for s in gen_samples) / n_eval
            g_std  = math.sqrt(g_var)
            avg_d  = d_loss / (2 * batch_size)
            avg_g  = g_loss / batch_size
            print(f"{epoch:>6}  {avg_d:>8.4f}  {avg_g:>8.4f}  {g_mean:>+8.3f}  {g_std:>8.3f}")

    return G, D


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def statistics(samples: list) -> tuple:
    n = len(samples)
    mean = sum(samples) / n
    std  = math.sqrt(sum((s - mean) ** 2 for s in samples) / n)
    return mean, std


def estimate_kl(p_samples: list, q_samples: list, n_bins: int = 50) -> float:
    """
    Estimate KL(P || Q) via histogram density matching.
    Lower = better; 0 means perfect match.
    """
    lo = min(min(p_samples), min(q_samples)) - 0.5
    hi = max(max(p_samples), max(q_samples)) + 0.5
    width = (hi - lo) / n_bins
    eps = 1e-10

    def to_hist(samples):
        counts = [0] * n_bins
        for s in samples:
            idx = int((s - lo) / width)
            idx = max(0, min(n_bins - 1, idx))
            counts[idx] += 1
        total = sum(counts)
        return [max(c / total, eps) for c in counts]

    p_hist = to_hist(p_samples)
    q_hist = to_hist(q_samples)
    return sum(p_hist[i] * math.log(p_hist[i] / q_hist[i]) for i in range(n_bins))


def ascii_hist(samples: list, label: str, n_bins: int = 20, width: int = 40):
    """Print a sideways ASCII histogram."""
    lo = min(samples) - 0.1
    hi = max(samples) + 0.1
    bw = (hi - lo) / n_bins
    counts = [0] * n_bins
    for s in samples:
        idx = int((s - lo) / bw)
        idx = max(0, min(n_bins - 1, idx))
        counts[idx] += 1
    max_c = max(counts)
    print(f"\n{label}")
    for i, c in enumerate(counts):
        mid = lo + (i + 0.5) * bw
        bar = "█" * int(c / max_c * width)
        print(f"  {mid:+5.2f} | {bar}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    G, D = train(epochs=3000, batch_size=64, lr=3e-3, latent_dim=4)

    print("\n" + "=" * 55)
    print("=== Final Evaluation ===")
    print("=" * 55)

    n_eval = 4000
    real_samples = [x[0] for x in sample_real(n_eval)]
    gen_samples  = [G.forward(z)[0] for z in sample_noise(n_eval, 4)]

    r_mean, r_std = statistics(real_samples)
    g_mean, g_std = statistics(gen_samples)
    kl = estimate_kl(real_samples, gen_samples)

    print(f"\nReal data   | mean: {r_mean:+7.3f} | std: {r_std:.3f}")
    print(f"Generated   | mean: {g_mean:+7.3f} | std: {g_std:.3f}")
    print(f"KL estimate (real || gen): {kl:.4f}  (lower is better)")

    ascii_hist(real_samples, "Real data distribution:")
    ascii_hist(gen_samples,  "Generated distribution:")

    left_frac  = sum(1 for s in gen_samples if s < 0) / n_eval
    right_frac = 1.0 - left_frac
    print(f"\nMode coverage | left (x<0): {left_frac:.1%}  right (x≥0): {right_frac:.1%}")

    if abs(left_frac - 0.5) < 0.15:
        print("✓ Generator learned both modes — no mode collapse.")
    else:
        print("✗ Mode collapse: generator skewed toward one mode.")

    print("\nTraining complete.")


if __name__ == "__main__":
    demo()
