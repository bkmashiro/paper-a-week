"""
Demo for paper 004: Adam optimizer.

Compares Adam vs SGD on the same ill-conditioned toy regression task.
Saves a loss curve if matplotlib is available; otherwise prints metrics only.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from adam import make_toy_regression, train


def print_separator(title=""):
    width = 62
    if title:
        pad = (width - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * (width - pad - len(title) - 2))
    else:
        print("─" * width)


def maybe_save_plot(adam_losses: np.ndarray, sgd_losses: np.ndarray) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    out_path = os.path.join(os.path.dirname(__file__), "loss_comparison.png")
    plt.figure(figsize=(7, 4))
    plt.plot(adam_losses, label="Adam", linewidth=2.2)
    plt.plot(sgd_losses, label="SGD", linewidth=2.2)
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("MSE loss (log scale)")
    plt.title("Adam vs SGD on ill-conditioned linear regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def main():
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Paper 004: Adam optimizer — NumPy training demo          ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()

    x, _, true_w = make_toy_regression()
    print_separator("Toy Task")
    print("  Model: y = XW + b")
    print(f"  Samples: {x.shape[0]}, features: {x.shape[1]}")
    print("  Feature scales: x[:,0] ~ 1e-2, x[:,1] ~ 1e1")
    print(f"  Ground-truth W: {true_w.ravel()}")
    print()

    adam = train("adam", steps=300, seed=0)
    sgd = train("sgd", steps=300, seed=0)

    checkpoints = [1, 5, 20, 100, 300]
    print_separator("Loss Comparison")
    print(f"  {'step':>6}  {'Adam':>14}  {'SGD':>14}")
    print(f"  {'─'*6}  {'─'*14}  {'─'*14}")
    for step in checkpoints:
        print(
            f"  {step:>6}  "
            f"{adam['losses'][step-1]:>14.6f}  "
            f"{sgd['losses'][step-1]:>14.6f}"
        )

    print()
    print_separator("Final Metrics")
    print(f"  Adam final loss:      {adam['final_loss']:.6f}")
    print(f"  SGD final loss:       {sgd['final_loss']:.6f}")
    print(f"  Adam weight error:    {adam['param_error']:.6f}")
    print(f"  SGD weight error:     {sgd['param_error']:.6f}")
    print(f"  Adam learned W:       {adam['W'].ravel().round(4)}")
    print(f"  SGD learned W:        {sgd['W'].ravel().round(4)}")
    print()

    plot_path = maybe_save_plot(adam["losses"], sgd["losses"])
    print_separator("Summary")
    print("  ✓ Adam keeps per-parameter first and second moment estimates")
    print("  ✓ Bias correction matters in the first few steps")
    print("  ✓ On this badly scaled toy problem, Adam converges much faster than SGD")
    if plot_path:
        print(f"  ✓ Saved loss curve: {os.path.basename(plot_path)}")
    else:
        print("  • matplotlib not installed, skipped plot export")
    print()


if __name__ == "__main__":
    main()
