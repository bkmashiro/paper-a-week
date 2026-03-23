"""
Demo: Dropout vs. No-Dropout on a toy classification task.

Shows the classic dropout benefit: the model without dropout memorizes
the small training set (low training loss, high validation loss), while
the model with dropout generalizes much better.

Run:
    python demo.py
"""

import numpy as np
from implementation import DropoutMLP, train

# ---------------------------------------------------------------------------
# Toy dataset: 2-class Gaussian blobs, heavily downsampled to provoke overfit
# ---------------------------------------------------------------------------

def make_dataset(n_train: int = 200, n_val: int = 500, n_features: int = 50, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_classes = 4

    # Mean vectors per class in a 50-d space
    means = rng.normal(0.0, 3.0, size=(n_classes, n_features))

    def sample(n):
        labels = rng.integers(0, n_classes, size=n)
        x = means[labels] + rng.normal(0.0, 1.5, size=(n, n_features))
        return x.astype(np.float64), labels

    x_train, y_train = sample(n_train)
    x_val, y_val = sample(n_val)
    return x_train, y_train, x_val, y_val


def run_experiment(drop_p: float, label: str, x_tr, y_tr, x_val, y_val):
    print(f"\n{'='*50}")
    print(f"  {label}  (drop_p={drop_p})")
    print(f"{'='*50}")
    model = DropoutMLP(
        in_dim=x_tr.shape[1],
        hidden=256,
        out_dim=4,
        drop_p=drop_p,
        seed=7,
    )
    history = train(model, x_tr, y_tr, x_val, y_val, epochs=40, lr=0.05, report_every=10)
    final_val = model.accuracy(x_val, y_val)
    train_acc = model.accuracy(x_tr, y_tr)   # evaluated with drop_p=0 (inference mode)
    print(f"\n  Final  train_acc={train_acc:.3f}  val_acc={final_val:.3f}")
    return history, final_val


def main():
    print("Paper 005: Dropout — Comparison Demo\n")
    print("Dataset: 200 training samples, 500 validation, 50 features, 4 classes")
    print("Architecture: 50 → 256 → 256 → 4  (ReLU + optional Dropout)\n")

    x_tr, y_tr, x_val, y_val = make_dataset()

    h_nodrop, acc_nodrop = run_experiment(0.0, "No Dropout (baseline)", x_tr, y_tr, x_val, y_val)
    h_drop,   acc_drop   = run_experiment(0.5, "With Dropout (p=0.5)",  x_tr, y_tr, x_val, y_val)

    print("\n\n" + "="*50)
    print("  Summary")
    print("="*50)
    print(f"  No Dropout  val_acc = {acc_nodrop:.3f}")
    print(f"  Dropout 0.5 val_acc = {acc_drop:.3f}")
    delta = acc_drop - acc_nodrop
    sign = "+" if delta >= 0 else ""
    print(f"  Delta       = {sign}{delta:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = range(1, len(h_nodrop["val_acc"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(epochs, h_nodrop["train_loss"], label="No Dropout", color="tomato")
        ax1.plot(epochs, h_drop["train_loss"],   label="Dropout 0.5", color="steelblue")
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross-Entropy")
        ax1.legend()

        ax2.plot(epochs, h_nodrop["val_acc"], label="No Dropout", color="tomato")
        ax2.plot(epochs, h_drop["val_acc"],   label="Dropout 0.5", color="steelblue")
        ax2.set_title("Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()

        plt.tight_layout()
        plt.savefig("dropout_comparison.png", dpi=120)
        print("\n  Plot saved to dropout_comparison.png")
    except ImportError:
        print("\n  (matplotlib not installed — skipping plot)")


if __name__ == "__main__":
    main()
