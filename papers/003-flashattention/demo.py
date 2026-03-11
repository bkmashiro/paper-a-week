"""
Demo for paper 003: FlashAttention

Verifies the tiled forward pass matches standard attention, then
compares asymptotic memory and rough CPU runtime.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from flash_attention import flash_attention_forward, standard_attention


def print_separator(title=""):
    width = 62
    if title:
        pad = (width - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * (width - pad - len(title) - 2))
    else:
        print("─" * width)


def benchmark(fn, *args, repeat=3, **kwargs):
    times = []
    out = None
    for _ in range(repeat):
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        times.append(time.perf_counter() - start)
    return out, min(times)


def memory_report(seq_len: int, d_model: int, block_size: int, dtype=np.float64):
    itemsize = np.dtype(dtype).itemsize
    standard_scores = seq_len * seq_len * itemsize
    standard_probs = seq_len * seq_len * itemsize
    flash_tiles = 2 * block_size * d_model * itemsize
    flash_running = (2 * seq_len + seq_len * d_model) * itemsize
    return {
        "standard_bytes": standard_scores + standard_probs,
        "flash_bytes": flash_tiles + flash_running,
    }


def format_bytes(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def demo_correctness_and_timing():
    print_separator("Correctness vs Standard Attention")
    rng = np.random.default_rng(42)
    d_model = 64
    block_size = 64

    print(f"  d_model={d_model}, block_size={block_size}, dtype=float64\n")
    print(f"  {'seq':>6}  {'max error':>12}  {'standard':>12}  {'flash':>12}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*12}")

    for seq_len in [64, 256, 1024]:
        Q = rng.normal(size=(seq_len, d_model))
        K = rng.normal(size=(seq_len, d_model))
        V = rng.normal(size=(seq_len, d_model))

        out_std, t_std = benchmark(standard_attention, Q, K, V)
        out_flash, t_flash = benchmark(
            flash_attention_forward,
            Q,
            K,
            V,
            block_size=block_size,
        )

        max_err = np.max(np.abs(out_std - out_flash))
        print(f"  {seq_len:>6}  {max_err:>12.2e}  {t_std*1000:>9.2f} ms  {t_flash*1000:>9.2f} ms")
        assert np.allclose(out_std, out_flash, atol=1e-9, rtol=1e-9)

    print("\n  ✓ FlashAttention forward matches standard attention")
    print("  Note: on CPU/NumPy, tiled Python loops are usually slower than dense matmul.")
    print("  The paper's 2-4x speedup comes from the fused CUDA kernel on A100 GPUs.\n")


def demo_memory_complexity():
    print_separator("Memory Comparison")
    d_model = 64
    block_size = 64

    print(f"  {'seq':>6}  {'standard O(N^2)':>18}  {'flash O(N)':>15}  {'ratio':>10}")
    print(f"  {'─'*6}  {'─'*18}  {'─'*15}  {'─'*10}")

    for seq_len in [64, 256, 1024, 4096]:
        report = memory_report(seq_len, d_model, block_size)
        ratio = report["standard_bytes"] / report["flash_bytes"]
        print(
            f"  {seq_len:>6}  "
            f"{format_bytes(report['standard_bytes']):>18}  "
            f"{format_bytes(report['flash_bytes']):>15}  "
            f"{ratio:>9.1f}x"
        )

    print("\n  Standard attention stores full score/probability matrices in memory.")
    print("  FlashAttention keeps only tiles plus running softmax statistics.\n")


def main():
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Paper 003: FlashAttention — tiled forward pass demo      ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()

    demo_correctness_and_timing()
    demo_memory_complexity()

    print_separator("Summary")
    print("  ✓ Exact output match with standard softmax attention")
    print("  ✓ Extra memory drops from O(N^2) to O(N)")
    print("  ✓ Tiling + online softmax are the core ideas")
    print("  ✓ GPU speedups in the paper come from IO-aware CUDA kernels")
    print()
    print("  Implementation: papers/003-flashattention/flash_attention.py")
    print("  Run: python demo.py")
    print()


if __name__ == "__main__":
    main()
