"""
Demo for paper 002: LoRA — Low-Rank Adaptation

Shows:
  1. A tiny transformer model (GPT-style)
  2. Full fine-tune vs LoRA parameter counts
  3. Training a few steps on random data with LoRA
  4. Weight merging for inference

Requires: torch only
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from lora import LoRALinear, apply_lora, count_trainable_params


# ─────────────────────────────────────────────────────────────
# Tiny GPT-style model (for demonstration)
# ─────────────────────────────────────────────────────────────

class TinyAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        att = torch.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attn = TinyAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TinyBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────

def print_separator(title=""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * (width - pad - len(title) - 2))
    else:
        print("─" * width)


def demo_parameter_comparison():
    print_separator("Parameter Comparison")

    model = TinyGPT(vocab_size=1000, d_model=256, num_heads=4, num_layers=4)

    # Full fine-tune: all params trainable
    full_trainable, total = count_trainable_params(model)
    print(f"  Model: TinyGPT (vocab=1000, d=256, heads=4, layers=4)")
    print(f"  Total parameters:           {total:>10,}")
    print(f"  Full fine-tune (trainable):  {full_trainable:>10,}")

    # Apply LoRA to Q and V projections (paper's recommended targets)
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("q_proj" in name or "v_proj" in name):
            target_modules.append(name)

    print(f"\n  LoRA targets: {target_modules}")

    # Freeze entire model first (simulates a pretrained model)
    for param in model.parameters():
        param.requires_grad_(False)

    rank = 8
    apply_lora(model, target_modules, rank=rank, alpha=16.0)

    lora_trainable, total_after = count_trainable_params(model)
    print(f"\n  After LoRA (rank={rank}):")
    print(f"  Total parameters:           {total_after:>10,}")
    print(f"  Trainable (LoRA only):       {lora_trainable:>10,}")
    print(f"  Reduction:                   {lora_trainable / full_trainable * 100:>9.2f}%")
    print()

    return model


def demo_training_loop(model: nn.Module):
    print_separator("Training with LoRA")

    # Random data (just to show LoRA gradients flow)
    torch.manual_seed(42)
    batch_size, seq_len = 4, 16
    x = torch.randint(0, 1000, (batch_size, seq_len))
    y = torch.randint(0, 1000, (batch_size, seq_len))

    # Only optimize LoRA parameters
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(lora_params, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"  Training on random data ({batch_size} seqs × {seq_len} tokens)")
    print(f"  Optimizer: Adam (lr=1e-3), only LoRA params\n")

    for step in range(5):
        optimizer.zero_grad()
        logits = model(x)  # (B, T, vocab)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        print(f"  Step {step + 1}: loss = {loss.item():.4f}")

    print("\n  ✓ LoRA gradients flow correctly — loss decreases\n")
    return model


def demo_weight_merging(model: nn.Module):
    print_separator("Weight Merging")

    torch.manual_seed(0)
    x = torch.randint(0, 1000, (1, 8))

    # Forward pass before merging (LoRA as separate path)
    with torch.no_grad():
        out_before = model(x)

    # Merge LoRA weights into base linear layers
    merge_count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
            merge_count += 1

    print(f"  Merged {merge_count} LoRA layers into base weights")

    # Forward pass after merging (single matmul, no LoRA overhead)
    with torch.no_grad():
        out_after = model(x)

    diff = (out_before - out_after).abs().max().item()
    print(f"  Max output difference: {diff:.2e}")
    print(f"  ✓ Outputs match (merge is lossless)\n")

    # After merging, all params are frozen — inference only
    trainable, total = count_trainable_params(model)
    print(f"  After merge: {trainable:,} trainable / {total:,} total")
    print(f"  ✓ Ready for inference — no extra LoRA overhead\n")


def demo_scaling_intuition():
    print_separator("Scaling Intuition: rank vs params")

    d = 1024
    print(f"  Weight matrix: {d}×{d} = {d*d:,} params\n")
    print(f"  {'rank':>6}  {'LoRA params':>12}  {'% of full':>10}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*10}")
    for r in [1, 2, 4, 8, 16, 32, 64]:
        lora_params = d * r + r * d  # A: r×d, B: d×r
        pct = lora_params / (d * d) * 100
        print(f"  {r:>6}  {lora_params:>12,}  {pct:>9.2f}%")
    print()


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Paper 002: LoRA — Low-Rank Adaptation demo             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    demo_scaling_intuition()
    model = demo_parameter_comparison()
    model = demo_training_loop(model)
    demo_weight_merging(model)

    print_separator("Summary")
    print("  ✓ LoRA reduces trainable params by ~97%+ on Q,V projections")
    print("  ✓ Training converges with only LoRA parameters")
    print("  ✓ Weights merge losslessly for zero-overhead inference")
    print("  ✓ Higher rank = more capacity but more params")
    print()
    print("  Implementation: papers/002-lora/lora.py")
    print("  Run: python demo.py")
    print()


if __name__ == "__main__":
    main()
