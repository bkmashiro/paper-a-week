"""
LoRA: Low-Rank Adaptation of Large Language Models
Hu et al., 2021 | https://arxiv.org/abs/2106.09685

Key idea: Instead of fine-tuning W (d×d), train two small matrices
A (d×r) and B (r×d) where r << d.
The adapted weight is W + BA × (alpha/r).

Implements:
  - LoRALinear     — wraps nn.Linear with low-rank adaptation
  - apply_lora     — inject LoRA into target layers of any model
  - count_trainable_params — compare full vs LoRA parameter counts

Requires: torch
Total: ~100 lines (including docstrings)
"""

import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────────────────────
# LoRA Linear Layer
# ─────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with LoRA adaptation.

    Forward:  y = Wx + (x @ A^T @ B^T) * (alpha / r)
              = base_output + lora_output

    A is initialized with Kaiming uniform, B with zeros,
    so ΔW = BA = 0 at start (preserves pretrained behavior).
    """

    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha

        d_in = linear.in_features
        d_out = linear.out_features

        # LoRA matrices: A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze original weights
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path: frozen pretrained weights
        base = self.linear(x)
        # LoRA path: low-rank adaptation
        lora = (x @ self.lora_A.T @ self.lora_B.T) * (self.alpha / self.rank)
        return base + lora

    def merge_weights(self):
        """
        Merge LoRA weights into base linear layer (for inference speed).
        After merging, the forward pass is just a single matmul — no extra latency.
        """
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * (self.alpha / self.rank)
            self.linear.weight.add_(delta)
            # Zero out LoRA matrices so forward() returns just the base path
            self.lora_A.zero_()
            self.lora_B.zero_()
        self.lora_A.requires_grad_(False)
        self.lora_B.requires_grad_(False)


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def apply_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 4,
    alpha: float = 1.0,
) -> nn.Module:
    """
    Replace target Linear layers in model with LoRA-wrapped versions.

    Args:
        model: any nn.Module
        target_modules: list of dotted module names (e.g. ["layer.0.attn.q_proj"])
        rank: LoRA rank r
        alpha: scaling factor α

    Returns:
        model with LoRA layers injected (in-place)
    """
    for name, module in model.named_modules():
        if name in target_modules and isinstance(module, nn.Linear):
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)
    return model


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Returns (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
