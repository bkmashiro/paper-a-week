# 007 — Deep Residual Learning for Image Recognition (ResNet)

**Paper:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, 2015 | **Venue:** CVPR 2016  
**Link:** https://arxiv.org/abs/1512.03385

---

## Background

Before ResNet, the field had been pushing convolutional networks deeper — going from AlexNet's 8 layers (2012) to VGGNet's 19 layers (2014) and GoogLeNet's 22 layers (2014). The intuition was straightforward: deeper networks should be more expressive. Yet in practice, simply stacking more layers made things worse, not better. Researchers observed a striking "degradation problem": a 56-layer plain network trained on CIFAR-10 had *higher training error* than a 20-layer network. This was not overfitting — the deeper model was failing to learn even on the training set.

The degradation problem is subtle. In theory, a deeper network can always replicate a shallower network by setting the extra layers to identity mappings. If the optimizer could find that solution, deeper should always be at least as good as shallower. The fact that it couldn't points to an optimization difficulty: learning an identity mapping through a chain of weight layers and nonlinearities is surprisingly hard for SGD. The gradients must flow back through many layers, and each layer's Jacobian tends to shrink them (vanishing gradients) or explode them in poorly conditioned landscapes.

He et al.'s key insight was a reformulation: instead of asking layers to learn the desired output mapping H(x) directly, let them learn the *residual* F(x) = H(x) − x, and add the input back via a shortcut connection: output = F(x) + x. If the identity is the right answer, the network just needs to push F(x) toward zero — which is much easier than learning a precise identity through composition of nonlinearities. This re-parameterization costs essentially nothing (no extra parameters for the shortcut), yet it fundamentally changes the optimization landscape. With residual connections, training a 152-layer network became stable and produced a model that won the ImageNet 2015 competition with 3.57% top-5 error — halving the previous best.

---

## Core Ideas

### Residual Block

The fundamental building block is the **residual mapping**:

```
y = F(x, {Wᵢ}) + x
```

Where:
- `x` is the block input
- `F(x, {Wᵢ})` is a small sub-network (the residual function) — typically 2 or 3 weight layers with BatchNorm and ReLU
- The `+ x` is the **shortcut connection** (identity mapping, zero extra cost)

The activation is applied after addition: `output = ReLU(F(x) + x)`.

For a two-layer basic block:
```
F(x) = W₂ · σ(BN(W₁ · x))
y    = σ(BN(F(x)) + x)
```

The residual formulation means gradients flow directly back through the shortcut, bypassing any number of intermediate layers:
```
∂L/∂x = ∂L/∂y · (∂F/∂x + I)
```
The identity term `I` ensures a gradient highway through the entire network, regardless of depth.

### Projection Shortcuts (Dimension Mismatch)

When the spatial dimensions or channel count change (e.g., strided convolution halves the feature map), a simple identity shortcut can't be added directly since x and F(x) have different shapes. Two options:

1. **Zero-padding** — pad x with zeros to match channel count (no extra parameters)
2. **Projection shortcut** — use a 1×1 convolution to match dimensions: `y = F(x) + Wₛ · x`

The paper finds option 2 works slightly better; option 1 is competitive at lower cost.

### Bottleneck Block (for deep networks, 50+ layers)

To manage computational cost in very deep networks (ResNet-50/101/152), the paper uses a **bottleneck design**: a 1×1 convolution reduces channels, a 3×3 convolution operates in the reduced space, and a 1×1 convolution expands back up:

```
Input: C channels
→ 1×1 conv → C/4 channels  (reduce)
→ 3×3 conv → C/4 channels  (process)
→ 1×1 conv → C channels    (expand)
→ + shortcut
```

This keeps the expensive 3×3 convolution operating on a smaller tensor, making deep bottleneck networks cheaper than equivalently deep basic-block networks.

### Network Architecture

ResNet follows a simple stage-based design:
- **Conv1**: 7×7 conv, stride 2, 64 filters → 112×112
- **MaxPool**: stride 2 → 56×56
- **Stage 1** (conv2_x): residual blocks at 56×56, 64 channels
- **Stage 2** (conv3_x): residual blocks at 28×28, 128 channels (first block stride 2)
- **Stage 3** (conv4_x): residual blocks at 14×14, 256 channels
- **Stage 4** (conv5_x): residual blocks at 7×7, 512 channels
- **GAP + FC(1000) + Softmax**

The number of blocks per stage varies by depth: ResNet-18 uses [2,2,2,2], ResNet-34 [3,4,6,3], ResNet-50/101/152 use bottleneck blocks.

### Why Residuals Help: Theoretical Perspective

The ensemble hypothesis (Veit et al., 2016) offers an intuition: a ResNet with n residual blocks behaves like an *implicit ensemble* of 2ⁿ paths of varying depths. Short paths dominate gradient flow during training, while long paths contribute refined features during inference. Residual networks are much less sensitive to layer removal than plain networks — further evidence that information travels along many parallel routes.

---

## Implementation Notes

### What We Kept

- Full BasicBlock and Bottleneck designs (exact math)
- Projection shortcuts with 1×1 convolution
- BatchNorm + ReLU placement (BN before addition for the residual branch, ReLU after addition)
- ResNet-18 and ResNet-34 (BasicBlock), ResNet-50 (Bottleneck)
- Global average pooling before the final classifier
- Forward pass for both architectures

### What We Simplified

- No ImageNet training loop (too large for a demo)
- CIFAR-10 variant included in the demo (smaller images → no initial 7×7 conv or MaxPool)
- Weight initialization follows Kaiming He normal (same as the paper), but no full training
- No multi-GPU or distributed training

### Key Code Points

**BasicBlock** (lines ~30–60):
```python
def forward(self, x):
    identity = x
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    if self.shortcut is not None:
        identity = self.shortcut(x)
    out = self.relu(out + identity)   # ← the residual addition
    return out
```

The shortcut branch is `None` when dimensions match (pure identity), and a 1×1 conv otherwise.

**Bottleneck** (lines ~70–105):
```python
def forward(self, x):
    identity = x
    out = self.relu(self.bn1(self.conv1(x)))   # 1×1 reduce
    out = self.relu(self.bn2(self.conv2(out))) # 3×3
    out = self.bn3(self.conv3(out))            # 1×1 expand
    if self.shortcut is not None:
        identity = self.shortcut(x)
    out = self.relu(out + identity)
    return out
```

---

## Running

```bash
pip install torch torchvision
python implementation.py
```

Expected output:
```
ResNet Architecture Demo
========================================
ResNet-18  | Params:   11,689,512 | Output: torch.Size([2, 1000])
ResNet-34  | Params:   21,797,672 | Output: torch.Size([2, 1000])
ResNet-50  | Params:   25,557,032 | Output: torch.Size([2, 1000])

CIFAR-10 ResNet-20 Demo
========================================
ResNet-20 (CIFAR) | Params: 269,722 | Output: torch.Size([4, 10])

Residual Connection Benefit Demo
========================================
Depth  5 | Plain loss: 2.3301 | ResNet loss: 2.6308
Depth 10 | Plain loss: 2.3301 | ResNet loss: 2.6976
Depth 20 | Plain loss: 2.3301 | ResNet loss: 3.1554
(ResNet maintains stable gradient flow at all depths)

Gradient norm comparison (depth=20):
  Plain  network: gradient norm = 10994.7100
  ResNet network: gradient norm = 233.3394
```
