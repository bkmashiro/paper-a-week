# 013 — EfficientNet: Rethinking Model Scaling for CNNs

**Authors:** Mingxing Tan, Quoc V. Le  
**Published:** 2019 (arXiv: 1905.11946)  
**Venue:** ICML 2019  
**Category:** Deep Learning / Architecture Design  
**Link:** https://arxiv.org/abs/1905.11946

---

## Background

Practitioners have long wanted to scale up CNNs to get better accuracy. The question is *how*. The three obvious dimensions are:

- **Depth** — more layers (ResNet-50 → ResNet-152)
- **Width** — more channels per layer (WideResNet)
- **Resolution** — higher input image size (299×299 in Inception vs 224×224 in ResNet)

The standard practice was to scale one dimension at a time. Add more layers until accuracy plateaus, or throw GPU memory at higher resolution. Tan & Le asked the deeper question: *do these three dimensions interact, and is there a principled way to scale all three together?*

Their answer is **compound scaling**: a simple rule that ties depth, width, and resolution together via a single scalar coefficient φ (phi). This produces the **EfficientNet** family — models that achieve better accuracy per FLOP than any previous architecture at the time of publication.

---

## Core Ideas

### 1. Why Scaling Dimensions Interact

Scaling depth alone hits diminishing returns because deeper networks are harder to optimize (vanishing gradients). Scaling width alone gives more channels but shallow features. Resolution scaling only helps if the network has enough capacity (depth + width) to exploit fine-grained detail.

The insight: **the three dimensions are not independent**. Higher-resolution images benefit from deeper networks (more receptive field) and wider networks (more capacity to detect fine features). They must grow together.

### 2. Compound Scaling Rule

Fix a user-defined compound coefficient **φ ≥ 0** that controls how many more resources you want to use. Then scale:

```
depth:      d = α^φ
width:      w = β^φ
resolution: r = γ^φ
```

Subject to the constraint:

```
α · β² · γ² ≈ 2
α ≥ 1,  β ≥ 1,  γ ≥ 1
```

The constraint ensures that total FLOPs roughly **double for each unit increase in φ** (FLOPs scale as d · w² · r² ∝ (αβ²γ²)^φ ≈ 2^φ).

The optimal values α, β, γ are found once via a small NAS-style grid search on the baseline model (EfficientNet-B0). The paper finds **α=1.2, β=1.1, γ=1.15** (satisfying 1.2 × 1.1² × 1.15² ≈ 2.0).

### 3. EfficientNet-B0: The Baseline Architecture

Rather than apply compound scaling to an existing architecture, the authors first design an efficient baseline using Neural Architecture Search (NAS) optimizing for accuracy per FLOP. The result — **EfficientNet-B0** — uses:

- **MBConv blocks** (mobile inverted bottleneck convolutions, from MobileNetV2)
- **Squeeze-and-Excitation (SE)** modules for channel attention
- **Swish activation** (`x · σ(x)`) instead of ReLU
- Relatively small: 5.3M parameters, ~0.39B FLOPs on 224×224 inputs

### 4. EfficientNet-B1 through B7

Scaling B0 with increasing φ produces the B1–B7 family:

| Model | φ | Resolution | Params | Top-1 Acc |
|-------|---|------------|--------|-----------|
| B0 | 0 | 224 | 5.3M | 77.1% |
| B1 | 1 | 240 | 7.8M | 79.1% |
| B3 | 3 | 300 | 12M  | 81.6% |
| B5 | 5 | 456 | 30M  | 83.6% |
| B7 | 7 | 600 | 66M  | 84.3% |

EfficientNet-B7 achieved **84.3% top-1 accuracy on ImageNet** — the state of the art in 2019 — while being **8.4× smaller and 6.1× faster** than GPT-4V-era contemporaries like AmoebaNet-C.

### 5. MBConv: The Building Block

The core operation is the **Mobile Inverted Bottleneck Convolution**:

```
Input (C_in channels)
  → 1×1 Conv (expand to C_in × expansion_factor, e.g. ×6)
  → Depthwise Conv k×k (spatial mixing, one filter per channel)
  → Squeeze-and-Excitation (global avg pool → FC → FC → scale)
  → 1×1 Conv (project back to C_out channels)
  → Dropout (stochastic depth)
  → Add residual (if C_in == C_out and stride == 1)
```

Depthwise separable convolution is the key efficiency trick: instead of C_out filters of shape (k, k, C_in), you use C_in filters of shape (k, k, 1) + a 1×1 pointwise conv. FLOPs drop by ~k²-fold.

### 6. Squeeze-and-Excitation

SE adds channel attention with almost no parameters:

```
z = GlobalAvgPool(x)              # (C,)
s = σ(W₂ · ReLU(W₁ · z))        # (C,) squeeze ratio r=4
y = s · x                        # channel-wise scaling
```

The SE module lets the network dynamically emphasize important feature channels and suppress irrelevant ones.

---

## Implementation Details

### MBConv Forward Pass (Pseudocode)

```python
def mbconv(x, expand_ratio, k, out_channels, se_ratio=0.25):
    C = x.shape[1]
    hidden = C * expand_ratio

    # Expansion
    if expand_ratio != 1:
        x = conv2d(x, 1, hidden) + bn + swish

    # Depthwise conv
    x = depthwise_conv2d(x, k, hidden) + bn + swish

    # Squeeze-and-Excitation
    se_dim = max(1, int(C * se_ratio))
    s = adaptive_avg_pool(x)            # (B, hidden, 1, 1)
    s = swish(linear(s, se_dim))
    s = sigmoid(linear(s, hidden))
    x = x * s

    # Projection
    x = conv2d(x, 1, out_channels) + bn   # no activation

    return x + residual  # if shapes match
```

### Swish Activation

```python
def swish(x):
    return x * sigmoid(x)
```

Swish is smooth, non-monotonic, and outperforms ReLU on deeper networks. It is now also known as **SiLU** (Sigmoid Linear Unit) and widely used in modern architectures.

### Stochastic Depth

EfficientNet uses stochastic depth (drop path): randomly skip entire residual blocks during training with probability `p_drop`. This acts as a regularizer and allows training very deep networks. At inference all blocks are active (outputs scaled by `1 - p_drop`).

```python
def stochastic_depth(x, residual, drop_prob, training):
    if not training or drop_prob == 0.0:
        return x + residual
    keep_prob = 1.0 - drop_prob
    mask = (torch.rand(x.shape[0], 1, 1, 1) < keep_prob).float()
    return x + residual * mask / keep_prob
```

---

## Results and Impact

### ImageNet Performance (2019)

EfficientNets dominated the accuracy-efficiency frontier at publication:

- **EfficientNet-B0** matched MobileNetV2 accuracy with 5× fewer parameters
- **EfficientNet-B3** matched ResNet-50 accuracy with 6× fewer FLOPs
- **EfficientNet-B7** set a new SOTA at 84.3% top-1, beating the previous record (AmoebaNet + NAS at 83.9%) with 8× fewer parameters

### Transfer Learning

EfficientNets transferred exceptionally well to other datasets:
- **CIFAR-100**: 91.7% (B0), 93.6% (B7)
- **Flowers102**: 98.8% (B7)
- **Stanford Cars**: 94.7% (B7)

The efficient architecture forces the model to learn generalizable features rather than memorizing dataset-specific patterns.

### Legacy

The compound scaling principle became immediately influential:

- **EfficientNetV2** (2021): improved MBConv with Fused-MBConv and progressive training
- **CoAtNet** (2021): combined EfficientNet scaling with Transformers
- The NAS + compound scaling methodology influenced the design of all subsequent efficient vision models
- Used as a backbone in object detection (EfficientDet), segmentation, and medical imaging

---

## Key Takeaways

1. **Model scaling has three dimensions — and they interact**: naive single-dimension scaling is suboptimal; compound scaling jointly increases depth, width, and resolution for better accuracy per FLOP.

2. **The compound coefficient φ is remarkably practical**: a single hyperparameter trades off compute vs. accuracy across an entire model family, with no per-model architecture search.

3. **Architecture quality matters more than scale**: EfficientNet-B0 (5.3M params) is more carefully designed than ResNet-50 (25M params). Scale amplifies architectural quality — start with a good base.

4. **MBConv + depthwise separable conv is a powerful primitive**: the efficiency of depthwise conv combined with the expressiveness of channel attention (SE) forms the basis of essentially all efficient vision architectures since 2019.

5. **Swish / SiLU beats ReLU at depth**: a small activation change with meaningful accuracy gains on deep networks — it propagates non-zero gradients for negative inputs, unlike ReLU.

---

## References

- Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018
- Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
- Tan & Le, "EfficientNetV2: Smaller Models and Faster Training", ICML 2021
- Ramachandran et al., "Searching for Activation Functions" (Swish), arXiv 2017
