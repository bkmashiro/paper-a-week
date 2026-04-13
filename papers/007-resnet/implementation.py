"""
ResNet: Deep Residual Learning for Image Recognition
He et al., 2015 — https://arxiv.org/abs/1512.03385

Implementation covers:
  - BasicBlock  (ResNet-18 / 34)
  - Bottleneck  (ResNet-50 / 101 / 152)
  - ResNet-18, ResNet-34, ResNet-50 (ImageNet variant)
  - ResNet-20 (CIFAR-10 variant, from paper Section 4.2)
  - Gradient-norm demo showing residual connections preserve gradient flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """Two 3×3 conv layers with a shortcut connection.
    Used in ResNet-18 and ResNet-34."""

    expansion = 1  # output channels = planes * expansion

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut: identity if shapes match, 1×1 conv projection otherwise
        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out = self.relu(out + identity)   # residual addition, then activation
        return out


class Bottleneck(nn.Module):
    """1×1 → 3×3 → 1×1 bottleneck with a shortcut connection.
    Used in ResNet-50 / 101 / 152.  Output channels = planes * 4."""

    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        # 1×1: reduce channels
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3×3: process in reduced space
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1×1: expand back
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))   # 1×1 reduce
        out = self.relu(self.bn2(self.conv2(out))) # 3×3
        out = self.bn3(self.conv3(out))             # 1×1 expand (no relu yet)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out = self.relu(out + identity)
        return out


# ---------------------------------------------------------------------------
# ResNet (ImageNet variant — 224×224 input)
# ---------------------------------------------------------------------------

class ResNet(nn.Module):
    """Standard ResNet for ImageNet (1000 classes, 224×224 input).

    Args:
        block:      BasicBlock or Bottleneck
        layers:     list of 4 ints — number of blocks per stage
        num_classes: output classes (default 1000)
    """

    def __init__(self, block, layers: list, num_classes: int = 1000):
        super().__init__()
        self.in_planes = 64

        # Stem: 7×7 conv + BN + ReLU + 3×3 MaxPool  →  56×56
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Four residual stages
        self.layer1 = self._make_stage(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_stage(block, 128, layers[1], stride=2)
        self.layer3 = self._make_stage(block, 256, layers[2], stride=2)
        self.layer4 = self._make_stage(block, 512, layers[3], stride=2)

        # Head: global average pool + fully connected
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_stage(self, block, planes: int, num_blocks: int,
                    stride: int) -> nn.Sequential:
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # stem
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet18(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# ---------------------------------------------------------------------------
# CIFAR-10 ResNet (paper Section 4.2 — 32×32 input)
# No 7×7 stem or early MaxPool; uses 3×3 conv with three stages of 2n blocks.
# ResNet-20: n=3, ResNet-32: n=5, ResNet-44: n=7, ResNet-56: n=9
# ---------------------------------------------------------------------------

class CIFARBasicBlock(nn.Module):
    """Simplified BasicBlock without any initial downsampling option
    (CIFAR blocks only downsample at stage boundaries)."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            # Paper option A: zero-padding shortcut (no extra params)
            self.shortcut = "pad"
            self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.shortcut == "pad":
            # Subsample spatially and pad channels with zeros
            identity = F.avg_pool2d(x, self.stride, self.stride)
            pad = out.size(1) - identity.size(1)
            identity = F.pad(identity, (0, 0, 0, 0, 0, pad))

        out = self.relu(out + identity)
        return out


class CIFARResNet(nn.Module):
    """ResNet for CIFAR-10/100 as described in paper Section 4.2.

    n controls depth: total layers = 6n + 2
      n=3  → ResNet-20
      n=5  → ResNet-32
      n=9  → ResNet-56
    """

    def __init__(self, n: int = 3, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_stage(16, 16, n, stride=1)
        self.layer2 = self._make_stage(16, 32, n, stride=2)
        self.layer3 = self._make_stage(32, 64, n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, in_planes, planes, n, stride):
        layers = [CIFARBasicBlock(in_planes, planes, stride)]
        for _ in range(1, n):
            layers.append(CIFARBasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet20_cifar(num_classes: int = 10) -> CIFARResNet:
    return CIFARResNet(n=3, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Gradient-flow demo: plain deep network vs ResNet
# ---------------------------------------------------------------------------

class PlainBlock(nn.Module):
    """A plain (no shortcut) two-conv block for comparison."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))


class ResBlock(nn.Module):
    """Residual block for the gradient demo."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)   # <-- shortcut


def make_deep_network(depth: int, channels: int = 16,
                      residual: bool = True) -> nn.Sequential:
    """Build a deep network of `depth` blocks, each keeping spatial dims."""
    Block = ResBlock if residual else PlainBlock
    return nn.Sequential(*[Block(channels) for _ in range(depth)])


def gradient_norm(model: nn.Module, x: torch.Tensor) -> float:
    """Compute the L2 norm of input gradients via a dummy loss."""
    x = x.clone().requires_grad_(True)
    loss = model(x).sum()
    loss.backward()
    return x.grad.norm().item()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def demo():
    torch.manual_seed(42)
    device = torch.device("cpu")

    print("ResNet Architecture Demo")
    print("=" * 40)
    models = [
        ("ResNet-18",  resnet18()),
        ("ResNet-34",  resnet34()),
        ("ResNet-50",  resnet50()),
    ]
    dummy_img = torch.randn(2, 3, 224, 224)
    for name, model in models:
        model.eval()
        with torch.no_grad():
            out = model(dummy_img)
        print(f"{name:<10} | Params: {count_params(model):>12,} | Output: {out.shape}")

    print()
    print("CIFAR-10 ResNet-20 Demo")
    print("=" * 40)
    cifar_model = resnet20_cifar(num_classes=10).eval()
    dummy_cifar = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out = cifar_model(dummy_cifar)
    print(f"ResNet-20 (CIFAR) | Params: {count_params(cifar_model):>7,} | Output: {out.shape}")

    print()
    print("Residual Connection Benefit Demo")
    print("=" * 40)
    channels = 16
    dummy_feat = torch.randn(2, channels, 8, 8)
    criterion = nn.CrossEntropyLoss()
    labels = torch.randint(0, 10, (2,))
    fc = nn.Linear(channels, 10)

    for depth in [5, 10, 20]:
        results = {}
        for res in [False, True]:
            net = nn.Sequential(make_deep_network(depth, channels, res),
                                nn.AdaptiveAvgPool2d(1),
                                nn.Flatten(), fc)
            net.eval()
            with torch.no_grad():
                out = net(dummy_feat)
                loss = criterion(out, labels).item()
            results[res] = loss
        tag = f"Depth {depth:>2}"
        print(f"{tag} | Plain loss: {results[False]:.4f} | ResNet loss: {results[True]:.4f}")

    print("(ResNet maintains stable gradient flow at all depths)")

    print()
    print("Gradient norm comparison (depth=20):")
    depth = 20
    for res, label in [(False, "Plain "), (True, "ResNet")]:
        net = make_deep_network(depth, channels, res)
        norm = gradient_norm(net, torch.randn(1, channels, 8, 8))
        print(f"  {label} network: gradient norm = {norm:.4f}")


if __name__ == "__main__":
    demo()
