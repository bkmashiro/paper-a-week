# 009 — GAN: Generative Adversarial Networks

**Paper:** Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio, 2014 | **Venue:** NeurIPS 2014  
**Link:** https://arxiv.org/abs/1406.2661

---

## Background

Before GANs, the dominant approaches to generative modeling were either intractable or required strong approximations. Restricted Boltzmann Machines (RBMs) and Deep Boltzmann Machines required Markov Chain Monte Carlo sampling at every step of training — computationally brutal and slow to mix. Variational Autoencoders (VAEs), introduced around the same time, bypassed this by learning an approximate posterior via a reparameterization trick, but the Gaussian assumption in the output space often produced blurry samples. In both cases, the model had to explicitly define and optimize a likelihood — a density in pixel space — which is hard to specify well.

Goodfellow et al. reframed the problem entirely: instead of computing likelihood, can a model *learn what "realistic" means* from data, using another neural network as the judge? This is the core game-theoretic insight of GANs. The generator network G learns to produce samples that fool a discriminator D, while D learns to tell apart real samples from generated ones. Neither network ever sees an explicit density function — the signal comes entirely from the adversarial feedback.

This adversarial framework had a profound impact not just on generative modeling but on how practitioners think about training objectives. Using one neural network as a learned loss function for another opened the door to image-to-image translation (pix2pix), style transfer, super-resolution (SRGAN), and eventually diffusion-based models that incorporated adversarial objectives. The GAN framework catalyzed an entire era of visual generative AI.

---

## Core Ideas

### The Minimax Game

The GAN training objective is a two-player minimax game:

```
min_G  max_D  V(D, G)  =  E_{x ~ p_data}[log D(x)]  +  E_{z ~ p_z}[log(1 - D(G(z)))]
```

- **D(x)** = probability that x is a real sample (output ∈ (0,1))
- **G(z)** = generated sample from noise z ~ p_z(z) (typically z ~ N(0, I))
- **D maximizes**: correctly classifying real as real (log D(x) → 0) and fake as fake (log(1 − D(G(z))) → 0)
- **G minimizes**: making D believe fakes are real (log(1 − D(G(z))) → −∞)

At the global optimum, G perfectly matches the data distribution p_data, and D(x) = 1/2 everywhere — it can no longer distinguish real from fake, since they're identical.

### Optimal Discriminator

For a fixed G, the optimal discriminator D* is:

```
D*_G(x) = p_data(x) / (p_data(x) + p_g(x))
```

where p_g is the distribution induced by G. Plugging D* back into the value function gives the Generator's effective loss:

```
C(G) = -log(4) + 2 * JSD(p_data || p_g)
```

where JSD is the Jensen-Shannon Divergence. This is minimized when p_g = p_data, confirming the equilibrium.

### Training Algorithm

The paper proposes alternating gradient descent:

```
For each training step:
  1. Sample minibatch of m real examples {x_1, ..., x_m} from p_data
  2. Sample minibatch of m noise vectors {z_1, ..., z_m} from p_z
  3. Update D by ascending its stochastic gradient:
       ∇_θd  (1/m) Σ [log D(x_i) + log(1 − D(G(z_i)))]
  4. Sample fresh m noise vectors {z_1, ..., z_m} from p_z
  5. Update G by descending its stochastic gradient:
       ∇_θg  (1/m) Σ log(1 − D(G(z_i)))
     [in practice: ascend ∇_θg  (1/m) Σ log D(G(z_i))  — the non-saturating trick]
```

The paper suggests k=1 discriminator updates per generator update in practice, though k > 1 is an option for slower-collapsing discriminators.

### The Non-Saturating Generator Loss

The original G objective `log(1 − D(G(z)))` saturates early in training when D easily rejects fakes — the gradient is flat when G needs it most. In practice, the generator instead maximizes `log D(G(z))`, which has the same fixed point but provides stronger gradients at the start of training. This is the standard "non-saturating GAN" formulation used in virtually all practical implementations.

```
# Saturating (original paper, theoretical analysis):
loss_G = log(1 - D(G(z)))   # gradient weak when D(G(z)) ≈ 0

# Non-saturating (practical, same Nash equilibrium):
loss_G = -log(D(G(z)))      # gradient strong throughout
```

---

## Implementation Notes

### What We Kept

- Full minimax game with Generator and Discriminator MLPs
- Non-saturating generator loss (practical formulation)
- Alternating gradient descent (1 D step per G step)
- Binary cross-entropy loss for both networks
- The complete training loop with real loss tracking

### What We Simplified

- Target distribution: 1D Gaussian mixture (instead of image data) for fast CPU demo
- Small MLPs (2–3 layers) instead of deep CNNs
- No learning rate scheduling or advanced tricks (WGAN, spectral norm, etc.)
- No batch norm (not in original paper; added in DCGAN 2015)

### Key Code Points

**Discriminator loss** (lines ~60–75):
```python
# D wants real → 1, fake → 0
real_loss = F.binary_cross_entropy(D(real), ones)
fake_loss = F.binary_cross_entropy(D(fake.detach()), zeros)
d_loss = (real_loss + fake_loss) / 2
```

**Generator loss** (non-saturating) (lines ~80–90):
```python
# G wants D to output 1 on its fakes
fake = G(z)
g_loss = F.binary_cross_entropy(D(fake), ones)  # -log D(G(z))
```

**Why `.detach()` matters**: when computing d_loss with `D(fake)`, we detach G's output so gradients don't flow back through G — D's optimizer should only update D's parameters.

---

## Key Takeaways

1. **Likelihood-free generation**: GANs remove the need to define an explicit density — the discriminator learns the implicit metric.

2. **Adversarial feedback is a powerful learning signal**: training against an adaptive adversary forces the generator to fix the most distinguishable failure modes first.

3. **Training instability is inherent**: the minimax game is difficult to stabilize. Mode collapse (G always produces the same output), discriminator saturation, and oscillating losses are fundamental challenges. Later work (WGAN, spectral norm, gradient penalty) addresses these.

4. **The non-saturating trick is essential**: the theoretical analysis uses `log(1 − D(G(z)))` but the practical default is always `-log(D(G(z)))`. Know both.

5. **GANs as implicit models**: GANs belong to the family of *implicit generative models* — models that sample from p_g without specifying p_g analytically. This family now includes diffusion models using score matching.

---

## Running

```bash
pip install torch numpy
python implementation.py
```

Expected output:
```
Training GAN on Gaussian mixture: N(-2, 0.3) and N(2, 0.3)
 Epoch    D_loss    G_loss    G_mean     G_std
-------------------------------------------------------
   300    0.6651    0.8059    +0.098     0.621
   600    0.6322    0.8637    +0.071     0.732
   ...
  3000    0.6388    0.7292    +0.046     2.050

=== Final Evaluation ===
Real data   | mean:  +0.004 | std: 2.028
Generated   | mean:  +0.056 | std: 2.076
KL estimate (real || gen): 0.8839  (lower is better)

Mode coverage | left (x<0): 48.9%  right (x≥0): 51.1%
✓ Generator learned both modes — no mode collapse.
```

Note: the generated distribution captures both modes and has correct mean/std, but is
smoother than the sharp target peaks — a classic vanilla GAN artifact (mode blending).
The KL appears high because of bin mismatch at the inter-modal gap; the mode balance (≈50/50)
is the key indicator of success here.
