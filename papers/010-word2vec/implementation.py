"""
Word2Vec: Distributed Representations of Words and Phrases (Mikolov et al., 2013)

Skip-gram with Negative Sampling (SGNS) — full training from scratch.
Under 300 lines, numpy only.

Paper: https://arxiv.org/abs/1310.4546
"""

import numpy as np
from collections import Counter
import re


# ── Vocabulary ────────────────────────────────────────────────────────────────

def build_vocab(tokens: list[str], min_count: int = 2):
    """Build word→index mappings, keeping words with freq >= min_count."""
    counts = Counter(tokens)
    vocab = [w for w, c in counts.most_common() if c >= min_count]
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    freq = np.array([counts[i2w[i]] for i in range(len(vocab))], dtype=np.float32)
    return w2i, i2w, freq


def subsample_mask(tokens: list[str], freq: np.ndarray, w2i: dict, t: float = 1e-4) -> list[bool]:
    """
    Subsampling of frequent words (Sec 2.3).

    P(keep w) = sqrt(t / f(w))   [simplified formula from original code]

    High-frequency words like 'the', 'a' are dropped with high probability,
    speeding up training and improving quality of rare-word representations.
    """
    total = freq.sum()
    keep = []
    for tok in tokens:
        if tok not in w2i:
            keep.append(False)
            continue
        f = freq[w2i[tok]] / total
        p_keep = min(1.0, np.sqrt(t / f))
        keep.append(np.random.random() < p_keep)
    return keep


# ── Noise Distribution ────────────────────────────────────────────────────────

def make_noise_dist(freq: np.ndarray) -> np.ndarray:
    """
    Unigram distribution raised to 3/4 power (Sec 2.2).

    P_n(w) ∝ f(w)^(3/4)

    This smooths the distribution, giving rare words more chance to be
    drawn as negatives than pure unigram sampling would.
    """
    p = freq ** 0.75
    return p / p.sum()


# ── Model ─────────────────────────────────────────────────────────────────────

class SkipGramNS:
    """
    Skip-gram with Negative Sampling.

    Two embedding matrices:
      W       (V × D): center-word (input) embeddings
      W_prime (V × D): context-word (output) embeddings

    Only W is used at inference; W_prime is discarded.
    """

    def __init__(self, vocab_size: int, dim: int = 64):
        self.V = vocab_size
        self.D = dim
        # Init: small random center embeddings, zero context embeddings
        self.W       = np.random.randn(vocab_size, dim).astype(np.float32) * 0.01
        self.W_prime = np.zeros((vocab_size, dim), dtype=np.float32)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))

    def train_pair(self, center: int, context: int, negatives: np.ndarray, lr: float) -> float:
        """
        One SGD step on a (center, context) pair with K negative samples.

        Objective (maximise):
          log σ(v'_context · v_center)
          + Σ_k log σ(-v'_{neg_k} · v_center)

        Returns the loss (negated objective) for monitoring.
        """
        h = self.W[center]                         # (D,)

        # ── Positive sample ──────────────────────────────────────────────────
        s_pos = self._sigmoid(self.W_prime[context] @ h)   # scalar
        # dL/dW'_context = (σ - 1) · h
        g_pos = (s_pos - 1.0) * h
        # ── Negative samples ─────────────────────────────────────────────────
        s_neg = self._sigmoid(self.W_prime[negatives] @ h) # (K,)
        # dL/dW'_{neg_k} = σ_k · h
        g_neg = s_neg[:, None] * h[None, :]                # (K, D)

        # ── Gradient w.r.t. center embedding h ───────────────────────────────
        # Chain-rule sum from both positive and all negatives
        g_h = ((s_pos - 1.0) * self.W_prime[context]
               + (s_neg[:, None] * self.W_prime[negatives]).sum(axis=0))

        # ── Parameter updates ─────────────────────────────────────────────────
        self.W_prime[context]  -= lr * g_pos
        self.W_prime[negatives] -= lr * g_neg
        self.W[center]          -= lr * g_h

        loss = -np.log(s_pos + 1e-9) - np.log(1 - s_neg + 1e-9).sum()
        return float(loss)


# ── Training ──────────────────────────────────────────────────────────────────

def generate_pairs(tokens: list[int], window: int):
    """Yield (center, context) index pairs for Skip-gram."""
    n = len(tokens)
    for i, center in enumerate(tokens):
        lo = max(0, i - window)
        hi = min(n - 1, i + window)
        for j in range(lo, hi + 1):
            if j != i:
                yield center, tokens[j]


def train(model: SkipGramNS,
          tokens: list[int],
          noise_dist: np.ndarray,
          window: int = 5,
          neg_samples: int = 5,
          lr: float = 0.025,
          epochs: int = 3,
          report_every: int = 50_000) -> list[float]:
    """Full training loop. Returns per-epoch average losses."""
    losses = []
    rng = np.random.default_rng(42)

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for center, context in generate_pairs(tokens, window):
            # Sample K negatives (avoid center and context — best-effort)
            negatives = rng.choice(model.V, size=neg_samples, p=noise_dist, replace=True)
            loss = model.train_pair(center, context, negatives, lr)
            total_loss += loss
            count += 1
            if count % report_every == 0:
                print(f"  epoch {epoch+1}  pair {count:,}  avg_loss {total_loss/count:.4f}")

        avg = total_loss / max(count, 1)
        losses.append(avg)
        print(f"Epoch {epoch+1}/{epochs}  avg_loss={avg:.4f}")
        # Simple LR decay
        lr *= 0.9
    return losses


# ── Evaluation ────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def most_similar(model: SkipGramNS, i2w: dict, word_idx: int, top_k: int = 5):
    """Brute-force cosine nearest neighbors."""
    vec = model.W[word_idx]
    # Normalise entire matrix once
    norms = np.linalg.norm(model.W, axis=1, keepdims=True) + 1e-9
    sims  = (model.W / norms) @ (vec / (np.linalg.norm(vec) + 1e-9))
    sims[word_idx] = -1.0   # exclude self
    top = np.argsort(sims)[::-1][:top_k]
    return [(i2w[idx], float(sims[idx])) for idx in top]


def analogy(model: SkipGramNS, w2i: dict, i2w: dict,
            a: str, b: str, c: str, top_k: int = 5):
    """
    Vector arithmetic analogy: a - b + c ≈ ?
    Classic example: king - man + woman ≈ queen
    """
    try:
        va, vb, vc = model.W[w2i[a]], model.W[w2i[b]], model.W[w2i[c]]
    except KeyError as e:
        return f"Word not in vocab: {e}"
    target = va - vb + vc
    exclude = {w2i[a], w2i[b], w2i[c]}
    norms = np.linalg.norm(model.W, axis=1, keepdims=True) + 1e-9
    sims  = (model.W / norms) @ (target / (np.linalg.norm(target) + 1e-9))
    for idx in exclude:
        sims[idx] = -1.0
    top = np.argsort(sims)[::-1][:top_k]
    return [(i2w[idx], float(sims[idx])) for idx in top]


# ── Demo ──────────────────────────────────────────────────────────────────────

CORPUS = """
machine learning is a field of artificial intelligence that uses statistical techniques
to give computer systems the ability to learn from data without being explicitly programmed
deep learning is a subset of machine learning that uses neural networks with many layers
neural networks are inspired by the structure and function of the brain
the brain contains billions of neurons that communicate through synapses
a neural network learns by adjusting weights through a process called backpropagation
gradient descent is an optimization algorithm used to minimize the loss function
the loss function measures how well the model fits the training data
training data is used to teach the model to make predictions
predictions are made by feeding input data through the neural network
word embeddings capture semantic relationships between words
similar words have similar vector representations in the embedding space
the word king is similar to the word queen in the embedding space
man and woman are similar words in human language
computer and machine are related words in the field of technology
language models predict the next word in a sequence
sequence modeling is important for natural language processing
natural language processing enables computers to understand human language
artificial intelligence is transforming technology and society
machine learning models learn patterns from large amounts of data
data science combines statistics programming and domain knowledge
deep neural networks have many hidden layers between input and output
the transformer architecture revolutionized natural language processing
attention mechanisms allow models to focus on relevant parts of the input
recurrent neural networks process sequential data one step at a time
convolutional neural networks are widely used for image recognition
image recognition systems can identify objects in photographs
computer vision is the field of teaching computers to understand images
reinforcement learning trains agents to maximize rewards through interaction
the agent learns by trial and error in an environment
supervised learning uses labeled data to train classification models
unsupervised learning finds patterns in data without labels
clustering algorithms group similar data points together
dimensionality reduction techniques compress high dimensional data
word vectors encode meaning in a low dimensional continuous space
distributed representations capture rich linguistic structure
language and meaning are encoded in neural network weights
training embeddings requires large text corpora and compute
the model learns to predict surrounding words from a center word
context words provide information about the meaning of a word
""".lower()


def demo():
    print("=" * 60)
    print("Word2Vec: Skip-gram with Negative Sampling — Demo")
    print("=" * 60)

    # ── Tokenise ──────────────────────────────────────────────────
    tokens_raw = re.findall(r"[a-z]+", CORPUS)
    print(f"\nCorpus: {len(tokens_raw):,} tokens")

    w2i, i2w, freq = build_vocab(tokens_raw, min_count=2)
    print(f"Vocabulary: {len(w2i):,} words")

    # ── Subsample ─────────────────────────────────────────────────
    mask = subsample_mask(tokens_raw, freq, w2i, t=1e-4)
    tokens_idx = [w2i[t] for t, keep in zip(tokens_raw, mask) if keep and t in w2i]
    print(f"After subsampling: {len(tokens_idx):,} tokens")

    # ── Build noise distribution ───────────────────────────────────
    noise_dist = make_noise_dist(freq)

    # ── Train ──────────────────────────────────────────────────────
    np.random.seed(0)
    model = SkipGramNS(vocab_size=len(w2i), dim=50)
    print("\nTraining Skip-gram (3 epochs, window=4, neg=5) ...")
    train(model, tokens_idx, noise_dist,
          window=4, neg_samples=5, lr=0.05, epochs=5,
          report_every=999_999)   # silence per-step output for small corpus

    # ── Most similar words ─────────────────────────────────────────
    print("\n── Most similar words ───────────────────────────────────")
    for query in ["learning", "neural", "data", "language"]:
        if query in w2i:
            neighbours = most_similar(model, i2w, w2i[query], top_k=4)
            nb_str = ", ".join(f"{w}({s:.2f})" for w, s in neighbours)
            print(f"  {query:12s} → {nb_str}")

    # ── Analogy (best-effort on tiny corpus) ──────────────────────
    print("\n── Analogy: machine - learning + data ≈ ? ──────────────")
    results = analogy(model, w2i, i2w, "machine", "learning", "data")
    if isinstance(results, str):
        print(" ", results)
    else:
        for w, s in results[:3]:
            print(f"  {w}  ({s:.3f})")

    # ── Show a few raw vectors ─────────────────────────────────────
    print("\n── Cosine similarity pairs ──────────────────────────────")
    pairs = [("learning", "training"), ("neural", "network"),
             ("machine", "computer"), ("language", "words")]
    for a, b in pairs:
        if a in w2i and b in w2i:
            sim = cosine_similarity(model.W[w2i[a]], model.W[w2i[b]])
            print(f"  sim({a}, {b}) = {sim:.4f}")

    print("\nDone. Embeddings shape:", model.W.shape)


if __name__ == "__main__":
    demo()
