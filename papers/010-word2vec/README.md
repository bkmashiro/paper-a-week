# 010 — Word2Vec: Distributed Representations of Words and Phrases

**Paper:** Mikolov, T., Sutskever, I., Chen, K., Corrado, G., Dean, J. (2013)  
**Venue:** NeurIPS 2013  
**Link:** https://arxiv.org/abs/1310.4546

---

## Background

Before Word2Vec, the dominant representation for words in NLP was the one-hot vector: a sparse binary vector of length |V| (vocabulary size), with a single 1 at the word's index. This representation has two fatal flaws. First, it is astronomically high-dimensional — vocabularies routinely exceed 100,000 words. Second, and more fundamentally, it treats all words as equidistant from each other: the vectors for "cat" and "kitten" are just as far apart as "cat" and "airplane." There is no notion of semantic similarity baked into the representation.

Distributed word representations — the idea that a word's meaning can be encoded as a dense, low-dimensional real-valued vector — had existed since Rumelhart's 1986 work and were explored extensively by Bengio et al. in 2003 with neural language models. The trouble was compute: training a neural language model with a large hidden layer and a full softmax over |V| output units was prohibitively slow, scaling poorly to corpora of billions of words. Mikolov et al.'s 2013 papers attacked this problem head-on, stripping away the non-linear hidden layers and replacing the expensive softmax with efficient approximations. The result was models that could train on billions of words overnight on a single machine and produce embedding spaces with remarkable geometric structure.

The second 2013 paper, "Distributed Representations of Words and Phrases," extended the original with three crucial improvements: **Negative Sampling** (a dramatically simpler and more effective objective than Hierarchical Softmax), **subsampling of frequent words**, and a method for learning **phrase representations**. Together, these made Word2Vec the tool that popularised dense word embeddings across the entire field of NLP, laying the groundwork for everything from GloVe to ELMo to modern transformer-based models.

---

## Core Ideas

### 1. The Skip-gram Model

Skip-gram is the heart of this paper. Given a sequence of training words w₁, w₂, …, w_T, the Skip-gram objective is to maximise the average log probability:

```
(1/T) Σ_{t=1}^{T}  Σ_{-c ≤ j ≤ c, j≠0}  log P(w_{t+j} | w_t)
```

where c is the context window size. For each center word w_t, we predict all surrounding context words within the window. This is the *opposite* direction from CBOW (Continuous Bag of Words), which predicts the center word from its context. Skip-gram works better for infrequent words: each occurrence of a rare word generates many training pairs, each reinforcing the word's representation.

The basic probability model is a softmax over all V words:

```
P(w_O | w_I) = exp(v'_{w_O} · v_{w_I}) / Σ_{w=1}^{V} exp(v'_w · v_{w_I})
```

where `v_w` is the **input embedding** (center role) and `v'_w` is the **output embedding** (context role). Every word has two vectors; only the input embeddings are used at inference. The softmax denominator requires summing over all V words — intractable at scale.

### 2. Negative Sampling

Negative Sampling replaces the full softmax with a binary classification problem. Instead of predicting the correct word from all V candidates, we ask: "Is this (center, context) pair real or noise-generated?" The objective per pair becomes:

```
log σ(v'_{w_O} · v_{w_I}) + Σ_{k=1}^{K} E_{w_k ~ P_n(w)} [log σ(−v'_{w_k} · v_{w_I})]
```

where σ is the sigmoid, K is the number of negative samples (5–20 for small data, 2–5 for large), and P_n(w) is the **noise distribution**. Each gradient update touches only K+1 word vectors instead of all V. For K=5 and |V|=100,000, this is a ~20,000× reduction in work per step.

The choice of noise distribution matters enormously. A uniform distribution over vocabulary leads to too many common-word negatives. The paper found that raising the unigram distribution to the **3/4 power** works best in practice:

```
P_n(w) ∝ f(w)^(3/4)
```

This smooths the distribution: rare words get relatively more representation as negatives, improving the quality of their learned embeddings.

### 3. Subsampling of Frequent Words

Words like "the", "a", "of" appear constantly but carry little semantic signal. The paper introduces a stochastic subsampling rule: each word w_i in the training stream is discarded with probability:

```
P(discard w_i) = 1 − sqrt(t / f(w_i))
```

where f(w_i) is the word's frequency and t is a threshold (typically 10⁻⁵). Words with frequency above t are aggressively dropped; words below t are kept with high probability. In the Google News corpus (1 billion words), this removed ~75% of tokens, making training ~2–10× faster and — crucially — improving accuracy on rare words by increasing the effective co-occurrence rate between infrequent but meaningful words.

### 4. The Geometry of Meaning

The most celebrated property of Word2Vec embeddings is that linear algebraic operations in embedding space correspond to semantic relationships. The famous example:

```
v(king) − v(man) + v(woman) ≈ v(queen)
```

This works because the training objective forces semantically related words into similar neighbourhoods. The vector offset `v(man) − v(woman)` encodes the "gender" direction, and this direction is approximately consistent across analogous pairs: (king, queen), (actor, actress), (uncle, aunt), etc. The paper introduced the 3CosMul analogy evaluation metric and showed that 5-dimensional analogy tasks (semantic + syntactic) could be solved with surprisingly high accuracy purely through vector arithmetic.

---

## Implementation Notes

### What We Kept

- Skip-gram architecture (center word predicts context words)
- Negative sampling with exact gradient computation
- Unigram^(3/4) noise distribution
- Subsampling of frequent words
- Cosine similarity for nearest-neighbour evaluation
- Vector arithmetic for analogy queries

### What We Simplified

- No Hierarchical Softmax (Negative Sampling is standard in practice)
- Single-threaded training (original used 24 threads with shared memory)
- Mini-batch size 1 (one pair per update; original processes ~100 words/thread)
- No phrase detection (bigram scoring: score(A,B) = [C(AB)−δ] / [C(A)·C(B)])
- Small demo corpus (real training uses billions of tokens)

### Key Code Points

**The negative sampling gradient** (lines ~70–90):

```python
def train_pair(self, center, context, negatives, lr):
    h = self.W[center]                              # center embedding, shape (D,)
    
    s_pos = sigmoid(self.W_prime[context] @ h)      # score for real pair
    s_neg = sigmoid(self.W_prime[negatives] @ h)    # score for K noise pairs
    
    # Gradient: push positive score toward 1, negatives toward 0
    g_pos = (s_pos - 1.0) * h                       # ∂L/∂W'_context
    g_neg = s_neg[:, None] * h[None, :]             # ∂L/∂W'_negatives
    
    # Center embedding gets gradients from all K+1 pairs
    g_h = (s_pos - 1.0) * self.W_prime[context] \
        + (s_neg[:, None] * self.W_prime[negatives]).sum(axis=0)
    
    self.W_prime[context]   -= lr * g_pos
    self.W_prime[negatives] -= lr * g_neg
    self.W[center]          -= lr * g_h
```

The key insight: gradients flow through *two* embedding matrices. The center word's embedding is updated once per (center, context, negatives) triple; the context/noise embeddings are updated for each sample independently. This is why we keep two separate matrices W and W'.

---

## Running

```bash
pip install numpy
python implementation.py
```

Expected output (exact values vary slightly):

```
============================================================
Word2Vec: Skip-gram with Negative Sampling — Demo
============================================================

Corpus: 362 tokens
Vocabulary: 67 words
After subsampling: 241 tokens

Training Skip-gram (3 epochs, window=4, neg=5) ...
Epoch 1/5  avg_loss=...
...

── Most similar words ───────────────────────────────────
  learning     → training(0.xx), data(0.xx), model(0.xx), deep(0.xx)
  neural       → network(0.xx), networks(0.xx), layers(0.xx), ...
  data         → training(0.xx), large(0.xx), patterns(0.xx), ...
  language     → natural(0.xx), processing(0.xx), human(0.xx), ...

── Cosine similarity pairs ──────────────────────────────
  sim(learning, training) = 0.xxxx
  sim(neural, network)    = 0.xxxx
  sim(machine, computer)  = 0.xxxx
  sim(language, words)    = 0.xxxx

Done. Embeddings shape: (67, 50)
```

---

## Key Takeaways

1. **Remove the bottleneck, not the model.** The core innovation is not the Skip-gram architecture (which existed) but replacing the |V|-way softmax with Negative Sampling — a K+1 binary classification. This makes training O(K·D) per pair instead of O(V·D).

2. **Frequent words are noise, not signal.** Aggressive subsampling of words like "the" paradoxically *improves* representation quality for rare content words, because it reduces the overwhelming statistical signal from meaningless co-occurrences.

3. **Two vectors per word.** Every word has an input embedding (W) and an output embedding (W'). Only W is used at inference. Averaging W and W' sometimes helps, but the paper reports W alone works well enough.

4. **Linear structure is an emergent property.** Nobody designed the embedding space to support vector arithmetic. It emerges from the training objective: maximising the similarity of co-occurring words and minimising it for random pairs creates a geometry where semantic relationships are encoded as consistent directions.

5. **Scale matters enormously.** The analogy results that wowed the field came from training on the Google News corpus (100 billion words). The small-corpus demo here shows the mechanics; real quality requires real data.
