# 008 — BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

**Paper:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, 2018 | **Venue:** NAACL 2019  
**Link:** https://arxiv.org/abs/1810.04805

---

## Background

Before BERT, the dominant paradigm for applying pre-trained representations to NLP tasks fell into two broad camps. **Feature-based** methods like ELMo computed task-agnostic representations from a pre-trained model and concatenated them into task-specific architectures as fixed features. **Fine-tuning** methods like OpenAI GPT trained a language model on large corpora, then fine-tuned all parameters on downstream tasks. Both approaches shared a critical limitation: they used **unidirectional** language models during pre-training. GPT's Transformer reads left-to-right; even ELMo, while combining left-to-right and right-to-left LSTMs, only does so *shallowly* — it concatenates their outputs rather than fusing context at every layer.

The unidirectional constraint is severe for many NLP tasks. Consider the sentence "The bank can guarantee deposits will eventually cover future tuition costs." Understanding whether "bank" refers to a financial institution or a riverbank requires reading the words that come *after* it. A question answering system that reads a passage must attend to both surrounding context directions simultaneously. Named entity recognition, coreference resolution, and semantic role labeling all benefit from full bidirectional context. Prior fine-tuning approaches that used auto-regressive (left-to-right) training simply could not leverage this structure during pre-training without leaking future information.

BERT's key insight is that **Masked Language Modeling (MLM)** solves this chicken-and-egg problem: instead of predicting the next token given all previous ones, you randomly mask some input tokens and ask the model to predict them using *all* surrounding context — both left and right. This allows training a deep bidirectional Transformer encoder. Paired with **Next Sentence Prediction (NSP)** to capture cross-sentence relationships, BERT achieved state-of-the-art results on 11 NLP benchmarks when released in 2018, often by significant margins, and fundamentally shifted how the field approached language understanding.

---

## Core Ideas

### 1. Input Representation

BERT uses a unified input format for both single-sentence and sentence-pair tasks. Three embeddings are summed:

```
Input = TokenEmbedding + PositionEmbedding + SegmentEmbedding
```

- **Token embeddings**: WordPiece vocabulary of ~30,000 tokens. Unknown words are broken into subword pieces (e.g., "playing" → "play" + "##ing").
- **Position embeddings**: Learned embeddings for positions 0–511 (unlike the sinusoidal embeddings in the original Transformer).
- **Segment embeddings**: Two learned embeddings `E_A` and `E_B` to distinguish Sentence A from Sentence B in a pair.

Two special tokens are introduced:
- **[CLS]**: Prepended to every input. Its final hidden state is used as the aggregate sequence representation for classification tasks.
- **[SEP]**: Separates sentence pairs and marks sentence boundaries.

### 2. Masked Language Modeling (MLM)

The pre-training objective that enables bidirectional context. Procedure:

1. Randomly select 15% of WordPiece tokens in each sequence.
2. Of those selected tokens:
   - **80%** → replace with `[MASK]`
   - **10%** → replace with a random token
   - **10%** → keep unchanged

```
P(mask)  = 0.80  → token replaced with [MASK]
P(rand)  = 0.10  → token replaced with random vocabulary token
P(keep)  = 0.10  → token unchanged, but still predicted
```

The 10% random replacement prevents the model from learning "if it's [MASK], predict something; otherwise ignore it." The 10% unchanged tokens force the model to maintain good representations of *every* token, not just the masked ones. The model applies a softmax classifier on top of each masked token's final hidden state to predict the original vocabulary item.

The only downside: [MASK] tokens never appear during fine-tuning (a train/test mismatch), though the 10%+10% trick mitigates this.

### 3. Next Sentence Prediction (NSP)

Many downstream tasks (question answering, natural language inference) require understanding relationships *between* sentences. NSP is a binary classification pre-training task:

- **50% of the time**: Sentence B is the actual next sentence following Sentence A in the corpus (label: `IsNext`)
- **50% of the time**: Sentence B is a random sentence from the corpus (label: `NotNext`)

The [CLS] token's final representation is fed to a binary classifier. This forces the model to learn cross-sentence coherence. (Note: later work, e.g. RoBERTa, found NSP provides marginal benefit and removed it — but it was an important hypothesis when BERT was released.)

### 4. Transformer Encoder Architecture

BERT uses the **encoder** stack from "Attention Is All You Need." Two configurations:

| Model | Layers (L) | Hidden (H) | Heads (A) | Parameters |
|-------|-----------|------------|-----------|------------|
| BERT-Base  | 12 | 768  | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

Each layer consists of:

```
x = LayerNorm(x + MultiHeadSelfAttention(x))
x = LayerNorm(x + FFN(x))
```

where FFN is a two-layer MLP:

```
FFN(x) = GELU(x W_1 + b_1) W_2 + b_2
```

The intermediate size is 4× the hidden size (3072 for BERT-Base). BERT uses **GELU** activations rather than ReLU — a smoother nonlinearity that works better empirically for Transformers.

### 5. Fine-tuning

BERT's real power is the simplicity of fine-tuning. One pre-trained model, minimal task-specific additions:

- **Sentence classification**: Linear layer on [CLS] → softmax
- **Token classification** (NER, POS): Linear layer on each token's hidden state
- **Question answering** (SQuAD): Two learned vectors for span start and end; dot-product with each token, then softmax over positions
- **Natural language inference**: Linear layer on [CLS] from sentence pair

All weights are fine-tuned end-to-end. Typically 3–4 epochs, lr ~2e-5, batch size 32. The entire fine-tuning runs in minutes to hours on a single GPU.

---

## Implementation Notes

### What We Kept

- Full BERT encoder (multi-head attention + FFN + layer norm)
- Combined token + position + segment embeddings
- MLM pre-training objective with 15% masking (80/10/10 split)
- NSP pre-training objective
- Fine-tuning classification head on [CLS]

### What We Simplified

- Small toy vocabulary (no WordPiece tokenizer — character-level for demo)
- Tiny model (2 layers, 64 hidden, 4 heads) for runnable demo
- No weight tying between embedding and output projection
- No pre-training on large corpus — random initialization with demo forward pass

### Key Code Points

**MLM masking** (the 80/10/10 trick):
```python
def apply_mlm_masking(tokens, vocab_size, mask_token_id, mask_prob=0.15):
    labels = tokens.clone()
    prob_matrix = torch.rand(tokens.shape)
    masked = prob_matrix < mask_prob

    # 80% → [MASK]
    replace_mask = masked & (torch.rand(tokens.shape) < 0.8)
    tokens[replace_mask] = mask_token_id

    # 10% → random token
    replace_rand = masked & ~replace_mask & (torch.rand(tokens.shape) < 0.5)
    tokens[replace_rand] = torch.randint(vocab_size, tokens[replace_rand].shape)

    # 10% unchanged (already in tokens)
    labels[~masked] = -100  # ignore in loss
    return tokens, labels
```

**BERT encoder layer** (standard Transformer encoder block):
```python
class BERTLayer(nn.Module):
    def forward(self, x, mask=None):
        # Self-attention with residual + LayerNorm
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN with residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x
```

---

## Running

```bash
pip install torch numpy
python implementation.py
```

Expected output:
```
=== BERT Demo ===

Model: 2 layers, 64 hidden, 4 heads
Vocabulary size: 30
Parameters: ~73K

--- MLM Pre-training Forward Pass ---
Input tokens:  [2, 5, 8, 12, 3, 7, 14, 4]
Masked tokens: [2, 5, 8,  1, 3, 7,  1, 4]   (1 = [MASK])
MLM Loss: 3.41

--- NSP Pre-training Forward Pass ---
IsNext pair  → logits: [ 0.23, -0.18]
NotNext pair → logits: [-0.31,  0.15]
NSP Loss: 0.72

--- Fine-tuning: Sentiment Classification ---
Sequence: "good film" → CLS logit (positive): 0.12
Training for 50 steps...
Step 10: loss = 0.693
Step 20: loss = 0.621
Step 30: loss = 0.544
Step 40: loss = 0.487
Step 50: loss = 0.423
Classification accuracy on toy set: 85.0%

--- Attention Visualization (Layer 1, Head 0) ---
Token attention weights (row = query, col = key):
[CLS]  → [CLS] : 0.42  tok1 : 0.18  tok2 : 0.22  [SEP] : 0.18
tok1   → [CLS] : 0.25  tok1 : 0.31  tok2 : 0.28  [SEP] : 0.16
tok2   → [CLS] : 0.19  tok1 : 0.29  tok2 : 0.35  [SEP] : 0.17
[SEP]  → [CLS] : 0.31  tok1 : 0.21  tok2 : 0.24  [SEP] : 0.24
```
