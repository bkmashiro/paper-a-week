"""
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Devlin et al., 2018 — https://arxiv.org/abs/1810.04805

Core implementation of:
  - BERT encoder (multi-head self-attention + FFN + layer norm)
  - Input embeddings (token + position + segment)
  - Masked Language Modeling (MLM) pre-training objective
  - Next Sentence Prediction (NSP) pre-training objective
  - Fine-tuning classification head on [CLS]

Under 300 lines. Requires: torch, numpy
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Special token IDs (reserved in vocabulary)
# ─────────────────────────────────────────────────────────────────────────────
PAD_ID  = 0
MASK_ID = 1
CLS_ID  = 2
SEP_ID  = 3


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Head Self-Attention
# ─────────────────────────────────────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        B, T, H = x.shape
        nh, dh = self.num_heads, self.head_dim

        # Project and split into heads: (B, T, H) → (B, nh, T, dh)
        Q = self.q(x).view(B, T, nh, dh).transpose(1, 2)
        K = self.k(x).view(B, T, nh, dh).transpose(1, 2)
        V = self.v(x).view(B, T, nh, dh).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, nh, T, T)
        if attn_mask is not None:
            scores = scores + attn_mask  # attn_mask: 0 or -1e9

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum and reshape
        out = torch.matmul(attn_weights, V)           # (B, nh, T, dh)
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        return self.out(out), attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# Position-wise Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BERT uses GELU, not ReLU
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


# ─────────────────────────────────────────────────────────────────────────────
# Single BERT Encoder Layer
# ─────────────────────────────────────────────────────────────────────────────
class BERTLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        attn_out, weights = self.attention(x, attn_mask)
        x = self.norm1(x + self.dropout(attn_out))   # residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))     # residual + LayerNorm
        return x, weights


# ─────────────────────────────────────────────────────────────────────────────
# BERT Embeddings (Token + Position + Segment)
# ─────────────────────────────────────────────────────────────────────────────
class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.token_emb    = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_ID)
        self.position_emb = nn.Embedding(max_len, hidden_size)
        self.segment_emb  = nn.Embedding(2, hidden_size)   # 0 = sent A, 1 = sent B
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor, segment_ids: torch.Tensor = None):
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, -1)
        if segment_ids is None:
            segment_ids = torch.zeros_like(token_ids)

        emb = self.token_emb(token_ids) + self.position_emb(positions) + \
              self.segment_emb(segment_ids)
        return self.dropout(self.norm(emb))


# ─────────────────────────────────────────────────────────────────────────────
# Full BERT Model
# ─────────────────────────────────────────────────────────────────────────────
class BERT(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 768, num_layers: int = 12,
                 num_heads: int = 12, intermediate_size: int = 3072,
                 max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embeddings = BERTEmbeddings(vocab_size, hidden_size, max_len, dropout)
        self.layers = nn.ModuleList([
            BERTLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        # MLM head: hidden → vocab logits
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        # NSP head: [CLS] hidden → 2-class logits
        self.nsp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, token_ids: torch.Tensor, segment_ids: torch.Tensor = None,
                attn_mask: torch.Tensor = None):
        """
        token_ids:   (B, T) — input token IDs (with [CLS], [SEP], possible [MASK])
        segment_ids: (B, T) — 0 for sent A tokens, 1 for sent B tokens
        attn_mask:   (B, 1, 1, T) — additive mask, -1e9 for pad tokens
        Returns:
            hidden: (B, T, H)      — contextual representations for every token
            mlm_logits: (B, T, V) — per-token vocab logits (for MLM)
            nsp_logits: (B, 2)    — IsNext / NotNext logits (from [CLS])
            all_attn_weights: list of (B, nh, T, T) per layer
        """
        x = self.embeddings(token_ids, segment_ids)
        all_attn_weights = []
        for layer in self.layers:
            x, w = layer(x, attn_mask)
            all_attn_weights.append(w)

        cls_hidden = x[:, 0, :]          # [CLS] token representation
        mlm_logits = self.mlm_head(x)
        nsp_logits = self.nsp_head(cls_hidden)
        return x, mlm_logits, nsp_logits, all_attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# MLM masking: 80% [MASK], 10% random, 10% unchanged
# ─────────────────────────────────────────────────────────────────────────────
def apply_mlm_masking(tokens: torch.Tensor, vocab_size: int,
                      mask_token_id: int = MASK_ID, mask_prob: float = 0.15,
                      special_ids=(PAD_ID, CLS_ID, SEP_ID)):
    tokens = tokens.clone()
    labels = tokens.clone()
    prob_matrix = torch.rand(tokens.shape)

    # Don't mask special tokens
    for sid in special_ids:
        prob_matrix[tokens == sid] = 1.0

    selected = prob_matrix < mask_prob   # 15% selected

    # 80% of selected → [MASK]
    replace_with_mask = selected & (torch.rand(tokens.shape) < 0.8)
    tokens[replace_with_mask] = mask_token_id

    # 10% of selected → random vocabulary token (excluding specials)
    replace_with_rand = selected & ~replace_with_mask & (torch.rand(tokens.shape) < 0.5)
    rand_tokens = torch.randint(low=4, high=vocab_size, size=tokens.shape)
    tokens[replace_with_rand] = rand_tokens[replace_with_rand]

    # remaining 10% of selected: kept unchanged (already in tokens)

    # Only compute loss on selected positions; -100 = ignore_index in CrossEntropyLoss
    labels[~selected] = -100
    return tokens, labels


# ─────────────────────────────────────────────────────────────────────────────
# Classification fine-tuning head (added on top of BERT)
# ─────────────────────────────────────────────────────────────────────────────
class BERTClassifier(nn.Module):
    def __init__(self, bert: BERT, num_classes: int, hidden_size: int):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, token_ids, segment_ids=None, attn_mask=None):
        _, _, _, _ = self.bert(token_ids, segment_ids, attn_mask)
        # Re-run to get hidden states cleanly
        x = self.bert.embeddings(token_ids, segment_ids)
        for layer in self.bert.layers:
            x, _ = layer(x, attn_mask)
        cls_hidden = x[:, 0, :]
        return self.classifier(cls_hidden)


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────
def demo():
    torch.manual_seed(42)
    np.random.seed(42)

    # Tiny BERT config for demo
    VOCAB_SIZE  = 30
    HIDDEN      = 64
    LAYERS      = 2
    HEADS       = 4
    INTERMEDIATE = HIDDEN * 4
    MAX_LEN     = 16

    model = BERT(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN,
        num_layers=LAYERS,
        num_heads=HEADS,
        intermediate_size=INTERMEDIATE,
        max_len=MAX_LEN,
        dropout=0.0    # deterministic for demo
    )

    total_params = sum(p.numel() for p in model.parameters())
    print("=== BERT Demo ===\n")
    print(f"Model: {LAYERS} layers, {HIDDEN} hidden, {HEADS} heads")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Parameters: ~{total_params//1000}K\n")

    # ── MLM forward pass ─────────────────────────────────────────────────────
    print("--- MLM Pre-training Forward Pass ---")
    # Construct: [CLS] tok tok tok tok tok tok [SEP]
    seq = torch.tensor([[CLS_ID, 5, 8, 12, 17, 7, 14, SEP_ID]])
    masked_seq, labels = apply_mlm_masking(seq, VOCAB_SIZE)

    print(f"Input tokens:  {seq[0].tolist()}")
    print(f"Masked tokens: {masked_seq[0].tolist()}   (1 = [MASK])")

    _, mlm_logits, _, _ = model(masked_seq)
    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, VOCAB_SIZE), labels.view(-1), ignore_index=-100
    )
    print(f"MLM Loss: {mlm_loss.item():.4f}\n")

    # ── NSP forward pass ─────────────────────────────────────────────────────
    print("--- NSP Pre-training Forward Pass ---")
    # IsNext pair: [CLS] A A A [SEP] B B B [SEP]
    #   segment_ids: 0 0 0 0 0 1 1 1 1
    is_next_tokens = torch.tensor([[CLS_ID, 5, 8, 12, SEP_ID, 17, 22, 9, SEP_ID]])
    seg_ids        = torch.tensor([[0,      0, 0,  0,       0,  1,  1, 1,       1]])
    # NotNext pair: random B
    not_next_tokens = torch.tensor([[CLS_ID, 5, 8, 12, SEP_ID, 4, 25, 11, SEP_ID]])

    _, _, nsp_is_next,  _ = model(is_next_tokens, seg_ids)
    _, _, nsp_not_next, _ = model(not_next_tokens, seg_ids)

    nsp_labels_is  = torch.tensor([0])   # 0 = IsNext
    nsp_labels_not = torch.tensor([1])   # 1 = NotNext
    nsp_loss = (F.cross_entropy(nsp_is_next,  nsp_labels_is) +
                F.cross_entropy(nsp_not_next, nsp_labels_not)) / 2

    fmt = lambda t: "[" + ", ".join(f"{v:5.2f}" for v in t[0].tolist()) + "]"
    print(f"IsNext pair  → logits: {fmt(nsp_is_next)}")
    print(f"NotNext pair → logits: {fmt(nsp_not_next)}")
    print(f"NSP Loss: {nsp_loss.item():.4f}\n")

    # ── Fine-tuning: Sentiment Classification ────────────────────────────────
    print("--- Fine-tuning: Sentiment Classification ---")
    classifier = BERTClassifier(model, num_classes=2, hidden_size=HIDDEN)
    optimizer  = torch.optim.Adam(classifier.parameters(), lr=2e-4)

    # Toy dataset: positive (label 0) and negative (label 1) "sentences"
    # Each sequence is [CLS] + tokens + [SEP]; tokens are arbitrary IDs here
    pos_samples = [
        [CLS_ID, 10, 15, 20, SEP_ID],
        [CLS_ID, 11, 16, 21, SEP_ID],
        [CLS_ID,  9, 14, 19, SEP_ID],
        [CLS_ID, 12, 17, 22, SEP_ID],
        [CLS_ID, 13, 18, 23, SEP_ID],
    ]
    neg_samples = [
        [CLS_ID,  5, 7,  9, SEP_ID],
        [CLS_ID,  4, 6,  8, SEP_ID],
        [CLS_ID,  5, 8, 11, SEP_ID],
        [CLS_ID,  6, 9, 12, SEP_ID],
        [CLS_ID,  4, 7, 10, SEP_ID],
    ]
    X = torch.tensor(pos_samples + neg_samples)
    y = torch.tensor([0]*5 + [1]*5)

    print("Training for 50 steps...")
    classifier.train()
    for step in range(1, 51):
        idx = torch.randperm(len(X))
        X_shuf, y_shuf = X[idx], y[idx]
        logits = classifier(X_shuf)
        loss = F.cross_entropy(logits, y_shuf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"Step {step:2d}: loss = {loss.item():.4f}")

    classifier.eval()
    with torch.no_grad():
        preds = classifier(X).argmax(dim=1)
    acc = (preds == y).float().mean().item()
    print(f"Classification accuracy on toy set: {acc*100:.1f}%\n")

    # ── Attention Visualization ───────────────────────────────────────────────
    print("--- Attention Visualization (Layer 0, Head 0) ---")
    seq2 = torch.tensor([[CLS_ID, 10, 15, SEP_ID]])
    token_names = ["[CLS]", "tok1", "tok2", "[SEP]"]
    model.eval()
    with torch.no_grad():
        _, _, _, attn_all_layers = model(seq2)
    w = attn_all_layers[0][0, 0]   # layer 0, batch 0, head 0 → (T, T)
    print(f"Token attention weights (row = query, col = key):")
    col_header = "       " + "  ".join(f"{n:6s}" for n in token_names)
    print(col_header)
    for i, row_name in enumerate(token_names):
        row_str = f"{row_name:6s} →  " + "  ".join(f"{w[i,j].item():.4f}" for j in range(len(token_names)))
        print(row_str)
    print()
    print("Key insight: [CLS] attends broadly — its representation aggregates")
    print("information from the full sequence, enabling classification tasks.")


if __name__ == "__main__":
    demo()
