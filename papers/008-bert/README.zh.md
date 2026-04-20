# 008 — BERT：深度双向 Transformer 的预训练方法

**论文：** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, 2018 | **发表于：** NAACL 2019  
**链接：** https://arxiv.org/abs/1810.04805

---

## 背景

在 BERT 出现之前，将预训练语言模型应用于下游 NLP 任务主要有两种范式：一种是**基于特征**的方法（如 ELMo），将预训练模型的输出作为固定特征拼接到任务专属架构中；另一种是**微调**方法（如 OpenAI GPT），先在大规模语料上训练语言模型，再对所有参数进行端到端微调。

这两种方法都有一个根本性的缺陷：预训练时只能使用**单向**语言模型。GPT 是从左到右逐词预测的自回归模型；ELMo 虽然同时训练了正向和反向 LSTM，但只是在输出层做浅层融合，并没有在每一层都实现真正的双向上下文感知。

这个限制在很多任务中非常致命。以问答任务为例，理解一个词的含义往往需要同时看它前后的语境；命名实体识别、指代消解等任务同样依赖完整的双向上下文。单向预训练根本无法充分利用这些信息。

BERT 的核心洞察是用**掩码语言模型（MLM）**来绕开这个问题。它不再预测下一个词，而是随机遮盖输入序列中的部分词，让模型利用**左右两侧所有上下文**来预测被遮盖的词。这样就可以训练一个深度双向 Transformer 编码器，从根本上解决了单向限制。加上用于捕捉句间关系的**下句预测（NSP）**任务，BERT 在发布时一举刷新了 11 项 NLP 基准测试的纪录，标志着 NLP 预训练时代的真正到来。

---

## 核心思想

### 1. 输入表示

BERT 把词嵌入（Token Embedding）、位置嵌入（Position Embedding）和分句嵌入（Segment Embedding）三者相加作为最终输入：

```
输入向量 = TokenEmbedding + PositionEmbedding + SegmentEmbedding
```

两个特殊 token 至关重要：
- **[CLS]**：加在每个输入序列开头，其最终隐藏态用于分类任务
- **[SEP]**：用来分隔两个句子，或标记单个句子的结尾

位置嵌入是**学习得到**的（而非原始 Transformer 中的正弦函数），支持最长 512 个 token。分句嵌入用两个可学习的向量 $E_A$ 和 $E_B$ 来区分句子 A 和句子 B。

### 2. 掩码语言模型（MLM）

预训练时，随机选取 15% 的 token 进行处理：

- **80%** 的情况：替换为 `[MASK]`
- **10%** 的情况：替换为词表中的随机词
- **10%** 的情况：保持原词不变

```
P([MASK])  = 0.15 × 0.80
P(random)  = 0.15 × 0.10
P(不变)    = 0.15 × 0.10
```

这种三分策略很有意思：纯粹用 `[MASK]` 替换的话，模型会学会"不是 MASK 的 token 就不用管"，而随机替换和保持原词的比例强迫模型对**每一个** token 都维持良好的上下文表示。损失只在被选中的 15% 上计算，其余位置不参与梯度更新。

### 3. 下句预测（NSP）

问答、自然语言推理等任务需要理解**两个句子之间的关系**。NSP 任务简单直接：

- **50% 的情况**：句子 B 确实是语料中紧跟句子 A 的下一句（标签：`IsNext`）
- **50% 的情况**：句子 B 是从语料中随机抽取的（标签：`NotNext`）

取 [CLS] token 的最终表示接一个二分类器来判断。这让模型在预训练时就学会了捕捉跨句子的语义关系。

> **注**：后续工作（RoBERTa、ALBERT）发现 NSP 任务的帮助有限，甚至可能有害，因此将其移除。这是 BERT 之后研究的一个重要方向。

### 4. Transformer 编码器

BERT 使用标准 Transformer 编码器，有两种规格：

| 模型 | 层数 (L) | 隐藏维度 (H) | 注意力头数 (A) | 参数量 |
|------|---------|------------|-------------|------|
| BERT-Base  | 12 | 768  | 12 | 1.1亿 |
| BERT-Large | 24 | 1024 | 16 | 3.4亿 |

每层的结构：

```
x = LayerNorm(x + MultiHeadSelfAttention(x))
x = LayerNorm(x + FFN(x))
```

FFN 的中间层维度是隐藏维度的 4 倍，激活函数用 **GELU**（高斯误差线性单元）而非 ReLU，在 Transformer 结构上实践效果更好。

### 5. 微调范式

BERT 的另一大亮点是微调的**极度简洁**。同一个预训练模型，加上最小的任务专属结构，就能应对各类任务：

- **句子分类**（情感分析等）：[CLS] → 线性层 → softmax
- **序列标注**（命名实体识别）：每个 token 的隐藏态 → 线性层
- **阅读理解**（SQuAD）：两个可学习向量分别预测答案的起始和结束位置
- **自然语言推理**：句对输入，[CLS] → 三分类

整个微调过程通常只需 3-4 个 epoch，学习率约 2e-5，在单卡 GPU 上几分钟到几小时即可完成。这种"一个模型走天下"的简洁性是 BERT 成功的关键因素之一。

---

## 实现说明

### 保留的内容

- 完整的 BERT 编码器（多头自注意力 + 前馈网络 + 层归一化）
- Token + 位置 + 分句三合一嵌入
- MLM 预训练目标（含 80/10/10 掩码策略）
- NSP 预训练目标
- 基于 [CLS] 的微调分类头

### 简化的内容

- 用小型玩具词表替代 WordPiece 分词器
- 使用微型模型（2 层，64 维，4 头）以便快速运行
- 未实现嵌入矩阵与输出投影的权重绑定
- 不在大语料上做真实预训练，仅演示前向传播和优化过程

### 关键代码片段

**MLM 掩码的 80/10/10 策略：**
```python
def apply_mlm_masking(tokens, vocab_size, mask_token_id, mask_prob=0.15):
    labels = tokens.clone()
    prob_matrix = torch.rand(tokens.shape)
    masked = prob_matrix < mask_prob

    # 80% → [MASK]
    replace_mask = masked & (torch.rand(tokens.shape) < 0.8)
    tokens[replace_mask] = mask_token_id

    # 10% → 随机词
    replace_rand = masked & ~replace_mask & (torch.rand(tokens.shape) < 0.5)
    tokens[replace_rand] = torch.randint(vocab_size, tokens[replace_rand].shape)

    # 剩余 10% 保持不变，但仍参与损失计算
    labels[~masked] = -100  # 不计算损失的位置标记为 -100
    return tokens, labels
```

---

## 运行方式

```bash
pip install torch numpy
python implementation.py
```

---

## 延伸思考

BERT 开创了 NLP 的**预训练-微调范式**，彻底改变了该领域的研究方式。但它也有几个值得关注的局限性：

1. **[MASK] 的训练-推理不一致**：微调和推理时不会出现 [MASK] token，预训练和实际使用存在分布偏移。
2. **NSP 的有效性存疑**：RoBERTa 的消融实验表明，去掉 NSP 反而能提升性能。
3. **计算成本高**：BERT-Large 的预训练需要 64 块 TPU 运行 4 天。
4. **单向生成能力弱**：BERT 的双向编码器设计使其不适合做文本生成，这是后来 GPT 系列崛起的主要原因。

理解这些局限性，有助于理解 RoBERTa、ALBERT、XLNet、GPT-3 等后续工作的动机。每一篇都是在 BERT 基础上针对某个具体缺陷做的针对性改进。
