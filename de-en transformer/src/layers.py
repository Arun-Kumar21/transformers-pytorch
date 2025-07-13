import torch
import torch.nn as nn

import math
from .utils import clones

# LayerNorm: Applies layer normalization to stabilize and accelerate training.
class LayerNorm(nn.Module):
    "Normalize features"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# SublayerConnection: Implements residual connection followed by layer normalization.
class SublayerConnection(nn.Module):
    "Residual connection followed by layer norm"

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))

# attention: Computes scaled dot-product attention for input queries, keys, and values.
def attention(q, k, v, mask=None, dropout=None):
    "Compute Scaled Dot Product Attention (Attention score)"
    dim_k = k.size(-1)
    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dim_k)
    if mask is not None:
        score = score.masked_fill(mask == 0, -1e9)
    att_w = score.softmax(dim=-1)
    if dropout is not None:
        att_w = dropout(att_w)

    return torch.matmul(att_w, v), att_w

# MultiHeadedAttention: Allows the model to jointly attend to information from different representation subspaces.
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_embd, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert n_embd % n_head == 0, "can't divide n_embd by n_head"
        self.n_head = n_head
        self.n_embd = n_embd
        self.dim_k = n_embd // n_head
        self.Ws = clones(nn.Linear(n_embd, n_embd), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = q.size(0)

        Q = self.Ws[0](q).view(n_batches, -1, self.n_head, self.dim_k).transpose(1, 2)
        K = self.Ws[1](k).view(n_batches, -1, self.n_head, self.dim_k).transpose(1, 2)
        V = self.Ws[2](v).view(n_batches, -1, self.n_head, self.dim_k).transpose(1, 2)

        x, self.attn = attention(Q, K, V, mask=mask, dropout=self.dropout)

        "Concatenating all heads"
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.n_head * self.dim_k)

        return self.Ws[-1](x)

# FeedForwardLayer: Position-wise feed-forward network applied to each position separately.
class FeedForwardLayer(nn.Module):
    def __init__(self, n_embd, dropout):
        super(FeedForwardLayer, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)

# Embeddings: Converts input token indices to dense vector representations.
class Embeddings(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.scale = n_embd**0.5

    def forward(self, x):
        return self.embedding(x) * self.scale

# PositionalEncoding: Adds positional information to token embeddings to enable order awareness.
class PositionalEncoding(nn.Module):
    def __init__(self, n_embd, max_len=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, n_embd)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_embd, 2) * -(math.log(10000.0) / n_embd)
        )  # e ^ [-ln(10000) * (2i/ n_embd)]
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

# EncoderLayer: Single encoder block with self-attention and feed-forward sublayers.
class EncoderLayer(nn.Module):
    "Consist of multi-head attention and feed forward"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.add_and_norm = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.add_and_norm[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.add_and_norm[1](x, self.feed_forward)
        return x

# DecoderLayer: Single decoder block with masked self-attention, encoder-decoder attention, and feed-forward sublayers.
class DecoderLayer(nn.Module):
    "Consist of multi-head attn, src_attn, feed forward"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.add_and_norm = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, encoder_op, src_mask, tgt_mask):
        x = self.add_and_norm[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.add_and_norm[1](
            x, lambda x: self.src_attn(x, encoder_op, encoder_op, src_mask)
        )
        x = self.add_and_norm[2](x, self.feed_forward)
        return x

# Encoder: Stacks multiple encoder layers and applies final layer normalization.
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# Decoder: Stacks multiple decoder layers and applies final layer normalization.
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, encoder_op, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_op, src_mask, tgt_mask)
        return self.norm(x)
