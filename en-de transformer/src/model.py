import copy

import torch.nn as nn

from .layers import (
    MultiHeadedAttention,
    FeedForwardLayer,
    PositionalEncoding,
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    Embeddings,
)
from .utils import make_tgt_mask

pad_idx = 0


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        n_embd,
        n_head,
        n_layer,
        dropout,
        max_seq_len,
    ):
        super(Transformer, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, n_embd, dropout)
        ff = FeedForwardLayer(n_embd, dropout)
        position = PositionalEncoding(n_embd, max_seq_len)

        self.encoder = Encoder(EncoderLayer(n_embd, c(attn), c(ff), dropout), n_layer)
        self.decoder = Decoder(
            DecoderLayer(n_embd, c(attn), c(attn), c(ff), dropout), n_layer
        )
        self.src_embed = nn.Sequential(Embeddings(src_vocab_size, n_embd), c(position))
        self.tgt_embed = nn.Sequential(Embeddings(tgt_vocab_size, n_embd), c(position))
        self.generator = nn.Linear(n_embd, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Process src and tgt sequences."
        encoded_src = self.encode(src, src_mask)
        decoded_tgt = self.decode(encoded_src, src_mask, tgt, tgt_mask)

        return self.generator(decoded_tgt)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, encoder_op, src_mask, tgt, tgt_mask):
        tgt_mask = make_tgt_mask(tgt, pad_idx).to(tgt.device)
        return self.decoder(self.tgt_embed(tgt), encoder_op, src_mask, tgt_mask)
