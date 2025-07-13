import copy

import torch
import torch.nn as nn


def clones(module, N):
  "Create N identical layers."
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def make_src_mask(src, pad_idx):
    "Create a mask to hide padding tokens in the source sequence."
    return (src != pad_idx).unsqueeze(1)

def make_tgt_mask(tgt, pad_idx):
    "Create a mask to hide padding tokens in the target sequence and future tokens."
    # (B, seq_len, seq_len)
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(-2)
    tgt_seq_len = tgt.size(-1)
    subsequent_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, dtype=torch.bool, device=tgt.device))
    # (B, seq_len, seq_len) & (1, seq_len, seq_len) -> (B, seq_len, seq_len)
    return tgt_pad_mask & subsequent_mask