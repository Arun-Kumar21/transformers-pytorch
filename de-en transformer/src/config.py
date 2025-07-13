import torch


class Config:
    def __init__(self):
        self.vocab_size = 8000
        self.n_embd = 512
        self.n_head = 4
        self.n_layer = 4
        self.dropout = 0.1
        self.max_seq_len = 100
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = 3e-4