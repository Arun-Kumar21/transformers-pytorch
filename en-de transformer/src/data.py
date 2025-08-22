import os
import pandas as pd
from .config import Config
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import tokenizer


class EnDePairDataset(Dataset):
    def __init__(self, en_sentences, de_sentences, tokenizer):
        self.en_sentences = en_sentences
        self.de_sentences = de_sentences

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        "return (En tensor, De tensor)"
        self.en_sentence = self.en_sentences[idx]
        self.de_sentence = self.de_sentences[idx]

        en_encoded = self.tokenizer.encode(self.en_sentence)
        de_encoded = self.tokenizer.encode(self.de_sentence)
        return torch.tensor(en_encoded, dtype=torch.long), torch.tensor(de_encoded, dtype=torch.long)

hyperparams = Config()

max_seq_len = hyperparams.max_seq_len
pad_idx = tokenizer.pad_token_id

def collate_fn(batch):
    "Return fixed size tensor"
    srcs, tgts = zip(*batch)
    batch_size = len(batch)

    src_batch = []
    tgt_batch = []

    for src in srcs:
        if len(src) < max_seq_len:
            "Padding"
            padded = torch.cat([src, torch.tensor([pad_idx] * (max_seq_len - len(src)), dtype=torch.long)])
        else:
            "Truncate"
            padded = src[:max_seq_len]
        src_batch.append(padded)

    for tgt in tgts:
        if len(tgt) < max_seq_len:
            padded = torch.cat([tgt, torch.tensor([pad_idx] * (max_seq_len - len(tgt)), dtype=torch.long)])
        else:
            padded = tgt[:max_seq_len]
        tgt_batch.append(padded)

    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)

    return src_batch, tgt_batch

data_dir = '../data/input/'

df = pd.read_csv(data_dir + '/de_en.xls') 
en_sentences = df['ENGLISH'].astype(str).tolist()
de_sentences = df['GERMAN'].astype(str).tolist()


train_ds = EnDePairDataset(en_sentences, de_sentences, tokenizer)
val_ds = EnDePairDataset(en_sentences, de_sentences, tokenizer)

print("Length of train_dataset, val_dataset:")
print(len(train_ds), len(val_ds))

print("First example:")

print(train_ds[0])
print(tokenizer.decode(train_ds[0][0]))
print(tokenizer.decode(train_ds[0][1]))


train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

# (batch_size, seq)-> (64, 2150)
len(train_loader)