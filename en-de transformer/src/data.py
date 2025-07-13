import os
from .config import Config
import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm


class DeEnPairDataset(Dataset):
    def __init__(self, en_file, de_file, sp_en, sp_de):
        "Read file from drive"
        with open(en_file, "r") as f:
            self.en_sentences = [line.strip() for line in f.readlines()]
        with open(de_file, "r") as f:
            self.de_sentences = [line.strip() for line in f.readlines()]
        self.sp_en = sp_en
        self.sp_de = sp_de

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        "return (De tensor, En tensor)"
        en_encoded = self.sp_en.encode(self.en_sentences[idx], out_type=int)
        de_encoded = self.sp_de.encode(self.de_sentences[idx], out_type=int)
        return torch.tensor(en_encoded, dtype=torch.long), torch.tensor(
            de_encoded, dtype=torch.long
        )


hyperparams = Config()
pad_idx = 0


def collate_fn(batch):
    "Return fixed size tensor"
    srcs, tgts = zip(*batch)
    batch_size = len(batch)

    src_batch = []
    tgt_batch = []

    for src in srcs:
        if len(src) < hyperparams.max_seq_len:
            "Padding"
            padded = torch.cat(
                [
                    src,
                    torch.tensor(
                        [pad_idx] * (hyperparams.max_seq_len - len(src)),
                        dtype=torch.long,
                    ),
                ]
            )
        else:
            "Truncate"
            padded = src[: hyperparams.max_seq_len]
        src_batch.append(padded)

    for tgt in tgts:
        if len(tgt) < hyperparams.max_seq_len:
            padded = torch.cat(
                [
                    tgt,
                    torch.tensor(
                        [pad_idx] * (hyperparams.max_seq_len - len(tgt)),
                        dtype=torch.long,
                    ),
                ]
            )
        else:
            padded = tgt[: hyperparams.max_seq_len]
        tgt_batch.append(padded)

    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)

    return src_batch, tgt_batch


data_dir = "/content/drive/My Drive/data/"
spm_dir = "/content/drive/My Drive/data/spm/"

sp_en = spm.SentencePieceProcessor()
sp_en.load(f"{spm_dir}en.model")

sp_de = spm.SentencePieceProcessor()
sp_de.load(f"{spm_dir}de.model")


train_ds = DeEnPairDataset(
    os.path.join(data_dir, "train.en"), os.path.join(data_dir, "train.de"), sp_en, sp_de
)
val_ds = DeEnPairDataset(
    os.path.join(data_dir, "valid.en"), os.path.join(data_dir, "valid.de"), sp_en, sp_de
)

print(len(train_ds), len(val_ds))

print(train_ds[0])

sp_en.decode(train_ds[0][0].tolist()), sp_de.decode(train_ds[0][1].tolist())

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

# (batch_size, seq)-> (32, 625)
len(train_loader)
