import time

import torch
import torch.nn as nn

from .model import Transformer
from .config import Config
from .utils import make_src_mask, make_tgt_mask

from .data import train_loader, val_loader

hyperparams = Config()

model = Transformer(
    src_vocab_size=hyperparams.vocab_size,
    tgt_vocab_size=hyperparams.vocab_size,
    n_embd=hyperparams.n_embd,
    n_head=hyperparams.n_head,
    n_layer=hyperparams.n_layer,
    dropout=hyperparams.dropout,
    max_seq_len=hyperparams.max_seq_len,
)

pad_idx = 0

# loss function
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=hyperparams.lr, betas=(0.9, 0.98), eps=1e-9
)

# lr decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

epochs = 10
model_save_path = "/content/drive/My Drive/transformer_En-De.pth"

train_losses = []
val_losses = []

try:
    for epoch in range(epochs):
        start_time = time.time()
        model.train()

        total_train_loss = 0

        for i, (src, tgt) in enumerate(train_loader):
            src = src.to(hyperparams.device)
            tgt = tgt.to(hyperparams.device)

            src_mask = make_src_mask(src, pad_idx).to(hyperparams.device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = make_tgt_mask(tgt_input, pad_idx).to(hyperparams.device)

            optimizer.zero_grad()
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt_output.contiguous().view(-1),
            )
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if i % 100 == 0:
                print(f"Batch {i} / {len(train_loader)}, training loss: {loss.item()}")

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}, training loss: {avg_train_loss}")

        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved after epoch {epoch + 1} to {model_save_path}")

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for i, (src, tgt) in enumerate(val_loader):
                src = src.to(hyperparams.device)
                tgt = tgt.to(hyperparams.device)

                src_mask = make_src_mask(src, pad_idx).to(hyperparams.device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                tgt_mask = make_tgt_mask(tgt_input, pad_idx).to(hyperparams.device)

                output = model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt_output.contiguous().view(-1),
                )
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}, validation loss: {avg_val_loss}")

        end_time = time.time()
        print(f"Epoch {epoch + 1}, time: {end_time - start_time}")
        scheduler.step()


except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

print("Training finished.")
