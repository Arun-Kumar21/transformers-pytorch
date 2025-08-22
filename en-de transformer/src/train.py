import time
import torch
import torch.nn as nn

from config import Config
from data import train_loader, val_loader
from tokenizer import tokenizer
from model import Transfomer as model
from utils import make_src_mask, make_tgt_mask

hyperparams = Config()

lr = hyperparams.lr
pad_idx = tokenizer.pad_token_id
device = hyperparams.device

# loss function
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

# lr decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

epochs = 50
model_save_path = '/kaggle/working/' + 'transfomer_En-de.pth'

train_losses = []
val_losses = []

try:
  for epoch in range(epochs):
    start_time = time.time()
    model.train()

    total_train_loss = 0

    for i, (src, tgt) in enumerate(train_loader):
      src = src.to(device)
      tgt = tgt.to(device)

      src_mask = make_src_mask(src, pad_idx).to(device)
      tgt_input = tgt[:, :-1]
      tgt_output = tgt[:, 1:]
      tgt_mask = make_tgt_mask(tgt_input, pad_idx).to(device)

      optimizer.zero_grad()
      output = model(src, tgt_input, src_mask, tgt_mask)
      loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
      loss.backward()
      optimizer.step()

      total_train_loss += loss.item()

      if i % 100 == 0:
        print(f"Batch {i} / {len(train_loader)}, training loss: {loss.item()}")

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1}, training loss: {avg_train_loss}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved after epoch {epoch+1} to {model_save_path}")

    model.eval()
    total_val_loss = 0

    with torch.no_grad():
      for i, (src, tgt) in enumerate(val_loader):
        src = src.to(device)
        tgt = tgt.to(device)

        src_mask = make_src_mask(src, pad_idx).to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask = make_tgt_mask(tgt_input, pad_idx).to(device)

        output = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
        total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}, validation loss: {avg_val_loss}")

    end_time = time.time()
    print(f"Epoch {epoch+1}, time: {end_time - start_time}")
    scheduler.step()


except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

print("Training finished.")